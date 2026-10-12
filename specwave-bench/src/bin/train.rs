//! Fits SpecWave's predictors with the real model, and writes the guesses they make for test queries.
//!
//! ```text
//! train --model arctic [--texts FILE] [--layers 0,3,6,8,9,10] [--lambdas 0.001,0.01,0.1] [--split 0.85]
//!       [--max-tokens N] [--models models] [--out results/train.jsonl]
//!       [--queries FILE --guesses DIR --label NAME [--compare NAME] [--embeddings FILE]]
//! ```
//!
//! The model runs on the processor (CPU). With `--texts` it reads every text and fits one predictor per layer of
//! `--layers`, then saves them at `<models>/<model id>/predictors.bin`, where `<models>` is this repository's
//! `models` folder unless `--models` names another. Without `--texts` it loads the predictors saved there. With
//! `--queries` it then reads each query and guesses its final embedding at every layer with a predictor, the way
//! SpecWave does at its fire layer. The guesses from layer l go to `<DIR>/<NAME>.L<l>.fbin`, one row per query in the
//! order of the file, which is how `spec_coverage` names them. `--compare` names other guesses in that folder to
//! compare with row by row, eg. `openings` for `openings.L<l>.fbin`, and `--embeddings` names a `.fbin` file of the
//! queries' final embeddings, to compare the guesses and the model's own embeddings with.
//!
//! Texts and queries are JavaScript Object Notation (JSON) lines with a "text" field. `--max-tokens` cuts every text at
//! that many tokens, counting the query prompt and the special tokens. `--model` takes a short name of specwave's models
//! list, eg. `arctic` or `bge-small`, which also picks the list's layers for it, or a Hugging Face repository id found
//! in the local cache. Each run appends its result lines to `--out`, one per fitted layer and one per guessed layer.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Error, ErrorKind};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use specwave::encoder::{Backend, CandleEncoder, LayerEncoder, LayerHook, models, snapshot_dir};
use specwave::predictor::{self, Predictors};
use specwave::train::{Pairs, TrainParams, refit};
use specwave_bench::data::{Fbin, write_fbin};
use specwave_bench::results::{Json, append, provenance};

/// Every argument name.
const NAMES: [&str; 13] = [
    "model",
    "texts",
    "layers",
    "lambdas",
    "split",
    "max-tokens",
    "models",
    "out",
    "queries",
    "guesses",
    "label",
    "compare",
    "embeddings",
];

/// How often the fit reports how far it got, in texts.
const PROGRESS: usize = 1000;

fn main() {
    if let Err(e) = Settings::parse(std::env::args().skip(1)).and_then(|s| run(&s)) {
        eprintln!("train: {e}");
        std::process::exit(if e.kind() == ErrorKind::InvalidInput { 2 } else { 1 });
    }
}

/// The command line's settings.
#[derive(Debug, PartialEq)]
struct Settings {
    repo: String,
    texts: Option<PathBuf>,
    layers: Vec<usize>,
    lambdas: Vec<f64>,
    split: f64,
    max_tokens: Option<usize>,
    models: PathBuf,
    out: PathBuf,
    guess: Option<Guess>,
}

/// What `--queries` asks for.
#[derive(Debug, PartialEq)]
struct Guess {
    queries: PathBuf,
    dir: PathBuf,
    label: String,
    compare: Option<String>,
    embeddings: Option<PathBuf>,
}

impl Settings {
    /// Reads `args`, the arguments after the program's name. A name that `NAMES` doesn't list, a name given twice, a
    /// bad value, or an argument that needs another one gives `InvalidInput`.
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Settings, Error> {
        let mut values: HashMap<String, String> = HashMap::new();

        while let Some(arg) = args.next() {
            let name = arg
                .strip_prefix("--")
                .filter(|n| NAMES.contains(n))
                .ok_or_else(|| invalid(format!("unknown argument {arg}")))?;
            let value = args.next().ok_or_else(|| invalid(format!("--{name} needs a value")))?;
            if values.insert(name.to_string(), value).is_some() {
                return Err(invalid(format!("--{name} is given twice")));
            }
        }

        let get = |name: &str| values.get(name).map(String::as_str);
        let model = get("model").ok_or_else(|| invalid("--model is required".to_string()))?;
        let listed = models::find(model)?;
        let repo = listed.map_or(model, |m| m.repo.as_str()).to_string();
        let layers = match (get("layers"), listed) {
            (Some(v), _) => list(v, "layers")?,
            (None, Some(listed)) => listed.layers.clone(),
            (None, None) => return Err(invalid(format!("--layers is required for {model}"))),
        };
        let lambdas: Vec<f64> = get("lambdas").map_or(Ok(vec![1e-3, 1e-2, 1e-1]), |v| list(v, "lambdas"))?;
        let split: f64 = get("split").map_or(Ok(0.85), |v| one(v, "split"))?;
        let max_tokens = get("max-tokens").map(|v| one(v, "max-tokens")).transpose()?;
        let path = |name: &str, default: &str| match get(name) {
            Some(path) => PathBuf::from(path),
            None => Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(default),
        };
        let given = |name: &str| get(name).map(PathBuf::from);

        if layers.is_empty() || lambdas.iter().any(|&l| !(l.is_finite() && l > 0.0)) {
            return Err(invalid(
                "--layers names at least one layer, and every penalty of --lambdas is above 0".to_string(),
            ));
        }

        if !(split > 0.0 && split < 1.0) {
            return Err(invalid("--split is a share above 0 and below 1".to_string()));
        }

        let guess = match (get("queries"), get("guesses"), get("label")) {
            (Some(queries), Some(dir), Some(label)) => Some(Guess {
                queries: PathBuf::from(queries),
                dir: PathBuf::from(dir),
                label: label.to_string(),
                compare: get("compare").map(str::to_string),
                embeddings: given("embeddings"),
            }),
            (None, None, None) if get("compare").is_none() && get("embeddings").is_none() => None,
            _ => {
                return Err(invalid(
                    "--queries, --guesses and --label go together, and --compare and --embeddings need them"
                        .to_string(),
                ));
            }
        };

        Ok(Settings {
            repo,
            texts: given("texts"),
            layers,
            lambdas,
            split,
            max_tokens,
            models: path("models", "models"),
            out: path("out", "results/train.jsonl"),
            guess,
        })
    }
}

fn run(s: &Settings) -> Result<(), Error> {
    let run = run_id();
    let mut encoder = CandleEncoder::open(&snapshot_dir(&s.repo)?, Backend::Cpu)?;

    if let Some(max) = s.max_tokens {
        encoder.set_max_tokens(max)?;
    }

    let path = predictor::path(&s.models, &s.repo)?;
    let predictors = match &s.texts {
        Some(texts) => fit(s, &mut encoder, texts, &path, &run)?,
        None => Predictors::load(&path)?,
    };

    match &s.guess {
        Some(g) => guess(s, &mut encoder, &predictors, g, &run),
        None => Ok(()),
    }
}

/// Fits the predictors on the texts in `texts_path`, saves them at `path` and returns them.
fn fit(
    s: &Settings,
    encoder: &mut CandleEncoder,
    texts_path: &Path,
    path: &Path,
    run: &str,
) -> Result<Predictors, Error> {
    if s.layers.iter().any(|&l| l > encoder.layers()) {
        return Err(invalid(format!("{} has only {} layers", s.repo, encoder.layers())));
    }

    let texts = read_texts(texts_path)?;
    let mut pairs = Pairs::new(encoder.dim(), &s.layers)?;
    let (mut tokens, mut at_max) = (0, 0);
    let started = Instant::now();

    for (i, text) in texts.iter().enumerate() {
        let count = encoder.tokens(text)?;
        tokens += count;
        at_max += usize::from(count >= encoder.max_tokens());
        pairs.encode(encoder, text)?;
        if (i + 1) % PROGRESS == 0 {
            eprintln!(
                "{}: {} of {} texts in {:.0} s",
                s.repo,
                i + 1,
                texts.len(),
                started.elapsed().as_secs_f64()
            );
        }
    }

    let encode_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let params = TrainParams {
        layers: s.layers.clone(),
        lambdas: s.lambdas.clone(),
        split: s.split,
        threads: 0,
    };
    let fit = refit(&s.repo, &pairs, &params)?;
    let fit_seconds = started.elapsed().as_secs_f64();
    fit.predictors.save(path)?;
    let mut lines = Vec::with_capacity(fit.choices.len());

    for choice in &fit.choices {
        eprintln!(
            "{} L{}: penalty {}, held-out cosines {:?}",
            s.repo, choice.layer, choice.lambda, choice.held_out
        );
        let json = Json::new()
            .str("bench", "train")
            .str("run", run)
            .str("model", &s.repo)
            .int("layer", choice.layer as u64)
            .num("lambda", choice.lambda)
            .nums("lambdas", &s.lambdas)
            .nums("held_out", &choice.held_out)
            .num("split", s.split)
            .int("pairs", pairs.len() as u64)
            .str("texts", &texts_path.display().to_string())
            .int("max_tokens", encoder.max_tokens() as u64)
            .num("tokens_mean", tokens as f64 / texts.len().max(1) as f64)
            .int("at_max_tokens", at_max as u64)
            .num("encode_seconds", encode_seconds)
            .num("fit_seconds", fit_seconds)
            .str("predictors", &path.display().to_string());
        lines.push(provenance(json).line());
    }

    append(&s.out, &lines)?;
    eprintln!("{}: {} texts, saved {}", s.repo, texts.len(), path.display());
    Ok(fit.predictors)
}

/// Guesses every query's final embedding at each layer with a predictor, writes the guesses and compares them.
fn guess(
    s: &Settings,
    encoder: &mut CandleEncoder,
    predictors: &Predictors,
    g: &Guess,
    run: &str,
) -> Result<(), Error> {
    let layers: Vec<usize> = predictors.layers().collect();
    let dim = predictors.dim();

    if dim != encoder.dim() || layers.iter().any(|&l| l > encoder.layers()) {
        return Err(invalid(format!("the predictors don't fit {}", s.repo)));
    }

    let queries = read_texts(&g.queries)?;
    let mut tables = vec![Vec::with_capacity(queries.len() * dim); layers.len()];
    let mut finals = Vec::with_capacity(queries.len() * dim);
    let mut hook = Guesser::new(predictors, &layers);

    for text in &queries {
        hook.done.iter_mut().for_each(|d| *d = false);
        let embedding = encoder.encode(text, &mut hook)?;
        if let Some(e) = hook.error.take() {
            return Err(e);
        }
        if !hook.done.iter().all(|&d| d) {
            return Err(Error::other("the encoder skipped a layer with a predictor"));
        }
        for (table, guess) in tables.iter_mut().zip(&hook.guesses) {
            table.extend_from_slice(guess);
        }
        finals.extend_from_slice(&embedding);
    }

    let embeddings = g.embeddings.as_deref().map(Fbin::open).transpose()?;
    let reference = match &embeddings {
        Some(e) if e.dim() == dim && e.len() >= queries.len() => Some(&e.values()[..finals.len()]),
        Some(_) => return Err(invalid("--embeddings doesn't hold one embedding per query".to_string())),
        None => None,
    };
    std::fs::create_dir_all(&g.dir)?;
    let mut lines = Vec::with_capacity(layers.len() + 1);

    if let Some(reference) = reference {
        let c = Cosines::of(&finals, reference, dim);
        eprintln!(
            "final embeddings: 1 - cosine {:.1e} at worst, {:.1e} on average",
            1.0 - c.min,
            1.0 - c.mean
        );
        let json = Json::new()
            .str("bench", "finals")
            .str("run", run)
            .str("model", &s.repo)
            .str("queries", &g.queries.display().to_string())
            .int("rows", queries.len() as u64)
            .int("max_tokens", encoder.max_tokens() as u64);
        lines.push(provenance(c.add(json, "cos")).line());
    }

    for (&layer, table) in layers.iter().zip(&tables) {
        let path = g.dir.join(format!("{}.L{layer}.fbin", g.label));
        write_fbin(&path, table, dim)?;
        let own = Cosines::of(table, &finals, dim);
        let mut json = Json::new()
            .str("bench", "guesses")
            .str("run", run)
            .str("model", &s.repo)
            .int("layer", layer as u64)
            .num("lambda", predictors.get(layer).map_or(f64::NAN, |r| r.penalty()))
            .str("queries", &g.queries.display().to_string())
            .int("rows", queries.len() as u64)
            .int("max_tokens", encoder.max_tokens() as u64)
            .str("path", &path.display().to_string());
        json = own.add(json, "cos_model");
        let mut shown = format!("L{layer}: cosine {:.4} with the model's embeddings", own.mean);
        if let Some(reference) = reference {
            let c = Cosines::of(table, reference, dim);
            shown += &format!(", {:.4} with --embeddings", c.mean);
            json = c.add(json, "cos_embeddings");
        }
        if let Some(name) = &g.compare {
            let other_path = g.dir.join(format!("{name}.L{layer}.fbin"));
            let other = Fbin::open(&other_path)?;
            if other.dim() != dim || other.len() < queries.len() {
                return Err(invalid(format!(
                    "{} doesn't hold a guess per query",
                    other_path.display()
                )));
            }
            let c = Cosines::of(table, &other.values()[..table.len()], dim);
            shown += &format!(", 1 - cosine with {name} {:.1e} at worst", 1.0 - c.min);
            json = c.add(json.str("compare", name), "cos_compare");
        }
        eprintln!("{shown}");
        lines.push(provenance(json).line());
    }

    append(&s.out, &lines)
}

/// A hook that guesses the final embedding at every layer with a predictor, the way SpecWave does at its fire layer.
struct Guesser<'a> {
    predictors: &'a Predictors,
    layers: &'a [usize],
    /// The latest guess from each layer, in the order of `layers`.
    guesses: Vec<Vec<f32>>,
    /// Whether each layer has made its guess for the current query.
    done: Vec<bool>,
    error: Option<Error>,
}

impl<'a> Guesser<'a> {
    fn new(predictors: &'a Predictors, layers: &'a [usize]) -> Guesser<'a> {
        Guesser {
            predictors,
            layers,
            guesses: vec![vec![0.0; predictors.dim()]; layers.len()],
            done: vec![false; layers.len()],
            error: None,
        }
    }
}

impl LayerHook for Guesser<'_> {
    fn wants(&self, layer: usize) -> bool {
        self.predictors.get(layer).is_some()
    }

    fn on_layer(&mut self, layer: usize, pooled: &[f32]) {
        let (Some(i), Some(ridge)) = (self.layers.iter().position(|&l| l == layer), self.predictors.get(layer)) else {
            return;
        };

        match ridge.predict(pooled, &mut self.guesses[i]) {
            Ok(()) => self.done[i] = true,
            Err(e) => self.error = Some(e),
        }
    }
}

/// The cosines between the matching rows of two tables of `dim` values per row, summed up.
#[derive(Debug, PartialEq)]
struct Cosines {
    min: f64,
    /// The 10th percentile, by nearest rank.
    p10: f64,
    mean: f64,
}

impl Cosines {
    fn of(a: &[f32], b: &[f32], dim: usize) -> Cosines {
        let mut all: Vec<f64> = a
            .chunks_exact(dim)
            .zip(b.chunks_exact(dim))
            .map(|(x, y)| cosine(x, y))
            .collect();
        all.sort_by(f64::total_cmp);
        let n = all.len();
        Cosines {
            min: all.first().copied().unwrap_or(f64::NAN),
            p10: all.get(n.div_ceil(10).max(1) - 1).copied().unwrap_or(f64::NAN),
            mean: all.iter().sum::<f64>() / n as f64,
        }
    }

    /// Adds the numbers to `json` as `<name>_min`, `<name>_p10` and `<name>_mean`.
    fn add(&self, json: Json, name: &str) -> Json {
        json.num(&format!("{name}_min"), self.min)
            .num(&format!("{name}_p10"), self.p10)
            .num(&format!("{name}_mean"), self.mean)
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(&p, &q)| f64::from(p) * f64::from(q)).sum::<f64>();
    dot(a, b) / (dot(a, a) * dot(b, b)).sqrt()
}

/// The "text" field of every line of the JSON lines file at `path`.
fn read_texts(path: &Path) -> Result<Vec<String>, Error> {
    let file = File::open(path).map_err(|e| Error::new(e.kind(), format!("{}: {e}", path.display())))?;
    let mut texts = Vec::new();

    for (i, line) in BufReader::new(file).lines().enumerate() {
        let value: serde_json::Value = serde_json::from_str(&line?).map_err(|e| {
            Error::new(
                ErrorKind::InvalidData,
                format!("{} line {}: {e}", path.display(), i + 1),
            )
        })?;
        let text = value["text"].as_str().ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidData,
                format!("{} line {} has no text", path.display(), i + 1),
            )
        })?;
        texts.push(text.to_string());
    }

    Ok(texts)
}

/// An id shared by the lines of one run, from the time and the process id.
fn run_id() -> String {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    format!("{}-{}", now.as_millis(), std::process::id())
}

fn one<T: FromStr>(value: &str, name: &str) -> Result<T, Error> {
    value
        .trim()
        .parse()
        .map_err(|_| invalid(format!("--{name} has a bad value: {value}")))
}

/// The comma-separated values of `--name`, none of them empty.
fn list<T: FromStr>(value: &str, name: &str) -> Result<Vec<T>, Error> {
    value.split(',').map(|v| one(v, name)).collect()
}

fn invalid(message: String) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

#[cfg(test)]
#[path = "../../tests/unit/train.rs"]
mod tests;
