//! Makes synthetic guesses for an index whose queries have no model states, as the synthetic module describes.
//!
//! ```text
//! synthetic calibrate --index FILE --queries FILE --corpus FILE --coverage FILES [--dataset fiqa] [--fit openings]
//!           [--clustering l2] [--nprobe 9] [--rerank 29] [--layers 8,9,10] [--documents 64] [--seed 1]
//!           [--out results/synthetic.jsonl]
//! synthetic guesses --noise FILE --queries FILE --corpus FILE --dir DIR --label NAME [--documents 64] [--seed 1]
//! ```
//!
//! `calibrate` fits each layer's noise on a dataset whose guesses came from the model, eg. FiQA's index, test queries
//! and corpus. The targets come from the coverage results of the model's guesses, for every layer that passes the
//! verdict module's rule at the base nprobe and R, the number of candidates that a normal search rescores, or for each
//! of `--layers`. They are the mean cosine of the layer's guesses with the final embeddings and their posting coverage
//! at the layer's β. It appends a line per layer to `--out`. `guesses` reads those lines and writes each layer's
//! guesses of the final embeddings in `--queries`, with noise shaped like `--corpus`, as `DIR/NAME.L<layer>.fbin`, the
//! files that the `spec` binary's `--guesses` reads. `--documents` is the number of documents that one draw of the data
//! noise adds up. A query's draws depend only on `--seed` and its row.

use std::io::{Error, ErrorKind};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use coldivf::{Index, SearchParams};
use specwave_bench::data::{Fbin, write_fbin};
use specwave_bench::results::{Json, append, provenance};
use specwave_bench::synthetic::{Command, DataNoise, Draws, calibrate, fitted, shares, targets};
use specwave_bench::trials::layer_settings;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");

    if let Err(e) = Command::parse(&args, &repo).and_then(|c| run(&c)) {
        eprintln!("synthetic: {e}");
        std::process::exit(if e.kind() == ErrorKind::InvalidInput { 2 } else { 1 });
    }
}

fn run(command: &Command) -> Result<(), Error> {
    match command {
        Command::Calibrate {
            index,
            queries,
            corpus,
            coverage,
            key,
            rerank,
            layers,
            documents,
            seed,
            out,
        } => {
            let mut lines = Vec::new();
            for file in coverage {
                let text = std::fs::read_to_string(file).map_err(|e| Error::new(e.kind(), format!("{file}: {e}")))?;
                lines.extend(text.lines().map(str::to_string));
            }
            let base = SearchParams {
                k: 10,
                nprobe: key.nprobe,
                rerank: *rerank,
                reach: None,
            };
            let mut chosen = layer_settings(&lines, &key.dataset, &key.fit, &key.clustering, &base)?;
            chosen.retain(|l| layers.as_ref().is_none_or(|wanted| wanted.contains(&l.layer)));
            if chosen.is_empty() {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "no layer to fit passes in the coverage results",
                ));
            }
            let documents_file = Fbin::open(Path::new(corpus))?;
            let data = DataNoise::new(&documents_file, *documents)?;
            let embeddings = Fbin::open(Path::new(queries))?;
            let draws = Draws::new(embeddings.values(), &data, *seed)?;
            let opened = Index::open(Path::new(index))?;
            let run = run_id();
            let mut results = Vec::with_capacity(chosen.len());
            println!(
                "{:>5} {:>6} {:>10} {:>10} {:>7} {:>8} {:>8} {:>9} {:>9} {:>9}",
                "layer", "beta", "cos real", "cov real", "share", "scale", "cos", "coverage", "isotropic", "shaped"
            );
            for l in &chosen {
                let Some(target) = targets(&lines, key, l.layer, l.beta) else {
                    eprintln!("synthetic: the coverage results have no targets for layer {}", l.layer);
                    continue;
                };
                let fit = calibrate(
                    &opened,
                    embeddings.values(),
                    &draws,
                    target,
                    key.nprobe,
                    l.beta,
                    &shares(),
                )?;
                println!(
                    "{:>5} {:>6} {:>10.4} {:>10.4} {:>7.2} {:>8.4} {:>8.4} {:>9.4} {:>9.4} {:>9.4}",
                    l.layer,
                    l.beta,
                    target.0,
                    target.1,
                    fit.noise.share,
                    fit.noise.scale,
                    fit.cosine,
                    fit.coverage,
                    fit.isotropic,
                    fit.shaped
                );
                let json = Json::new()
                    .str("bench", "synthetic")
                    .str("record", "noise")
                    .str("run", &run)
                    .str("dataset", &key.dataset)
                    .str("fit", &key.fit)
                    .str("clustering", &key.clustering)
                    .int("nprobe", key.nprobe as u64)
                    .str("r_label", &key.r_label)
                    .int("layer", l.layer as u64)
                    .num("beta", l.beta)
                    .num("cos_target", target.0)
                    .num("coverage_target", target.1)
                    .num("share", fit.noise.share)
                    .num("scale", fit.noise.scale)
                    .num("cos", fit.cosine)
                    .num("coverage", fit.coverage)
                    .num("coverage_isotropic", fit.isotropic)
                    .num("coverage_shaped", fit.shaped)
                    .int("queries", draws.len() as u64)
                    .int("documents", *documents as u64)
                    .int("seed", *seed)
                    .str("index", index)
                    .str("queries", queries)
                    .str("corpus", corpus);
                results.push(provenance(json).line());
            }
            append(Path::new(out), &results)
        }
        Command::Guesses {
            noise,
            queries,
            corpus,
            dir,
            label,
            documents,
            seed,
        } => {
            let text = std::fs::read_to_string(noise).map_err(|e| Error::new(e.kind(), format!("{noise}: {e}")))?;
            let lines: Vec<String> = text.lines().map(str::to_string).collect();
            let layers = fitted(&lines);
            if layers.is_empty() {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!("{noise} holds no fitted noise"),
                ));
            }
            let documents_file = Fbin::open(Path::new(corpus))?;
            let data = DataNoise::new(&documents_file, *documents)?;
            let embeddings = Fbin::open(Path::new(queries))?;
            let draws = Draws::new(embeddings.values(), &data, *seed)?;
            let dim = embeddings.dim();
            std::fs::create_dir_all(dir)?;
            for (layer, noise) in layers {
                let mut guesses = vec![0f32; embeddings.len() * dim];
                for (q, out) in guesses.chunks_exact_mut(dim).enumerate() {
                    draws.guess(q, embeddings.row(q), &noise, out);
                }
                let path = Path::new(dir).join(format!("{label}.L{layer}.fbin"));
                write_fbin(&path, &guesses, dim)?;
                eprintln!(
                    "synthetic: L{layer}, share {:.2}, scale {:.4}, mean cosine {:.4}, wrote {}",
                    noise.share,
                    noise.scale,
                    draws.mean_cosine(&noise),
                    path.display()
                );
            }
            Ok(())
        }
    }
}

/// An id shared by the lines of one run, from the time and the process id.
fn run_id() -> String {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    format!("{}-{}", now.as_millis(), std::process::id())
}
