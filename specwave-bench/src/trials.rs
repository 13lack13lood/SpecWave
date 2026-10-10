//! The end-to-end cold trials of the `spec` binary, which time SpecWave from a query's text to its hits. Each trial is
//! a fresh process, which
//! 1. loads the model, from a file dropped from the system's cache first in the model-cold variant,
//! 2. runs the model alone on at least 20 texts, to warm it up,
//! 3. times each layer with 20 more calls, loads the wave model of the index's disk volume and picks the fire layer,
//! 4. opens the index, drops it from the system's cache and checks that no page of it stays,
//! 5. waits for the text in its start state, with the index's warm-up started its lead before the text if it has one,
//! 6. times one query, or the first few, and appends a line for each, with the coverages of each guess measured once
//!    the last query is done.
//!
//! The loader reads the whole weights file into memory, and the model runs at least 21 times before the first query.
//! So the model-cold variant changes only the load, and its lines differ in the load's time, its disk reads and the
//! weights' residency, not in the queries.
//!
//! Speculation on and off alternate over the same queries. Each query runs once with each in every setting, i.e. a
//! start state and a warm-up lead, and the two runs of a query sit a pass of trials apart, since pages read soon after
//! a first read come back faster even when none shows as cached.
//!
//! Two kinds of guesses can drive the reads.
//! - **The model's own,** from the predictors of each layer, with the model's final embeddings, eg. on FiQA.
//! - **Guesses from files,** one per query and layer, with the final embeddings from a file too, eg. Cohere's 1M
//!   embeddings with synthetic guesses. The model still runs on a text for each query, so the reads still hide under
//!   its work, but its states and its output are replaced. Predictors that pass a summary's first d values through
//!   turn each layer's guess file into the guess.

use std::collections::BTreeMap;
use std::fmt::Write;
use std::fs::File;
use std::io::{BufRead, BufReader, Error, ErrorKind};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{Duration, Instant};

use coldivf::{Hit, Index, Limits, SearchCtx, SearchParams, Trace, data_file};
use specwave::encoder::{LayerEncoder, LayerHook};
use specwave::predictor::{self, Predictors, Ridge};
use specwave::scheduler::{
    Layer0, LayerProfile, LayerSettings, ReadTimes, Schedule, Volume, WaveModel, choose, support_dir,
};
use specwave::speculate::Worker;
use specwave::{Report, SpecWave};

use crate::coldcache::{Mapped, Usage, make_cold};
use crate::data::{Fbin, Ibin};
use crate::results::{Json, provenance};
use crate::schedule::{Base, settings};
use crate::synthetic::{coverage_of, rerank_coverage_of};
use crate::verdict::{Rule, read};

/// How long a trial watches a file it dropped from the cache, for anything that reads it back.
pub const COLD_WAIT: Duration = Duration::from_millis(500);

/// How long the `prewarm` start state keeps the core busy, so it runs at full clock when the text arrives.
pub const PREWARM: Duration = Duration::from_millis(150);

/// How long the `idle` start state sleeps, so the cores go idle before the text arrives.
pub const IDLE: Duration = Duration::from_millis(50);

/// The searches that measure an index's read sizes.
pub const SIZE_SAMPLES: usize = 32;

/// The machine's state when the text arrives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Start {
    /// After a sleep, so the query starts on idle cores, which run their first few milliseconds slowly.
    Idle,
    /// After the core was kept busy, so it runs at full clock.
    Prewarm,
}

impl Start {
    pub fn name(self) -> &'static str {
        match self {
            Start::Idle => "idle",
            Start::Prewarm => "prewarm",
        }
    }
}

impl FromStr for Start {
    type Err = Error;

    fn from_str(s: &str) -> Result<Start, Error> {
        match s {
            "idle" => Ok(Start::Idle),
            "prewarm" => Ok(Start::Prewarm),
            _ => Err(invalid(format!("a start state is idle or prewarm, not {s}"))),
        }
    }
}

/// What a trial's query meets, besides speculation on or off.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Setting {
    pub start: Start,
    /// How long before the text the index's warm-up starts, if it does.
    pub warm_ms: Option<u64>,
}

/// One trial.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Trial {
    pub number: usize,
    /// The query of its first search. The later ones of a trial take the queries after it.
    pub query: usize,
    pub on: bool,
    pub setting: Setting,
}

/// Every trial of a run, in order. There are two passes over the queries `first` to `first + count`, wrapping around
/// `queries`, and each pass runs the queries in every setting in turn. Within a pass speculation is on for every other
/// query, and the second pass turns it the other way, so each query runs once with each in every setting. No queries
/// or no settings give no trials.
pub fn plan(first: usize, count: usize, queries: usize, settings: &[Setting]) -> Vec<Trial> {
    let mut trials = Vec::with_capacity(2 * count * settings.len());

    if queries == 0 {
        return trials;
    }

    for pass in 0..2 {
        for &setting in settings {
            for i in 0..count {
                trials.push(Trial {
                    number: trials.len(),
                    query: (first + i) % queries,
                    on: (i + pass).is_multiple_of(2),
                    setting,
                });
            }
        }
    }

    trials
}

/// What one index's reads take, measured on a few searches, for the scheduler's rule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sizes {
    /// The read of the cluster tables in milliseconds (ms), from a cold start.
    pub tables_ms: f64,
    /// A route, which scores every cluster center, in ms.
    pub route_ms: f64,
    /// The bytes read per cluster, and per rescored candidate, rounded out to read pages.
    pub cluster_bytes: f64,
    pub row_bytes: f64,
    /// The scan time per cluster, in ms.
    pub scan_ms: f64,
}

impl Sizes {
    /// The sizes as the command line takes them, `tables_ms:route_ms:cluster_bytes:row_bytes:scan_ms`.
    pub fn arg(&self) -> String {
        format!(
            "{}:{}:{}:{}:{}",
            self.tables_ms, self.route_ms, self.cluster_bytes, self.row_bytes, self.scan_ms
        )
    }
}

impl FromStr for Sizes {
    type Err = Error;

    fn from_str(s: &str) -> Result<Sizes, Error> {
        let values: Vec<f64> = s.split(':').map(|v| one(v, "reads")).collect::<Result<_, _>>()?;

        match values[..] {
            [tables_ms, route_ms, cluster_bytes, row_bytes, scan_ms]
                if values.iter().all(|v| v.is_finite() && *v >= 0.0) =>
            {
                Ok(Sizes {
                    tables_ms,
                    route_ms,
                    cluster_bytes,
                    row_bytes,
                    scan_ms,
                })
            }
            _ => Err(invalid(format!(
                "--reads takes five values of 0 or more, tables_ms:route_ms:cluster_bytes:row_bytes:scan_ms, not {s}"
            ))),
        }
    }
}

/// Read times from the wave model of the index's volume and the index's measured sizes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IndexReads {
    pub wave: WaveModel,
    pub sizes: Sizes,
}

impl ReadTimes for IndexReads {
    fn tables_ms(&self) -> f64 {
        self.sizes.tables_ms
    }

    fn chain_ms(&self, clusters: usize, rows: usize) -> f64 {
        let (clusters, s) = (clusters as f64, &self.sizes);
        s.route_ms
            + self.wave.read_ms(clusters * s.cluster_bytes)
            + clusters * s.scan_ms
            + self.wave.read_ms(rows as f64 * s.row_bytes)
    }
}

/// The read sizes of the index at `path`, from its first search after it was dropped from the cache, which reads the
/// cluster tables, and from `samples` searches of the rows of `queries` with `base`, and the name of the read backend
/// that the searches went through, as their trace gives it. It drops the index from the cache again at the end.
/// Searches whose trace holds no read, eg. of a warm index, give `InvalidData`.
pub fn measure_sizes(
    path: &Path,
    queries: &Fbin,
    base: &SearchParams,
    samples: usize,
) -> Result<(Sizes, &'static str), Error> {
    let map = Mapped::open(&data_file(path)?)?;
    make_cold(&[&map], COLD_WAIT)?;
    let index = Index::open(path)?;
    let limits = Limits {
        max_k: base.k,
        max_nprobe: base.nprobe,
        max_rerank: base.rerank,
    };
    let mut ctx = SearchCtx::new(limits);
    let n = queries.len().max(1);
    index.search(queries.row(0), base, &mut ctx)?;
    let tables_ms = ms(ctx.trace().tables);
    let (mut clusters, mut cluster_bytes, mut rows, mut row_bytes) = (0, 0, 0, 0);
    let (mut route_ms, mut scan_ms) = (0.0, 0.0);

    for i in 0..samples {
        index.search(queries.row((1 + i) % n), base, &mut ctx)?;
        let trace = ctx.trace();
        (clusters, cluster_bytes) = (clusters + base.nprobe, cluster_bytes + trace.cluster_bytes);
        (rows, row_bytes) = (rows + base.rerank, row_bytes + trace.row_bytes);
        (route_ms, scan_ms) = (route_ms + ms(trace.route), scan_ms + ms(trace.scan));
    }

    let backend = ctx.trace().read_path.backend;
    drop(index);
    make_cold(&[&map], COLD_WAIT)?;

    if cluster_bytes == 0 || row_bytes == 0 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "the searches read nothing, so the index can't be measured",
        ));
    }

    let sizes = Sizes {
        tables_ms,
        route_ms: route_ms / samples as f64,
        cluster_bytes: cluster_bytes as f64 / clusters as f64,
        row_bytes: row_bytes as f64 / rows as f64,
        scan_ms: scan_ms / clusters as f64,
    };
    Ok((sizes, backend))
}

/// Whether a fetch through the read backend `backend`, named as a search's trace names it, returns while its reads
/// run, so that a wave from layer 0 reads ahead without holding SpecWave's worker. The thread pool's fetch waits for
/// every read, and the portable backend reads only once a scan asks for the clusters.
pub fn reads_ahead(backend: &str) -> bool {
    matches!(backend, "rdadvise" | "io_uring" | "fadvise")
}

/// One step of a trial's wait for its text, once the index is out of the cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wait {
    /// A sleep, so the cores go idle.
    Sleep(Duration),
    /// A spin, which keeps the core busy.
    Spin(Duration),
    /// The start of the index's warm-up, which runs on a thread of its own.
    WarmUp,
}

/// The steps of the wait for the text in `setting`. The start state fills the end of the wait for its whole length,
/// and a warm-up starts exactly its lead before the text. A lead shorter than the start state starts inside it, and a
/// longer one starts first, with a sleep before the start state.
pub fn before_text(setting: &Setting) -> Vec<Wait> {
    let (state, length): (fn(Duration) -> Wait, Duration) = match setting.start {
        Start::Idle => (Wait::Sleep, IDLE),
        Start::Prewarm => (Wait::Spin, PREWARM),
    };
    let steps = match setting.warm_ms.map(Duration::from_millis) {
        None => vec![state(length)],
        Some(lead) if lead <= length => vec![state(length - lead), Wait::WarmUp, state(lead)],
        Some(lead) => vec![Wait::WarmUp, Wait::Sleep(lead - length), state(length)],
    };
    steps
        .into_iter()
        .filter(|step| !matches!(step, Wait::Sleep(d) | Wait::Spin(d) if d.is_zero()))
        .collect()
}

/// Waits for the text in `setting`, through the steps of `before_text`, and returns when the index's warm-up started,
/// if it did.
pub fn wait_for_text(index: &Index, setting: &Setting) -> Result<Option<Instant>, Error> {
    let mut warmed = None;

    for step in before_text(setting) {
        match step {
            Wait::Sleep(d) => std::thread::sleep(d),
            Wait::Spin(d) => {
                let start = Instant::now();
                while start.elapsed() < d {
                    std::hint::spin_loop();
                }
            }
            Wait::WarmUp => {
                warmed = Some(Instant::now());
                index.warm_up()?;
            }
        }
    }

    Ok(warmed)
}

/// Each layer that may guess, with its β and m, for searches with `base` on the fit `fit` of `dataset` with
/// `clustering`, from the coverage results `lines` under the verdict module's default rule. The rerank coverage is the
/// one of a normal search's rows.
pub fn layer_settings(
    lines: &[String],
    dataset: &str,
    fit: &str,
    clustering: &str,
    base: &SearchParams,
) -> Result<Vec<LayerSettings>, Error> {
    let cells = read(lines)?;
    let base = Base {
        dataset: dataset.to_string(),
        fit: fit.to_string(),
        clustering: clustering.to_string(),
        nprobe: base.nprobe,
        r_label: base.rerank.to_string(),
        rerank: base.rerank,
    };
    Ok(settings(&cells, &Rule::default(), 1, &base))
}

/// Layer 0's wave for searches with `base`, from the coverage results `lines`, i.e. the mean posting coverage of layer
/// 0's guesses of `fit` on `dataset` with `clustering` at β = 1 over every seed and multiple. Without such cells there
/// is none. The guesses have no calibrated β, so the wave reads with β = 1.
pub fn layer0(lines: &[String], dataset: &str, fit: &str, clustering: &str, base: &SearchParams) -> Option<Layer0> {
    let coverages: Vec<f64> = lines
        .iter()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|v| {
            v["record"] == "grid"
                && v["dataset"] == dataset
                && v["clustering"] == clustering
                && v["layer"].as_u64() == Some(0)
                && v["guess"].as_str().is_some_and(|g| crate::verdict::fit_of(g) == fit)
                && v["nprobe"].as_u64() == Some(base.nprobe as u64)
                && v["r_label"] == base.rerank.to_string().as_str()
                && v["beta"].as_f64() == Some(1.0)
        })
        .filter_map(|v| v["posting_coverage"].as_f64())
        .collect();

    match coverages.is_empty() {
        true => None,
        false => Some(Layer0 {
            beta: 0.0,
            coverage: coverages.iter().sum::<f64>() / coverages.len() as f64,
        }),
    }
}

/// Layer settings as the command line takes them, `layer:beta:mult` separated by commas, eg. `9:0:2,10:0.25:2`.
pub fn layers_arg(layers: &[LayerSettings]) -> String {
    let parts: Vec<String> = layers
        .iter()
        .map(|s| format!("{}:{}:{}", s.layer, s.beta, s.rerank_mult))
        .collect();
    parts.join(",")
}

/// The layer settings of `--layers`.
pub fn parse_layers(value: &str) -> Result<Vec<LayerSettings>, Error> {
    value
        .split(',')
        .map(|part| {
            let fields: Vec<&str> = part.split(':').collect();
            let [layer, beta, mult] = fields[..] else {
                return Err(invalid(format!("--layers takes layer:beta:mult, not {part}")));
            };
            Ok(LayerSettings {
                layer: one(layer, "layers")?,
                beta: one(beta, "layers")?,
                rerank_mult: one(mult, "layers")?,
            })
        })
        .collect()
}

/// Predictors at `layers` that give back the first `dim` values of a summary, scaled to length 1, so a summary that
/// starts with a guess gives that guess.
pub fn pass_through(model: &str, dim: usize, layers: &[usize]) -> Result<Predictors, Error> {
    let ridges = layers.iter().map(|&layer| {
        let mut weights = vec![0.0; 2 * dim * dim];
        (0..dim).for_each(|i| weights[i * 2 * dim + i] = 1.0);
        Ridge::new(layer, dim, weights, vec![0.0; dim], 0.0)
    });
    Predictors::new(model, ridges.collect::<Result<_, _>>()?)
}

/// The final embeddings and each layer's guesses of the queries, one row per query.
pub struct Guessed {
    pub embeddings: Fbin,
    /// By layer.
    pub guesses: Vec<(usize, Fbin)>,
}

impl Guessed {
    /// The guesses of `layers` in the folder `dir`, named `<label>.L<layer>.fbin`, for the final embeddings in
    /// `embeddings`. Files with another size of row or fewer rows give `InvalidData`.
    pub fn open(embeddings: &Path, dir: &Path, label: &str, layers: &[usize]) -> Result<Guessed, Error> {
        let embeddings = Fbin::open(embeddings)?;
        let mut guesses = Vec::with_capacity(layers.len());

        for &layer in layers {
            let path = dir.join(format!("{label}.L{layer}.fbin"));
            let file = Fbin::open(&path)?;
            if file.dim() != embeddings.dim() || file.len() < embeddings.len() {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!("{} doesn't hold a guess for each query", path.display()),
                ));
            }
            guesses.push((layer, file));
        }

        Ok(Guessed { embeddings, guesses })
    }

    /// The guess of `query` at `layer`, if a file holds that layer's guesses.
    pub fn guess(&self, layer: usize, query: usize) -> Option<&[f32]> {
        self.guesses
            .iter()
            .find(|(l, _)| *l == layer)
            .map(|(_, file)| file.row(query))
    }
}

/// An encoder that runs `inner` on each text. With guesses from files, the hook gets the query's guess at each layer
/// instead of the layer's state, as a summary of the guess followed by d zeros, and the query's final embedding comes
/// from the file too. The query is the one of the last `set_query`.
pub struct Relay<E> {
    pub inner: E,
    guessed: Option<Guessed>,
    query: usize,
    /// The summary that the hook gets, whose last d values stay 0.
    summary: Vec<f32>,
}

impl<E: LayerEncoder> Relay<E> {
    pub fn new(inner: E, guessed: Option<Guessed>) -> Relay<E> {
        let dim = guessed.as_ref().map_or(0, |g| g.embeddings.dim());
        Relay {
            inner,
            guessed,
            query: 0,
            summary: vec![0.0; 2 * dim],
        }
    }

    pub fn set_query(&mut self, query: usize) {
        self.query = query;
    }

    /// The guesses from files, if they drive the reads.
    pub fn guessed(&self) -> Option<&Guessed> {
        self.guessed.as_ref()
    }
}

impl<E: LayerEncoder> LayerEncoder for Relay<E> {
    fn layers(&self) -> usize {
        self.inner.layers()
    }

    fn dim(&self) -> usize {
        self.guessed.as_ref().map_or(self.inner.dim(), |g| g.embeddings.dim())
    }

    fn encode(&mut self, text: &str, hook: &mut dyn LayerHook) -> Result<Vec<f32>, Error> {
        let Relay {
            inner,
            guessed,
            query,
            summary,
        } = self;
        let Some(guessed) = guessed.as_ref() else {
            return inner.encode(text, hook);
        };
        let mut swap = Swap {
            outer: hook,
            guessed,
            query: *query,
            summary,
            missing: None,
        };
        inner.encode(text, &mut swap)?;

        match swap.missing {
            Some(layer) => Err(Error::new(
                ErrorKind::InvalidInput,
                format!("the hook wants layer {layer}, which has no guess file"),
            )),
            None => Ok(guessed.embeddings.row(*query).to_vec()),
        }
    }

    fn queues_work(&self) -> bool {
        self.inner.queues_work()
    }
}

/// The hook between the model and SpecWave's, which hands over each layer's guess from its file.
struct Swap<'a> {
    outer: &'a mut dyn LayerHook,
    guessed: &'a Guessed,
    query: usize,
    /// The guess followed by d zeros.
    summary: &'a mut [f32],
    /// A layer that the outer hook wanted and no file has a guess for.
    missing: Option<usize>,
}

impl LayerHook for Swap<'_> {
    fn wants(&self, layer: usize) -> bool {
        self.outer.wants(layer)
    }

    fn on_layer(&mut self, layer: usize, _pooled: &[f32]) {
        let Some(guess) = self.guessed.guess(layer, self.query) else {
            self.missing = Some(layer);
            return;
        };
        self.summary[..guess.len()].copy_from_slice(guess);
        self.outer.on_layer(layer, self.summary);
    }
}

/// One timed query.
pub struct Timed {
    pub start: Instant,
    pub end: Instant,
    pub report: Report,
    pub hits: Vec<Hit>,
    pub trace: Trace,
    pub before: Usage,
    pub after: Usage,
    /// Whether the index was warm when the text arrived, and how many of its pages were in the cache then, out of how
    /// many.
    pub warm: bool,
    pub resident: usize,
    pub pages: usize,
}

/// Times the search of `text` on `index` with `base`, from the text to the hits. `map` maps the index's file, whose
/// cached pages it counts first.
pub fn time_query<E: LayerEncoder>(
    sw: &mut SpecWave<E>,
    index: &Index,
    map: &Mapped,
    text: &str,
    base: &SearchParams,
) -> Result<Timed, Error> {
    let resident = map.resident(0..map.len())?;
    let pages = map.len().div_ceil(crate::coldcache::page_size());
    let warm = index.is_warm();
    let mut hits = Vec::with_capacity(base.k);
    let before = Usage::now();
    let start = Instant::now();
    sw.search(text, index, base, &mut hits)?;
    let end = Instant::now();
    let after = Usage::now();
    Ok(Timed {
        start,
        end,
        report: *sw.last(),
        hits,
        trace: sw.worker().trace(),
        before,
        after,
        warm,
        resident,
        pages,
    })
}

/// The share of the first `k` true neighbors `truth` among the first `k` hits, by id.
pub fn recall(hits: &[Hit], truth: &[u32], k: usize) -> f64 {
    let truth = &truth[..k.min(truth.len())];
    let found = hits
        .iter()
        .take(k)
        .filter(|h| truth.iter().any(|&t| u64::from(t) == h.id))
        .count();
    found as f64 / k.max(1) as f64
}

/// Adds a timed query's fields to `json`. Times are in microseconds (µs) from the text's arrival.
pub fn timed_fields(json: Json, t: &Timed) -> Json {
    let at = |instant: Option<Instant>| instant.map_or(f64::NAN, |i| us(i.saturating_duration_since(t.start)));
    let (report, served, trace) = (&t.report, &t.report.served, &t.trace);
    let between = |a: Option<Instant>, b: Option<Instant>| match (a, b) {
        (Some(a), Some(b)) => us(b.saturating_duration_since(a)),
        _ => f64::NAN,
    };
    json.num("total_us", us(t.end - t.start))
        .num("encode_us", at(report.encoded))
        .num("post_embedding_us", between(report.encoded, Some(t.end)))
        .num("fired_us", at(report.fired))
        .num("guess_taken_us", at(served.guess_taken))
        .num("guess_ready_us", at(served.guess_ready))
        .num("chain_us", between(served.guess_taken, served.guess_ready))
        .num("worker_taken_us", at(served.taken))
        .num("worker_done_us", at(served.done))
        .num("finish_wait_us", between(report.encoded, served.taken))
        .str("ran", &format!("{:?}", report.ran).to_lowercase())
        .str("ranked", &format!("{:?}", served.ranked).to_lowercase())
        .bool("skipped_rows", served.skipped_rows)
        .bool("layer0_sent", report.layer0)
        .opt("fired_layer", report.fire_layer.map(|l| l as u64))
        .bool("warm_at_start", t.warm)
        .int("pages_resident_at_start", t.resident as u64)
        .int("pages_checked", t.pages as u64)
        .num("tables_us", us(trace.tables))
        .num("buffers_us", us(trace.buffers))
        .num("route_us", us(trace.route))
        .num("submit_us", us(trace.submit))
        .num("wave_us", us(trace.wave))
        .num("scan_us", us(trace.scan))
        .num("rows_us", us(trace.rows))
        .num("rerank_us", us(trace.rerank))
        .int("scanned", trace.scanned as u64)
        .int("cluster_bytes", trace.cluster_bytes as u64)
        .int("cluster_ranges", trace.cluster_ranges as u64)
        .int("row_bytes", trace.row_bytes as u64)
        .int("row_ranges", trace.row_ranges as u64)
        .str("read_path", trace.read_path.backend)
        .num("footprint_delta", t.after.footprint as f64 - t.before.footprint as f64)
        .int("peak_footprint", t.after.peak_footprint)
        .int("disk_bytes", t.after.disk_bytes.saturating_sub(t.before.disk_bytes))
        .int("pageins", t.after.pageins.saturating_sub(t.before.pageins))
        .ints("ids", &t.hits.iter().map(|h| h.id).collect::<Vec<u64>>())
}

/// Adds a schedule's fields to `json`, with nulls when it never guesses.
pub fn schedule_fields(json: Json, schedule: &Schedule) -> Json {
    let fire = schedule.fire;
    let pick = |f: fn(&specwave::scheduler::Fire) -> f64| fire.as_ref().map_or(f64::NAN, f);
    json.opt("fire_layer", fire.map(|f| f.settings.layer as u64))
        .num("beta", pick(|f| f.settings.beta))
        .opt("mult", fire.map(|f| u64::from(f.settings.rerank_mult)))
        .opt("clusters", fire.map(|f| f.clusters as u64))
        .opt("rows", fire.map(|f| f.rows as u64))
        .num("left_ms", pick(|f| f.left_ms))
        .num("needed_ms", pick(|f| f.needed_ms))
        .bool("fits", fire.is_some_and(|f| f.fits()))
        .num("layer0_beta", schedule.layer0_beta.unwrap_or(f64::NAN))
}

/// What a trial measured about its model before the queries.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Model {
    /// The load's time in ms, the bytes that the process read from disks during it, and how much its footprint grew.
    pub load_ms: f64,
    pub load_disk_bytes: u64,
    pub load_footprint: f64,
    /// The first model call after the load, in ms.
    pub first_call_ms: f64,
    /// The share of the weights file in the cache before and after the load.
    pub resident_before: f64,
    pub resident_loaded: f64,
}

/// The guesses from files of `s`, if it names them, and the predictors that turn each layer's summary into its guess,
/// for the layers that may guess, `layers`, and layer 0 for a wave from there. Guesses from files take predictors that
/// pass them through, and the model's own guesses take the model's predictors file.
pub fn predictors(s: &Settings, layers: &[LayerSettings]) -> Result<(Option<Guessed>, Predictors), Error> {
    let mut guess_layers: Vec<usize> = layers.iter().map(|l| l.layer).collect();

    if s.layer0.is_some() && !guess_layers.contains(&0) {
        guess_layers.insert(0, 0);
    }

    match &s.guesses {
        Some((dir, label)) => {
            let guessed = Guessed::open(&s.queries, dir, label, &guess_layers)?;
            let dim = guessed.embeddings.dim();
            Ok((Some(guessed), pass_through("synthetic", dim, &guess_layers)?))
        }
        None => Ok((None, Predictors::load(&predictor::path(&s.models, &s.model)?)?)),
    }
}

/// Runs the queries of the trial `one` and gives a result line for each. `encoder` has loaded and warmed up its model
/// as `model` says, and `profile` holds its time at each layer. It picks the schedule, opens the index, drops it from
/// the cache, waits for the text in the trial's setting, then times each query. Each guess's coverages come once the
/// last query is done, since their searches read the index. `tokens` counts the tokens of a text.
pub fn run<E: LayerEncoder>(
    s: &Settings,
    one: &One,
    encoder: Relay<E>,
    predictors: Predictors,
    profile: &LayerProfile,
    model: &Model,
    tokens: impl Fn(&E, &str) -> Result<usize, Error>,
) -> Result<Vec<String>, Error> {
    let Layers::Given(layers) = &s.layers else {
        return Err(invalid("a trial takes --layers".to_string()));
    };
    let texts = read_texts(&s.texts)?;
    let truth = Ibin::open(&s.truth)?;
    let count = encoder.guessed().map_or(texts.len(), |g| g.embeddings.len());

    if truth.len() < count || truth.width() < s.base.k {
        return Err(damaged("--truth needs k true neighbors for every query".to_string()));
    }

    let wave = WaveModel::for_volume(&s.index);
    let wave_cached = support_dir()
        .and_then(|dir| Volume::of(&s.index).ok().map(|v| (dir, v)))
        .and_then(|(dir, v)| WaveModel::cached(&dir, &v).ok().flatten())
        .is_some();
    let reads = IndexReads { wave, sizes: one.sizes };
    let chosen = choose(profile, &reads, &s.base, layers, s.layer0)?;
    let schedule = if one.trial.on { chosen } else { Schedule::NONE };
    let index = Index::open(&s.index)?;
    let map = Mapped::open(&data_file(&s.index)?)?;
    let worker = Worker::new(chosen.limits(&s.base))?;
    let mut sw = SpecWave::new(encoder, predictors, schedule, worker)?;
    make_cold(&[&map], COLD_WAIT)?;
    let warmed = wait_for_text(&index, &one.trial.setting)?;
    let mut queries = Vec::with_capacity(s.per_trial);

    for seq in 0..s.per_trial {
        let q = (one.trial.query + seq) % count;
        sw.encoder_mut().set_query(q);
        let timed = time_query(&mut sw, &index, &map, &texts[q % texts.len()], &s.base)?;
        let guess = sw.guess().map(<[f32]>::to_vec);
        queries.push((q, timed, sw.embedding().to_vec(), guess));
    }

    let fire = chosen.fire.map(|f| f.params(s.base));
    let (wide, rows) = fire.map_or((s.base.nprobe, s.base.rerank), |p| {
        (p.guess_nprobe(), p.prefetch_rows())
    });
    let mut ctx = SearchCtx::new(Limits {
        max_k: rows.max(s.base.rerank),
        max_nprobe: wide.max(s.base.nprobe),
        max_rerank: rows.max(s.base.rerank),
    });
    let mut lines = Vec::with_capacity(queries.len());

    for (seq, (q, timed, e, guess)) in queries.iter().enumerate() {
        let (posting, rerank) = match (guess, &fire) {
            (Some(guess), Some(p)) => (
                coverage_of(&index, &mut ctx, e, guess, s.base.nprobe, wide)?,
                rerank_coverage_of(&index, &mut ctx, e, guess, p)?,
            ),
            _ => (f64::NAN, f64::NAN),
        };
        let lead = warmed.map_or(f64::NAN, |w| ms(timed.start.saturating_duration_since(w)));
        let json = Json::new()
            .str("bench", "spec")
            .str("run", &one.run)
            .int("trial", one.trial.number as u64)
            .int("seq", seq as u64)
            .int("query", *q as u64)
            .str("mode", if one.trial.on { "on" } else { "off" })
            .str("start", one.trial.setting.start.name())
            .opt("warm_ms", one.trial.setting.warm_ms)
            .num("warm_lead_ms", lead)
            .bool("model_cold", s.model_cold)
            .num("recall", recall(&timed.hits, truth.row(*q), s.base.k))
            .num("posting_coverage", posting)
            .num("rerank_coverage", rerank);
        let tokens = tokens(&sw.encoder().inner, &texts[q % texts.len()])?;
        let json = trial_fields(json, s, &one.trial, tokens)
            .num("load_ms", model.load_ms)
            .int("load_disk_bytes", model.load_disk_bytes)
            .num("load_footprint_delta", model.load_footprint)
            .num("first_call_ms", model.first_call_ms)
            .num("model_resident_before", model.resident_before)
            .num("model_resident_loaded", model.resident_loaded)
            .nums(
                "profile_ms",
                &profile.times().iter().map(|&(_, t)| t).collect::<Vec<f64>>(),
            )
            .num("wave_a_ms", wave.a_ms)
            .num("wave_b_ms_per_mb", wave.b_ms_per_mb)
            .bool("wave_cached", wave_cached)
            .num("tables_ms", one.sizes.tables_ms)
            .num("route_ms", one.sizes.route_ms)
            .num("cluster_bytes_each", one.sizes.cluster_bytes)
            .num("row_bytes_each", one.sizes.row_bytes)
            .num("scan_ms_each", one.sizes.scan_ms)
            .str("layers", &layers_arg(layers));
        lines.push(provenance(timed_fields(schedule_fields(json, &chosen), timed)).line());
    }

    Ok(lines)
}

/// Adds what the trial ran with to `json`.
fn trial_fields(json: Json, s: &Settings, t: &Trial, tokens: usize) -> Json {
    json.str("index", &s.index.display().to_string())
        .str("texts", &s.texts.display().to_string())
        .str("guesses", s.guesses.as_ref().map_or("", |(_, label)| label.as_str()))
        .str("model", &s.model)
        .int("threads", s.threads as u64)
        .int("nprobe", s.base.nprobe as u64)
        .int("rerank", s.base.rerank as u64)
        .int("k", s.base.k as u64)
        .int("per_trial", s.per_trial as u64)
        .int("warmup", s.warmup as u64)
        .int("first_query", t.query as u64)
        .int("tokens", tokens as u64)
}

/// The texts of a JSON (JavaScript Object Notation) lines file, from each line's "text" field.
pub fn read_texts(path: &Path) -> Result<Vec<String>, Error> {
    let file = File::open(path).map_err(|e| Error::new(e.kind(), format!("{}: {e}", path.display())))?;
    let mut texts = Vec::new();

    for (i, line) in BufReader::new(file).lines().enumerate() {
        let value: serde_json::Value =
            serde_json::from_str(&line?).map_err(|e| damaged(format!("{} line {}: {e}", path.display(), i + 1)))?;
        let text = value["text"]
            .as_str()
            .ok_or_else(|| damaged(format!("{} line {} has no text", path.display(), i + 1)))?;
        texts.push(text.to_string());
    }

    match texts.is_empty() {
        true => Err(damaged(format!("{} has no texts", path.display()))),
        false => Ok(texts),
    }
}

/// The command line's settings.
#[derive(Clone, Debug)]
pub struct Settings {
    /// The index, the query texts, and the true neighbors of each query by row.
    pub index: PathBuf,
    pub texts: PathBuf,
    pub truth: PathBuf,
    /// The queries' final embeddings. With guess files they are the embeddings that the search finishes with, and
    /// otherwise only the vectors that measure the index's read sizes.
    pub queries: PathBuf,
    /// The folder and the label of the guess files, if the guesses come from files.
    pub guesses: Option<(PathBuf, String)>,
    /// The model's Hugging Face repository id, and the folder of the predictors.
    pub model: String,
    pub models: PathBuf,
    pub max_tokens: usize,
    pub base: SearchParams,
    /// The layers that may guess, given or from the coverage results.
    pub layers: Layers,
    pub layer0: Option<Layer0>,
    /// The trials' queries, the searches per trial, and the model's threads.
    pub trials: usize,
    pub first: usize,
    pub per_trial: usize,
    pub threads: usize,
    /// The model calls that warm it up.
    pub warmup: usize,
    pub settings: Vec<Setting>,
    pub model_cold: bool,
    pub out: PathBuf,
    /// Whether to print the plan and stop.
    pub plan_only: bool,
    /// The one trial to run, in a child process.
    pub one: Option<One>,
}

/// Where the layers that may guess come from.
#[derive(Clone, Debug, PartialEq)]
pub enum Layers {
    Given(Vec<LayerSettings>),
    /// The coverage results files, and the dataset, fit and clustering of their cells.
    Grid {
        files: Vec<String>,
        dataset: String,
        fit: String,
        clustering: String,
    },
}

/// The trial that a child process runs, with what the parent measured once.
#[derive(Clone, Debug, PartialEq)]
pub struct One {
    pub run: String,
    pub trial: Trial,
    pub sizes: Sizes,
}

/// Every argument name that takes a value.
const NAMES: [&str; 31] = [
    "index",
    "texts",
    "truth",
    "queries",
    "guesses",
    "label",
    "model",
    "models",
    "max-tokens",
    "nprobe",
    "rerank",
    "k",
    "layers",
    "layer0",
    "coverage",
    "dataset",
    "fit",
    "clustering",
    "trials",
    "first",
    "per-trial",
    "threads",
    "warmup",
    "start",
    "warm-ms",
    "out",
    "run-id",
    "trial",
    "query",
    "mode",
    "reads",
];

/// The arguments that take no value.
const FLAGS: [&str; 3] = ["model-cold", "one", "plan"];

impl Settings {
    /// Reads `args`, the arguments after the program's name. `repo` is this repository's folder, which holds the
    /// default results file and the predictors. A name that isn't known, a name given twice, a missing or bad value,
    /// or a child's trial without its parts gives `InvalidInput`.
    pub fn parse(args: &[String], repo: &Path) -> Result<Settings, Error> {
        let mut values: Vec<(&str, &str)> = Vec::new();
        let mut flags: Vec<&str> = Vec::new();
        let mut it = args.iter();

        while let Some(arg) = it.next() {
            let name = arg
                .strip_prefix("--")
                .ok_or_else(|| invalid(format!("unknown argument {arg}")))?;
            if FLAGS.contains(&name) {
                if flags.contains(&name) {
                    return Err(invalid(format!("--{name} is given twice")));
                }
                flags.push(name);
                continue;
            }
            if !NAMES.contains(&name) {
                return Err(invalid(format!("unknown argument {arg}")));
            }
            let value = it.next().ok_or_else(|| invalid(format!("--{name} needs a value")))?;
            if values.iter().any(|(n, _)| *n == name) {
                return Err(invalid(format!("--{name} is given twice")));
            }
            values.push((name, value));
        }

        let get = |name: &str| values.iter().find(|(n, _)| *n == name).map(|&(_, v)| v);
        let need = |name: &str| get(name).ok_or_else(|| invalid(format!("--{name} is required")));
        let number = |name: &str, default: usize| get(name).map_or(Ok(default), |v| one(v, name));
        let path = |name: &str, default: &str| get(name).map_or_else(|| repo.join(default), PathBuf::from);
        let base = SearchParams {
            k: number("k", 10)?,
            nprobe: number("nprobe", 9)?,
            rerank: number("rerank", 29)?,
            reach: None,
        };
        let layers = match (get("layers"), get("coverage")) {
            (Some(layers), None) => Layers::Given(parse_layers(layers)?),
            (None, Some(files)) => Layers::Grid {
                files: files.split(',').map(str::to_string).collect(),
                dataset: get("dataset").unwrap_or("fiqa").to_string(),
                fit: get("fit").unwrap_or("openings").to_string(),
                clustering: get("clustering").unwrap_or("l2").to_string(),
            },
            _ => return Err(invalid("give either --layers or --coverage".to_string())),
        };
        let layer0 = get("layer0")
            .map(|v| match v.split(':').collect::<Vec<_>>()[..] {
                [beta, coverage] => Ok(Layer0 {
                    beta: one(beta, "layer0")?,
                    coverage: one(coverage, "layer0")?,
                }),
                _ => Err(invalid(format!("--layer0 takes beta:coverage, not {v}"))),
            })
            .transpose()?;
        let guesses = match (get("guesses"), get("label")) {
            (Some(dir), Some(label)) => Some((PathBuf::from(dir), label.to_string())),
            (None, None) => None,
            _ => return Err(invalid("--guesses and --label go together".to_string())),
        };
        let starts: Vec<Start> = list(get("start").unwrap_or("idle,prewarm"), "start")?;
        let warm: Vec<Option<u64>> = get("warm-ms")
            .unwrap_or("none")
            .split(',')
            .map(|v| match v.trim() {
                "none" => Ok(None),
                ms => one(ms, "warm-ms").map(Some),
            })
            .collect::<Result<_, _>>()?;
        let mut settings: Vec<Setting> = starts
            .iter()
            .flat_map(|&start| warm.iter().map(move |&warm_ms| Setting { start, warm_ms }))
            .collect();
        settings.sort();
        settings.dedup();
        let one_trial = match flags.contains(&"one") {
            false => None,
            true => {
                let mode = need("mode")?;
                if !matches!(mode, "on" | "off") || settings.len() != 1 {
                    return Err(invalid(
                        "a child's trial takes --mode on or off and one --start and --warm-ms".to_string(),
                    ));
                }
                Some(One {
                    run: need("run-id")?.to_string(),
                    trial: Trial {
                        number: one(need("trial")?, "trial")?,
                        query: one(need("query")?, "query")?,
                        on: mode == "on",
                        setting: settings[0],
                    },
                    sizes: need("reads")?.parse()?,
                })
            }
        };
        let s = Settings {
            index: PathBuf::from(need("index")?),
            texts: PathBuf::from(need("texts")?),
            truth: PathBuf::from(need("truth")?),
            queries: PathBuf::from(need("queries")?),
            guesses,
            model: repo_id(get("model").unwrap_or("arctic")).to_string(),
            models: path("models", "models"),
            max_tokens: number("max-tokens", 128)?,
            base,
            layers,
            layer0,
            trials: number("trials", 100)?,
            first: number("first", 0)?,
            per_trial: number("per-trial", 1)?,
            threads: number("threads", 5)?,
            warmup: number("warmup", 20)?,
            settings,
            model_cold: flags.contains(&"model-cold"),
            out: path("out", "results/spec.jsonl"),
            plan_only: flags.contains(&"plan"),
            one: one_trial,
        };

        if s.trials == 0 || s.per_trial == 0 || s.threads == 0 || s.warmup == 0 || s.base.k == 0 {
            return Err(invalid(
                "--trials, --per-trial, --threads, --warmup and --k are at least 1".to_string(),
            ));
        }

        if s.base.nprobe == 0 || s.base.rerank < s.base.k {
            return Err(invalid("--nprobe is at least 1 and --rerank at least --k".to_string()));
        }

        if s.layer0
            .is_some_and(|l| !(l.beta.is_finite() && l.beta >= 0.0 && (0.0..=1.0).contains(&l.coverage)))
        {
            return Err(invalid(
                "--layer0's beta is at least 0 and its coverage from 0 to 1".to_string(),
            ));
        }

        Ok(s)
    }

    /// The arguments of the child process that runs `trial`, with the layers and the sizes the parent resolved.
    pub fn child_args(&self, run: &str, trial: &Trial, layers: &[LayerSettings], sizes: &Sizes) -> Vec<String> {
        let warm_ms = trial.setting.warm_ms.map_or("none".to_string(), |w| w.to_string());
        let mut pairs = vec![
            ("run-id", run.to_string()),
            ("trial", trial.number.to_string()),
            ("query", trial.query.to_string()),
            ("mode", if trial.on { "on" } else { "off" }.to_string()),
            ("start", trial.setting.start.name().to_string()),
            ("warm-ms", warm_ms),
            ("reads", sizes.arg()),
            ("layers", layers_arg(layers)),
            ("index", self.index.display().to_string()),
            ("texts", self.texts.display().to_string()),
            ("truth", self.truth.display().to_string()),
            ("queries", self.queries.display().to_string()),
            ("model", self.model.clone()),
            ("models", self.models.display().to_string()),
            ("max-tokens", self.max_tokens.to_string()),
            ("nprobe", self.base.nprobe.to_string()),
            ("rerank", self.base.rerank.to_string()),
            ("k", self.base.k.to_string()),
            ("per-trial", self.per_trial.to_string()),
            ("threads", self.threads.to_string()),
            ("warmup", self.warmup.to_string()),
            ("out", self.out.display().to_string()),
        ];

        if let Some((dir, label)) = &self.guesses {
            pairs.push(("guesses", dir.display().to_string()));
            pairs.push(("label", label.clone()));
        }

        if let Some(l) = self.layer0 {
            pairs.push(("layer0", format!("{}:{}", l.beta, l.coverage)));
        }

        let mut args = vec!["--one".to_string()];

        if self.model_cold {
            args.push("--model-cold".to_string());
        }

        for (name, value) in pairs {
            args.push(format!("--{name}"));
            args.push(value);
        }

        args
    }
}

/// The summary of the result lines of run `run` in `lines`, as two Markdown tables. The first gives each setting's
/// times with speculation on and off, and the second the time that speculation saved for each query against its run
/// with speculation off. A first query without a warm-up should meet a cold index, so one whose index had pages in
/// the cache at its start is left out, and a note says how many were.
pub fn summary(lines: &[String], run: &str) -> String {
    type Key = (String, String, bool, String);
    let mut groups: BTreeMap<(Key, bool), Vec<serde_json::Value>> = BTreeMap::new();
    let mut not_cold = 0;

    for line in lines {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if v["bench"] != "spec" || v["run"] != run {
            continue;
        }
        let cached = v["pages_resident_at_start"].as_u64().is_some_and(|pages| pages > 0);
        if v["seq"].as_u64() == Some(0) && v["warm_ms"].is_null() && cached {
            not_cold += 1;
            continue;
        }
        let key = (
            v["start"].as_str().unwrap_or("?").to_string(),
            v["warm_ms"].as_u64().map_or("none".to_string(), |w| w.to_string()),
            v["model_cold"].as_bool().unwrap_or(false),
            match v["seq"].as_u64() {
                Some(0) => "first".to_string(),
                _ => "later".to_string(),
            },
        );
        groups.entry((key, v["mode"] == "on")).or_default().push(v);
    }

    let mut out = String::new();
    let _ = writeln!(
        out,
        "| Start | Warm-Up Lead (ms) | Model Cold | Queries | Speculation | Trials | Total p50 / p90 (ms) | \
         Post-Embedding p50 / p90 (ms) | Model p50 (ms) | Load p50 (ms) | Recall@k | Posting Coverage | \
         Rerank Coverage | Guessed |"
    );
    let _ = writeln!(out, "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    let values =
        |rows: &[serde_json::Value], name: &str| -> Vec<f64> { rows.iter().filter_map(|r| r[name].as_f64()).collect() };
    let field =
        |rows: &[serde_json::Value], name: &str| -> Vec<f64> { values(rows, name).iter().map(|v| v / 1e3).collect() };
    // The mean of a field over the rows that have it, or a dash for none.
    let mean = |rows: &[serde_json::Value], name: &str| -> String {
        let found = values(rows, name);
        match found.is_empty() {
            true => "-".to_string(),
            false => format!("{:.4}", found.iter().sum::<f64>() / found.len() as f64),
        }
    };

    for ((key, on), rows) in groups.iter().rev() {
        let (total, post, model) = (
            field(rows, "total_us"),
            field(rows, "post_embedding_us"),
            field(rows, "encode_us"),
        );
        let load = match values(rows, "load_ms") {
            loads if loads.is_empty() => "-".to_string(),
            loads => format!("{:.1}", rank(&loads, 50.0)),
        };
        let recalls = values(rows, "recall");
        let guessed = rows.iter().filter(|r| r["ran"] == "guessed").count();
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {} | {} | {:.2} / {:.2} | {:.3} / {:.3} | {:.2} | {} | {:.4} | {} | {} | {} of {} |",
            key.0,
            key.1,
            if key.2 { "yes" } else { "no" },
            key.3,
            if *on { "on" } else { "off" },
            rows.len(),
            rank(&total, 50.0),
            rank(&total, 90.0),
            rank(&post, 50.0),
            rank(&post, 90.0),
            rank(&model, 50.0),
            load,
            recalls.iter().sum::<f64>() / recalls.len().max(1) as f64,
            mean(rows, "posting_coverage"),
            mean(rows, "rerank_coverage"),
            guessed,
            rows.len()
        );
    }

    let _ = writeln!(
        out,
        "\n| Start | Warm-Up Lead (ms) | Model Cold | Queries | Pairs | Saved p10 / p50 / p90 (ms) | Recall Gap |"
    );
    let _ = writeln!(out, "|---|---|---|---|---|---|---|");
    let keys: Vec<&Key> = groups.keys().filter(|(_, on)| *on).map(|(k, _)| k).collect();

    for key in keys {
        let by_query = |on: bool| -> BTreeMap<(u64, u64), (f64, f64)> {
            let rows = groups.get(&(key.clone(), on)).map_or(&[][..], Vec::as_slice);
            rows.iter()
                .filter_map(|r| {
                    let id = (r["query"].as_u64()?, r["seq"].as_u64()?);
                    Some((id, (r["total_us"].as_f64()? / 1e3, r["recall"].as_f64()?)))
                })
                .collect()
        };
        let (on, off) = (by_query(true), by_query(false));
        let pairs: Vec<(f64, f64)> = on
            .iter()
            .filter_map(|(id, &(t_on, r_on))| off.get(id).map(|&(t_off, r_off)| (t_off - t_on, r_on - r_off)))
            .collect();
        let saved: Vec<f64> = pairs.iter().map(|p| p.0).collect();
        let gap = pairs.iter().map(|p| p.1).sum::<f64>() / pairs.len().max(1) as f64;
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {} | {:.2} / {:.2} / {:.2} | {:+.4} |",
            key.0,
            key.1,
            if key.2 { "yes" } else { "no" },
            key.3,
            pairs.len(),
            rank(&saved, 10.0),
            rank(&saved, 50.0),
            rank(&saved, 90.0),
            gap
        );
    }

    if not_cold > 0 {
        let _ = writeln!(
            out,
            "\n{not_cold} first queries without a warm-up are left out, since their index had pages in the cache at \
             their start."
        );
    }

    out
}

/// The value at percentile `p` of `values` by nearest rank, i.e. the smallest that at least `p` percent of them are
/// at or below, or NaN (not a number) for no values.
pub fn rank(values: &[f64], p: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);

    match sorted.len() {
        0 => f64::NAN,
        n => sorted[((p / 100.0 * n as f64).ceil() as usize).clamp(1, n) - 1],
    }
}

/// The repository id of a model's short name, `arctic`, or the name itself.
pub fn repo_id(model: &str) -> &str {
    match model {
        "arctic" => "Snowflake/snowflake-arctic-embed-m-v1.5",
        other => other,
    }
}

fn us(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn one<T: FromStr>(value: &str, name: &str) -> Result<T, Error> {
    value
        .trim()
        .parse()
        .map_err(|_| invalid(format!("--{name} has a bad value: {value}")))
}

/// The values of a list separated by commas.
fn list<T: FromStr<Err = Error>>(value: &str, name: &str) -> Result<Vec<T>, Error> {
    value
        .split(',')
        .map(|v| v.trim().parse().map_err(|e: Error| invalid(format!("--{name}: {e}"))))
        .collect()
}

fn invalid(message: String) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

fn damaged(message: String) -> Error {
    Error::new(ErrorKind::InvalidData, message)
}

#[cfg(test)]
#[path = "../tests/unit/trials.rs"]
mod tests;
