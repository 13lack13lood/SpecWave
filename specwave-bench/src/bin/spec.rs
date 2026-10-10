//! Times SpecWave end to end with the real model (feature `candle`), in cold trials with speculation on and off, as
//! the trials module describes. Each trial is a fresh process with the model loaded and the index out of memory.
//!
//! ```text
//! spec --index FILE --texts FILE --truth FILE --queries FILE
//!      (--layers L:B:M,... | --coverage FILES [--dataset fiqa] [--fit openings] [--clustering l2]) [--layer0 B:C]
//!      [--guesses DIR --label NAME] [--model arctic] [--models models] [--max-tokens 128]
//!      [--nprobe 9] [--rerank 29] [--k 10] [--trials 100] [--first 0] [--per-trial 1] [--threads 5] [--warmup 20]
//!      [--start idle,prewarm] [--warm-ms none,0,50,200,1000] [--model-cold] [--out results/spec.jsonl] [--plan]
//! ```
//!
//! - **Data.** `--index` is the index file, `--texts` a JavaScript Object Notation (JSON) lines file of the query texts
//!   with a "text" field, `--truth` an `.ibin` file of each query's true neighbors by row, and `--queries` an `.fbin`
//!   file of the queries' final embeddings. Without guess files the model gives the final embeddings, and `--queries`
//!   only measures the index's read sizes. With `--guesses DIR --label NAME` the guess of each layer l comes from
//!   `DIR/NAME.L<l>.fbin` and the final embeddings from `--queries`, and query i runs the model on text i modulo the
//!   texts, so the reads still hide under the model's work.
//! - **Layers.** `--layers` gives each layer that may guess with its β and m, eg. `9:0:2,10:0:2`. `--coverage` takes
//!   them from coverage results instead, from each layer's first passing cell for the base nprobe and R, the number of
//!   candidates that a normal search rescores. `--layer0 B:C` adds a wave from layer 0 with the calibrated β B and the
//!   posting coverage C for drives too slow for any layer, which `--coverage` also finds when its grid has layer 0 at
//!   β = 1.
//! - **Trials.** Queries `--first` on, `--trials` of them, run once with speculation on and once off in every setting,
//!   i.e. every `--start` state with every `--warm-ms` lead, where `none` starts no warm-up. A lead starts the index's
//!   warm-up that long before the text, and each line records the lead measured. `--per-trial` searches that many
//!   queries per process, the first one cold. `--model-cold` drops the model's weights from the cache before each load,
//!   which changes only the load, as the trials module says. `--threads` sets the model's thread variables, and
//!   `--warmup` the model calls before the profile.
//! - **Output.** Each child appends a line per search to `--out`, and the parent prints a summary of the run. `--plan`
//!   prints the trials and stops. A line holds the schedule that speculation picks, `fire_layer` and the rest, also
//!   when speculation is off, and `fired_layer`, the layer whose guess read the clusters, if one did. For that guess it
//!   also holds the posting coverage, the share of the final embedding's clusters that its read covered, and the
//!   rerank coverage, the share of a normal search's rescored rows among the rows that it prefetched.
//!
//! The parent measures the index's read sizes once, from a cold first search and 32 more, which the children's
//! schedules then use with the cached wave model of the index's volume. Where those searches read through a backend
//! whose fetch waits for its reads, eg. on a drive that refuses read hints, it leaves out the wave from layer 0, which
//! would hold the worker there. Run it on an idle machine, since the times belong to this machine and its load.

use std::io::{Error, ErrorKind};
use std::path::Path;
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use specwave::encoder::{Backend, CandleEncoder, LayerEncoder, NoHook, snapshot_dir};
use specwave::scheduler::{LayerProfile, LayerSettings, PROFILE_CALLS};
use specwave_bench::coldcache::{Mapped, Usage, make_cold};
use specwave_bench::data::Fbin;
use specwave_bench::results::append;
use specwave_bench::trials::{
    self, COLD_WAIT, Layers, Model, One, Relay, SIZE_SAMPLES, Settings, layer_settings, layer0, layers_arg,
    measure_sizes, plan, predictors, read_texts, reads_ahead, summary,
};

/// The environment variables that set the model's thread counts, for candle's pools and for Accelerate.
const THREAD_VARS: [&str; 4] = [
    "RAYON_NUM_THREADS",
    "CANDLE_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OMP_NUM_THREADS",
];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let result = Settings::parse(&args, &repo).and_then(|s| match &s.one {
        Some(one) => trial(&s, one),
        None => run(&s),
    });

    if let Err(e) = result {
        eprintln!("spec: {e}");
        std::process::exit(if e.kind() == ErrorKind::InvalidInput { 2 } else { 1 });
    }
}

/// Plans the trials, measures the index's read sizes, runs each trial in a fresh process and prints the summary.
fn run(s: &Settings) -> Result<(), Error> {
    let (layers, layer0) = resolve(s)?;
    let count = match &s.guesses {
        Some(_) => Fbin::open(&s.queries)?.len(),
        None => read_texts(&s.texts)?.len(),
    };
    let trials = plan(s.first, s.trials, count, &s.settings);
    eprintln!("spec: layers {}, layer 0 {layer0:?}", layers_arg(&layers));

    if s.plan_only {
        for t in &trials {
            println!(
                "trial {} query {} speculation {} start {} warm-up lead {:?}",
                t.number,
                t.query,
                if t.on { "on" } else { "off" },
                t.setting.start.name(),
                t.setting.warm_ms
            );
        }
        return Ok(());
    }

    let (sizes, backend) = measure_sizes(&s.index, &Fbin::open(&s.queries)?, &s.base, SIZE_SAMPLES)?;
    eprintln!("spec: read sizes {sizes:?}, through {backend}");
    let layer0 = layer0.filter(|_| reads_ahead(backend));

    if s.layer0.is_some() && layer0.is_none() {
        eprintln!("spec: no wave from layer 0, since a fetch through {backend} waits for its reads");
    }

    let run = run_id();
    let exe = std::env::current_exe()?;
    let resolved = Settings { layer0, ..s.clone() };
    let mut failed = Vec::new();

    for t in &trials {
        let threads = s.threads.to_string();
        let status = Command::new(&exe)
            .args(resolved.child_args(&run, t, &layers, &sizes))
            .envs(THREAD_VARS.iter().map(|var| (var, &threads)))
            .status()?;
        match status.success() {
            true => eprintln!("spec: trial {} of {} done", t.number + 1, trials.len()),
            false => {
                eprintln!("spec: trial {} failed", t.number);
                failed.push(t.number);
            }
        }
    }

    let text = std::fs::read_to_string(&s.out).unwrap_or_default();
    let lines: Vec<String> = text.lines().map(str::to_string).collect();
    print!("{}", summary(&lines, &run));

    match failed.is_empty() {
        true => Ok(()),
        false => Err(Error::other(format!("{} trials failed: {failed:?}", failed.len()))),
    }
}

/// The layers that may guess, and the wave from layer 0, from the command line or the coverage results.
fn resolve(s: &Settings) -> Result<(Vec<LayerSettings>, Option<specwave::scheduler::Layer0>), Error> {
    let (files, dataset, fit, clustering) = match &s.layers {
        Layers::Given(layers) => return Ok((layers.clone(), s.layer0)),
        Layers::Grid {
            files,
            dataset,
            fit,
            clustering,
        } => (files, dataset, fit, clustering),
    };
    let mut lines = Vec::new();

    for file in files {
        let text = std::fs::read_to_string(file).map_err(|e| Error::new(e.kind(), format!("{file}: {e}")))?;
        lines.extend(text.lines().map(str::to_string));
    }

    let layers = layer_settings(&lines, dataset, fit, clustering, &s.base)?;

    if layers.is_empty() {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "the coverage results have no passing layer for these settings",
        ));
    }

    let layer0 = s.layer0.or_else(|| layer0(&lines, dataset, fit, clustering, &s.base));
    Ok((layers, layer0))
}

/// Runs one trial in this process and appends its lines. It loads the model, warms it up and times its layers, and
/// the trials module runs the queries.
fn trial(s: &Settings, one: &One) -> Result<(), Error> {
    set_user_interactive();
    let Layers::Given(layers) = &s.layers else {
        return Err(Error::new(ErrorKind::InvalidInput, "a child's trial takes --layers"));
    };
    let texts = read_texts(&s.texts)?;
    let snapshot = snapshot_dir(&s.model)?;
    let weights = Mapped::open(&snapshot.join("model.safetensors"))?;

    if s.model_cold {
        make_cold(&[&weights], COLD_WAIT)?;
    }

    let resident_before = weights.residency()?;
    let (before, started) = (Usage::now(), Instant::now());
    let mut encoder = CandleEncoder::open(&snapshot, Backend::Cpu)?;
    encoder.set_max_tokens(s.max_tokens)?;
    let load_ms = ms_since(started);
    let after = Usage::now();
    let resident_loaded = weights.residency()?;
    // The warm-up and the profile run on the last texts, which the trials reach last.
    let spare: Vec<&str> = (0..s.warmup.max(PROFILE_CALLS))
        .map(|i| texts[texts.len() - 1 - i % texts.len()].as_str())
        .collect();
    let first = Instant::now();
    encoder.encode(spare[0], &mut NoHook)?;
    let first_call_ms = ms_since(first);

    for text in spare.iter().take(s.warmup).skip(1) {
        encoder.encode(text, &mut NoHook)?;
    }

    let profile = LayerProfile::measure(&mut encoder, &spare, PROFILE_CALLS)?;
    let (guessed, predictors) = predictors(s, layers)?;
    let model = Model {
        load_ms,
        load_disk_bytes: after.disk_bytes.saturating_sub(before.disk_bytes),
        load_footprint: after.footprint as f64 - before.footprint as f64,
        first_call_ms,
        resident_before,
        resident_loaded,
    };
    let tokens = |encoder: &CandleEncoder, text: &str| encoder.tokens(text);
    let lines = trials::run(
        s,
        one,
        Relay::new(encoder, guessed),
        predictors,
        &profile,
        &model,
        tokens,
    )?;
    append(&s.out, &lines)
}

/// The milliseconds since `start`.
fn ms_since(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}

#[cfg(target_os = "macos")]
fn set_user_interactive() {
    // SAFETY: the call changes only the calling thread's scheduling class, and it takes no pointers.
    let _ = unsafe { libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0) };
}

#[cfg(not(target_os = "macos"))]
fn set_user_interactive() {}

/// An id shared by the lines of one run, from the time and the process id.
fn run_id() -> String {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    format!("{}-{}", now.as_millis(), std::process::id())
}
