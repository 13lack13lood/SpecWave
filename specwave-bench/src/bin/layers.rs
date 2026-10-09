//! Times the embedding model layer by layer. T_l is the time from a query's text to the end of layer l, where layer 0
//! is the embedding output and layer L is the last layer. T_L - T_l is the model time still to come after layer l,
//! the time that reads started at layer l can hide under.
//!
//! ```text
//! layers [--model arctic,minilm] [--queries data/fiqa_queries.jsonl] [--split test] [--runs 50] [--warmup 20]
//!        [--out results/layers.jsonl] [--device cpu|metal --threads N]
//! ```
//!
//! Without `--device` it times each setup in a process of its own, with the thread counts set in that process's
//! environment. The setups are the processor (CPU) with P - 1 and with P threads, where P is the number of performance
//! cores (`hw.perflevel0.physicalcpu`), and on macOS the graphics processor (GPU) through Metal. Then it prints one
//! table of every setup. With `--device` it times one setup, and the four thread variables of its environment must
//! all be `--threads`. A name that isn't listed above, or one given twice, is an error.
//!
//! A setup makes 20 warm-up calls first. Then each of the 50 runs takes the next query, and for every layer l it times
//! these calls on that one text, batch 1.
//! - T_l, running the model only up to layer l and then waiting for the device. Each call waits for one layer only,
//!   since on Metal a wait stalls the queue of work.
//! - The whole model with the hook at layer l, the way SpecWave runs it. This gives the time the hook gets layer l's
//!   summary, and the total time with that one copy back to the host.
//!
//! Each run then times the whole model with no hook. The tables and the results file hold the medians over the runs,
//! by nearest rank.
//!
//! `--model` takes short names (`arctic`, `minilm`) or Hugging Face repository ids, found in the local cache.
//! `--queries` is a JSON (JavaScript Object Notation) lines file of {row, id, text}, and `--split` keeps the rows that
//! the `.splits.json` file next to it lists under that name, eg. the test split of BeIR (Benchmarking Information
//! Retrieval). `--split all` keeps every row. The timings belong to this machine and its load, so time an idle machine.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Error, ErrorKind};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str::FromStr;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use specwave::encoder::{Backend, CandleEncoder, LayerEncoder, LayerHook, NoHook, snapshot_dir};
use specwave_bench::results::{Json, append, provenance};

/// The environment variables that set the processor's thread counts, for candle's pools and for Accelerate.
const THREAD_VARS: [&str; 4] = [
    "RAYON_NUM_THREADS",
    "CANDLE_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OMP_NUM_THREADS",
];

/// Every argument name. The parent passes `run-id` to each child process, so all their lines share the run's id.
const NAMES: [&str; 9] = [
    "model", "queries", "split", "runs", "warmup", "out", "device", "threads", "run-id",
];

fn main() {
    let result = Args::parse(std::env::args().skip(1)).and_then(|args| match args.str("device") {
        Some(device) => time_setup(&args, device),
        None => time_all(&args),
    });

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(if e.kind() == ErrorKind::InvalidInput { 2 } else { 1 });
    }
}

/// Times every setup of every model, each in a fresh process, then prints them side by side.
fn time_all(args: &Args) -> Result<(), Error> {
    let perf = sysctl("hw.perflevel0.physicalcpu")
        .parse::<usize>()
        .unwrap_or_else(|_| std::thread::available_parallelism().map_or(1, |n| n.get()));
    let run = args.str("run-id").map_or_else(run_id, str::to_string);
    let exe = std::env::current_exe()?;
    let out = args.path("out", "results/layers.jsonl");
    let mut setups = vec![("cpu", perf.saturating_sub(1).max(1)), ("cpu", perf)];
    let mut failed = Vec::new();
    setups.dedup();

    if cfg!(target_os = "macos") {
        setups.push(("metal", perf));
    }

    for model in args.str("model").unwrap_or("arctic,minilm").split(',') {
        for &(device, threads) in &setups {
            let threads = threads.to_string();
            let mut child = Command::new(&exe);
            for (name, value) in args
                .0
                .iter()
                .filter(|(name, _)| !matches!(name.as_str(), "model" | "run-id"))
            {
                child.arg(format!("--{name}")).arg(value);
            }
            child.args([
                "--model",
                model,
                "--device",
                device,
                "--threads",
                &threads,
                "--run-id",
                &run,
            ]);
            child.envs(THREAD_VARS.iter().map(|var| (var, &threads)));
            if !child.status()?.success() {
                failed.push(format!("{model} on {device} with {threads} threads"));
            }
        }
    }

    // The failures come first, so they show even when no setup wrote a line.
    if !failed.is_empty() {
        eprintln!("these setups failed: {}", failed.join(", "));
    }

    print!("{}", summary(&out, &run)?);

    match failed.is_empty() {
        true => Ok(()),
        false => Err(Error::other(format!("{} setups failed", failed.len()))),
    }
}

/// A hook that notes when it gets one layer.
struct AtLayer {
    layer: usize,
    start: Instant,
    at: Option<Duration>,
}

impl LayerHook for AtLayer {
    fn wants(&self, layer: usize) -> bool {
        layer == self.layer
    }

    fn on_layer(&mut self, _layer: usize, _pooled: &[f32]) {
        self.at = Some(self.start.elapsed());
    }
}

/// A hook that wants every layer, so the warm-up runs every kind of work the timed calls do.
struct EveryLayer;

impl LayerHook for EveryLayer {
    fn wants(&self, _layer: usize) -> bool {
        true
    }

    fn on_layer(&mut self, _layer: usize, _pooled: &[f32]) {}
}

/// The times of one layer over the runs, in milliseconds.
#[derive(Default)]
struct Samples {
    to_layer: Vec<f64>,
    hook_at: Vec<f64>,
    hooked_total: Vec<f64>,
    hooked_left: Vec<f64>,
}

/// Times one model on one device, in this process, and appends one line per layer to the results file.
fn time_setup(args: &Args, device: &str) -> Result<(), Error> {
    let threads: usize = args.get("threads", 0)?;
    let runs: usize = args.get("runs", 50)?;
    let warmup: usize = args.get("warmup", 20)?;
    let model = args.str("model").unwrap_or("arctic");
    let split = args.str("split").unwrap_or("test");
    let queries_path = args.path("queries", "data/fiqa_queries.jsonl");
    let out = args.path("out", "results/layers.jsonl");
    let run = args.str("run-id").map_or_else(run_id, str::to_string);
    let backend = match device {
        "cpu" => Backend::Cpu,
        _ => Backend::Metal,
    };

    if model.contains(',') {
        return Err(Error::new(ErrorKind::InvalidInput, "--device takes one --model"));
    }

    for var in THREAD_VARS {
        if threads == 0 || std::env::var(var).ok() != Some(threads.to_string()) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!(
                    "set --threads N and {} to N, or leave out --device",
                    THREAD_VARS.join(", ")
                ),
            ));
        }
    }

    set_user_interactive();
    let repo = repo_id(model);
    let queries = read_queries(&queries_path, split)?;
    let started = Instant::now();
    let mut encoder = CandleEncoder::open(&snapshot_dir(repo)?, backend)?;
    let load_ms = ms(started.elapsed());
    let layers = encoder.layers();
    let load_before = sysctl("vm.loadavg");
    let mut samples: Vec<Samples> = (0..=layers).map(|_| Samples::default()).collect();
    let mut plain = Vec::with_capacity(runs);
    let mut tokens = 0;
    eprintln!(
        "{repo} on {device} with {threads} threads: {} queries, loaded in {load_ms:.0} ms",
        queries.len()
    );

    for i in 0..warmup {
        encoder.encode(&queries[(runs + i) % queries.len()], &mut EveryLayer)?;
    }

    for r in 0..runs {
        let text = &queries[r % queries.len()];
        tokens += encoder.tokens(text)?;
        for (layer, s) in samples.iter_mut().enumerate() {
            let start = Instant::now();
            encoder.run_to(text, layer)?;
            s.to_layer.push(ms(start.elapsed()));
            let mut hook = AtLayer {
                layer,
                start: Instant::now(),
                at: None,
            };
            encoder.encode(text, &mut hook)?;
            let total = ms(hook.start.elapsed());
            let at = ms(hook.at.ok_or_else(|| Error::other("the hook never got its layer"))?);
            s.hook_at.push(at);
            s.hooked_total.push(total);
            s.hooked_left.push(total - at);
        }
        let start = Instant::now();
        encoder.encode(text, &mut NoHook)?;
        plain.push(ms(start.elapsed()));
    }

    let load_after = sysctl("vm.loadavg");
    let machine = Machine::now();
    let plain_ms = median(&plain);
    let last_ms = median(&samples[layers].to_layer);
    let tokens_mean = tokens as f64 / runs as f64;
    let mut lines = Vec::with_capacity(layers + 1);
    println!(
        "\n{repo} on {device} with {threads} threads: {runs} runs per layer after {warmup} warm-up calls, \
         {tokens_mean:.1} tokens on average, medians in ms"
    );
    println!(
        "{:>5} {:>8} {:>8} {:>8} {:>8} {:>13} {:>13} {:>9}",
        "layer", "T_l", "p10", "p90", "T_L-T_l", "hook at l", "total, hook", "readback"
    );

    for (layer, s) in samples.iter().enumerate() {
        let t = median(&s.to_layer);
        let (hook_ms, hooked_ms) = (median(&s.hook_at), median(&s.hooked_total));
        let (p10, p90) = (percentile(&s.to_layer, 10.0), percentile(&s.to_layer, 90.0));
        println!(
            "{layer:>5} {t:>8.3} {p10:>8.3} {p90:>8.3} {:>8.3} {hook_ms:>13.3} {hooked_ms:>13.3} {:>9.3}",
            last_ms - t,
            hooked_ms - plain_ms
        );
        let json = Json::new()
            .str("bench", "layers")
            .str("run", &run)
            .str("model", repo)
            .int("model_layers", layers as u64)
            .int("dim", encoder.dim() as u64)
            .str("device", device)
            .int("threads", threads as u64)
            .bool("accelerate", cfg!(target_os = "macos") && backend == Backend::Cpu)
            .int("layer", layer as u64)
            .num("t_ms", t)
            .num("t_p10_ms", p10)
            .num("t_p90_ms", p90)
            .num("t_min_ms", percentile(&s.to_layer, 0.0))
            .num("headroom_ms", last_ms - t)
            .num("hook_ms", hook_ms)
            .num("hooked_total_ms", hooked_ms)
            .num("hooked_headroom_ms", median(&s.hooked_left))
            .num("plain_total_ms", plain_ms)
            .num("readback_ms", hooked_ms - plain_ms)
            .int("runs", runs as u64)
            .int("warmup", warmup as u64)
            .num("tokens_mean", tokens_mean)
            .str("queries", &queries_path.display().to_string())
            .str("split", split)
            .num("load_ms", load_ms)
            .str("loadavg_before", &load_before)
            .str("loadavg_after", &load_after);
        lines.push(provenance(machine.add(json)).line());
    }

    println!("whole model without a hook: {plain_ms:.3} ms");
    append(&out, &lines)
}

/// A setup's name, and its (T_l, T_L - T_l) for each layer.
type Setup = (String, Vec<(f64, f64)>);

/// The layer times of every setup of run `run` in the results file `out`, side by side, one table per model. There
/// are none when the file doesn't exist, eg. because every setup failed before it wrote a line.
fn summary(out: &Path, run: &str) -> Result<String, Error> {
    let mut models: Vec<(String, Vec<Setup>)> = Vec::new();
    let mut text = String::new();
    let file = match File::open(out) {
        Ok(file) => file,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(text),
        Err(e) => return Err(e),
    };

    for line in BufReader::new(file).lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&line?) else {
            continue;
        };
        if v["bench"] != "layers" || v["run"] != run {
            continue;
        }
        let model = v["model"].as_str().unwrap_or("?").to_string();
        let setup = match v["device"].as_str() {
            Some("cpu") => format!("cpu, {} threads", v["threads"]),
            other => other.unwrap_or("?").to_string(),
        };
        let times = (
            v["t_ms"].as_f64().unwrap_or(f64::NAN),
            v["headroom_ms"].as_f64().unwrap_or(f64::NAN),
        );
        let at = models.iter().position(|(m, _)| *m == model).unwrap_or_else(|| {
            models.push((model, Vec::new()));
            models.len() - 1
        });
        let setups = &mut models[at].1;
        match setups.iter_mut().find(|(s, _)| *s == setup) {
            Some((_, rows)) => rows.push(times),
            None => setups.push((setup, vec![times])),
        }
    }

    for (model, setups) in &models {
        text.push_str(&format!("\n{model}: T_l and T_L-T_l in ms, medians\n"));
        let header: Vec<String> = setups.iter().map(|(s, _)| format!("{s:>19}")).collect();
        text.push_str(&format!("{:>5} | {}\n", "", header.join(" | ")));
        let rows = setups.iter().map(|(_, rows)| rows.len()).max().unwrap_or(0);
        for layer in 0..rows {
            let cells: Vec<String> = setups
                .iter()
                .map(|(_, r)| {
                    r.get(layer)
                        .map_or(String::new(), |(t, h)| format!("{t:>9.3} {h:>9.3}"))
                })
                .map(|c| format!("{c:>19}"))
                .collect();
            text.push_str(&format!("{layer:>5} | {}\n", cells.join(" | ")));
        }
    }

    Ok(text)
}

/// The texts of the queries file, keeping the rows of one split.
fn read_queries(path: &Path, split: &str) -> Result<Vec<String>, Error> {
    let keep: Option<HashSet<u64>> = match split {
        "all" => None,
        _ => {
            let splits = path.with_extension("splits.json");
            let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&splits)?).map_err(damaged)?;
            let rows = v[split]
                .as_array()
                .ok_or_else(|| damaged(format!("{} has no split {split}", splits.display())))?;
            Some(rows.iter().filter_map(serde_json::Value::as_u64).collect())
        }
    };
    let file = File::open(path).map_err(|e| Error::new(e.kind(), format!("{}: {e}", path.display())))?;
    let mut texts = Vec::new();

    for (i, line) in BufReader::new(file).lines().enumerate() {
        let v: serde_json::Value = serde_json::from_str(&line?).map_err(damaged)?;
        let row = v["row"].as_u64().unwrap_or(i as u64);
        if keep.as_ref().is_none_or(|k| k.contains(&row)) {
            let text = v["text"]
                .as_str()
                .ok_or_else(|| damaged(format!("line {i} has no text")))?;
            texts.push(text.to_string());
        }
    }

    match texts.is_empty() {
        true => Err(damaged(format!("{} has no queries in split {split}", path.display()))),
        false => Ok(texts),
    }
}

/// The machine's state that the results record, beyond what `provenance` adds.
struct Machine {
    perf_cores: String,
    efficiency_cores: String,
    gpu_cores: String,
    os: String,
    power: String,
}

impl Machine {
    fn now() -> Machine {
        let accelerator = command("ioreg", &["-r", "-d", "1", "-c", "AGXAccelerator", "-w0"]);
        let gpu_cores = accelerator
            .split("\"gpu-core-count\" = ")
            .nth(1)
            .and_then(|s| s.split(|c: char| !c.is_ascii_digit()).next())
            .unwrap_or_default();
        let os = format!(
            "{} {}",
            command("sw_vers", &["-productVersion"]),
            command("sw_vers", &["-buildVersion"])
        );
        let power = command("pmset", &["-g", "batt"]);
        Machine {
            perf_cores: sysctl("hw.perflevel0.physicalcpu"),
            efficiency_cores: sysctl("hw.perflevel1.physicalcpu"),
            gpu_cores: gpu_cores.to_string(),
            os: os.trim().to_string(),
            power: power.lines().next().unwrap_or_default().to_string(),
        }
    }

    fn add(&self, json: Json) -> Json {
        json.str("perf_cores", &self.perf_cores)
            .str("efficiency_cores", &self.efficiency_cores)
            .str("gpu_cores", &self.gpu_cores)
            .str("os", &self.os)
            .str("power", &self.power)
    }
}

#[cfg(target_os = "macos")]
fn set_user_interactive() {
    // SAFETY: the call changes only the calling thread's scheduling class, and it takes no pointers.
    let _ = unsafe { libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0) };
}

#[cfg(not(target_os = "macos"))]
fn set_user_interactive() {}

fn repo_id(model: &str) -> &str {
    match model {
        "arctic" => "Snowflake/snowflake-arctic-embed-m-v1.5",
        "minilm" => "sentence-transformers/all-MiniLM-L6-v2",
        other => other,
    }
}

/// An id shared by the lines of one invocation, from the time and the process id.
fn run_id() -> String {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    format!("{}-{}", now.as_millis(), std::process::id())
}

fn sysctl(name: &str) -> String {
    command("sysctl", &["-n", name])
}

fn command(program: &str, args: &[&str]) -> String {
    Command::new(program)
        .args(args)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn median(values: &[f64]) -> f64 {
    percentile(values, 50.0)
}

/// The nearest-rank percentile `p` (0 to 100), the smallest value that at least `p` percent of the values are at or
/// below.
fn percentile(values: &[f64], p: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);

    match sorted.len() {
        0 => f64::NAN,
        n => sorted[((p / 100.0 * n as f64).ceil() as usize).clamp(1, n) - 1],
    }
}

fn damaged(e: impl ToString) -> Error {
    Error::new(ErrorKind::InvalidData, e.to_string())
}

/// Arguments of the form `--name value`.
struct Args(HashMap<String, String>);

impl Args {
    /// Reads `args`, the arguments after the program's name. A name that `NAMES` doesn't list, a name given twice, or
    /// a bad value gives `InvalidInput`.
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Args, Error> {
        let mut values = HashMap::new();

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

        let args = Args(values);

        if args.get::<usize>("runs", 50)? == 0 || args.get::<usize>("threads", 1)? == 0 {
            return Err(invalid("--runs and --threads are at least 1".to_string()));
        }

        args.get::<usize>("warmup", 20)?;

        match (args.str("device"), args.str("threads")) {
            (Some("cpu" | "metal"), _) | (None, None) => Ok(args),
            (Some(_), _) => Err(invalid("--device is cpu or metal".to_string())),
            (None, Some(_)) => Err(invalid(
                "--threads goes with --device, since each setup sets its own otherwise".to_string(),
            )),
        }
    }

    fn str(&self, name: &str) -> Option<&str> {
        self.0.get(name).map(String::as_str)
    }

    /// The value of `--name`, or `default` when it isn't given.
    fn get<T: FromStr>(&self, name: &str, default: T) -> Result<T, Error> {
        match self.str(name) {
            Some(v) => v.parse().map_err(|_| invalid(format!("--{name} has a bad value: {v}"))),
            None => Ok(default),
        }
    }

    /// The path given as `--name`, else `default` inside this repository.
    fn path(&self, name: &str, default: &str) -> PathBuf {
        match self.str(name) {
            Some(path) => PathBuf::from(path),
            None => Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(default),
        }
    }
}

fn invalid(message: String) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

#[cfg(test)]
#[path = "../../tests/unit/layers.rs"]
mod tests;
