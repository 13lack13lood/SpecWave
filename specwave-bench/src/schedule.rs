//! The fire layer that SpecWave's scheduler picks from measured inputs, i.e. layer times, cold read times and the
//! coverage grid.
//!
//! - **Profiles** come from the lines of the `layers` binary, with T_l, the median time to the end of layer l, for each
//!   model, device and thread count.
//! - **Read times** come from the cold trials of ColdIVF's `cold` harness, which time a search with the true query at
//!   the clusters and rows of a guess. A source is a start state and a read, eg. `prewarm` and `first`. Its chain for a
//!   number of clusters and rows is the median route time (`route_us`), plus the median wave time (`cluster_wave_us`),
//!   plus the median scan time (`scan_us`), plus the median row time (`rows_us`), over the trials that waited for the
//!   cluster tables first. Its tables' read is the median of `prepare_us` + `tables_us` over the trials that didn't
//!   wait. Medians are by nearest rank.
//! - **Each layer's β and m** come from the coverage grid. They are the layer's first passing cell, by β and then by m,
//!   under the verdict module's rule, so m is the smallest multiple whose rerank coverage reaches the target at the
//!   first β whose recall passes. A layer with no passing cell may not guess.

use std::collections::BTreeMap;
use std::fmt::Write;
use std::io::{Error, ErrorKind};

use coldivf::SearchParams;
use specwave::encoder::models;
use specwave::scheduler::{Fire, LayerProfile, LayerSettings, ReadTimes, Schedule, choose, evaluate};
use specwave::speculate::SpecParams;

use crate::results::Json;
use crate::verdict::{Cell, DEFINITIONS, Rule, Verdict, fit_of, read};

/// The time profile of one model on one device.
#[derive(Clone, Debug, PartialEq)]
pub struct Setup {
    pub model: String,
    pub device: String,
    pub threads: u64,
    pub profile: LayerProfile,
}

impl Setup {
    /// The setup's name, eg. "cpu, 5 threads" or "metal".
    pub fn name(&self) -> String {
        match self.device.as_str() {
            "cpu" => format!("cpu, {} threads", self.threads),
            device => device.to_string(),
        }
    }
}

/// The profile of each model, device and thread count in `lines`, the result lines of the `layers` binary. Other lines
/// are skipped, and when two lines time the same layer of a setup, the later one counts. A layer line without a field
/// the profile needs, or a setup whose times don't make a profile, gives `InvalidData`.
pub fn profiles(lines: &[String]) -> Result<Vec<Setup>, Error> {
    let mut times: BTreeMap<(String, String, u64), BTreeMap<usize, f64>> = BTreeMap::new();

    for (i, line) in lines.iter().enumerate() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if v["bench"] != "layers" {
            continue;
        }
        let missing = |key: &str| damaged(&format!("layer line {} has no {key}", i + 1));
        let text = |key: &str| v[key].as_str().map(str::to_string).ok_or_else(|| missing(key));
        let whole = |key: &str| v[key].as_u64().ok_or_else(|| missing(key));
        let key = (text("model")?, text("device")?, whole("threads")?);
        let layer = whole("layer")? as usize;
        let t = v["t_ms"].as_f64().ok_or_else(|| missing("t_ms"))?;
        times.entry(key).or_default().insert(layer, t);
    }

    times
        .into_iter()
        .map(|((model, device, threads), layers)| {
            let profile = LayerProfile::new(layers.into_iter().collect())
                .map_err(|e| damaged(&format!("{model} on {device} with {threads} threads: {e}")))?;
            Ok(Setup {
                model,
                device,
                threads,
                profile,
            })
        })
        .collect()
}

/// A start state and a read of the cold trials, eg. `prewarm` and `first`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Source {
    /// The `variant` field, `idle` for a search started from sleep or `prewarm` for one on a core kept busy first.
    pub variant: String,
    /// The `read` field, `first` for the first read of the trial's pages or `repeat` for a read soon after another.
    pub read: String,
}

impl Source {
    pub fn new(variant: &str, read: &str) -> Source {
        Source {
            variant: variant.to_string(),
            read: read.to_string(),
        }
    }

    pub fn name(&self) -> String {
        format!("{} {}", self.variant, self.read)
    }
}

/// The medians of one cell's cold trials, in ms.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Chain {
    pub route_ms: f64,
    pub wave_ms: f64,
    pub scan_ms: f64,
    pub rows_ms: f64,
    pub trials: usize,
}

impl Chain {
    pub fn ms(&self) -> f64 {
        self.route_ms + self.wave_ms + self.scan_ms + self.rows_ms
    }
}

/// The medians of the cold trials, by source.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ColdReads {
    /// By source, clusters and rows.
    chains: BTreeMap<(Source, usize, usize), Chain>,
    /// The tables' read by source, in ms.
    tables: BTreeMap<Source, f64>,
}

/// The medians of the cold trials in `lines`, the result lines of ColdIVF's `cold` harness. Lines without `variant` are
/// skipped. A trial line without a field the medians need gives `InvalidData`.
pub fn cold_reads(lines: &[String]) -> Result<ColdReads, Error> {
    type Samples = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);
    let mut chains: BTreeMap<(Source, usize, usize), Samples> = BTreeMap::new();
    let mut tables: BTreeMap<Source, Vec<f64>> = BTreeMap::new();

    for (i, line) in lines.iter().enumerate() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let Some(variant) = v["variant"].as_str() else {
            continue;
        };
        let missing = |key: &str| damaged(&format!("trial line {} has no {key}", i + 1));
        let number = |key: &str| v[key].as_f64().ok_or_else(|| missing(key));
        let source = Source::new(variant, v["read"].as_str().ok_or_else(|| missing("read"))?);
        let clusters = v["nprobe"].as_u64().ok_or_else(|| missing("nprobe"))? as usize;
        let rows = v["rerank"].as_u64().ok_or_else(|| missing("rerank"))? as usize;

        if number("gap_ms")? > 0.0 {
            let (route, wave, scan, row) = chains.entry((source, clusters, rows)).or_default();
            route.push(number("route_us")? / 1e3);
            wave.push(number("cluster_wave_us")? / 1e3);
            scan.push(number("scan_us")? / 1e3);
            row.push(number("rows_us")? / 1e3);
        } else {
            let read = number("prepare_us")? + number("tables_us")?;
            tables.entry(source).or_default().push(read / 1e3);
        }
    }

    Ok(ColdReads {
        chains: chains
            .into_iter()
            .map(|(key, (route, wave, scan, rows))| {
                let chain = Chain {
                    route_ms: median(&route),
                    wave_ms: median(&wave),
                    scan_ms: median(&scan),
                    rows_ms: median(&rows),
                    trials: wave.len(),
                };
                (key, chain)
            })
            .collect(),
        tables: tables.into_iter().map(|(source, t)| (source, median(&t))).collect(),
    })
}

impl ColdReads {
    /// Every source with a tables' read, in order.
    pub fn sources(&self) -> Vec<Source> {
        self.tables.keys().cloned().collect()
    }

    pub fn chain(&self, source: &Source, clusters: usize, rows: usize) -> Option<Chain> {
        self.chains.get(&(source.clone(), clusters, rows)).copied()
    }

    /// The read times of `source`, if it has a tables' read.
    pub fn of(&self, source: &Source) -> Option<Measured<'_>> {
        Some(Measured {
            reads: self,
            source: source.clone(),
            tables_ms: *self.tables.get(source)?,
        })
    }
}

/// The read times of one source. A chain that the trials didn't measure has no time, which the scheduler refuses.
pub struct Measured<'a> {
    reads: &'a ColdReads,
    source: Source,
    tables_ms: f64,
}

impl Measured<'_> {
    /// Whether the trials measured the chain of every layer of `layers` for searches with `base`.
    pub fn covers(&self, base: &SearchParams, layers: &[LayerSettings]) -> bool {
        layers.iter().all(|s| {
            let params = SpecParams {
                base: *base,
                beta: s.beta,
                rerank_mult: s.rerank_mult,
            };
            self.reads
                .chain(&self.source, params.guess_nprobe(), params.prefetch_rows())
                .is_some()
        })
    }
}

impl ReadTimes for Measured<'_> {
    fn tables_ms(&self) -> f64 {
        self.tables_ms
    }

    fn chain_ms(&self, clusters: usize, rows: usize) -> f64 {
        self.reads
            .chain(&self.source, clusters, rows)
            .map_or(f64::NAN, |c| c.ms())
    }
}

/// The layers that may guess for `base`, each with its β and m. They come from each layer's first passing cell under
/// `rule` with the rerank coverage `definition`, a position in the verdict module's `DEFINITIONS`. Layers whose cells
/// don't pass are left out, and so are the cells of guesses without a layer.
pub fn settings(cells: &[Cell], rule: &Rule, definition: usize, base: &Base) -> Vec<LayerSettings> {
    let mut groups: BTreeMap<usize, Vec<&Cell>> = BTreeMap::new();

    for c in cells {
        let Some(layer) = c.layer else {
            continue;
        };
        if (c.dataset.as_str(), c.clustering.as_str(), c.nprobe, c.r_label.as_str())
            == (
                base.dataset.as_str(),
                base.clustering.as_str(),
                base.nprobe,
                base.r_label.as_str(),
            )
            && fit_of(&c.guess) == base.fit
        {
            groups.entry(layer).or_default().push(c);
        }
    }

    groups
        .into_iter()
        .filter_map(|(layer, group)| match rule.decide(&group, definition) {
            (Verdict::Pass, Some(c)) => Some(LayerSettings {
                layer,
                beta: c.beta,
                rerank_mult: u8::try_from(c.mult).unwrap_or(u8::MAX),
            }),
            _ => None,
        })
        .collect()
}

/// The searches and guesses that a schedule is for.
#[derive(Clone, Debug, PartialEq)]
pub struct Base {
    pub dataset: String,
    /// The guesses' fit, eg. "openings" for the guesses "openings.L9" and the others of that fit.
    pub fit: String,
    pub clustering: String,
    pub nprobe: usize,
    pub r_label: String,
    /// R, the rescoring depth of the label.
    pub rerank: usize,
}

impl Base {
    pub fn params(&self) -> SearchParams {
        SearchParams {
            k: 10,
            nprobe: self.nprobe,
            rerank: self.rerank,
            reach: None,
        }
    }
}

/// Every base of the cells of `fit` on `dataset` with `clustering`, in order of nprobe and then R.
pub fn bases(cells: &[Cell], dataset: &str, fit: &str, clustering: &str) -> Vec<Base> {
    let mut found: Vec<Base> = Vec::new();

    for c in cells {
        if c.dataset != dataset || c.clustering != clustering || fit_of(&c.guess) != fit || c.layer.is_none() {
            continue;
        }
        let base = Base {
            dataset: dataset.to_string(),
            fit: fit.to_string(),
            clustering: clustering.to_string(),
            nprobe: c.nprobe,
            r_label: c.r_label.clone(),
            rerank: c.rerank,
        };
        if !found.contains(&base) {
            found.push(base);
        }
    }

    found.sort_by_key(|b| (b.nprobe, b.rerank));
    found
}

/// What the scheduler picks for one setup, source and base.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub setup: String,
    pub source: Source,
    pub base: Base,
    /// The layers that may guess, with their β and m.
    pub layers: Vec<LayerSettings>,
    /// `None` when the trials didn't measure the chain of every layer that may guess.
    pub schedule: Option<Schedule>,
    /// The layers after the fire layer that may guess, which the rule turned down.
    pub later: Vec<Fire>,
}

/// The scheduler's pick for every setup, source and base, with each layer's β and m from `cells` under `rule` with the
/// rerank coverage `definition`. A source's base whose layers' chains the trials didn't all measure gets no schedule.
pub fn table(
    setups: &[Setup],
    reads: &ColdReads,
    cells: &[Cell],
    rule: &Rule,
    definition: usize,
    bases: &[Base],
) -> Result<Vec<Row>, Error> {
    let mut rows = Vec::new();

    for setup in setups {
        for source in reads.sources() {
            let Some(measured) = reads.of(&source) else {
                continue;
            };
            for base in bases {
                let layers = settings(cells, rule, definition, base);
                let params = base.params();
                let (schedule, later) = match measured.covers(&params, &layers) {
                    true => {
                        let schedule = choose(&setup.profile, &measured, &params, &layers, None)?;
                        let fired = schedule.fire.map_or(0, |f| f.settings.layer);
                        let later = layers
                            .iter()
                            .filter(|s| schedule.fire.is_some() && s.layer > fired)
                            .map(|s| evaluate(&setup.profile, &measured, &params, s))
                            .collect::<Result<Vec<Fire>, Error>>()?;
                        (Some(schedule), later)
                    }
                    false => (None, Vec::new()),
                };
                rows.push(Row {
                    setup: setup.name(),
                    source: source.clone(),
                    base: base.clone(),
                    layers,
                    schedule,
                    later,
                });
            }
        }
    }

    Ok(rows)
}

/// The rows as a Markdown table, with times in ms to two decimals.
pub fn markdown(rows: &[Row]) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "| Setup | Source | nprobe | R | Fire Layer | β, m | Clusters Read, Rows Prefetched | T_L − T_l | Needed | \
         Later Layers |"
    );
    let _ = writeln!(out, "|---|---|---|---|---|---|---|---|---|---|");

    for row in rows {
        let head = format!(
            "| {} | {} | {} | {} |",
            row.setup,
            row.source.name(),
            row.base.nprobe,
            row.base.r_label
        );
        let Some(schedule) = row.schedule else {
            let _ = writeln!(out, "{head} not measured | | | | | |");
            continue;
        };
        let Some(fire) = schedule.fire else {
            let _ = writeln!(out, "{head} none | | | | | |");
            continue;
        };
        let later: Vec<String> = row
            .later
            .iter()
            .map(|f| format!("L{} needs {:.2} of {:.2}", f.settings.layer, f.needed_ms, f.left_ms))
            .collect();
        let _ = writeln!(
            out,
            "{head} L{}{} | {}, {} | {}, {} | {:.2} | {:.2} | {} |",
            fire.settings.layer,
            if fire.fits() { "" } else { " (reads run over)" },
            fire.settings.beta,
            fire.settings.rerank_mult,
            fire.clusters,
            fire.rows,
            fire.left_ms,
            fire.needed_ms,
            if later.is_empty() {
                "none".to_string()
            } else {
                later.join(", ")
            }
        );
    }

    out
}

/// The result line of `row`, with the fire layer's settings and times, which are null without one.
pub fn json(row: &Row) -> Json {
    let fire = row.schedule.and_then(|s| s.fire);
    let pick = |f: fn(&Fire) -> f64| fire.as_ref().map_or(f64::NAN, f);
    Json::new()
        .str("record", "schedule")
        .str("setup", &row.setup)
        .str("source", &row.source.name())
        .str("dataset", &row.base.dataset)
        .str("fit", &row.base.fit)
        .str("clustering", &row.base.clustering)
        .int("nprobe", row.base.nprobe as u64)
        .str("r_label", &row.base.r_label)
        .int("rerank", row.base.rerank as u64)
        .ints("layers", &row.layers.iter().map(|s| s.layer as u64).collect::<Vec<_>>())
        .bool("measured", row.schedule.is_some())
        .opt("fire_layer", fire.map(|f| f.settings.layer as u64))
        .num("beta", pick(|f| f.settings.beta))
        .opt("mult", fire.map(|f| u64::from(f.settings.rerank_mult)))
        .opt("clusters", fire.map(|f| f.clusters as u64))
        .opt("rows", fire.map(|f| f.rows as u64))
        .num("left_ms", pick(|f| f.left_ms))
        .num("needed_ms", pick(|f| f.needed_ms))
        .bool("fits", fire.is_some_and(|f| f.fits()))
}

/// The command line of the `schedule` binary.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// The coverage results files, judged with the verdict module's default rule.
    pub coverage: Vec<String>,
    /// The results file of the `layers` binary.
    pub layers: String,
    /// The results file of ColdIVF's cold trials.
    pub cold: String,
    pub dataset: String,
    pub fit: String,
    pub clustering: String,
    /// The model whose setups get rows, by its repository id, eg. "Snowflake/snowflake-arctic-embed-m-v1.5".
    pub model: String,
    /// The rerank coverage that the rule judges, a position in the verdict module's `DEFINITIONS`.
    pub definition: usize,
    /// The results file that a line for each row is appended to, if any.
    pub out: Option<String>,
}

/// Every argument name of the `schedule` binary.
const NAMES: [&str; 9] = [
    "coverage",
    "layers",
    "cold",
    "dataset",
    "fit",
    "clustering",
    "model",
    "definition",
    "out",
];

impl Settings {
    /// Reads `args`, the arguments after the program's name. `--coverage`, `--layers` and `--cold` are required, and
    /// `--coverage` takes files separated by commas. A name that isn't known, a name given twice, a missing value or a
    /// definition that isn't one of `DEFINITIONS` gives `InvalidInput`.
    pub fn parse(args: &[String]) -> Result<Settings, Error> {
        let mut values: Vec<(&str, &str)> = Vec::new();
        let mut it = args.iter();

        while let Some(arg) = it.next() {
            let name = arg
                .strip_prefix("--")
                .filter(|n| NAMES.contains(n))
                .ok_or_else(|| invalid(&format!("unknown argument {arg}")))?;
            let value = it.next().ok_or_else(|| invalid(&format!("--{name} needs a value")))?;
            if values.iter().any(|(n, _)| *n == name) {
                return Err(invalid(&format!("--{name} is given twice")));
            }
            values.push((name, value));
        }

        let get = |name: &str| values.iter().find(|(n, _)| *n == name).map(|&(_, v)| v.to_string());
        let need = |name: &str| get(name).ok_or_else(|| invalid(&format!("--{name} is required")));
        let definition = get("definition").unwrap_or_else(|| DEFINITIONS[1].to_string());
        let definition = DEFINITIONS
            .iter()
            .position(|d| *d == definition)
            .ok_or_else(|| invalid(&format!("--definition is one of {}", DEFINITIONS.join(", "))))?;
        Ok(Settings {
            coverage: need("coverage")?.split(',').map(str::to_string).collect(),
            layers: need("layers")?,
            cold: need("cold")?,
            dataset: get("dataset").unwrap_or_else(|| "fiqa".to_string()),
            fit: get("fit").unwrap_or_else(|| "openings".to_string()),
            clustering: get("clustering").unwrap_or_else(|| "l2".to_string()),
            model: models::repo(&get("model").unwrap_or_else(|| "arctic".to_string()))?.to_string(),
            definition,
            out: get("out"),
        })
    }
}

/// The rows of the files that `s` names. Files that hold no layer times for the model, or no grid cells for the
/// dataset, fit and clustering, give `InvalidData`.
pub fn rows(s: &Settings) -> Result<Vec<Row>, Error> {
    let lines = |files: &[String]| -> Result<Vec<String>, Error> {
        let mut lines = Vec::new();
        for file in files {
            let text = std::fs::read_to_string(file).map_err(|e| Error::new(e.kind(), format!("{file}: {e}")))?;
            lines.extend(text.lines().map(str::to_string));
        }
        Ok(lines)
    };
    let cells = read(&lines(&s.coverage)?)?;
    let setups: Vec<Setup> = profiles(&lines(std::slice::from_ref(&s.layers))?)?
        .into_iter()
        .filter(|setup| setup.model == s.model)
        .collect();
    let reads = cold_reads(&lines(std::slice::from_ref(&s.cold))?)?;
    let bases = bases(&cells, &s.dataset, &s.fit, &s.clustering);

    if setups.is_empty() || bases.is_empty() {
        return Err(damaged(
            "the files hold no layer times for the model, or no grid cells for the dataset, fit and clustering",
        ));
    }

    table(&setups, &reads, &cells, &Rule::default(), s.definition, &bases)
}

/// The median of `values` by nearest rank.
fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);

    match sorted.len() {
        0 => f64::NAN,
        n => sorted[n.div_ceil(2) - 1],
    }
}

fn invalid(message: &str) -> Error {
    Error::new(ErrorKind::InvalidInput, message.to_string())
}

fn damaged(message: &str) -> Error {
    Error::new(ErrorKind::InvalidData, message.to_string())
}

#[cfg(test)]
#[path = "../tests/unit/schedule.rs"]
mod tests;
