//! The run of the `spec_coverage` binary, from its command line to its result lines.
//!
//! For each dataset, clustering and rotation seed, it builds the index from the documents and their clusters, as a
//! file for routes and reads and in memory for normal searches. It first sweeps the rescoring depth R of normal
//! searches at each base nprobe, for the smallest R that reaches the target recall. On Quora, with 384 dimensions,
//! that depth is R384, which the sweeps of the first seed give. Every dataset's grid tries it next to the depths of
//! `--reranks`, at the same share of clusters and with the same kind of clustering. Then for every guess file, base
//! nprobe, beta, R and prefetch multiple m, it measures the cells that the coverage module defines.

use std::collections::{HashMap, HashSet};
use std::io::{Error, ErrorKind};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{Instant, UNIX_EPOCH};

use coldivf::Clusters;

use crate::coverage::{self, Base, Cell, Depth, Exact, Grid, Indexes, K, Queries, RESAMPLE_SEED, RESAMPLES, Resamples};
use crate::data::{COSINE, Fbin, Files, Guess, Ibin, SPHERICAL, clusters, clusters_of};
use crate::results::{BUILD, Json, append, names_its_code, provenance};

/// The dataset whose sweeps give R384.
pub const R384_FROM: &str = "quora";

/// The share of the ceiling, the recall with every vector of the clusters rescored, that a sweep also finds a depth
/// for.
const NEAR_CEILING: f64 = 0.99;

/// Every argument name. `redo` alone takes no value.
const NAMES: [&str; 15] = [
    "out",
    "data",
    "datasets",
    "kinds",
    "guesses",
    "queries",
    "fractions",
    "betas",
    "reranks",
    "mults",
    "target",
    "r384",
    "seeds",
    "threads",
    "redo",
];

/// Where R384 comes from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum R384 {
    /// The sweeps of the Quora dataset, which runs first.
    Measured,
    Fixed(usize),
    /// No grid tries it.
    Skipped,
}

/// The command line's settings.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub out: PathBuf,
    pub data: PathBuf,
    pub datasets: Vec<String>,
    pub kinds: Vec<String>,
    pub guesses: Option<Vec<String>>,
    pub queries: Option<usize>,
    pub fractions: Vec<f64>,
    pub betas: Vec<f64>,
    pub reranks: Vec<usize>,
    pub mults: Vec<usize>,
    pub target: f64,
    pub r384: R384,
    /// The seeds of the indexes' rotations. Each one builds its own pair of indexes.
    pub seeds: Vec<u64>,
    pub threads: usize,
    pub redo: bool,
}

impl Settings {
    /// Reads `args`, the arguments after the program's name. A bad argument gives `InvalidInput`.
    pub fn parse(args: &[String]) -> Result<Settings, Error> {
        let mut values: HashMap<&str, &str> = HashMap::new();
        let mut redo = false;
        let mut it = args.iter();

        while let Some(arg) = it.next() {
            let name = arg
                .strip_prefix("--")
                .filter(|n| NAMES.contains(n))
                .ok_or_else(|| invalid(&format!("unknown argument {arg}")))?;
            match name {
                "redo" => redo = true,
                _ => {
                    let value = it.next().ok_or_else(|| invalid(&format!("--{name} needs a value")))?;
                    if values.insert(name, value).is_some() {
                        return Err(invalid(&format!("--{name} is given twice")));
                    }
                }
            }
        }

        let get = |name: &str| values.get(name).copied();
        let out = get("out").ok_or_else(|| invalid("--out FILE is required"))?;
        let mut datasets = list(&values, "datasets", &["quora".to_string(), "fiqa".to_string()])?;
        // The sweeps that give R384 run first.
        datasets.sort_by_key(|d| d != R384_FROM);
        let r384 = match get("r384") {
            None => R384::Measured,
            Some("none") => R384::Skipped,
            Some(r) => R384::Fixed(one(r, "r384")?),
        };
        let settings = Settings {
            out: PathBuf::from(out),
            data: get("data").map_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../data"), PathBuf::from),
            datasets,
            kinds: list(
                &values,
                "kinds",
                &["l2".to_string(), COSINE.to_string(), SPHERICAL.to_string()],
            )?,
            guesses: values
                .contains_key("guesses")
                .then(|| list(&values, "guesses", &[]))
                .transpose()?,
            queries: get("queries").map(|q| one(q, "queries")).transpose()?,
            fractions: list(&values, "fractions", &[0.017, 0.0256, 0.0307])?,
            betas: list(&values, "betas", &[0.0, 0.25, 0.5, 1.0])?,
            reranks: list(&values, "reranks", &[29, 39])?,
            mults: list(&values, "mults", &[2, 3, 4])?,
            target: get("target").map_or(Ok(0.95), |t| one(t, "target"))?,
            r384,
            seeds: list(&values, "seeds", &[7, 11, 23])?,
            threads: get("threads").map_or(Ok(4), |t| one(t, "threads"))?,
            redo,
        };
        settings.check()?;
        Ok(settings)
    }

    fn check(&self) -> Result<(), Error> {
        let share = |x: f64| x.is_finite() && x > 0.0 && x <= 1.0;
        let repeats = |v: &[String]| v.iter().collect::<HashSet<_>>().len() != v.len();

        if self.r384 == R384::Measured && !self.datasets.iter().any(|d| d == R384_FROM) {
            return Err(invalid(
                "R384 comes from the quora sweeps, so --datasets needs quora unless --r384 gives it",
            ));
        }

        if repeats(&self.datasets)
            || repeats(&self.kinds)
            || self.seeds.iter().collect::<HashSet<_>>().len() != self.seeds.len()
        {
            return Err(invalid("--datasets, --kinds and --seeds each name a value once"));
        }

        if !self.fractions.iter().all(|&f| share(f)) {
            return Err(invalid(
                "every share of --fractions is above 0 and at most 1, eg. 0.0256 for 2.56%",
            ));
        }

        if !share(self.target) {
            return Err(invalid("--target is a recall above 0 and at most 1"));
        }

        if !self.betas.iter().all(|b| b.is_finite() && *b >= 0.0) {
            return Err(invalid("every beta of --betas is finite and at least 0"));
        }

        let r384 = match self.r384 {
            R384::Fixed(r) => Some(r),
            _ => None,
        };

        if self.reranks.iter().chain(&r384).any(|&r| r < K) || self.mults.contains(&0) {
            return Err(invalid(
                "every depth of --reranks and --r384 is at least 10, and every multiple at least 1",
            ));
        }

        if self.threads == 0 || self.queries == Some(0) {
            return Err(invalid("--threads and --queries are at least 1"));
        }

        Ok(())
    }
}

fn one<T: FromStr>(value: &str, name: &str) -> Result<T, Error> {
    value
        .trim()
        .parse()
        .map_err(|_| invalid(&format!("--{name} has a bad value: {value}")))
}

/// The comma-separated list of `--name`, or `default` when it isn't given. Neither it nor any of its items can be
/// empty.
fn list<T: FromStr + Clone>(values: &HashMap<&str, &str>, name: &str, default: &[T]) -> Result<Vec<T>, Error> {
    let item = |x: &str| match x.trim().is_empty() {
        true => Err(invalid(&format!("--{name} has an empty item"))),
        false => one(x, name),
    };

    match values.get(name) {
        Some(v) => v.split(',').map(item).collect(),
        None if default.is_empty() => Err(invalid(&format!("--{name} needs at least one value"))),
        None => Ok(default.to_vec()),
    }
}

/// One run's settings, its id, and the lines its results file held when it started.
struct Job<'a> {
    s: &'a Settings,
    run: &'a str,
    done: &'a HashSet<String>,
}

/// Measures everything that `s` asks for and appends the results to `s.out`. `command` is the command line, for the
/// header line.
pub fn run(s: &Settings, command: &str) -> Result<(), Error> {
    let started = UNIX_EPOCH.elapsed().map_or(0, |d| d.as_secs());
    let run = format!("{started}-{}", std::process::id());
    let done = lines_of(&s.out)?;

    if !names_its_code(BUILD) {
        eprintln!("this build doesn't name its code ({BUILD}), so every guess is measured again");
    }

    let header = provenance(Json::new().str("run", &run).str("record", "run"))
        .str("command", command)
        .str("data", &s.data.display().to_string())
        .int("threads", s.threads as u64)
        .num("target", s.target)
        .ints("seeds", &s.seeds)
        .int("resamples", RESAMPLES as u64)
        .int("resample_seed", RESAMPLE_SEED)
        .num("interval", 0.95)
        .line();
    append(&s.out, &[header])?;
    let job = Job {
        s,
        run: &run,
        done: &done,
    };
    let mut r384 = HashMap::new();

    for name in &s.datasets {
        dataset(&job, name, &mut r384)?;
    }

    Ok(())
}

/// Every line of the file at `path`, or none when there is no file yet.
fn lines_of(path: &Path) -> Result<HashSet<String>, Error> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(text.lines().map(str::to_string).collect()),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(HashSet::new()),
        Err(e) => Err(e),
    }
}

/// One dataset's opened files and its chosen guesses.
struct Data<'a> {
    name: &'a str,
    files: &'a Files,
    corpus: &'a Fbin,
    queries: Queries<'a>,
    /// The number of queries in the files, which `--queries` may cut.
    all_queries: usize,
    guesses: &'a [Guess],
    resamples: &'a Resamples,
}

/// The R384 found for each clustering, one per share of clusters.
type R384s = HashMap<String, Vec<Option<usize>>>;

/// Runs every clustering and seed of the dataset `name`. Damaged files give `InvalidData`.
fn dataset(job: &Job, name: &str, r384: &mut R384s) -> Result<(), Error> {
    let files = Files::new(&job.s.data, name);
    let corpus = Fbin::open(&files.corpus())?;
    let e = Fbin::open(&files.queries())?;
    let truth = Ibin::open(&files.truth())?;

    if corpus.dim() != e.dim() || truth.len() != e.len() {
        return Err(damaged(&format!(
            "{name}: the documents, queries and true neighbors don't match"
        )));
    }

    if truth.width() < K {
        let path = files.truth();
        return Err(damaged(&format!(
            "{} holds fewer than {K} true neighbors per query",
            path.display()
        )));
    }

    if let Some(i) = truth.values().iter().position(|&id| id as usize >= corpus.len()) {
        let (path, row) = (files.truth(), i / truth.width());
        return Err(damaged(&format!(
            "{}: query {row} has a true neighbor past the last of the {} documents",
            path.display(),
            corpus.len()
        )));
    }

    if let Some(row) = coverage::bad_row(corpus.values(), corpus.dim()) {
        let path = files.corpus();
        return Err(damaged(&format!(
            "{}: document {row} is zero or not finite",
            path.display()
        )));
    }

    let n = job.s.queries.map_or(e.len(), |n| n.min(e.len()));
    let (dim, width) = (e.dim(), truth.width());
    let queries = Queries::new(dim, &e.values()[..n * dim], &truth.values()[..n * width], width)?;

    if let Some(row) = coverage::bad_row(queries.vectors, dim) {
        let path = files.queries();
        return Err(damaged(&format!(
            "{}: query {row} is zero or not finite",
            path.display()
        )));
    }

    let guesses: Vec<Guess> = files
        .guesses()?
        .into_iter()
        .filter(|g| job.s.guesses.as_ref().is_none_or(|wanted| wanted.contains(&g.label)))
        .collect();

    if guesses.is_empty() {
        eprintln!("{name}: no guess files, so only the sweeps run");
    } else {
        check_targets(&files)?;
    }

    let resamples = Resamples::new(queries.len(), RESAMPLES, RESAMPLE_SEED);
    let data = Data {
        name,
        files: &files,
        corpus: &corpus,
        queries,
        all_queries: e.len(),
        guesses: &guesses,
        resamples: &resamples,
    };

    for kind in &job.s.kinds {
        let clusters = clusters(&files, kind, &corpus)?;
        for &seed in &job.s.seeds {
            clustering(job, &data, kind, &clusters, seed, r384)?;
        }
    }

    Ok(())
}

/// Fails with `InvalidData` unless `spec/<name>/e.fbin`, the final embeddings that the guesses were fitted to, holds the
/// same bytes as the test queries' file. The data step writes it last, once every guess and the true neighbors belong
/// to the test queries, so a match says that they do.
fn check_targets(files: &Files) -> Result<(), Error> {
    let (targets, queries) = (files.targets(), files.queries());
    let read = |path: &Path| std::fs::read(path).map_err(|e| Error::new(e.kind(), format!("{}: {e}", path.display())));

    if read(&targets)? != read(&queries)? {
        return Err(damaged(&format!(
            "{} differs from {}, so the guesses belong to other queries",
            targets.display(),
            queries.display()
        )));
    }

    Ok(())
}

/// Builds the indexes of one clustering with one seed, sweeps them, then measures every guess that isn't done yet.
fn clustering(
    job: &Job,
    data: &Data,
    kind: &str,
    clusters: &Clusters,
    seed: u64,
    r384: &mut R384s,
) -> Result<(), Error> {
    let s = job.s;
    let start = Instant::now();
    let path = data.files.index(kind);

    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }

    let indexes = Indexes::build(data.corpus, clusters, &path, s.threads, seed)?;
    let seconds = start.elapsed().as_secs_f64();
    eprintln!(
        "{} {kind} seed {seed}: indexed {} documents in {} clusters in {seconds:.1} s",
        data.name,
        data.corpus.len(),
        indexes.nlist
    );
    let index = Json::new()
        .str("run", job.run)
        .str("record", "index")
        .str("build", BUILD)
        .str("dataset", data.name)
        .str("clustering", kind)
        .int("seed", seed)
        .int("documents", data.corpus.len() as u64)
        .int("dim", data.queries.dim as u64)
        .int("nlist", indexes.nlist as u64)
        .str("cluster_files", clusters_of(kind).0)
        .bool("unit_centers", clusters_of(kind).1)
        .str("path", &path.display().to_string())
        .int("bytes", std::fs::metadata(&path)?.len())
        .num("build_seconds", seconds)
        .line();
    append(&s.out, &[index])?;
    let at = Clustering {
        kind,
        seed,
        indexes: &indexes,
    };
    let grid = Grid {
        bases: bases(job, data, &at, r384)?,
        betas: s.betas.clone(),
        mults: s.mults.clone(),
    };
    let mut pending = Vec::new();

    for guess in data.guesses {
        let marker = marker(data, &at, guess, &grid)?;
        if done_already(s, job.done, &marker, BUILD) {
            eprintln!("{} {kind} seed {seed} {}: done already", data.name, guess.label);
        } else {
            pending.push((guess, marker));
        }
    }

    if pending.is_empty() {
        return Ok(());
    }

    let exact = coverage::exact(&indexes, &data.queries, &grid, s.threads)?;

    for (guess, marker) in pending {
        measure(job, data, &at, &grid, &exact, guess, marker)?;
    }

    Ok(())
}

/// Whether a guess is done already, i.e. `done` holds its marker `marker` and `s` doesn't ask to redo it. The markers
/// of a build that doesn't name its code, eg. one made where git couldn't read a repository, don't change with the
/// code, so such a build measures every guess again.
fn done_already(s: &Settings, done: &HashSet<String>, marker: &str, build: &str) -> bool {
    !s.redo && names_its_code(build) && done.contains(marker)
}

/// One clustering's kind, rotation seed and indexes.
struct Clustering<'a> {
    kind: &'a str,
    seed: u64,
    indexes: &'a Indexes,
}

/// Sweeps the rescoring depth at each share of clusters, and returns the grid's bases, with R384 among their depths.
/// The first seed's sweeps of the Quora dataset give R384, and the other seeds and datasets reuse it.
fn bases(job: &Job, data: &Data, at: &Clustering, r384: &mut R384s) -> Result<Vec<Base>, Error> {
    let (s, queries, kind, indexes) = (job.s, &data.queries, at.kind, at.indexes);
    let depths = coverage::sweep_depths();
    let gives_r384 = data.name == R384_FROM && s.r384 == R384::Measured && s.seeds.first() == Some(&at.seed);
    let mut bases = Vec::with_capacity(s.fractions.len());

    for (i, &fraction) in s.fractions.iter().enumerate() {
        let nprobe = coverage::base_nprobe(fraction, indexes.nlist);
        let recalls = coverage::sweep(indexes, queries, nprobe, &depths, s.threads)?;
        let ceiling = coverage::ceiling(indexes, queries, nprobe, s.threads)?;
        let reached = coverage::depth_for(&depths, &recalls, s.target);
        let near = coverage::depth_for(&depths, &recalls, NEAR_CEILING * ceiling);
        let show = |r: Option<usize>| r.map_or_else(|| "none".to_string(), |r| r.to_string());
        eprintln!(
            "{} {kind} seed {}: nprobe {nprobe} ({:.2}%), recall {} from R {}, ceiling {ceiling:.4}, {}% of it from R {}",
            data.name,
            at.seed,
            fraction * 100.0,
            s.target,
            show(reached),
            NEAR_CEILING * 100.0,
            show(near)
        );
        let sweep = Json::new()
            .str("run", job.run)
            .str("record", "sweep")
            .str("build", BUILD)
            .str("dataset", data.name)
            .str("clustering", kind)
            .int("seed", at.seed)
            .int("dim", queries.dim as u64)
            .int("nlist", indexes.nlist as u64)
            .int("queries", queries.len() as u64)
            .num("fraction", fraction)
            .int("nprobe", nprobe as u64)
            .num("target", s.target)
            .opt("r_target", reached.map(|r| r as u64))
            .num("ceiling", ceiling)
            .num("near_ceiling", NEAR_CEILING)
            .opt("r_near_ceiling", near.map(|r| r as u64))
            .bool("gives_r384", gives_r384)
            .ints("reranks", &depths.iter().map(|&r| r as u64).collect::<Vec<_>>())
            .nums("recalls", &recalls)
            .line();
        append(&s.out, &[sweep])?;
        if gives_r384 {
            r384.entry(kind.to_string())
                .or_insert_with(|| vec![None; s.fractions.len()])[i] = reached;
        }
        let extra = match s.r384 {
            R384::Measured => r384
                .get(kind)
                .ok_or_else(|| invalid(&format!("R384 isn't known for the {kind} clustering")))?[i],
            R384::Fixed(r) => Some(r),
            R384::Skipped => None,
        };
        let mut chosen: Vec<Depth> = s
            .reranks
            .iter()
            .map(|&r| Depth {
                label: r.to_string(),
                rerank: r,
            })
            .collect();
        chosen.extend(extra.map(|r| Depth {
            label: "r384".to_string(),
            rerank: r,
        }));
        bases.push(Base {
            fraction,
            nprobe,
            depths: chosen,
        });
    }

    Ok(bases)
}

/// Measures one guess file over the grid and appends its cells, then the marker that says it is done.
fn measure(
    job: &Job,
    data: &Data,
    at: &Clustering,
    grid: &Grid,
    exact: &Exact,
    guess: &Guess,
    marker: String,
) -> Result<(), Error> {
    let vectors = Fbin::open(&guess.path)?;
    let dim = data.queries.dim;

    if vectors.dim() != dim || vectors.len() != data.all_queries {
        let path = guess.path.display();
        return Err(damaged(&format!(
            "{path} doesn't hold one guess of {dim} values per query"
        )));
    }

    let guesses = &vectors.values()[..data.queries.vectors.len()];

    if let Some(row) = coverage::bad_row(guesses, dim) {
        let path = guess.path.display();
        return Err(damaged(&format!("{path}: guess {row} is zero or not finite")));
    }

    let start = Instant::now();
    let cos = coverage::mean_cosine(guesses, data.queries.vectors, dim);
    let cells = coverage::speculate(
        at.indexes,
        &data.queries,
        guesses,
        exact,
        grid,
        data.resamples,
        job.s.threads,
    )?;
    eprintln!(
        "{} {} seed {} {}: cosine {cos:.3}, {} cells in {:.1} s",
        data.name,
        at.kind,
        at.seed,
        guess.label,
        cells.len(),
        start.elapsed().as_secs_f64()
    );
    let line = Line {
        job,
        data,
        at,
        grid,
        guess,
        cos,
    };
    let mut lines: Vec<String> = cells.iter().map(|c| line.of(c)).collect();
    lines.push(marker);
    append(&job.s.out, &lines)
}

/// What every result line of one guess shares.
struct Line<'a> {
    job: &'a Job<'a>,
    data: &'a Data<'a>,
    at: &'a Clustering<'a>,
    grid: &'a Grid,
    guess: &'a Guess,
    cos: f64,
}

impl Line<'_> {
    /// The result line of the cell `c`, i.e. the guess, the clustering and the base it measures, then the cell itself.
    fn of(&self, c: &Cell) -> String {
        let base = &self.grid.bases[c.base];
        let depth = &base.depths[c.depth];
        let json = Json::new()
            .str("run", self.job.run)
            .str("record", "grid")
            .str("build", BUILD)
            .str("dataset", self.data.name)
            .str("clustering", self.at.kind)
            .int("seed", self.at.seed)
            .str("guess", &self.guess.label)
            .opt("layer", self.guess.layer.map(|l| l as u64))
            .num("cos", self.cos)
            .int("queries", self.data.queries.len() as u64)
            .int("nlist", self.at.indexes.nlist as u64)
            .num("fraction", base.fraction)
            .int("nprobe", base.nprobe as u64);
        cell_fields(json, c, depth).line()
    }
}

/// Adds the fields of the cell `c` at the rescoring depth `depth`, from its beta to its rerank coverages, each under
/// its name in the result lines. Both rerank coverages are there. `rerank_coverage` counts the rows that the true
/// embedding rescores over the guess's clusters, and `rerank_coverage_normal` those that a normal search rescores over
/// its own.
fn cell_fields(json: Json, c: &Cell, depth: &Depth) -> Json {
    let json = json
        .num("beta", c.beta)
        .num("beta_effective", c.beta_effective)
        .int("nprobe_guess", c.wide as u64)
        .str("r_label", &depth.label)
        .int("rerank", depth.rerank as u64)
        .int("mult", c.mult as u64)
        .int("prefetch", (c.mult * depth.rerank) as u64)
        .num("posting_coverage", c.posting_coverage)
        .num("posting_coverage_p10", c.posting_coverage_p10)
        .num("fully_covered", c.fully_covered)
        .num("missed_mean", c.missed_mean)
        .num("missed_p99", c.missed_p99)
        .num("recall_normal", c.recall_normal)
        .num("recall_guess", c.recall_guess);
    let json = estimate(json, "recall_gap", &c.recall_gap);
    let json = estimate(json, "rerank_coverage", &c.rerank_coverage)
        .num("rerank_coverage_p10", c.rerank_coverage_p10)
        .num("residual_mean", c.residual_mean)
        .num("residual_p99", c.residual_p99);
    estimate(json, "rerank_coverage_normal", &c.rerank_coverage_normal)
        .num("rerank_coverage_normal_p10", c.rerank_coverage_normal_p10)
}

/// Adds the mean of `e` as `name`, and its standard error and interval as `name` with `_se`, `_lo` and `_hi` after it.
fn estimate(json: Json, name: &str, e: &coverage::Estimate) -> Json {
    json.num(name, e.mean)
        .num(&format!("{name}_se"), e.se)
        .num(&format!("{name}_lo"), e.lo)
        .num(&format!("{name}_hi"), e.hi)
}

/// The line that marks a guess as measured, for one build of this program, one grid, one seed and one state of its
/// input files.
fn marker(data: &Data, at: &Clustering, guess: &Guess, grid: &Grid) -> Result<String, Error> {
    let bases: Vec<String> = grid
        .bases
        .iter()
        .map(|b| {
            let depths: Vec<String> = b.depths.iter().map(|d| format!("{}={}", d.label, d.rerank)).collect();
            format!("{}@{}:{}", b.nprobe, b.fraction, depths.join("/"))
        })
        .collect();
    let files = data.files;
    let inputs = [
        files.corpus(),
        files.queries(),
        files.truth(),
        files.targets(),
        files.centroids(at.kind),
        files.assignment(at.kind),
        guess.path.clone(),
    ];
    let stamps: Vec<String> = inputs.iter().map(|p| stamp(p)).collect::<Result<_, Error>>()?;
    let joined = |v: &[String]| v.join(",");
    let betas: Vec<String> = grid.betas.iter().map(f64::to_string).collect();
    let mults: Vec<String> = grid.mults.iter().map(usize::to_string).collect();
    let settings = format!(
        "bases={} betas={} mults={} queries={} resamples={RESAMPLES}@{RESAMPLE_SEED} inputs={}",
        joined(&bases),
        joined(&betas),
        joined(&mults),
        data.queries.len(),
        joined(&stamps)
    );

    Ok(Json::new()
        .str("record", "done")
        .str("build", BUILD)
        .str("dataset", data.name)
        .str("clustering", at.kind)
        .int("seed", at.seed)
        .str("guess", &guess.label)
        .str("grid", &settings)
        .line())
}

/// A file's size and modification time, which change when it is written again.
fn stamp(path: &Path) -> Result<String, Error> {
    let meta = std::fs::metadata(path)?;
    let modified = meta.modified()?.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
    Ok(format!("{}:{modified}", meta.len()))
}

fn invalid(message: &str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

fn damaged(message: &str) -> Error {
    Error::new(ErrorKind::InvalidData, message)
}

#[cfg(test)]
#[path = "../tests/unit/spec_coverage.rs"]
mod tests;
