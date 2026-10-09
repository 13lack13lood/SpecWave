//! Unit tests for the coverage run, on small random datasets written to files.

use std::fs;
use std::time::{Duration, SystemTime};

use serde_json::Value;

use super::*;
use crate::testutil::{Shape, perturbed, scratch, synthetic, write_table};

/// The smallest dimension that the index takes.
const SHAPE: Shape = Shape {
    dim: 128,
    n: 1200,
    nlist: 8,
    queries: 16,
    width: 12,
};

/// The guesses of every dataset, with the noise that makes each from the true embeddings.
const GUESSES: [(&str, f64); 2] = [("near.L2", 0.3), ("far.L1", 0.9)];

fn strings(args: &[&str]) -> Vec<String> {
    args.iter().map(|a| a.to_string()).collect()
}

fn parse(args: &[&str]) -> Result<Settings, Error> {
    Settings::parse(&strings(args))
}

/// Writes the dataset `name` into the data folder `dir`. It has its documents, test queries and their true neighbors,
/// the guesses' targets, which are the test queries, `l2` and `sph` clusterings with the same clusters, and the guess
/// files of `GUESSES`.
fn write_dataset(dir: &Path, name: &str, seed: u64) {
    let data = synthetic(seed, &SHAPE);
    let (files, dim) = (Files::new(dir, name), SHAPE.dim);
    write_table(&files.corpus(), dim, &data.rows.values, f32::to_bits);
    write_table(&files.queries(), dim, &data.queries, f32::to_bits);
    write_table(&files.truth(), SHAPE.width, &data.truth, |v| v);
    write_table(&files.targets(), dim, &data.queries, f32::to_bits);

    for kind in ["l2", "sph"] {
        write_table(&files.centroids(kind), dim, &data.clusters.centroids, f32::to_bits);
        write_table(&files.assignment(kind), 1, &data.clusters.assignment, |v| v);
    }

    for (i, (label, noise)) in GUESSES.iter().enumerate() {
        let guesses = perturbed(&data.queries, dim, seed + i as u64, *noise);
        let path = dir.join("spec").join(name).join("ehat").join(format!("{label}.fbin"));
        write_table(&path, dim, &guesses, f32::to_bits);
    }
}

/// The small grid that the tests measure, one seed and two threads, as pairs of an argument and its value.
const SMALL_GRID: [(&str, &str); 6] = [
    ("--fractions", "0.25"),
    ("--betas", "0,0.75"),
    ("--reranks", "10,20"),
    ("--mults", "1,2"),
    ("--seeds", "7"),
    ("--threads", "2"),
];

/// Runs the measurement on the data folder `dir` with `extra` and the small grid, where `extra` names no other value,
/// and returns the result lines.
fn measure(dir: &Path, extra: &[&str]) -> Result<Vec<Value>, Error> {
    let out = dir.join("out.jsonl");
    let (data, out_path) = (dir.display().to_string(), out.display().to_string());
    let mut args = vec!["--out", &out_path, "--data", &data];

    for (name, value) in SMALL_GRID {
        if !extra.contains(&name) {
            args.extend([name, value]);
        }
    }

    args.extend(extra);
    run(&parse(&args)?, "spec_coverage test")?;
    Ok(fs::read_to_string(&out)?
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect())
}

/// The lines of `lines` whose record is `record`.
fn records<'a>(lines: &'a [Value], record: &str) -> Vec<&'a Value> {
    lines.iter().filter(|v| v["record"] == record).collect()
}

#[test]
fn settings_have_defaults_and_bad_values_fail_with_invalid_input() {
    let s = parse(&["--out", "x.jsonl"]).unwrap();
    let expected = Settings {
        out: PathBuf::from("x.jsonl"),
        data: Path::new(env!("CARGO_MANIFEST_DIR")).join("../data"),
        datasets: strings(&["quora", "fiqa"]),
        kinds: strings(&["l2", "cos", "sph"]),
        guesses: None,
        queries: None,
        fractions: vec![0.017, 0.0256, 0.0307],
        betas: vec![0.0, 0.25, 0.5, 1.0],
        reranks: vec![29, 39],
        mults: vec![2, 3, 4],
        target: 0.95,
        r384: R384::Measured,
        seeds: vec![7, 11, 23],
        threads: 4,
        redo: false,
    };
    assert_eq!(s, expected);

    let s = parse(&[
        "--datasets",
        "fiqa,quora",
        "--out",
        "x",
        "--seeds",
        "13,7",
        "--redo",
        "--r384",
        "30",
    ])
    .unwrap();
    assert_eq!(s.datasets, ["quora", "fiqa"], "the sweeps that give R384 run first");
    assert_eq!((s.seeds, s.redo, s.r384), (vec![13, 7], true, R384::Fixed(30)));
    let s = parse(&[
        "--out",
        "x",
        "--datasets",
        "fiqa",
        "--r384",
        "none",
        "--guesses",
        "a.L1,b",
    ])
    .unwrap();
    assert_eq!((s.r384, s.guesses), (R384::Skipped, Some(strings(&["a.L1", "b"]))));

    let bad: [&[&str]; 22] = [
        &[],
        &["--out"],
        &["--out", "x", "--frac", "0.1"],
        &["--out", "x", "--out", "y"],
        &["--out", "x", "--fractions", "2.56"],
        &["--out", "x", "--fractions", "0"],
        &["--out", "x", "--fractions", "nan"],
        &["--out", "x", "--target", "0"],
        &["--out", "x", "--target", "1.5"],
        &["--out", "x", "--threads", "0"],
        &["--out", "x", "--queries", "0"],
        &["--out", "x", "--betas", "-0.25"],
        &["--out", "x", "--betas", "inf"],
        &["--out", "x", "--betas", "0,,1"],
        &["--out", "x", "--reranks", "9"],
        &["--out", "x", "--mults", "0"],
        &["--out", "x", "--r384", "5"],
        &["--out", "x", "--seeds", ""],
        &["--out", "x", "--seeds", "7,7"],
        &["--out", "x", "--kinds", "l2,l2"],
        &["--out", "x", "--datasets", "fiqa"],
        &["--out", "x", "--guesses", ""],
    ];

    for args in bad {
        let found = parse(args).err().map(|e| e.kind());
        assert_eq!(found, Some(ErrorKind::InvalidInput), "{args:?}");
    }
}

#[test]
fn a_run_measures_each_dataset_clustering_seed_and_guess_once_per_build() {
    let dir = scratch("run");
    write_dataset(&dir, "quora", 1);
    write_dataset(&dir, "fiqa", 2);
    let args = ["--seeds", "7,11", "--kinds", "l2,cos", "--target", "0.5"];
    let first = measure(&dir, &args).unwrap();

    let header = records(&first, "run");
    assert_eq!(header.len(), 1);
    assert_eq!(
        (header[0]["build"].as_str(), header[0]["seeds"].clone()),
        (Some(BUILD), serde_json::json!([7, 11]))
    );
    assert_eq!(
        (header[0]["resamples"].as_u64(), header[0]["interval"].as_f64()),
        (Some(1000), Some(0.95))
    );

    let index = records(&first, "index");
    assert_eq!(index.len(), 2 * 2 * 2);

    for v in &index {
        let unit = v["clustering"] == "cos";
        assert_eq!(
            (v["cluster_files"].as_str(), v["unit_centers"].as_bool()),
            (Some("l2"), Some(unit)),
            "{v}"
        );
        assert_eq!(v["build"].as_str(), Some(BUILD), "{v}");
    }

    // The first seed's Quora sweeps give R384, and every grid uses it.
    let sweeps = records(&first, "sweep");
    assert_eq!(sweeps.len(), 2 * 2 * 2);
    assert!(sweeps.iter().all(|v| v["build"] == BUILD));

    for kind in ["l2", "cos"] {
        let given: Vec<&&Value> = sweeps
            .iter()
            .filter(|v| v["gives_r384"] == true && v["clustering"] == kind)
            .collect();
        assert_eq!(given.len(), 1, "{kind}");
        assert_eq!(
            (given[0]["dataset"].as_str(), given[0]["seed"].as_u64()),
            (Some("quora"), Some(7))
        );
        let r384 = given[0]["r_target"].as_u64().expect("the target is reached");
        let tried: Vec<&Value> = records(&first, "grid")
            .into_iter()
            .filter(|v| v["clustering"] == kind && v["r_label"] == "r384")
            .collect();
        assert!(tried.iter().all(|v| v["rerank"].as_u64() == Some(r384)), "{kind}");
        for dataset in ["quora", "fiqa"] {
            assert_eq!(
                tried.iter().filter(|v| v["dataset"] == dataset).count(),
                2 * 2 * 2 * 2,
                "{kind} {dataset}"
            );
        }
    }

    let grid = records(&first, "grid");
    // 2 datasets, 2 clusterings, 2 seeds and 2 guesses, each with 2 betas, 3 depths and 2 multiples.
    assert_eq!(grid.len(), 2 * 2 * 2 * 2 * (2 * 3 * 2));

    for v in &grid {
        assert_eq!(v["build"].as_str(), Some(BUILD));
        let (wide, nprobe) = (v["nprobe_guess"].as_f64().unwrap(), v["nprobe"].as_f64().unwrap());
        assert_eq!(v["beta_effective"].as_f64(), Some(wide / nprobe - 1.0), "{v}");
        assert!(
            v["beta_effective"].as_f64().unwrap() <= v["beta"].as_f64().unwrap(),
            "{v}"
        );
        for name in ["recall_gap", "rerank_coverage", "rerank_coverage_normal"] {
            let get = |end: &str| {
                v[format!("{name}{end}")]
                    .as_f64()
                    .unwrap_or_else(|| panic!("{name}{end} in {v}"))
            };
            assert!(
                get("_lo") <= get("") && get("") <= get("_hi") && get("_se") >= 0.0,
                "{v}"
            );
        }
        assert!(v["rerank_coverage_normal_p10"].as_f64().is_some(), "{v}");
    }

    // The seeds rotate the codes differently, so some cell differs between them.
    let cell = |v: &Value| {
        format!(
            "{} {} {} {} {} {}",
            v["dataset"], v["clustering"], v["guess"], v["beta"], v["rerank"], v["mult"]
        )
    };
    let by_seed = |seed: u64| -> HashMap<String, f64> {
        grid.iter()
            .filter(|v| v["seed"] == seed)
            .map(|v| (cell(v), v["rerank_coverage"].as_f64().unwrap()))
            .collect()
    };
    let (seven, eleven) = (by_seed(7), by_seed(11));
    assert_eq!(seven.len(), eleven.len());
    assert!(seven.iter().any(|(k, x)| eleven[k] != *x));

    let done = records(&first, "done");
    assert_eq!(done.len(), 2 * 2 * 2 * 2);
    assert!(done.iter().all(|v| v["build"] == BUILD && v["seed"].as_u64().is_some()));

    // A second run finds every guess done, but a marker from another build doesn't count.
    let second = measure(&dir, &args).unwrap();
    assert_eq!(records(&second, "grid").len(), grid.len());
    assert_eq!(records(&second, "run").len(), 2);
    let out = dir.join("out.jsonl");
    let text = fs::read_to_string(&out).unwrap();
    let other: Vec<String> = text
        .lines()
        .map(|line| match line.contains("\"record\":\"done\"") {
            true => line.replace(&format!("\"build\":\"{BUILD}\""), "\"build\":\"other\""),
            false => line.to_string(),
        })
        .collect();
    fs::write(&out, other.join("\n") + "\n").unwrap();
    let third = measure(&dir, &args).unwrap();
    assert_eq!(records(&third, "grid").len(), 2 * grid.len());

    // A guess file written again is measured again, with each seed and clustering, and nothing else is.
    let path = dir.join("spec/fiqa/ehat/far.L1.fbin");
    let later = SystemTime::now() + Duration::from_secs(5);
    fs::File::options()
        .write(true)
        .open(&path)
        .unwrap()
        .set_modified(later)
        .unwrap();
    let fourth = measure(&dir, &args).unwrap();
    let new: Vec<&Value> = records(&fourth, "grid").split_off(2 * grid.len());
    assert_eq!(new.len(), 2 * 2 * 12);
    assert!(new.iter().all(|v| v["dataset"] == "fiqa" && v["guess"] == "far.L1"));

    // --redo measures every guess again.
    let fifth = measure(&dir, &[&args[..], &["--redo"]].concat()).unwrap();
    assert_eq!(records(&fifth, "grid").len(), 3 * grid.len() + new.len());
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn another_grid_or_a_rewritten_input_measures_a_guess_again() {
    let dir = scratch("marker");
    write_dataset(&dir, "fiqa", 6);
    let files = Files::new(&dir, "fiqa");
    let one = [
        ("--datasets", "fiqa"),
        ("--kinds", "l2"),
        ("--r384", "30"),
        ("--guesses", "near.L2"),
    ];
    let out = dir.join("out.jsonl");
    // The grid lines that a run of one guess with `extra` adds.
    let added = |extra: &[&str]| {
        let before = match fs::read_to_string(&out) {
            Ok(text) => text.lines().filter(|line| line.contains("\"record\":\"grid\"")).count(),
            Err(_) => 0,
        };
        let mut args: Vec<&str> = one
            .iter()
            .filter(|(name, _)| !extra.contains(name))
            .flat_map(|&(name, value)| [name, value])
            .collect();
        args.extend(extra);
        records(&measure(&dir, &args).unwrap(), "grid").len() - before
    };
    let cells = added(&[]);
    assert_eq!(cells, 2 * 3 * 2);
    assert_eq!(added(&[]), 0, "done already");

    let changes: [&[&str]; 7] = [
        &["--betas", "0,0.5"],
        &["--mults", "1,3"],
        &["--fractions", "0.5"],
        &["--reranks", "10,30"],
        &["--r384", "40"],
        &["--queries", "8"],
        &["--seeds", "11"],
    ];

    for extra in changes {
        assert!(added(extra) > 0, "{extra:?}");
    }

    let later = SystemTime::now() + Duration::from_secs(5);
    let inputs = [
        files.corpus(),
        files.queries(),
        files.truth(),
        files.targets(),
        files.centroids("l2"),
        files.assignment("l2"),
    ];

    for path in inputs {
        let file = fs::File::options().write(true).open(&path).unwrap();
        file.set_modified(later).unwrap();
        assert_eq!(added(&[]), cells, "{} written again", path.display());
    }

    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn r384_can_be_fixed_left_out_or_missing_when_quora_never_reaches_the_target() {
    let dir = scratch("r384");
    write_dataset(&dir, "quora", 3);
    write_dataset(&dir, "fiqa", 4);
    let depths = |lines: &[Value]| -> Vec<u64> {
        let mut depths: Vec<u64> = records(lines, "grid")
            .iter()
            .filter(|v| v["r_label"] == "r384")
            .map(|v| v["rerank"].as_u64().unwrap())
            .collect();
        depths.dedup();
        depths
    };

    let fixed = measure(&dir, &["--kinds", "l2", "--r384", "30", "--guesses", "near.L2"]).unwrap();
    assert_eq!(depths(&fixed), [30]);
    let lines = records(&fixed, "grid").len();
    assert!(
        records(&fixed, "grid")
            .iter()
            .any(|v| v["dataset"] == "fiqa" && v["r_label"] == "r384")
    );

    let none = measure(
        &dir,
        &["--kinds", "l2", "--r384", "none", "--guesses", "near.L2", "--redo"],
    )
    .unwrap();
    assert!(depths(&none[none.len() - lines..]).is_empty());

    // The probed clusters can't hold every true neighbor, so a recall of 1 is never reached.
    let unreached = measure(
        &dir,
        &["--kinds", "l2", "--target", "1", "--guesses", "near.L2", "--redo"],
    )
    .unwrap();
    let sweeps = records(&unreached, "sweep");
    let quora = sweeps.iter().rev().find(|v| v["dataset"] == "quora").unwrap();
    assert!(quora["r_target"].is_null(), "{quora}");
    assert!(depths(&unreached[unreached.len() - lines..]).is_empty());
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn damaged_inputs_fail_with_invalid_data() {
    let dir = scratch("damaged");
    let files = Files::new(&dir, "fiqa");
    let kind = |dir: &Path| {
        measure(dir, &["--datasets", "fiqa", "--r384", "none", "--kinds", "l2"])
            .err()
            .map(|e| e.kind())
    };
    let data = synthetic(5, &SHAPE);
    let dim = SHAPE.dim;
    write_dataset(&dir, "fiqa", 5);
    assert_eq!(kind(&dir), None);

    let mut narrow = Vec::new();
    let (rows, _) = data.truth.as_chunks::<{ SHAPE.width }>();
    rows.iter().for_each(|row| narrow.extend(&row[..5]));
    write_table(&files.truth(), 5, &narrow, |v| v);
    assert_eq!(kind(&dir), Some(ErrorKind::InvalidData), "5 true neighbors per query");

    let mut past = data.truth.clone();
    past[SHAPE.width + 3] = SHAPE.n as u32;
    write_table(&files.truth(), SHAPE.width, &past, |v| v);
    assert_eq!(
        kind(&dir),
        Some(ErrorKind::InvalidData),
        "a true neighbor past the last document"
    );
    write_table(&files.truth(), SHAPE.width, &data.truth, |v| v);

    for (what, value) in [("not a number", f32::NAN), ("infinite", f32::INFINITY)] {
        let mut corpus = data.rows.values.clone();
        corpus[7 * dim + 1] = value;
        write_table(&files.corpus(), dim, &corpus, f32::to_bits);
        assert_eq!(
            kind(&dir),
            Some(ErrorKind::InvalidData),
            "a document value that is {what}"
        );
    }

    let mut corpus = data.rows.values.clone();
    corpus[9 * dim..10 * dim].fill(0.0);
    write_table(&files.corpus(), dim, &corpus, f32::to_bits);
    assert_eq!(kind(&dir), Some(ErrorKind::InvalidData), "a zero document");
    write_table(&files.corpus(), dim, &data.rows.values, f32::to_bits);

    let mut centers = data.clusters.centroids.clone();
    centers[2] = f32::NAN;
    write_table(&files.centroids("l2"), dim, &centers, f32::to_bits);
    assert_eq!(
        kind(&dir),
        Some(ErrorKind::InvalidData),
        "a center value that isn't a number"
    );
    write_table(&files.centroids("l2"), dim, &data.clusters.centroids, f32::to_bits);

    let other = perturbed(&data.queries, dim, 6, 0.1);
    write_table(&files.targets(), dim, &other, f32::to_bits);
    assert_eq!(
        kind(&dir),
        Some(ErrorKind::InvalidData),
        "guesses fitted to other queries"
    );
    fs::remove_file(files.targets()).unwrap();
    assert_eq!(kind(&dir), Some(ErrorKind::NotFound), "no targets for the guesses");
    write_table(&files.targets(), dim, &data.queries, f32::to_bits);

    // The queries go into the targets too, so only the queries' own checks can fail.
    let write_queries = |width: usize, queries: &[f32]| {
        for path in [files.queries(), files.targets()] {
            write_table(&path, width, queries, f32::to_bits);
        }
    };
    let half: Vec<f32> = data
        .queries
        .chunks_exact(dim)
        .flat_map(|q| q[..dim / 2].to_vec())
        .collect();
    write_queries(dim / 2, &half);
    assert_eq!(
        kind(&dir),
        Some(ErrorKind::InvalidData),
        "queries of half the documents' width"
    );
    write_queries(dim, &data.queries);
    write_table(&files.truth(), SHAPE.width, &data.truth[SHAPE.width..], |v| v);
    assert_eq!(
        kind(&dir),
        Some(ErrorKind::InvalidData),
        "the true neighbors of one query missing"
    );
    write_table(&files.truth(), SHAPE.width, &data.truth, |v| v);

    // A row that is zero, or that holds a value that isn't a number.
    let spoiled = |rows: &[f32], zero: bool| {
        let mut rows = rows.to_vec();
        match zero {
            true => rows[4 * dim..5 * dim].fill(0.0),
            false => rows[4 * dim + 7] = f32::NAN,
        }
        rows
    };
    let guess = dir.join("spec/fiqa/ehat/near.L2.fbin");

    for zero in [true, false] {
        write_queries(dim, &spoiled(&data.queries, zero));
        assert_eq!(kind(&dir), Some(ErrorKind::InvalidData), "a bad query, zero {zero}");
        write_queries(dim, &data.queries);
        write_table(&guess, dim, &spoiled(&data.queries, zero), f32::to_bits);
        assert_eq!(kind(&dir), Some(ErrorKind::InvalidData), "a bad guess, zero {zero}");
    }

    write_table(&guess, dim, &data.queries[dim..], f32::to_bits);
    assert_eq!(kind(&dir), Some(ErrorKind::InvalidData), "a guess file one query short");
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_grid_line_holds_each_value_of_its_cell_under_its_own_name() {
    let estimate = |mean: f64, se: f64, lo: f64, hi: f64| coverage::Estimate { mean, se, lo, hi };
    let c = Cell {
        base: 0,
        beta: 0.5,
        wide: 13,
        beta_effective: 0.444,
        depth: 0,
        mult: 3,
        posting_coverage: 0.81,
        posting_coverage_p10: 0.62,
        fully_covered: 0.33,
        missed_mean: 1.25,
        missed_p99: 4.5,
        recall_normal: 0.71,
        recall_guess: 0.69,
        recall_gap: estimate(-0.02, 0.003, -0.026, -0.014),
        rerank_coverage: estimate(0.91, 0.004, 0.902, 0.918),
        rerank_coverage_p10: 0.72,
        residual_mean: 2.75,
        residual_p99: 9.5,
        rerank_coverage_normal: estimate(0.86, 0.005, 0.85, 0.87),
        rerank_coverage_normal_p10: 0.64,
    };
    let depth = Depth {
        label: "r384".to_string(),
        rerank: 30,
    };
    let line: Value = serde_json::from_str(&cell_fields(Json::new(), &c, &depth).line()).unwrap();
    let expected = serde_json::json!({
        "beta": 0.5, "beta_effective": 0.444, "nprobe_guess": 13, "r_label": "r384", "rerank": 30, "mult": 3,
        "prefetch": 90, "posting_coverage": 0.81, "posting_coverage_p10": 0.62, "fully_covered": 0.33,
        "missed_mean": 1.25, "missed_p99": 4.5, "recall_normal": 0.71, "recall_guess": 0.69, "recall_gap": -0.02,
        "recall_gap_se": 0.003, "recall_gap_lo": -0.026, "recall_gap_hi": -0.014, "rerank_coverage": 0.91,
        "rerank_coverage_se": 0.004, "rerank_coverage_lo": 0.902, "rerank_coverage_hi": 0.918,
        "rerank_coverage_p10": 0.72, "residual_mean": 2.75, "residual_p99": 9.5, "rerank_coverage_normal": 0.86,
        "rerank_coverage_normal_se": 0.005, "rerank_coverage_normal_lo": 0.85, "rerank_coverage_normal_hi": 0.87,
        "rerank_coverage_normal_p10": 0.64
    });
    assert_eq!(line, expected);
}

#[test]
fn a_guess_is_done_already_only_when_the_build_names_its_code_and_no_redo_is_asked() {
    let (s, redo) = (
        parse(&["--out", "x"]).unwrap(),
        parse(&["--out", "x", "--redo"]).unwrap(),
    );
    let done = HashSet::from(["marker".to_string()]);
    let named = "specwave 1d2a8be0c3f1+0f3a5c7e9b1d2f40 coldivf d90edb1c2d3e";

    assert!(done_already(&s, &done, "marker", named));
    assert!(!done_already(&s, &done, "another marker", named));
    assert!(!done_already(&redo, &done, "marker", named));
    assert!(!done_already(
        &s,
        &done,
        "marker",
        "specwave unknown coldivf d90edb1c2d3e"
    ));
    assert!(!done_already(
        &s,
        &done,
        "marker",
        "specwave 1d2a8be0c3f1 coldivf unknown"
    ));
}
