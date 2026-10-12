//! Unit tests for the fire layer picked from measured inputs, on small synthetic lines. The test that picks the fire
//! layers from the stored layer times, cold reads and coverage grid in `results` is ignored, and run by hand.

use std::path::Path;

use serde_json::json;
use specwave::scheduler::LayerProfile;

use super::*;
use crate::verdict::{Interval, read};

/// A cell of fiqa's `l2` grid at R 29, whose tests pass or fail clearly, with an interval of ±0.002 around each value.
fn cell(guess: &str, nprobe: usize, beta: f64, mult: usize, gap: f64, coverage: f64) -> Cell {
    let around = |mean: f64| Interval {
        mean,
        lo: mean - 0.002,
        hi: mean + 0.002,
    };
    let wide = specwave::speculate::wide_nprobe(nprobe, beta);
    Cell {
        dataset: "fiqa".to_string(),
        clustering: "l2".to_string(),
        guess: guess.to_string(),
        layer: guess.rsplit_once(".L").and_then(|(_, l)| l.parse().ok()),
        nprobe,
        beta,
        beta_effective: wide as f64 / nprobe as f64 - 1.0,
        r_label: "29".to_string(),
        rerank: 29,
        mult,
        queries: 1548,
        seeds: vec![7],
        builds: vec!["b1".to_string()],
        gap: around(gap),
        coverage: [around(coverage), around(coverage)],
    }
}

fn base(nprobe: usize, r_label: &str, rerank: usize) -> Base {
    Base {
        dataset: "fiqa".to_string(),
        fit: "openings".to_string(),
        clustering: "l2".to_string(),
        nprobe,
        r_label: r_label.to_string(),
        rerank,
    }
}

/// The grid at nprobe 9 and R 29. Layer 8 first passes at β 0.25 and m 3, layers 9 and 10 at β 0 and m 2, and layer 6
/// never passes.
fn grid() -> Vec<Cell> {
    let mut cells = vec![
        cell("openings.L6", 9, 0.0, 4, -0.03, 0.80),
        cell("openings.L6", 9, 0.25, 4, 0.01, 0.86),
        cell("openings.L8", 9, 0.0, 2, -0.02, 0.85),
        cell("openings.L8", 9, 0.25, 2, 0.01, 0.88),
        cell("openings.L8", 9, 0.25, 3, 0.01, 0.92),
        cell("openings.L8", 9, 0.25, 4, 0.01, 0.96),
        cell("openings.L9", 9, 0.0, 2, 0.0, 0.93),
        cell("openings.L9", 9, 0.0, 3, 0.0, 0.96),
        cell("openings.L10", 9, 0.0, 2, 0.002, 0.97),
        // These belong to other fits, grids or bases, or have no layer.
        cell("queries.L7", 9, 0.0, 2, 0.0, 0.99),
        cell("openings", 9, 0.0, 2, 0.0, 0.99),
        Cell {
            clustering: "sph".to_string(),
            ..cell("openings.L7", 9, 0.0, 2, 0.0, 0.99)
        },
        Cell {
            dataset: "quora".to_string(),
            ..cell("openings.L7", 9, 0.0, 2, 0.0, 0.99)
        },
        cell("openings.L7", 11, 0.0, 2, 0.0, 0.99),
    ];
    let mut r39 = cell("openings.L7", 9, 0.0, 2, 0.0, 0.99);
    r39.r_label = "39".to_string();
    r39.rerank = 39;
    cells.push(r39);
    cells
}

fn layer(layer: usize, beta: f64, rerank_mult: u8) -> LayerSettings {
    LayerSettings {
        layer,
        beta,
        rerank_mult,
    }
}

#[test]
fn each_layer_guesses_with_its_first_passing_cell_and_failing_layers_dont_guess() {
    let found = settings(&grid(), &Rule::default(), 1, &base(9, "29", 29));

    assert_eq!(found, [layer(8, 0.25, 3), layer(9, 0.0, 2), layer(10, 0.0, 2)]);
    assert_eq!(settings(&grid(), &Rule::default(), 1, &base(6, "29", 29)), []);

    // With a target of 0.95, layer 8 needs m = 4 and layer 9 m = 3.
    let strict = Rule {
        min_coverage: 0.95,
        ..Rule::default()
    };
    assert_eq!(
        settings(&grid(), &strict, 1, &base(9, "29", 29)),
        [layer(8, 0.25, 4), layer(9, 0.0, 3), layer(10, 0.0, 2)]
    );
}

#[test]
fn the_bases_are_each_nprobe_and_r_of_the_fit_once_in_order() {
    let mut cells = grid();
    cells.push(cell("openings.L9", 6, 0.0, 2, 0.0, 0.93));

    assert_eq!(
        bases(&cells, "fiqa", "openings", "l2"),
        [
            base(6, "29", 29),
            base(9, "29", 29),
            base(9, "39", 39),
            base(11, "29", 29)
        ]
    );
    assert_eq!(
        bases(&cells, "fiqa", "queries", "l2"),
        [Base {
            fit: "queries".to_string(),
            ..base(9, "29", 29)
        }]
    );
    assert!(bases(&cells, "nfcorpus", "openings", "l2").is_empty());
}

/// A result line of the `layers` binary.
fn layer_line(model: &str, device: &str, threads: u64, layer: u64, t_ms: f64) -> String {
    json!({
        "bench": "layers", "model": model, "device": device, "threads": threads, "layer": layer, "t_ms": t_ms,
        "headroom_ms": 0.0,
    })
    .to_string()
}

#[test]
fn profiles_come_from_the_layer_lines_and_a_later_line_counts() {
    let lines = vec![
        layer_line("arctic", "metal", 6, 0, 0.3),
        layer_line("arctic", "metal", 6, 12, 9.3),
        "not json".to_string(),
        json!({"bench": "train", "layer": 3}).to_string(),
        layer_line("arctic", "cpu", 5, 9, 12.0),
        layer_line("arctic", "cpu", 5, 0, 0.1),
        layer_line("arctic", "cpu", 5, 12, 16.4),
        layer_line("arctic", "cpu", 5, 9, 12.3),
    ];
    let setups = profiles(&lines).unwrap();
    let cpu = LayerProfile::new(vec![(0, 0.1), (9, 12.3), (12, 16.4)]).unwrap();
    let metal = LayerProfile::new(vec![(0, 0.3), (12, 9.3)]).unwrap();

    assert_eq!(setups.len(), 2);
    assert_eq!(
        (setups[0].name(), &setups[0].profile),
        ("cpu, 5 threads".to_string(), &cpu)
    );
    assert_eq!((setups[1].name(), &setups[1].profile), ("metal".to_string(), &metal));
    assert_eq!((setups[0].model.as_str(), setups[0].threads), ("arctic", 5));

    let missing =
        vec![json!({"bench": "layers", "model": "arctic", "device": "cpu", "threads": 5, "layer": 0}).to_string()];
    let falling = vec![
        layer_line("arctic", "cpu", 5, 0, 2.0),
        layer_line("arctic", "cpu", 5, 12, 1.0),
    ];

    for lines in [missing, falling] {
        assert_eq!(
            profiles(&lines).unwrap_err().kind(),
            ErrorKind::InvalidData,
            "{lines:?}"
        );
    }
}

/// A trial line of the `cold` harness, with its stage times in microseconds.
fn trial(source: (&str, &str), nprobe: u64, rerank: u64, gap_ms: u64, stages: [f64; 6]) -> String {
    let [route, wave, scan, rows, prepare, tables] = stages;
    json!({
        "variant": source.0, "read": source.1, "nprobe": nprobe, "rerank": rerank, "gap_ms": gap_ms,
        "route_us": route, "cluster_wave_us": wave, "scan_us": scan, "rows_us": rows, "prepare_us": prepare,
        "tables_us": tables,
    })
    .to_string()
}

const PREWARM: (&str, &str) = ("prewarm", "first");
const IDLE: (&str, &str) = ("idle", "first");

fn trials() -> Vec<String> {
    vec![
        trial(PREWARM, 9, 58, 10, [20.0, 400.0, 30.0, 800.0, 10.0, 0.0]),
        trial(PREWARM, 9, 58, 10, [40.0, 600.0, 20.0, 700.0, 10.0, 0.0]),
        trial(PREWARM, 9, 58, 10, [30.0, 500.0, 40.0, 600.0, 10.0, 0.0]),
        trial(PREWARM, 11, 87, 10, [50.0, 700.0, 50.0, 900.0, 10.0, 0.0]),
        trial(PREWARM, 9, 29, 0, [10.0, 450.0, 25.0, 500.0, 40.0, 320.0]),
        trial(PREWARM, 9, 29, 0, [10.0, 450.0, 25.0, 500.0, 30.0, 300.0]),
        trial(IDLE, 9, 58, 10, [60.0, 900.0, 80.0, 1000.0, 50.0, 0.0]),
        trial(IDLE, 9, 58, 10, [80.0, 700.0, 60.0, 900.0, 50.0, 0.0]),
        trial(IDLE, 9, 29, 0, [10.0, 800.0, 70.0, 900.0, 100.0, 350.0]),
        trial(("prewarm", "repeat"), 9, 58, 10, [20.0, 300.0, 20.0, 300.0, 10.0, 0.0]),
        json!({"label": "no variant", "nprobe": 9}).to_string(),
        "not json".to_string(),
    ]
}

#[test]
fn the_reads_take_each_cells_medians_and_the_tables_of_the_trials_that_didnt_wait() {
    let reads = cold_reads(&trials()).unwrap();
    let prewarm = Source::new("prewarm", "first");
    let idle = Source::new("idle", "first");
    let close = |a: f64, b: f64| (a - b).abs() < 1e-12;

    // Sources need a tables' read, which prewarm repeat reads lack.
    assert_eq!(reads.sources(), [idle.clone(), prewarm.clone()]);
    assert!(reads.of(&Source::new("prewarm", "repeat")).is_none());

    let chain = reads.chain(&prewarm, 9, 58).unwrap();
    assert!(
        close(chain.route_ms, 0.03)
            && close(chain.wave_ms, 0.5)
            && close(chain.scan_ms, 0.03)
            && close(chain.rows_ms, 0.7),
        "{chain:?}"
    );
    assert_eq!(chain.trials, 3);
    assert!(close(chain.ms(), 1.26));
    assert!(close(reads.chain(&idle, 9, 58).unwrap().wave_ms, 0.7));
    assert!(reads.chain(&prewarm, 9, 29).is_none());

    let measured = reads.of(&prewarm).unwrap();
    assert!(close(measured.tables_ms(), 0.33));
    assert!(close(measured.chain_ms(9, 58), 1.26));
    assert!(measured.chain_ms(11, 58).is_nan());
    assert!(close(reads.of(&idle).unwrap().tables_ms(), 0.45));

    let search = base(9, "29", 29).params();
    assert!(measured.covers(&search, &[layer(9, 0.0, 2), layer(8, 0.25, 3)]));
    assert!(!measured.covers(&search, &[layer(9, 0.0, 2), layer(10, 0.25, 2)]));
    assert!(!reads.of(&idle).unwrap().covers(&search, &[layer(8, 0.25, 3)]));

    let mut broken = trials();
    broken.push(json!({"variant": "idle", "read": "first", "nprobe": 9, "rerank": 58, "gap_ms": 10}).to_string());
    assert_eq!(cold_reads(&broken).unwrap_err().kind(), ErrorKind::InvalidData);
}

#[test]
fn the_table_fires_the_latest_layer_that_fits_and_shows_the_later_ones() {
    let profile = LayerProfile::new(vec![(0, 0.1), (8, 11.0), (9, 12.0), (10, 13.5), (12, 16.0)]).unwrap();
    let setups = [Setup {
        model: "arctic".to_string(),
        device: "cpu".to_string(),
        threads: 5,
        profile,
    }];
    // Layers 9 and 10 read 9 clusters and 58 rows, 2.7 ms, which needs 2.97 ms, more than the 2.5 ms after layer 10.
    let lines = vec![
        trial(PREWARM, 9, 58, 10, [0.0, 1000.0, 200.0, 1500.0, 0.0, 0.0]),
        trial(PREWARM, 11, 87, 10, [0.0, 1000.0, 200.0, 1500.0, 0.0, 0.0]),
        trial(PREWARM, 9, 29, 0, [0.0, 0.0, 0.0, 0.0, 30.0, 300.0]),
    ];
    let reads = cold_reads(&lines).unwrap();
    let bases = [base(6, "29", 29), base(9, "29", 29)];
    let mut cells = grid();
    cells.push(cell("openings.L9", 6, 0.0, 2, 0.0, 0.93));
    let rows = table(&setups, &reads, &cells, &Rule::default(), 1, &bases).unwrap();

    assert_eq!(rows.len(), 2);
    assert_eq!(
        (rows[0].base.nprobe, rows[0].layers.clone(), rows[0].schedule),
        (6, vec![layer(9, 0.0, 2)], None)
    );

    let row = &rows[1];
    let fire = row.schedule.unwrap().fire.unwrap();
    assert_eq!(
        (row.setup.as_str(), row.source.name()),
        ("cpu, 5 threads", "prewarm first".to_string())
    );
    assert_eq!(
        (fire.settings, fire.clusters, fire.rows, fire.left_ms),
        (layer(9, 0.0, 2), 9, 58, 4.0)
    );
    assert!((fire.needed_ms - 2.97).abs() < 1e-9);
    assert_eq!(row.later.len(), 1);
    assert_eq!((row.later[0].settings.layer, row.later[0].left_ms), (10, 2.5));

    let text = markdown(&rows);
    assert!(
        text.contains("| cpu, 5 threads | prewarm first | 6 | 29 | not measured |"),
        "{text}"
    );
    assert!(
        text.contains(
            "| cpu, 5 threads | prewarm first | 9 | 29 | L9 | 0, 2 | 9, 58 | 4.00 | 2.97 | L10 needs 2.97 of 2.50 |"
        ),
        "{text}"
    );

    let measured: serde_json::Value = serde_json::from_str(&json(row).line()).unwrap();
    assert_eq!(measured["fire_layer"], 9);
    assert_eq!(
        (
            measured["clusters"].clone(),
            measured["rows"].clone(),
            measured["mult"].clone()
        ),
        (json!(9), json!(58), json!(2))
    );
    assert_eq!(
        (measured["measured"].clone(), measured["fits"].clone()),
        (json!(true), json!(true))
    );
    assert_eq!(measured["layers"], json!([8, 9, 10]));
    assert!((measured["needed_ms"].as_f64().unwrap() - 2.97).abs() < 1e-9);

    let unmeasured: serde_json::Value = serde_json::from_str(&json(&rows[0]).line()).unwrap();
    assert_eq!(
        (unmeasured["measured"].clone(), unmeasured["fire_layer"].clone()),
        (json!(false), json!(null))
    );
}

fn parse(args: &[&str]) -> Result<Settings, Error> {
    Settings::parse(&args.iter().map(|a| a.to_string()).collect::<Vec<String>>())
}

#[test]
fn the_command_line_needs_its_three_files_and_takes_each_name_once() {
    let files = [
        "--coverage",
        "a.jsonl,b.jsonl",
        "--layers",
        "l.jsonl",
        "--cold",
        "c.jsonl",
    ];
    let defaults = Settings {
        coverage: vec!["a.jsonl".to_string(), "b.jsonl".to_string()],
        layers: "l.jsonl".to_string(),
        cold: "c.jsonl".to_string(),
        dataset: "fiqa".to_string(),
        fit: "openings".to_string(),
        clustering: "l2".to_string(),
        model: "Snowflake/snowflake-arctic-embed-m-v1.5".to_string(),
        definition: 1,
        out: None,
    };
    assert_eq!(parse(&files).unwrap(), defaults);

    let mut given = files.to_vec();
    given.extend([
        "--definition",
        "rerank_coverage",
        "--dataset",
        "quora",
        "--fit",
        "queries",
        "--clustering",
        "sph",
        "--model",
        "m",
        "--out",
        "o.jsonl",
    ]);
    let expected = Settings {
        dataset: "quora".to_string(),
        fit: "queries".to_string(),
        clustering: "sph".to_string(),
        model: "m".to_string(),
        definition: 0,
        out: Some("o.jsonl".to_string()),
        ..defaults
    };
    assert_eq!(parse(&given).unwrap(), expected);

    // A short name of the models list gives its repository id.
    let mut short = files.to_vec();
    short.extend(["--model", "e5-base"]);
    assert_eq!(parse(&short).unwrap().model, "intfloat/e5-base-v2");

    let bad: [&[&str]; 7] = [
        &files[2..],
        &["--coverage", "a", "--cold", "c"],
        &["--coverage", "a", "--layers", "l"],
        &["--coverage", "a", "--layers", "l", "--cold", "c", "--cold", "d"],
        &["--coverage", "a", "--layers", "l", "--cold", "c", "--colder", "d"],
        &["--coverage", "a", "--layers", "l", "--cold"],
        &[
            "--coverage",
            "a",
            "--layers",
            "l",
            "--cold",
            "c",
            "--definition",
            "coverage",
        ],
    ];

    for args in bad {
        assert_eq!(parse(args).unwrap_err().kind(), ErrorKind::InvalidInput, "{args:?}");
    }
}

/// A grid line of `spec_coverage` for `c`, with its intervals.
fn grid_line(c: &Cell, seed: u64) -> String {
    let mut line = json!({
        "record": "grid", "build": "b1", "dataset": c.dataset, "clustering": c.clustering, "seed": seed,
        "guess": c.guess, "layer": c.layer, "queries": c.queries, "nprobe": c.nprobe, "beta": c.beta,
        "beta_effective": c.beta_effective, "r_label": c.r_label, "rerank": c.rerank, "mult": c.mult,
    });

    for (name, i) in [
        ("recall_gap", c.gap),
        ("rerank_coverage", c.coverage[0]),
        ("rerank_coverage_normal", c.coverage[1]),
    ] {
        line[name] = i.mean.into();
        line[format!("{name}_lo")] = i.lo.into();
        line[format!("{name}_hi")] = i.hi.into();
    }

    line.to_string()
}

#[test]
fn the_rows_come_from_the_files_that_the_command_line_names() {
    let dir = crate::testutil::scratch("schedule-rows");
    let write = |name: &str, lines: &[String]| {
        let path = dir.join(name);
        std::fs::write(&path, lines.join("\n")).unwrap();
        path.display().to_string()
    };
    let grid: Vec<String> = grid().iter().map(|c| grid_line(c, 7)).collect();
    let (first, second) = grid.split_at(5);
    // The layers binary names the model by its repository id, which the short name `arctic` stands for.
    let arctic = "Snowflake/snowflake-arctic-embed-m-v1.5";
    let layers = vec![
        layer_line(arctic, "cpu", 5, 0, 0.1),
        layer_line(arctic, "cpu", 5, 8, 11.0),
        layer_line(arctic, "cpu", 5, 9, 12.0),
        layer_line(arctic, "cpu", 5, 10, 13.5),
        layer_line(arctic, "cpu", 5, 12, 16.0),
    ];
    let cold = vec![
        trial(PREWARM, 9, 58, 10, [0.0, 500.0, 30.0, 700.0, 0.0, 0.0]),
        trial(PREWARM, 11, 87, 10, [0.0, 600.0, 40.0, 800.0, 0.0, 0.0]),
        trial(PREWARM, 9, 29, 0, [0.0, 0.0, 0.0, 0.0, 30.0, 300.0]),
    ];
    let names = [write("a.jsonl", first), write("b.jsonl", second)].join(",");
    let (layers, cold) = (write("layers.jsonl", &layers), write("cold.jsonl", &cold));
    let args = [
        "--coverage",
        &names,
        "--layers",
        &layers,
        "--cold",
        &cold,
        "--model",
        "arctic",
    ];
    let found = rows(&parse(&args).unwrap()).unwrap();
    let measured: Vec<(usize, &str, Option<usize>)> = found
        .iter()
        .map(|r| {
            (
                r.base.nprobe,
                r.base.r_label.as_str(),
                r.schedule.and_then(|s| s.fire).map(|f| f.settings.layer),
            )
        })
        .collect();

    // The cold trials measure only nprobe 9 with R 29, where layer 10's 1.23 ms fits its 2.5 ms.
    assert_eq!(measured, [(9, "29", Some(10)), (9, "39", None), (11, "29", None)]);

    assert_eq!(
        rows(&parse(&args[..6]).unwrap()).unwrap().len(),
        found.len(),
        "arctic is the default model"
    );
    let other_model = [
        "--coverage",
        &names,
        "--layers",
        &layers,
        "--cold",
        &cold,
        "--model",
        "bge-small",
    ];
    let other_dataset = [
        "--coverage",
        &names,
        "--layers",
        &layers,
        "--cold",
        &cold,
        "--model",
        "arctic",
        "--dataset",
        "nfcorpus",
    ];
    let missing = [
        "--coverage",
        &names,
        "--layers",
        &layers,
        "--cold",
        "nowhere.jsonl",
        "--model",
        "arctic",
    ];
    assert_eq!(
        rows(&parse(&other_model).unwrap()).unwrap_err().kind(),
        ErrorKind::InvalidData
    );
    assert_eq!(
        rows(&parse(&other_dataset).unwrap()).unwrap_err().kind(),
        ErrorKind::InvalidData
    );
    assert_eq!(rows(&parse(&missing).unwrap()).unwrap_err().kind(), ErrorKind::NotFound);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The lines of the result file `name` in this repository's `results` folder.
fn result_lines(name: &str) -> Vec<String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../results").join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .lines()
        .map(str::to_string)
        .collect()
}

/// One entry of the fire layer table that was worked out by hand from those results. It holds the setup, the source,
/// nprobe and R, then the fire layer, β, m, the clusters and rows it reads, the time left and the time needed, and the
/// later layer's time needed, if any.
type Expected = (
    &'static str,
    &'static str,
    usize,
    &'static str,
    usize,
    f64,
    u8,
    usize,
    usize,
    &'static str,
    &'static str,
    Option<&'static str>,
);

#[test]
#[ignore = "reads the stored results of earlier runs, run by hand"]
fn the_scheduler_picks_the_fire_layers_worked_out_from_the_stored_results() {
    let setups: Vec<Setup> = profiles(&result_lines("p4-layers-idle.jsonl"))
        .unwrap()
        .into_iter()
        .filter(|s| s.model == "Snowflake/snowflake-arctic-embed-m-v1.5")
        .collect();
    let reads = cold_reads(&result_lines("p4-cold-fiqa.jsonl")).unwrap();
    let pick = |coverage: &str, fit: &str| {
        let cells = read(&result_lines(coverage)).unwrap();
        let bases: Vec<Base> = bases(&cells, "fiqa", fit, "l2")
            .into_iter()
            .filter(|b| b.r_label != "r384")
            .collect();
        table(&setups, &reads, &cells, &Rule::default(), 1, &bases).unwrap()
    };
    let rows = pick("p4-coverage-v3.jsonl", "openings");
    let mut expected: Vec<Expected> = Vec::new();

    for (setup, left) in [("cpu, 5 threads", "2.78"), ("cpu, 6 threads", "2.90")] {
        expected.extend([
            (setup, "prewarm first", 9, "29", 10, 0.0, 2, 9, 58, left, "1.26", None),
            (setup, "prewarm first", 11, "29", 10, 0.0, 2, 11, 58, left, "1.41", None),
            (setup, "prewarm repeat", 6, "29", 10, 0.25, 2, 7, 58, left, "0.78", None),
            (setup, "idle first", 6, "29", 10, 0.25, 2, 7, 58, left, "1.41", None),
            (setup, "prewarm repeat", 6, "39", 10, 0.25, 2, 7, 78, left, "0.80", None),
            (setup, "idle first", 6, "39", 10, 0.25, 2, 7, 78, left, "1.43", None),
            (setup, "prewarm repeat", 9, "39", 10, 0.0, 2, 9, 78, left, "0.97", None),
            (setup, "idle first", 9, "39", 10, 0.0, 2, 9, 78, left, "1.69", None),
            (
                setup,
                "prewarm repeat",
                11,
                "39",
                10,
                0.0,
                2,
                11,
                78,
                left,
                "1.35",
                None,
            ),
            (setup, "idle first", 11, "39", 10, 0.0, 2, 11, 78, left, "2.11", None),
        ]);
    }

    expected.extend([
        ("metal", "idle first", 6, "29", 10, 0.25, 2, 7, 58, "1.44", "1.41", None),
        ("metal", "idle first", 6, "39", 10, 0.25, 2, 7, 78, "1.44", "1.43", None),
        (
            "metal",
            "idle first",
            9,
            "29",
            9,
            0.0,
            2,
            9,
            58,
            "2.15",
            "1.69",
            Some("1.69"),
        ),
        (
            "metal",
            "idle first",
            9,
            "39",
            9,
            0.0,
            2,
            9,
            78,
            "2.15",
            "1.69",
            Some("1.69"),
        ),
        (
            "metal",
            "idle first",
            11,
            "29",
            9,
            0.0,
            2,
            11,
            58,
            "2.15",
            "2.01",
            Some("2.01"),
        ),
        (
            "metal",
            "idle first",
            11,
            "39",
            9,
            0.0,
            2,
            11,
            78,
            "2.15",
            "2.11",
            Some("2.11"),
        ),
    ]);

    for (setup, source, nprobe, r, fired, beta, mult, clusters, prefetched, left, needed, later) in expected {
        let what = format!("{setup}, {source}, nprobe {nprobe}, R {r}");
        let row = rows
            .iter()
            .find(|row| {
                (
                    row.setup.as_str(),
                    row.source.name(),
                    row.base.nprobe,
                    row.base.r_label.as_str(),
                ) == (setup, source.to_string(), nprobe, r)
            })
            .unwrap_or_else(|| panic!("no row for {what}"));
        let fire = row
            .schedule
            .and_then(|s| s.fire)
            .unwrap_or_else(|| panic!("nothing fires for {what}"));
        assert_eq!(
            (fire.settings, fire.clusters, fire.rows),
            (layer(fired, beta, mult), clusters, prefetched),
            "{what}"
        );
        assert_eq!(
            (format!("{:.2}", fire.left_ms), format!("{:.2}", fire.needed_ms)),
            (left.to_string(), needed.to_string()),
            "{what}"
        );
        assert!(fire.fits(), "{what}");
        let later_needed: Vec<String> = row.later.iter().map(|f| format!("{:.2}", f.needed_ms)).collect();
        assert_eq!(
            later_needed,
            later.map(str::to_string).into_iter().collect::<Vec<_>>(),
            "{what}"
        );
        assert!(row.later.iter().all(|f| f.settings.layer == 10 && !f.fits()), "{what}");
    }

    // The processor's prewarmed first reads have the chains of every layer only at nprobe 9 and 11 with R 29.
    for row in rows
        .iter()
        .filter(|r| r.setup.starts_with("cpu") && r.source.name() == "prewarm first")
    {
        assert_eq!(
            row.schedule.is_some(),
            row.base.r_label == "29" && row.base.nprobe != 6,
            "{row:?}"
        );
    }

    // On the processor every measured base fires at layer 10, which fits, from every source.
    for row in rows.iter().filter(|r| r.setup.starts_with("cpu")) {
        if let Some(schedule) = row.schedule {
            let fire = schedule.fire.unwrap();
            assert!(
                fire.settings.layer == 10 && fire.fits() && row.later.is_empty(),
                "{row:?}"
            );
        }
    }

    // The Rust predictors' guesses give the same fire layers, β and m wherever the trials measure their chains.
    let rust = pick("p6-coverage.jsonl", "rust.openings");
    assert_eq!(rust.len(), rows.len());

    for (ours, theirs) in rust.iter().zip(&rows) {
        assert_eq!(
            (&ours.setup, &ours.source, ours.base.nprobe, &ours.base.r_label),
            (&theirs.setup, &theirs.source, theirs.base.nprobe, &theirs.base.r_label)
        );
        let settings = |row: &Row| row.schedule.and_then(|s| s.fire).map(|f| f.settings);
        assert_eq!(
            settings(ours),
            settings(theirs),
            "{} {} {}",
            ours.setup,
            ours.source.name(),
            ours.base.nprobe
        );
    }
}
