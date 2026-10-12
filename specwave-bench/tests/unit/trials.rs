//! Unit tests for the end-to-end trials: the plan, the read times, the layers from the coverage grid, the relay of
//! guesses from files, the timed query and its fields, the wait for the text, a trial's queries, the worker's reads
//! before a search, the command line and the summary. They run no model, and the index is small random data in
//! clusters.

use serde_json::json;
use specwave::speculate::Worker;

use super::*;
use crate::coverage::Indexes;
#[cfg(target_os = "macos")]
use crate::testutil::cold;
use crate::testutil::{Shape, scratch, synthetic, write_table};

const SMALL: Shape = Shape {
    dim: 128,
    n: 3000,
    nlist: 24,
    queries: 12,
    width: 20,
};

fn setting(start: Start, warm_ms: Option<u64>) -> Setting {
    Setting { start, warm_ms }
}

#[test]
fn the_plan_runs_each_query_once_with_speculation_on_and_once_off_in_every_setting_a_pass_apart() {
    let settings = [setting(Start::Idle, None), setting(Start::Prewarm, Some(50))];
    let trials = plan(3, 4, 5, &settings);
    assert_eq!(trials.len(), 16);
    assert!(trials.iter().enumerate().all(|(i, t)| t.number == i));
    let queries: Vec<usize> = trials[..4].iter().map(|t| t.query).collect();
    assert_eq!(queries, [3, 4, 0, 1]);

    for (i, t) in trials.iter().enumerate() {
        // Within a pass and a setting, speculation turns on and off by turns, and the second pass turns it around.
        let (pass, at) = (i / 8, i % 4);
        assert_eq!(t.on, (at + pass) % 2 == 0, "trial {i}");
        assert_eq!(t.setting, settings[(i / 4) % 2], "trial {i}");
        let twin = &trials[(i + 8) % 16];
        assert_eq!(
            (twin.query, twin.setting, twin.on),
            (t.query, t.setting, !t.on),
            "trial {i}"
        );
    }

    assert!(plan(0, 3, 0, &settings).is_empty());
    assert!(plan(0, 3, 5, &[]).is_empty());
}

#[test]
fn read_sizes_round_trip_through_the_command_line_and_bad_ones_are_refused() {
    let sizes = Sizes {
        tables_ms: 0.31,
        route_ms: 0.014,
        cluster_bytes: 104_857.6,
        row_bytes: 16_384.0,
        scan_ms: 0.0125,
    };
    assert_eq!(sizes.arg().parse::<Sizes>().unwrap(), sizes);

    for bad in ["1:2:3:4", "1:2:3:4:5:6", "1:2:x:4:5", "1:2:-3:4:5", "NaN:2:3:4:5", ""] {
        assert_eq!(
            bad.parse::<Sizes>().unwrap_err().kind(),
            ErrorKind::InvalidInput,
            "{bad}"
        );
    }
}

#[test]
fn the_read_times_add_the_route_the_cluster_wave_the_scan_and_the_row_wave() {
    let reads = IndexReads {
        wave: WaveModel {
            a_ms: 0.25,
            b_ms_per_mb: 0.5,
        },
        sizes: Sizes {
            tables_ms: 0.3,
            route_ms: 0.02,
            cluster_bytes: 100_000.0,
            row_bytes: 20_000.0,
            scan_ms: 0.01,
        },
    };
    assert_eq!(reads.tables_ms(), 0.3);
    let expected = 0.02 + (0.25 + 0.5 * 1.0) + 10.0 * 0.01 + (0.25 + 0.5 * 1.16);
    assert!((reads.chain_ms(10, 58) - expected).abs() < 1e-12);
    // No rows to read take no wave.
    assert!((reads.chain_ms(10, 0) - (0.02 + 0.75 + 0.1)).abs() < 1e-12);
    assert_eq!(reads.chain_ms(0, 0), 0.02);
}

/// A grid line of `spec_coverage` for a guess of fiqa's `l2` grid at R 29, with its recall gap and its rerank coverages
/// of the rows that the embedding and a normal search rescore, each with an interval of ±0.002.
fn grid_line(
    guess: &str,
    nprobe: usize,
    beta: f64,
    mult: usize,
    measured: (f64, f64, f64),
    posting: f64,
    seed: u64,
) -> String {
    let (gap, coverage, normal) = measured;
    let layer: Option<usize> = guess.rsplit_once(".L").and_then(|(_, l)| l.parse().ok());
    let wide = specwave::speculate::wide_nprobe(nprobe, beta);
    let mut line = json!({
        "record": "grid", "build": "b1", "dataset": "fiqa", "clustering": "l2", "seed": seed, "guess": guess,
        "layer": layer, "queries": 1548, "nprobe": nprobe, "beta": beta, "r_label": "29", "rerank": 29,
        "beta_effective": wide as f64 / nprobe as f64 - 1.0, "mult": mult, "posting_coverage": posting,
        "cos": 0.9,
    });

    for (name, mean) in [
        ("recall_gap", gap),
        ("rerank_coverage", coverage),
        ("rerank_coverage_normal", normal),
    ] {
        line[name] = mean.into();
        line[format!("{name}_lo")] = (mean - 0.002).into();
        line[format!("{name}_hi")] = (mean + 0.002).into();
    }

    line.to_string()
}

/// Layer 8 first passes at β 0.25 and m 3, by the rows of a normal search, though at m 2 already by the embedding's.
/// Layer 9 first passes at β 0 and m 2, and layer 0 never passes, with layer 0's posting coverage at β 1 of 0.8 with
/// seed 7 and 0.84 with seed 11. Another fit's layer 0 covers 0.1.
fn grid() -> Vec<String> {
    vec![
        grid_line("openings.L0", 9, 1.0, 4, (-0.1, 0.5, 0.5), 0.80, 7),
        grid_line("openings.L0", 9, 1.0, 4, (-0.1, 0.5, 0.5), 0.84, 11),
        grid_line("openings.L0", 9, 0.5, 4, (-0.2, 0.4, 0.4), 0.70, 7),
        grid_line("other.L0", 9, 1.0, 4, (-0.1, 0.5, 0.5), 0.10, 7),
        grid_line("openings.L8", 9, 0.0, 2, (-0.02, 0.85, 0.85), 0.85, 7),
        grid_line("openings.L8", 9, 0.25, 2, (0.01, 0.95, 0.88), 0.92, 7),
        grid_line("openings.L8", 9, 0.25, 3, (0.01, 0.92, 0.92), 0.92, 7),
        grid_line("openings.L9", 9, 0.0, 2, (0.0, 0.93, 0.93), 0.93, 7),
        grid_line("other.L9", 9, 0.0, 2, (0.0, 0.93, 0.93), 0.5, 7),
        grid_line("openings.L9", 11, 1.0, 2, (0.0, 0.93, 0.93), 0.5, 7),
    ]
}

fn base(nprobe: usize) -> SearchParams {
    SearchParams {
        k: 10,
        nprobe,
        rerank: 29,
        reach: None,
    }
}

#[test]
fn the_layers_and_the_wave_from_layer0_come_from_the_coverage_grid() {
    let lines = grid();
    let layers = layer_settings(&lines, "fiqa", "openings", "l2", &base(9)).unwrap();
    let found: Vec<(usize, f64, u8)> = layers.iter().map(|l| (l.layer, l.beta, l.rerank_mult)).collect();
    assert_eq!(found, [(8, 0.25, 3), (9, 0.0, 2)]);
    let at0 = layer0(&lines, "fiqa", "openings", "l2", &base(9)).unwrap();
    assert!((at0.coverage - 0.82).abs() < 1e-12 && at0.beta == 0.0, "{at0:?}");

    // Another fit has its own layer 0, and another nprobe, R or clustering has none.
    assert_eq!(
        layer0(&lines, "fiqa", "other", "l2", &base(9)).map(|l| l.coverage),
        Some(0.1)
    );
    assert!(layer0(&lines, "fiqa", "openings", "l2", &base(11)).is_none());
    assert!(layer0(&lines, "fiqa", "openings", "cos", &base(9)).is_none());
    assert!(
        layer0(
            &lines,
            "fiqa",
            "openings",
            "l2",
            &SearchParams { rerank: 39, ..base(9) }
        )
        .is_none()
    );
    assert!(
        layer_settings(&lines, "fiqa", "openings", "l2", &base(11))
            .unwrap()
            .is_empty()
    );
}

#[cfg(target_os = "macos")]
#[test]
fn the_read_sizes_come_from_traced_searches_of_the_index_dropped_from_the_cache() {
    let _cold = cold();
    let data = synthetic(33, &SMALL);
    let dir = scratch("trials-sizes");
    let path = dir.join("index.coldivf");
    // The build's handle goes, so the measure opens the index afresh and reads its cluster tables.
    drop(Indexes::build(&data.rows, &data.clusters, &path, 1, 7).unwrap());
    let queries_path = dir.join("queries.fbin");
    write_table(&queries_path, SMALL.dim, &data.queries, f32::to_bits);
    let queries = Fbin::open(&queries_path).unwrap();
    let p = base(4);
    let (sizes, backend) = measure_sizes(&path, &queries, &p, 8).unwrap();
    assert_eq!(Mapped::open(&path).unwrap().residency().unwrap(), 0.0);
    assert!(
        sizes.tables_ms > 0.0 && sizes.route_ms > 0.0 && sizes.scan_ms > 0.0,
        "{sizes:?}"
    );
    // The file system here takes read hints, so the fetches return while their reads run.
    assert_eq!(backend, "rdadvise");
    assert!(reads_ahead(backend));

    // The same searches, traced here, read the same bytes.
    let index = Index::open(&path).unwrap();
    let mut ctx = SearchCtx::new(Limits {
        max_k: 10,
        max_nprobe: 4,
        max_rerank: 29,
    });
    index.search(queries.row(0), &p, &mut ctx).unwrap();
    let (mut clusters, mut rows) = (0, 0);

    for i in 0..8 {
        index.search(queries.row(1 + i), &p, &mut ctx).unwrap();
        (clusters, rows) = (clusters + ctx.trace().cluster_bytes, rows + ctx.trace().row_bytes);
    }

    assert_eq!(sizes.cluster_bytes, clusters as f64 / 32.0);
    assert_eq!(sizes.row_bytes, rows as f64 / (8.0 * 29.0));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn layers_round_trip_through_the_command_line() {
    let layers = parse_layers("8:0.25:3,9:0:2,10:0.5:4").unwrap();
    let found: Vec<(usize, f64, u8)> = layers.iter().map(|l| (l.layer, l.beta, l.rerank_mult)).collect();
    assert_eq!(found, [(8, 0.25, 3), (9, 0.0, 2), (10, 0.5, 4)]);
    assert_eq!(parse_layers(&layers_arg(&layers)).unwrap(), layers);

    for bad in ["8:0.25", "8:0.25:3:1", "x:0:2", "8:y:2", "8:0:300", "8:0:2,"] {
        assert_eq!(parse_layers(bad).unwrap_err().kind(), ErrorKind::InvalidInput, "{bad}");
    }
}

#[test]
fn pass_through_predictors_give_back_the_first_half_of_a_summary_at_length_1() {
    let predictors = pass_through("test", 4, &[3, 0]).unwrap();
    assert_eq!(predictors.layers().collect::<Vec<_>>(), [0, 3]);
    let mut guess = vec![0.0; 4];
    predictors
        .get(3)
        .unwrap()
        .predict(&[3.0, 0.0, 4.0, 0.0, 9.0, 9.0, 9.0, 9.0], &mut guess)
        .unwrap();
    assert_eq!(guess, [0.6, 0.0, 0.8, 0.0]);
}

/// An encoder of 4 layers and `dim` values that runs no model. It hands each wanted layer l a summary of 2·dim values
/// that are all l, and its embedding is all 9.
struct Plain {
    dim: usize,
    calls: usize,
}

impl LayerEncoder for Plain {
    fn layers(&self) -> usize {
        4
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn encode(&mut self, _text: &str, hook: &mut dyn LayerHook) -> Result<Vec<f32>, Error> {
        self.calls += 1;

        for layer in 0..=4 {
            if hook.wants(layer) {
                hook.on_layer(layer, &vec![layer as f32; 2 * self.dim]);
            }
        }

        Ok(vec![9.0; self.dim])
    }
}

/// A hook that wants `layers` and keeps every summary it gets.
struct Keep {
    layers: Vec<usize>,
    got: Vec<(usize, Vec<f32>)>,
}

impl LayerHook for Keep {
    fn wants(&self, layer: usize) -> bool {
        self.layers.contains(&layer)
    }

    fn on_layer(&mut self, layer: usize, pooled: &[f32]) {
        self.got.push((layer, pooled.to_vec()));
    }
}

#[test]
fn the_relay_hands_over_each_layers_guess_from_its_file_and_the_files_embedding() {
    let dir = scratch("trials-relay");
    let (dim, queries) = (4, 3);
    let rows = |offset: f32| -> Vec<f32> { (0..queries * dim).map(|i| offset + i as f32).collect() };
    write_table(&dir.join("e.fbin"), dim, &rows(100.0), f32::to_bits);
    write_table(&dir.join("g.L1.fbin"), dim, &rows(10.0), f32::to_bits);
    write_table(&dir.join("g.L3.fbin"), dim, &rows(30.0), f32::to_bits);
    let guessed = Guessed::open(&dir.join("e.fbin"), &dir, "g", &[1, 3]).unwrap();
    // The model's own size differs, since its states are replaced.
    let mut relay = Relay::new(Plain { dim: 6, calls: 0 }, Some(guessed));
    assert_eq!((relay.dim(), relay.layers()), (4, 4));
    relay.set_query(2);
    let mut hook = Keep {
        layers: vec![1, 3],
        got: Vec::new(),
    };
    let embedding = relay.encode("text", &mut hook).unwrap();
    assert_eq!(embedding, [108.0, 109.0, 110.0, 111.0]);
    let expected = |offset: f32| -> Vec<f32> {
        let mut summary: Vec<f32> = (8..12).map(|i| offset + i as f32).collect();
        summary.extend([0.0; 4]);
        summary
    };
    assert_eq!(hook.got, [(1, expected(10.0)), (3, expected(30.0))]);
    assert_eq!(relay.inner.calls, 1);

    // A layer without a guess file gives an error, rather than a guess from the model's state.
    let mut wants_2 = Keep {
        layers: vec![2],
        got: Vec::new(),
    };
    assert_eq!(
        relay.encode("text", &mut wants_2).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    assert!(wants_2.got.is_empty());

    // Without guess files the relay is the model.
    let mut plain = Relay::new(Plain { dim: 6, calls: 0 }, None);
    let mut hook = Keep {
        layers: vec![2],
        got: Vec::new(),
    };
    assert_eq!(plain.encode("text", &mut hook).unwrap(), vec![9.0; 6]);
    assert_eq!((plain.dim(), hook.got), (6, vec![(2, vec![2.0; 12])]));

    // Guess files with another size of row or too few rows are refused.
    write_table(&dir.join("short.L1.fbin"), dim, &rows(10.0)[..2 * dim], f32::to_bits);
    write_table(&dir.join("wide.L1.fbin"), 3, &rows(10.0)[..9], f32::to_bits);

    for label in ["short", "wide"] {
        let error = Guessed::open(&dir.join("e.fbin"), &dir, label, &[1]).err().unwrap();
        assert_eq!(error.kind(), ErrorKind::InvalidData, "{label}");
    }

    assert_eq!(
        Guessed::open(&dir.join("e.fbin"), &dir, "g", &[2])
            .err()
            .unwrap()
            .kind(),
        ErrorKind::NotFound
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn recall_counts_the_true_neighbors_among_the_first_k_hits() {
    let hits: Vec<Hit> = [5, 3, 9, 1].iter().map(|&id| Hit { id, score: 0.0 }).collect();
    assert_eq!(recall(&hits, &[3, 5, 7, 8], 4), 0.5);
    assert_eq!(recall(&hits, &[3, 5, 7, 8], 2), 1.0);
    assert_eq!(recall(&hits, &[1, 9, 3, 5, 6], 3), 2.0 / 3.0);
    assert_eq!(recall(&hits[..1], &[5, 3], 2), 0.5);
}

/// An encoder of 4 layers whose texts name the queries of `queries`, rows of `dim` values, which are the embeddings and
/// the guesses at every layer.
struct ByRow {
    dim: usize,
    queries: Vec<f32>,
}

impl LayerEncoder for ByRow {
    fn layers(&self) -> usize {
        4
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn encode(&mut self, text: &str, hook: &mut dyn LayerHook) -> Result<Vec<f32>, Error> {
        let q: usize = text.parse().unwrap();
        let e = self.queries[q * self.dim..(q + 1) * self.dim].to_vec();
        let mut pooled = e.clone();
        pooled.extend(vec![0.0; self.dim]);

        for layer in 0..=4 {
            if hook.wants(layer) {
                hook.on_layer(layer, &pooled);
            }
        }

        Ok(e)
    }
}

#[test]
fn a_timed_query_holds_the_search_its_report_and_the_index_pages_cached_at_its_start() {
    let data = synthetic(31, &SMALL);
    let dir = scratch("trials-timed");
    let path = dir.join("index.coldivf");
    let indexes = Indexes::build(&data.rows, &data.clusters, &path, 1, 7).unwrap();
    let index = indexes.file;
    let p = base(4);
    let schedule = Schedule {
        fire: Some(specwave::scheduler::Fire {
            settings: LayerSettings {
                layer: 2,
                beta: 0.5,
                rerank_mult: 2,
            },
            clusters: 6,
            rows: 58,
            left_ms: 2.0,
            needed_ms: 1.0,
        }),
        layer0_beta: None,
    };
    let encoder = ByRow {
        dim: SMALL.dim,
        queries: data.queries.clone(),
    };
    let predictors = pass_through("test", SMALL.dim, &[2]).unwrap();
    let worker = Worker::new(schedule.limits(&p)).unwrap();
    let mut sw = SpecWave::new(encoder, predictors, schedule, worker).unwrap();
    let map = Mapped::open(&path).unwrap();
    let wide = SearchParams { nprobe: 6, ..p };

    for q in 0..SMALL.queries {
        let t = time_query(&mut sw, &index, &map, &q.to_string(), &p).unwrap();
        let mut ctx = SearchCtx::new(Limits {
            max_k: 10,
            max_nprobe: 6,
            max_rerank: 58,
        });
        let expected = index
            .search(&data.queries[q * SMALL.dim..(q + 1) * SMALL.dim], &wide, &mut ctx)
            .unwrap();
        let ids = |hits: &[Hit]| hits.iter().map(|h| h.id).collect::<Vec<u64>>();
        assert_eq!(ids(&t.hits), ids(expected), "query {q}");
        assert_eq!(t.report.ran, specwave::Ran::Guessed);
        assert!(t.start <= t.report.encoded.unwrap() && t.report.done.unwrap() <= t.end);
        assert_eq!(t.pages, map.len().div_ceil(crate::coldcache::page_size()));
        assert!(!t.warm && t.trace.cluster_bytes > 0 && t.trace.scanned > 0);

        let line = timed_fields(Json::new(), &t).line();
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["ran"], "guessed");
        assert_eq!(v["ranked"], "guess");
        assert_eq!(v["fired_layer"], 2);
        let us = |name: &str| v[name].as_f64().unwrap();
        assert!((us("total_us") - (t.end - t.start).as_secs_f64() * 1e6).abs() < 1e-6);
        assert!(us("encode_us") + us("post_embedding_us") <= us("total_us") + 1e-6);
        assert!(us("fired_us") <= us("encode_us") && us("guess_taken_us") <= us("worker_done_us"));
        assert_eq!(v["ids"].as_array().unwrap().len(), t.hits.len());
        assert_eq!(v["cluster_bytes"].as_u64(), Some(t.trace.cluster_bytes as u64));
        assert_eq!(v["pages_resident_at_start"].as_u64(), Some(t.resident as u64));
    }

    // A warm index runs no guess, and the timed query says it was warm.
    index.set_warm(true);
    let t = time_query(&mut sw, &index, &map, "0", &p).unwrap();
    assert!(t.warm);
    assert_eq!(t.report.ran, specwave::Ran::Warm);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_schedules_fields_are_null_without_a_fire_layer() {
    let v: serde_json::Value = serde_json::from_str(&schedule_fields(Json::new(), &Schedule::NONE).line()).unwrap();
    assert!(v["fire_layer"].is_null() && v["beta"].is_null() && v["layer0_beta"].is_null());
    assert_eq!(v["fits"], false);
    let fire = specwave::scheduler::Fire {
        settings: LayerSettings {
            layer: 10,
            beta: 0.25,
            rerank_mult: 2,
        },
        clusters: 11,
        rows: 58,
        left_ms: 2.78,
        needed_ms: 1.41,
    };
    let schedule = Schedule {
        fire: Some(fire),
        layer0_beta: Some(1.0),
    };
    let v: serde_json::Value = serde_json::from_str(&schedule_fields(Json::new(), &schedule).line()).unwrap();
    assert_eq!(
        (v["fire_layer"].as_u64(), v["mult"].as_u64(), v["clusters"].as_u64()),
        (Some(10), Some(2), Some(11))
    );
    assert_eq!(
        (v["beta"].as_f64(), v["layer0_beta"].as_f64(), v["fits"].as_bool()),
        (Some(0.25), Some(1.0), Some(true))
    );
    assert_eq!(
        (v["left_ms"].as_f64(), v["needed_ms"].as_f64()),
        (Some(2.78), Some(1.41))
    );
}

#[test]
fn the_texts_come_from_each_lines_text_field() {
    let dir = scratch("trials-texts");
    let path = dir.join("texts.jsonl");
    std::fs::write(&path, "{\"row\": 0, \"text\": \"a b\"}\n{\"text\": \"c\"}\n").unwrap();
    assert_eq!(read_texts(&path).unwrap(), ["a b", "c"]);
    std::fs::write(&path, "{\"row\": 0}\n").unwrap();
    assert_eq!(read_texts(&path).unwrap_err().kind(), ErrorKind::InvalidData);
    std::fs::write(&path, "").unwrap();
    assert_eq!(read_texts(&path).unwrap_err().kind(), ErrorKind::InvalidData);
    let _ = std::fs::remove_dir_all(&dir);
}

fn parse(args: &[&str]) -> Result<Settings, Error> {
    let args: Vec<String> = args.iter().map(|a| a.to_string()).collect();
    Settings::parse(&args, Path::new("/repo"))
}

const REQUIRED: [&str; 8] = [
    "--index",
    "i.coldivf",
    "--texts",
    "t.jsonl",
    "--truth",
    "g.ibin",
    "--queries",
    "q.fbin",
];

/// The required arguments, then `more`.
fn with<'a>(more: &[&'a str]) -> Vec<&'a str> {
    REQUIRED.iter().copied().chain(more.iter().copied()).collect()
}

#[test]
fn the_command_line_takes_each_known_name_once_with_good_values() {
    let s = parse(&with(&["--layers", "9:0:2,10:0:2"])).unwrap();
    assert_eq!((s.base.k, s.base.nprobe, s.base.rerank), (10, 9, 29));
    assert_eq!(
        (s.trials, s.first, s.per_trial, s.threads, s.warmup),
        (100, 0, 1, 5, 20)
    );
    assert_eq!(s.model, "Snowflake/snowflake-arctic-embed-m-v1.5");
    assert_eq!(
        parse(&with(&["--layers", "9:0:2", "--model", "bge-small"]))
            .unwrap()
            .model,
        "BAAI/bge-small-en-v1.5"
    );
    assert_eq!(
        (s.out.clone(), s.models.clone()),
        ("/repo/results/spec.jsonl".into(), "/repo/models".into())
    );
    assert_eq!(s.settings, [setting(Start::Idle, None), setting(Start::Prewarm, None)]);
    assert!(s.guesses.is_none() && s.layer0.is_none() && s.one.is_none() && !s.model_cold && !s.plan_only);

    let s = parse(&with(&[
        "--coverage",
        "a.jsonl,b.jsonl",
        "--fit",
        "rust.openings",
        "--start",
        "prewarm",
        "--warm-ms",
        "none,0,200",
        "--guesses",
        "dir",
        "--label",
        "synthetic",
        "--layer0",
        "0:0.83",
        "--model-cold",
        "--plan",
    ]))
    .unwrap();
    let grid = Layers::Grid {
        files: vec!["a.jsonl".to_string(), "b.jsonl".to_string()],
        dataset: "fiqa".to_string(),
        fit: "rust.openings".to_string(),
        clustering: "l2".to_string(),
    };
    assert_eq!(s.layers, grid);
    let warm: Vec<Option<u64>> = s.settings.iter().map(|s| s.warm_ms).collect();
    assert_eq!(warm, [None, Some(0), Some(200)]);
    assert_eq!(s.guesses, Some(("dir".into(), "synthetic".to_string())));
    assert_eq!(s.layer0.map(|l| (l.beta, l.coverage)), Some((0.0, 0.83)));
    assert!(s.model_cold && s.plan_only);

    let bad: [&[&str]; 17] = [
        &["--layers", "9:0:2", "--index", "j"],
        &["--layers", "9:0:2", "--nprobes", "9"],
        &["--layers", "9:0:2", "--plan", "--plan"],
        &["--layers", "9:0:2", "--trials", "0"],
        &["--layers", "9:0:2", "--per-trial", "0"],
        &["--layers", "9:0:2", "--rerank", "5"],
        &["--layers", "9:0:2", "--nprobe", "0"],
        &["--layers", "9:0:2", "--start", "busy"],
        &["--layers", "9:0:2", "--warm-ms", "soon"],
        &["--layers", "9:0:2", "--guesses", "dir"],
        &["--layers", "9:0:2", "--layer0", "1.0"],
        &["--layers", "9:0:2", "--layer0", "0:1.5"],
        &["--layers", "9:0:2", "--coverage", "a.jsonl"],
        &[],
        &["--layers", "9:0:2", "--one"],
        &["--layers", "9:0:2", "--k"],
        &["--layers", "9:0:2", "nprobe", "9"],
    ];

    for args in bad {
        let error = parse(&with(args)).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidInput, "{args:?}");
    }

    for missing in 0..4 {
        let mut args = REQUIRED.to_vec();
        args.drain(2 * missing..2 * missing + 2);
        args.extend(["--layers", "9:0:2"]);
        assert_eq!(parse(&args).unwrap_err().kind(), ErrorKind::InvalidInput, "{args:?}");
    }
}

#[test]
fn a_childs_command_line_carries_its_trial_and_what_the_parent_resolved() {
    let parent: Vec<&str> = REQUIRED
        .iter()
        .copied()
        .chain([
            "--coverage",
            "grid.jsonl",
            "--guesses",
            "dir",
            "--label",
            "synthetic",
            "--nprobe",
            "11",
            "--rerank",
            "39",
            "--k",
            "5",
            "--per-trial",
            "3",
            "--threads",
            "6",
            "--warmup",
            "25",
            "--model-cold",
            "--out",
            "out.jsonl",
            "--max-tokens",
            "64",
        ])
        .collect();
    let s = parse(&parent).unwrap();
    let s = Settings {
        layer0: Some(Layer0 {
            beta: 0.0,
            coverage: 0.81,
        }),
        ..s
    };
    let trial = Trial {
        number: 7,
        query: 42,
        on: false,
        setting: setting(Start::Prewarm, Some(200)),
    };
    let layers = parse_layers("8:0.25:3,10:0:2").unwrap();
    let sizes = Sizes {
        tables_ms: 0.3,
        route_ms: 0.03,
        cluster_bytes: 98_304.5,
        row_bytes: 16_384.0,
        scan_ms: 0.012,
    };
    let args = s.child_args("run-1", &trial, &layers, &sizes);
    let child = Settings::parse(&args, Path::new("/elsewhere")).unwrap();
    let one = child.one.clone().unwrap();
    assert_eq!(one.run, "run-1");
    assert_eq!(one.trial, trial);
    assert_eq!(one.sizes, sizes);
    assert_eq!(child.layers, Layers::Given(layers));
    assert_eq!(child.layer0, s.layer0);
    assert_eq!(
        (child.index.clone(), child.queries.clone()),
        (s.index.clone(), s.queries.clone())
    );
    assert_eq!(
        (child.guesses.clone(), child.model.clone(), child.models.clone()),
        (s.guesses.clone(), s.model.clone(), s.models.clone())
    );
    assert_eq!((child.base.k, child.base.nprobe, child.base.rerank), (5, 11, 39));
    assert_eq!(
        (child.per_trial, child.threads, child.warmup, child.max_tokens),
        (3, 6, 25, 64)
    );
    assert_eq!((child.out.clone(), child.model_cold), (s.out.clone(), true));

    // A child's trial needs its parts, and one setting.
    let without = |name: &str| -> Vec<String> {
        let at = args.iter().position(|a| a == name).unwrap();
        let mut fewer = args.clone();
        fewer.drain(at..at + 2);
        fewer
    };

    for name in ["--run-id", "--trial", "--query", "--mode", "--reads", "--layers"] {
        let error = Settings::parse(&without(name), Path::new("/elsewhere")).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidInput, "{name}");
    }

    let mut both = without("--start");
    both.extend(["--start".to_string(), "idle,prewarm".to_string()]);
    assert_eq!(
        Settings::parse(&both, Path::new("/x")).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
}

/// A result line of the trial of `query` in run `run`, with speculation `on` and the total and the recall given.
fn result(run: &str, query: u64, on: bool, total_ms: f64, recall: f64, ran: &str) -> String {
    json!({
        "bench": "spec", "run": run, "trial": 0, "seq": 0, "query": query, "mode": if on { "on" } else { "off" },
        "start": "idle", "warm_ms": null, "model_cold": false, "total_us": total_ms * 1e3,
        "post_embedding_us": total_ms * 100.0, "encode_us": total_ms * 900.0, "recall": recall, "ran": ran,
    })
    .to_string()
}

/// `line` with the posting coverage `coverage`.
fn covered(line: String, coverage: f64) -> String {
    let mut v: serde_json::Value = serde_json::from_str(&line).unwrap();
    v["posting_coverage"] = coverage.into();
    v.to_string()
}

/// `line` with the field `name` set to `value`.
fn with_field(line: String, name: &str, value: serde_json::Value) -> String {
    let mut v: serde_json::Value = serde_json::from_str(&line).unwrap();
    v[name] = value;
    v.to_string()
}

#[test]
fn the_summary_pairs_each_querys_runs_and_shows_the_medians() {
    let lines = vec![
        covered(result("r1", 0, true, 15.0, 0.9, "guessed"), 0.9),
        result("r1", 0, false, 17.0, 0.8, "unguessed"),
        with_field(
            covered(result("r1", 1, true, 16.0, 1.0, "guessed"), 1.0),
            "rerank_coverage",
            json!(0.96),
        ),
        with_field(result("r1", 1, false, 16.5, 1.0, "unguessed"), "load_ms", json!(410.0)),
        result("r1", 2, true, 14.0, 1.0, "unguessed"),
        result("r2", 2, false, 99.0, 0.0, "unguessed"),
        // A first query without a warm-up whose index had pages cached is left out, but not one with a warm-up.
        with_field(
            result("r1", 3, true, 50.0, 0.0, "guessed"),
            "pages_resident_at_start",
            json!(2),
        ),
        with_field(
            with_field(
                result("r1", 3, false, 8.0, 0.5, "warm"),
                "pages_resident_at_start",
                json!(9),
            ),
            "warm_ms",
            json!(0),
        ),
        // A later query of a trial finds what the first one read.
        with_field(
            with_field(
                result("r1", 4, true, 12.0, 1.0, "guessed"),
                "pages_resident_at_start",
                json!(5),
            ),
            "seq",
            json!(1),
        ),
        "not json".to_string(),
    ];
    let text = summary(&lines, "r1");
    assert!(
        text.contains(
            "| idle | none | no | first | on | 3 | 15.00 / 16.00 | 1.500 / 1.600 | 13.50 | - | 0.9667 | 0.9500 | \
             0.9600 | 2 of 3 |"
        ),
        "{text}"
    );
    assert!(
        text.contains(
            "| idle | none | no | first | off | 2 | 16.50 / 17.00 | 1.650 / 1.700 | 14.85 | 410.0 | 0.9000 | - | - | \
             0 of 2 |"
        ),
        "{text}"
    );
    assert!(
        text.contains("| idle | 0 | no | first | off | 1 | 8.00 / 8.00 | 0.800 / 0.800 | 7.20 | - | 0.5000 | - | - |"),
        "{text}"
    );
    assert!(
        text.contains("| idle | none | no | later | on | 1 | 12.00 / 12.00 |"),
        "{text}"
    );
    // Queries 0 and 1 ran both ways and saved 2 and 0.5 ms, with recall 0.05 better on average.
    assert!(
        text.contains("| idle | none | no | first | 2 | 0.50 / 0.50 / 2.00 | +0.0500 |"),
        "{text}"
    );
    assert!(!text.contains("99.00") && !text.contains("50.00"), "{text}");
    assert!(
        text.contains("\n1 first queries without a warm-up are left out"),
        "{text}"
    );
    assert!(!summary(&lines[..6], "r1").contains("left out"));
}

#[test]
fn ranks_are_by_nearest_rank() {
    let values = [5.0, 1.0, 4.0, 2.0, 3.0];
    assert_eq!(
        (rank(&values, 50.0), rank(&values, 90.0), rank(&values, 10.0)),
        (3.0, 5.0, 1.0)
    );
    assert_eq!(rank(&values, 0.0), 1.0);
    assert!(rank(&[], 50.0).is_nan());
}

#[test]
fn the_warm_up_starts_its_lead_before_the_text_and_the_start_state_ends_the_wait() {
    let length = |step: &Wait| match step {
        Wait::Sleep(d) | Wait::Spin(d) => *d,
        Wait::WarmUp => Duration::ZERO,
    };
    let total = |steps: &[Wait]| steps.iter().map(length).sum::<Duration>();

    for (start, state, wait) in [
        (Start::Idle, IDLE, Wait::Sleep(IDLE)),
        (Start::Prewarm, PREWARM, Wait::Spin(PREWARM)),
    ] {
        assert_eq!(before_text(&setting(start, None)), [wait]);
        for lead in [0, 1, 50, 149, 150, 151, 200, 1000] {
            let steps = before_text(&setting(start, Some(lead)));
            let lead = Duration::from_millis(lead);
            let at = steps.iter().position(|s| *s == Wait::WarmUp).unwrap();
            assert_eq!(total(&steps[at + 1..]), lead, "{start:?}, {lead:?}: {steps:?}");
            assert_eq!(total(&steps), state.max(lead), "{start:?}, {lead:?}: {steps:?}");
            // The start state's own steps end the wait, and fill at least its length.
            let kind = |s: &Wait| std::mem::discriminant(s) == std::mem::discriminant(&wait);
            let tail: Vec<&Wait> = steps
                .iter()
                .rev()
                .filter(|s| **s != Wait::WarmUp)
                .take_while(|s| kind(s))
                .collect();
            assert!(
                tail.iter().map(|s| length(s)).sum::<Duration>() >= state,
                "{start:?}, {lead:?}: {steps:?}"
            );
            assert!(
                steps.iter().all(|s| *s == Wait::WarmUp || !length(s).is_zero()),
                "{steps:?}"
            );
        }
    }
}

#[test]
fn only_the_backends_whose_fetch_returns_while_it_reads_read_ahead() {
    for backend in ["rdadvise", "io_uring", "fadvise"] {
        assert!(reads_ahead(backend), "{backend}");
    }

    for backend in ["pool", "portable", ""] {
        assert!(!reads_ahead(backend), "{backend}");
    }
}

#[cfg(target_os = "macos")]
#[test]
fn a_trial_times_each_query_on_a_cold_index_with_the_warm_up_started_its_lead_before_the_text() {
    let _cold = cold();
    let data = synthetic(37, &SMALL);
    let dir = scratch("trials-run");
    drop(Indexes::build(&data.rows, &data.clusters, &dir.join("index.coldivf"), 1, 7).unwrap());
    let texts: String = (0..SMALL.queries).map(|q| format!("{{\"text\": \"{q}\"}}\n")).collect();
    std::fs::write(dir.join("texts.jsonl"), texts).unwrap();
    write_table(&dir.join("truth.ibin"), SMALL.width, &data.truth, |v| v);
    write_table(&dir.join("queries.fbin"), SMALL.dim, &data.queries, f32::to_bits);
    let file = |name: &str| dir.join(name).display().to_string();
    let (index, texts, truth, queries) = (
        file("index.coldivf"),
        file("texts.jsonl"),
        file("truth.ibin"),
        file("queries.fbin"),
    );
    let s = parse(&[
        "--index",
        &index,
        "--texts",
        &texts,
        "--truth",
        &truth,
        "--queries",
        &queries,
        "--layers",
        "2:0.5:2",
        "--nprobe",
        "4",
        "--per-trial",
        "2",
    ])
    .unwrap();
    let sizes = Sizes {
        tables_ms: 0.1,
        route_ms: 0.01,
        cluster_bytes: 16_384.0,
        row_bytes: 16_384.0,
        scan_ms: 0.001,
    };
    // Layer 2 leaves 8 ms, which covers the reads.
    let profile = LayerProfile::new(vec![(0, 0.0), (2, 8.0), (4, 16.0)]).unwrap();
    let model = Model {
        load_ms: 400.0,
        ..Model::default()
    };
    // Each trial searches a copy of its own, so the reads of one can't land in the cache while the next drops its copy.
    let trial = |on: bool, warm_ms: Option<u64>| -> Vec<serde_json::Value> {
        let copy = dir.join(format!("index-{on}-{warm_ms:?}.coldivf"));
        std::fs::copy(dir.join("index.coldivf"), &copy).unwrap();
        let s = Settings {
            index: copy,
            ..s.clone()
        };
        let one = One {
            run: "r".to_string(),
            trial: Trial {
                number: 0,
                query: 3,
                on,
                setting: setting(Start::Idle, warm_ms),
            },
            sizes,
        };
        let encoder = ByRow {
            dim: SMALL.dim,
            queries: data.queries.clone(),
        };
        let predictors = pass_through("test", SMALL.dim, &[2]).unwrap();
        let tokens = |_: &ByRow, text: &str| Ok(text.len());
        let lines = run(
            &s,
            &one,
            Relay::new(encoder, None),
            predictors,
            &profile,
            &model,
            tokens,
        )
        .unwrap_or_else(|e| panic!("the trial with speculation {on} and warm-up {warm_ms:?} failed: {e}"));
        lines.iter().map(|l| serde_json::from_str(l).unwrap()).collect()
    };

    for (on, ran) in [(true, "guessed"), (false, "unguessed")] {
        let lines = trial(on, None);
        assert_eq!(lines.len(), 2);
        let first = &lines[0];
        assert_eq!((first["seq"].as_u64(), first["query"].as_u64()), (Some(0), Some(3)));
        assert_eq!(lines[1]["query"].as_u64(), Some(4));
        // The first query met a cold index, and speculation off still records the schedule that on would pick.
        assert_eq!(first["pages_resident_at_start"].as_u64(), Some(0));
        assert_eq!(
            (first["ran"].as_str(), first["fire_layer"].as_u64()),
            (Some(ran), Some(2))
        );
        assert_eq!(
            (first["load_ms"].as_f64(), first["route_ms"].as_f64()),
            (Some(400.0), Some(0.01))
        );
        assert_eq!(first["tokens"].as_u64(), Some(1));
        assert!(first["warm_lead_ms"].is_null() && first["recall"].as_f64().unwrap() > 0.5);
        // With speculation on, the guess was the embedding itself, so it covered every cluster and rescored row.
        match on {
            true => {
                assert_eq!(first["fired_layer"].as_u64(), Some(2));
                assert_eq!(first["posting_coverage"].as_f64(), Some(1.0));
                assert!(first["rerank_coverage"].as_f64().unwrap() >= 0.9, "{first}");
            }
            false => {
                assert!(first["fired_layer"].is_null() && first["posting_coverage"].is_null());
                assert!(first["rerank_coverage"].is_null());
            }
        }
    }

    // A warm-up starts its lead before the text, not before the start state. 100 ms is time enough for some of the
    // small index to arrive.
    for lead in [0, 100] {
        let first = &trial(true, Some(lead))[0];
        let measured = first["warm_lead_ms"].as_f64().unwrap();
        assert!(
            measured >= lead as f64 && measured < (lead + 50) as f64,
            "{measured} for {lead}"
        );
        assert_eq!(first["warm_ms"].as_u64(), Some(lead));
        let resident = first["pages_resident_at_start"].as_u64().unwrap();
        assert!(lead == 0 || resident > 0, "{resident} pages for {lead}");
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// Whether a page of `map` comes into the cache within 5 s.
#[cfg(target_os = "macos")]
fn a_page_arrives(map: &Mapped) -> bool {
    let start = Instant::now();

    while start.elapsed() < Duration::from_secs(5) {
        if map.resident(0..map.len()).unwrap() > 0 {
            return true;
        }
        std::thread::sleep(Duration::from_millis(1));
    }

    false
}

/// An encoder that runs `inner`, but first watches `map` for a page to come into the cache, and notes whether one did.
#[cfg(target_os = "macos")]
struct Watching {
    inner: ByRow,
    map: Mapped,
    arrived: Option<bool>,
}

#[cfg(target_os = "macos")]
impl LayerEncoder for Watching {
    fn layers(&self) -> usize {
        self.inner.layers()
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn encode(&mut self, text: &str, hook: &mut dyn LayerHook) -> Result<Vec<f32>, Error> {
        self.arrived = Some(a_page_arrives(&self.map));
        self.inner.encode(text, hook)
    }
}

#[cfg(target_os = "macos")]
#[test]
fn the_workers_prepare_starts_reading_the_cluster_tables_before_any_search() {
    let _cold = cold();
    let data = synthetic(39, &SMALL);
    let dir = scratch("trials-prepare");
    // Each part opens a file of its own, since the indexes of one file in a process share the tables they loaded.
    let (path, other) = (dir.join("index.coldivf"), dir.join("other.coldivf"));

    for file in [&path, &other] {
        drop(Indexes::build(&data.rows, &data.clusters, file, 1, 7).unwrap());
    }

    let index = Index::open(&path).unwrap();
    let map = Mapped::open(&path).unwrap();
    let mut worker = Worker::new(Schedule::NONE.limits(&base(4))).unwrap();
    make_cold(&[&map], COLD_WAIT).unwrap();
    worker.prepare(&index, SMALL.dim).unwrap();
    assert!(a_page_arrives(&map));

    // A search has the worker prepare the index before the model runs, so the tables' pages arrive meanwhile.
    let encoder = Watching {
        inner: ByRow {
            dim: SMALL.dim,
            queries: data.queries.clone(),
        },
        map: Mapped::open(&other).unwrap(),
        arrived: None,
    };
    let predictors = pass_through("test", SMALL.dim, &[2]).unwrap();
    let mut sw = SpecWave::new(encoder, predictors, Schedule::NONE, worker).unwrap();
    let index = Index::open(&other).unwrap();
    make_cold(&[&sw.encoder().map], COLD_WAIT).unwrap();
    sw.search("0", &index, &base(4), &mut Vec::new()).unwrap();
    assert_eq!(sw.encoder().arrived, Some(true));
    let _ = std::fs::remove_dir_all(&dir);
}
