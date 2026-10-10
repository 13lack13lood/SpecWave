//! Unit tests for the synthetic guesses: the noise shaped like a corpus, the guesses' cosine and posting coverage, the
//! calibration, and the command line. The data is small and random.

use serde_json::json;

use super::*;
use crate::coverage::Indexes;
use crate::testutil::{Shape, scratch, synthetic, write_table};

/// A corpus of `n` rows of `dim` values around (1, 0, ..., 0), spread 3 along the first axis and 0.1 along the others.
fn stretched(dir: &std::path::Path, n: usize, dim: usize) -> Fbin {
    let mut rng = Rng(5);
    let values: Vec<f32> = (0..n * dim)
        .map(|i| match i % dim {
            0 => (1.0 + 3.0 * rng.normal()) as f32,
            _ => (0.1 * rng.normal()) as f32,
        })
        .collect();
    let path = dir.join("corpus.fbin");
    write_table(&path, dim, &values, f32::to_bits);
    Fbin::open(&path).unwrap()
}

#[test]
fn noise_shaped_like_the_corpus_has_a_squared_length_of_1_and_the_corpuss_spread() {
    let dir = scratch("synthetic-shape");
    let corpus = stretched(&dir, 4000, 16);
    let data = DataNoise::new(&corpus, 64).unwrap();
    assert!(
        (data.mean[0] - 1.0).abs() < 0.15 && data.mean[1..].iter().all(|m| m.abs() < 0.02),
        "{:?}",
        data.mean
    );
    let (mut rng, mut out) = (Rng(9), vec![0f64; 16]);
    let (mut length, mut along) = (0.0, 0.0);
    let draws = 3000;

    for _ in 0..draws {
        data.draw(&mut rng, &mut out);
        length += out.iter().map(|x| x * x).sum::<f64>();
        along += out[0] * out[0];
    }

    let (length, along) = (length / draws as f64, along / draws as f64);
    assert!((length - 1.0).abs() < 0.06, "mean squared length {length}");
    let (mut fresh, mut used) = (vec![0f64; 16], vec![5f64; 16]);
    data.draw(&mut Rng(3), &mut fresh);
    data.draw(&mut Rng(3), &mut used);
    assert_eq!(fresh, used, "a draw depends on what its buffer held");
    // Nearly all of the corpus's spread is along the first axis, 9 of 9.15.
    assert!(
        (along / length - 9.0 / 9.15).abs() < 0.01,
        "share along the first axis {}",
        along / length
    );

    let same = dir.join("same.fbin");
    write_table(&same, 4, &[0.5f32; 40], f32::to_bits);
    assert_eq!(
        DataNoise::new(&Fbin::open(&same).unwrap(), 8).err().unwrap().kind(),
        ErrorKind::InvalidInput
    );
    assert_eq!(
        DataNoise::new(&corpus, 0).err().unwrap().kind(),
        ErrorKind::InvalidInput
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// `n` random queries of `dim` values, not of length 1.
fn queries(n: usize, dim: usize) -> Vec<f32> {
    let mut rng = Rng(17);
    (0..n * dim).map(|_| (2.0 * rng.normal()) as f32).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(&p, &q)| f64::from(p) * f64::from(q)).sum::<f64>();
    dot(a, b) / (dot(a, a) * dot(b, b)).sqrt()
}

#[test]
fn each_guess_has_length_1_and_the_cosine_that_the_formula_gives() {
    let dir = scratch("synthetic-guess");
    let corpus = stretched(&dir, 2000, 16);
    let data = DataNoise::new(&corpus, 32).unwrap();
    let e = queries(20, 16);
    let draws = Draws::new(&e, &data, 3).unwrap();
    assert_eq!(draws.len(), 20);
    let mut guess = vec![0f32; 16];

    for share in [0.0, 0.3, 1.0] {
        for scale in [0.0, 0.2, 0.7, 3.0] {
            let noise = Noise { share, scale };
            for (q, row) in e.as_chunks::<16>().0.iter().enumerate() {
                draws.guess(q, row, &noise, &mut guess);
                let length: f64 = guess.iter().map(|&x| f64::from(x).powi(2)).sum();
                assert!((length - 1.0).abs() < 1e-6);
                let real = cosine(&guess, row);
                assert!((draws.cosine(q, &noise) - real).abs() < 1e-6, "{noise:?} query {q}");
                if scale == 0.0 {
                    assert!(real > 1.0 - 1e-6);
                }
            }
        }
    }

    // A query's draws depend only on the seed and its row.
    let first = Draws::new(&e[..5 * 16], &data, 3).unwrap();
    let noise = Noise { share: 0.4, scale: 0.5 };
    assert!((0..5).all(|q| first.cosine(q, &noise) == draws.cosine(q, &noise)));
    assert!((0..5).any(|q| Draws::new(&e, &data, 4).unwrap().cosine(q, &noise) != draws.cosine(q, &noise)));
    assert_eq!(
        Draws::new(&e[..15], &data, 3).err().unwrap().kind(),
        ErrorKind::InvalidInput
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_scale_found_for_a_cosine_gives_that_mean_cosine() {
    let dir = scratch("synthetic-scale");
    let corpus = stretched(&dir, 2000, 16);
    let data = DataNoise::new(&corpus, 32).unwrap();
    let draws = Draws::new(&queries(50, 16), &data, 3).unwrap();

    for share in [0.0, 0.5, 1.0] {
        for target in [0.6, 0.9, 0.975] {
            let scale = draws.scale_for(share, target).unwrap();
            let got = draws.mean_cosine(&Noise { share, scale });
            assert!((got - target).abs() < 1e-9, "share {share}, target {target}, got {got}");
        }
    }

    for bad in [1.0, 0.0, -0.5, 1.5, f64::NAN] {
        assert_eq!(
            draws.scale_for(0.5, bad).unwrap_err().kind(),
            ErrorKind::InvalidInput,
            "{bad}"
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}

const SMALL: Shape = Shape {
    dim: 128,
    n: 3000,
    nlist: 24,
    queries: 40,
    width: 20,
};

/// The synthetic data's index file and its corpus as an `.fbin` file, in the folder `dir`.
fn index_and_corpus(dir: &std::path::Path, data: &crate::testutil::Synthetic) -> (Indexes, Fbin) {
    let indexes = Indexes::build(&data.rows, &data.clusters, &dir.join("index.coldivf"), 1, 7).unwrap();
    let path = dir.join("corpus.fbin");
    write_table(&path, SMALL.dim, &data.rows.values, f32::to_bits);
    (indexes, Fbin::open(&path).unwrap())
}

#[test]
fn posting_coverage_is_1_without_noise_and_falls_as_the_noise_grows() {
    let dir = scratch("synthetic-coverage");
    let data = synthetic(41, &SMALL);
    let (indexes, corpus) = index_and_corpus(&dir, &data);
    let noise_data = DataNoise::new(&corpus, 64).unwrap();
    let draws = Draws::new(&data.queries, &noise_data, 2).unwrap();
    let at = |share: f64, scale: f64, beta: f64| {
        posting_coverage(&indexes.file, &data.queries, &draws, &Noise { share, scale }, 3, beta).unwrap()
    };
    assert_eq!((at(0.5, 0.0, 0.0), at(0.5, 0.0, 1.0)), (1.0, 1.0));
    // The mean over the queries is each guess's own coverage, averaged.
    let noise = Noise { share: 0.5, scale: 0.8 };
    let mut ctx = SearchCtx::new(Limits {
        max_k: 1,
        max_nprobe: 6,
        max_rerank: 1,
    });
    let mut guess = vec![0f32; SMALL.dim];
    let each: Vec<f64> = (0..SMALL.queries)
        .map(|q| {
            let e = &data.queries[q * SMALL.dim..(q + 1) * SMALL.dim];
            draws.guess(q, e, &noise, &mut guess);
            coverage_of(&indexes.file, &mut ctx, e, &guess, 3, 4).unwrap()
        })
        .collect();
    assert!(
        each.iter()
            .all(|c| [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0].iter().any(|v| (c - v).abs() < 1e-12))
    );
    let mean = each.iter().sum::<f64>() / each.len() as f64;
    assert!((at(0.5, 0.8, 1.0 / 3.0) - mean).abs() < 1e-12);
    let (small, large) = (at(0.5, 0.3, 0.0), at(0.5, 1.5, 0.0));
    assert!(small < 1.0 && large < small, "{small} {large}");
    // Reading more clusters covers more.
    assert!(at(0.5, 1.5, 1.0) > large);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn rerank_coverage_counts_a_normal_searchs_rescored_rows_among_the_guesss_prefetched_ones() {
    let dir = scratch("synthetic-rerank");
    let data = synthetic(43, &SMALL);
    let (indexes, corpus) = index_and_corpus(&dir, &data);
    let noise_data = DataNoise::new(&corpus, 64).unwrap();
    let draws = Draws::new(&data.queries, &noise_data, 3).unwrap();
    let params = |rerank_mult: u8| SpecParams {
        base: coldivf::SearchParams {
            k: 10,
            nprobe: 3,
            rerank: 20,
            reach: None,
        },
        beta: 0.0,
        rerank_mult,
    };
    let mut ctx = SearchCtx::new(Limits {
        max_k: 40,
        max_nprobe: 3,
        max_rerank: 40,
    });
    let mut coverage =
        |e: &[f32], guess: &[f32], m: u8| rerank_coverage_of(&indexes.file, &mut ctx, e, guess, &params(m)).unwrap();
    let (mut guess, mut deeper_helps) = (vec![0f32; SMALL.dim], 0);

    for q in 0..SMALL.queries {
        let e = &data.queries[q * SMALL.dim..(q + 1) * SMALL.dim];
        // The embedding itself, reading the same clusters, prefetches exactly the rows that a normal search rescores.
        assert_eq!(coverage(e, e, 1), 1.0, "query {q}");
        draws.guess(q, e, &Noise { share: 0.5, scale: 0.3 }, &mut guess);
        let (once, twice) = (coverage(e, &guess, 1), coverage(e, &guess, 2));
        // The best 2R candidates hold the best R, so the deeper prefetch covers no less.
        assert!(twice >= once, "query {q}: {once} {twice}");
        deeper_helps += usize::from(twice > once);
    }

    assert!(deeper_helps > 0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_calibration_finds_the_share_that_made_the_guesses() {
    let dir = scratch("synthetic-calibrate");
    let data = synthetic(43, &SMALL);
    let (indexes, corpus) = index_and_corpus(&dir, &data);
    let noise_data = DataNoise::new(&corpus, 64).unwrap();
    let draws = Draws::new(&data.queries, &noise_data, 2).unwrap();
    let truth = Noise { share: 0.3, scale: 0.0 };
    let truth = Noise {
        scale: draws.scale_for(truth.share, 0.8).unwrap(),
        ..truth
    };
    let coverage = posting_coverage(&indexes.file, &data.queries, &draws, &truth, 3, 0.5).unwrap();
    let fit = calibrate(&indexes.file, &data.queries, &draws, (0.8, coverage), 3, 0.5, &shares()).unwrap();
    // Coverage moves in steps, so a neighboring share may tie with the one that made the guesses.
    assert!(
        (fit.noise.share - 0.3).abs() < 0.1 + 1e-9 && fit.coverage == coverage,
        "{fit:?}"
    );
    assert!((fit.cosine - 0.8).abs() < 1e-9);
    // Isotropic noise covers more of the query's clusters than noise shaped like the clustered corpus.
    assert!(fit.isotropic > coverage && coverage > fit.shaped, "{fit:?}");

    for bad in [&[][..], &[0.5, 1.2][..]] {
        let error = calibrate(&indexes.file, &data.queries, &draws, (0.8, coverage), 3, 0.5, bad).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// A grid line of fiqa's `l2` grid at nprobe 9 and R 29.
fn grid(guess: &str, layer: usize, beta: f64, seed: u64, cos: f64, posting: f64) -> String {
    json!({
        "record": "grid", "dataset": "fiqa", "clustering": "l2", "guess": guess, "layer": layer, "nprobe": 9,
        "r_label": "29", "beta": beta, "seed": seed, "cos": cos, "posting_coverage": posting,
    })
    .to_string()
}

fn key() -> Key {
    Key {
        dataset: "fiqa".to_string(),
        fit: "openings".to_string(),
        clustering: "l2".to_string(),
        nprobe: 9,
        r_label: "29".to_string(),
    }
}

#[test]
fn the_targets_are_the_layers_mean_cosine_and_its_posting_coverage_at_its_beta() {
    let lines = vec![
        grid("openings.L9", 9, 0.0, 7, 0.94, 0.90),
        grid("openings.L9", 9, 0.0, 11, 0.94, 0.92),
        grid("openings.L9", 9, 0.25, 7, 0.94, 0.97),
        grid("rust.openings.L9", 9, 0.0, 7, 0.5, 0.5),
        grid("openings.L8", 8, 0.0, 7, 0.9, 0.85),
        "{\"record\": \"sweep\"}".to_string(),
    ];
    let (cos, coverage) = targets(&lines, &key(), 9, 0.0).unwrap();
    assert!((cos - 0.94).abs() < 1e-12 && (coverage - 0.91).abs() < 1e-12);
    assert_eq!(targets(&lines, &key(), 9, 0.25), Some((0.94, 0.97)));
    assert!(targets(&lines, &key(), 9, 0.5).is_none());
    assert!(targets(&lines, &key(), 10, 0.0).is_none());
    let rust = Key {
        fit: "rust.openings".to_string(),
        ..key()
    };
    assert_eq!(targets(&lines, &rust, 9, 0.0), Some((0.5, 0.5)));
}

#[test]
fn the_fitted_noise_of_each_layer_is_its_last_line() {
    let line = |layer: u64, share: f64| {
        json!({"bench": "synthetic", "record": "noise", "layer": layer, "share": share, "scale": 0.4}).to_string()
    };
    let lines = vec![
        line(9, 0.2),
        line(8, 0.1),
        line(9, 0.3),
        json!({"bench": "spec", "record": "noise", "layer": 7, "share": 0.5, "scale": 0.4}).to_string(),
        "nothing".to_string(),
    ];
    let found = fitted(&lines);
    assert_eq!(
        found,
        [
            (8, Noise { share: 0.1, scale: 0.4 }),
            (9, Noise { share: 0.3, scale: 0.4 })
        ]
    );
}

#[test]
fn the_command_line_takes_calibrate_or_guesses_with_their_names() {
    let parse = |args: &[&str]| {
        let args: Vec<String> = args.iter().map(|a| a.to_string()).collect();
        Command::parse(&args, std::path::Path::new("/repo"))
    };
    let calibrate = parse(&[
        "calibrate",
        "--index",
        "i",
        "--queries",
        "q",
        "--corpus",
        "c",
        "--coverage",
        "a,b",
        "--layers",
        "9,10",
        "--nprobe",
        "11",
        "--rerank",
        "39",
    ])
    .unwrap();
    let Command::Calibrate {
        coverage,
        key,
        rerank,
        layers,
        documents,
        seed,
        out,
        ..
    } = calibrate
    else {
        panic!("{calibrate:?}");
    };
    assert_eq!(coverage, ["a", "b"]);
    assert_eq!((key.nprobe, key.r_label.as_str(), rerank), (11, "39", 39));
    assert_eq!((layers, documents, seed), (Some(vec![9, 10]), 64, 1));
    assert_eq!(out, "/repo/results/synthetic.jsonl");
    let guesses = parse(&[
        "guesses",
        "--noise",
        "n",
        "--queries",
        "q",
        "--corpus",
        "c",
        "--dir",
        "d",
        "--label",
        "syn",
        "--seed",
        "4",
    ])
    .unwrap();
    assert!(
        matches!(
            guesses,
            Command::Guesses {
                seed: 4,
                documents: 64,
                ..
            }
        ),
        "{guesses:?}"
    );

    let bad: [&[&str]; 8] = [
        &[],
        &["fit", "--index", "i"],
        &["calibrate", "--index", "i", "--queries", "q", "--corpus", "c"],
        &[
            "guesses",
            "--noise",
            "n",
            "--queries",
            "q",
            "--corpus",
            "c",
            "--dir",
            "d",
        ],
        &["guesses", "--noise", "n", "--noise", "m"],
        &["guesses", "--nois", "n"],
        &[
            "guesses",
            "--noise",
            "n",
            "--queries",
            "q",
            "--corpus",
            "c",
            "--dir",
            "d",
            "--label",
            "s",
            "--documents",
            "0",
        ],
        &[
            "calibrate",
            "--index",
            "i",
            "--queries",
            "q",
            "--corpus",
            "c",
            "--coverage",
            "a",
            "--layers",
            "x",
        ],
    ];

    for args in bad {
        assert_eq!(parse(args).unwrap_err().kind(), ErrorKind::InvalidInput, "{args:?}");
    }
}
