//! Unit tests for the fit of the predictors, the training pairs, their log and the openings. The test that compares
//! the fit with the stored FiQA predictions is ignored, and run by hand.

use std::io::ErrorKind;
use std::path::PathBuf;

use super::*;
use crate::testutil::{Rng, Scratch, cosine};

/// The fit written out from the recipe in plain f64 loops, with Gaussian elimination in place of the Cholesky
/// factorization. It holds the standardization, Ȳ, and Wᵀ row by row.
struct Plain {
    mean: Vec<f64>,
    scale: Vec<f64>,
    ybar: Vec<f64>,
    wt: Vec<f64>,
    dim: usize,
}

impl Plain {
    fn fit(x: &[f32], y: &[f32], rows: Range<usize>, dim: usize, lambda: f64) -> Plain {
        let width = 2 * dim;
        let n = rows.len() as f64;
        let (mut mean, mut scale, mut ybar) = (vec![0.0; width], vec![0.0; width], vec![0.0; dim]);

        for r in rows.clone() {
            for i in 0..width {
                mean[i] += f64::from(x[r * width + i]) / n;
            }
            for j in 0..dim {
                ybar[j] += f64::from(y[r * dim + j]) / n;
            }
        }

        for r in rows.clone() {
            for i in 0..width {
                scale[i] += (f64::from(x[r * width + i]) - mean[i]).powi(2) / n;
            }
        }

        scale.iter_mut().for_each(|s| *s = s.sqrt() + 1e-6);
        let (mut a, mut b) = (vec![0.0; width * width], vec![0.0; width * dim]);

        for r in rows {
            let xs: Vec<f64> = (0..width)
                .map(|i| (f64::from(x[r * width + i]) - mean[i]) / scale[i])
                .collect();
            for i in 0..width {
                for k in 0..width {
                    a[i * width + k] += xs[i] * xs[k];
                }
                for j in 0..dim {
                    b[i * dim + j] += xs[i] * (f64::from(y[r * dim + j]) - ybar[j]);
                }
            }
        }

        for i in 0..width {
            a[i * width + i] += lambda * n;
        }

        Plain {
            mean,
            scale,
            ybar,
            wt: gauss(a, b, width, dim),
            dim,
        }
    }

    /// The guess for the summary `z`, scaled to length 1.
    fn guess(&self, z: &[f32]) -> Vec<f64> {
        let mut out = self.ybar.clone();

        for (i, &v) in z.iter().enumerate() {
            let s = (f64::from(v) - self.mean[i]) / self.scale[i];
            for (j, o) in out.iter_mut().enumerate() {
                *o += s * self.wt[i * self.dim + j];
            }
        }

        let norm = out.iter().map(|v| v * v).sum::<f64>().sqrt();
        out.iter().map(|v| v / norm).collect()
    }

    /// The mean cosine between the guesses for the pairs in `rows` and their embeddings.
    fn held_out(&self, x: &[f32], y: &[f32], rows: Range<usize>) -> f64 {
        let (width, dim) = (2 * self.dim, self.dim);
        let count = rows.len() as f64;
        rows.map(|r| {
            let g = self.guess(&x[r * width..(r + 1) * width]);
            let e: Vec<f64> = y[r * dim..(r + 1) * dim].iter().map(|&v| f64::from(v)).collect();
            let dot: f64 = g.iter().zip(&e).map(|(a, b)| a * b).sum();
            dot / e.iter().map(|v| v * v).sum::<f64>().sqrt()
        })
        .sum::<f64>()
            / count
    }
}

/// Solves a·w = b for w, with `n` unknowns and `cols` right-hand sides, by Gaussian elimination with partial pivoting.
fn gauss(mut a: Vec<f64>, mut b: Vec<f64>, n: usize, cols: usize) -> Vec<f64> {
    for c in 0..n {
        let p = (c..n)
            .max_by(|&i, &j| a[i * n + c].abs().total_cmp(&a[j * n + c].abs()))
            .unwrap();
        for k in 0..n {
            a.swap(c * n + k, p * n + k);
        }
        for k in 0..cols {
            b.swap(c * cols + k, p * cols + k);
        }
        for r in c + 1..n {
            let f = a[r * n + c] / a[c * n + c];
            for k in c..n {
                a[r * n + k] -= f * a[c * n + k];
            }
            for k in 0..cols {
                b[r * cols + k] -= f * b[c * cols + k];
            }
        }
    }

    for c in (0..n).rev() {
        for k in 0..cols {
            let mut v = b[c * cols + k];
            for r in c + 1..n {
                v -= a[c * n + r] * b[r * cols + k];
            }
            b[c * cols + k] = v / a[c * n + c];
        }
    }

    b
}

/// `n` pairs of summaries of `dim` dimensions and their embeddings. Summary value 0 never changes, value 1 barely does
/// and value 2 sits far from 0. The embeddings are a linear map of the summaries plus noise, scaled to length 1.
fn synthetic(n: usize, dim: usize, noise: f64, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let width = 2 * dim;
    let mut rng = Rng(seed);
    let map: Vec<f64> = (0..width * dim).map(|_| rng.normal()).collect();
    let (mut x, mut y) = (Vec::with_capacity(n * width), Vec::with_capacity(n * dim));

    for _ in 0..n {
        let signal: Vec<f64> = (0..width).map(|_| rng.normal()).collect();
        let mut row = signal.clone();
        row[0] = 0.7;
        row[1] *= 1e-5;
        row[2] += 50.0;
        x.extend(row.iter().map(|&v| v as f32));
        let e: Vec<f64> = (0..dim)
            .map(|j| (0..width).map(|i| signal[i] * map[i * dim + j]).sum::<f64>() + noise * rng.normal())
            .collect();
        let norm = e.iter().map(|v| v * v).sum::<f64>().sqrt();
        y.extend(e.iter().map(|&v| (v / norm) as f32));
    }

    (x, y)
}

/// A new summary like those of `synthetic`.
fn new_summary(rng: &mut Rng, dim: usize) -> Vec<f32> {
    let mut row: Vec<f64> = (0..2 * dim).map(|_| rng.normal()).collect();
    row[0] = 0.7;
    row[1] *= 1e-5;
    row[2] += 50.0;
    row.iter().map(|&v| v as f32).collect()
}

/// The fit of one layer, numbered 7, of the pairs `x` and `y`.
fn fit_one(x: &[f32], y: &[f32], dim: usize, params: &TrainParams) -> Fit {
    let pairs = Pairs::from_tables(dim, &[7], vec![x.to_vec()], y.to_vec()).unwrap();
    refit("org/model", &pairs, params).unwrap()
}

/// Checks the predictor of `fit` against `plain`, i.e. W′, b′ and the guesses for new summaries.
fn check_predictor(fit: &Fit, plain: &Plain, dim: usize, seed: u64, what: &str) {
    let width = 2 * dim;
    let ridge = fit.predictors.get(7).unwrap();

    for j in 0..dim {
        let mut shift = 0.0;
        let mut size = 0.0;
        for i in 0..width {
            let w = plain.wt[i * dim + j] / plain.scale[i];
            let found = f64::from(ridge.weights()[j * width + i]);
            assert!(
                (found - w).abs() <= 1e-6 * (1.0 + w.abs()),
                "{what}: W′ {j}, {i} is {found}, not {w}"
            );
            shift += w * plain.mean[i];
            size += (w * plain.mean[i]).abs();
        }
        let b = plain.ybar[j] - shift;
        let found = f64::from(ridge.bias()[j]);
        assert!(
            (found - b).abs() <= 1e-6 * (1.0 + size),
            "{what}: b′ {j} is {found}, not {b}"
        );
    }

    let mut rng = Rng(seed);
    let mut guess = vec![0f32; dim];

    for _ in 0..20 {
        let z = new_summary(&mut rng, dim);
        ridge.predict(&z, &mut guess).unwrap();
        for (&g, &e) in guess.iter().zip(&plain.guess(&z)) {
            assert!(
                (f64::from(g) - e).abs() <= 5e-5,
                "{what}: a guess value is {g}, not {e}"
            );
        }
    }
}

#[test]
fn the_fit_matches_the_recipe_written_as_plain_loops() {
    for (n, dim, noise, seed) in [(200, 4, 0.0, 1), (40, 8, 2.0, 2), (300, 6, 0.5, 3)] {
        let what = format!("{n} pairs of {dim} dimensions");
        let (x, y) = synthetic(n, dim, noise, seed);
        let params = TrainParams::new(&[7]);
        let fit = fit_one(&x, &y, dim, &params);
        let cut = (0.85 * n as f64) as usize;
        let held_out: Vec<f64> = params
            .lambdas
            .iter()
            .map(|&lambda| Plain::fit(&x, &y, 0..cut, dim, lambda).held_out(&x, &y, cut..n))
            .collect();
        let choice = &fit.choices[0];
        assert_eq!((choice.layer, choice.held_out.len()), (7, 3), "{what}");

        for (found, expected) in choice.held_out.iter().zip(&held_out) {
            assert!(
                (found - expected).abs() <= 1e-10,
                "{what}: held out {found}, not {expected}"
            );
        }

        let best = (1..3).fold(0, |b, i| if held_out[i] > held_out[b] { i } else { b });
        assert_eq!(choice.lambda, params.lambdas[best], "{what}");
        assert_eq!(fit.predictors.get(7).unwrap().penalty(), params.lambdas[best], "{what}");
        let plain = Plain::fit(&x, &y, 0..n, dim, params.lambdas[best]);
        check_predictor(&fit, &plain, dim, seed + 100, &what);
        eprintln!("{what}: held-out cosines {held_out:?}, chose {}", params.lambdas[best]);
    }
}

#[test]
fn the_best_held_out_cosine_wins_and_the_first_of_equal_ones() {
    assert_eq!(best(&[0.3]), 0);
    assert_eq!(best(&[0.1, 0.2, 0.3]), 2);
    assert_eq!(best(&[0.5, 0.5, 0.4]), 0);
    assert_eq!(best(&[0.4, 0.6, 0.6]), 1);
    assert_eq!(best(&[0.4, 0.6, 0.5]), 1);

    // Without noise a heavy penalty only pulls the guesses toward the mean embedding, so the light one wins on either
    // side.
    let (x, y) = synthetic(200, 4, 0.0, 1);

    for lambdas in [vec![10.0, 1e-3], vec![1e-3, 10.0]] {
        let params = TrainParams {
            lambdas: lambdas.clone(),
            ..TrainParams::new(&[7])
        };
        let choice = &fit_one(&x, &y, 4, &params).choices[0];
        assert_eq!(choice.lambda, 1e-3, "{lambdas:?}");
    }
}

#[test]
fn a_single_penalty_fits_every_pair_with_no_split() {
    let (x, y) = synthetic(50, 3, 0.3, 4);
    let params = TrainParams {
        lambdas: vec![0.05],
        split: f64::NAN,
        ..TrainParams::new(&[7])
    };
    let fit = fit_one(&x, &y, 3, &params);
    assert_eq!(
        fit.choices,
        [Choice {
            layer: 7,
            lambda: 0.05,
            held_out: Vec::new()
        }]
    );
    check_predictor(&fit, &Plain::fit(&x, &y, 0..50, 3, 0.05), 3, 9, "one penalty");

    // With one pair every summary value has no spread, so every guess is that pair's embedding.
    let fit = fit_one(&x[..6], &y[..3], 3, &params);
    let mut guess = vec![0f32; 3];
    fit.predictors
        .get(7)
        .unwrap()
        .predict(&new_summary(&mut Rng(1), 3), &mut guess)
        .unwrap();
    assert!(cosine(&guess, &y[..3]) > 1.0 - 1e-6);
}

/// The bits of every value of the predictors of `fit`.
fn bits(fit: &Fit) -> Vec<u32> {
    let p = &fit.predictors;
    p.layers()
        .flat_map(|l| {
            let r = p.get(l).unwrap();
            r.weights()
                .iter()
                .chain(r.bias())
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
fn a_fit_gives_the_same_bits_every_time_and_on_any_number_of_threads() {
    let (x, y) = synthetic(600, 16, 0.5, 5);
    let fit = |threads: usize| {
        let params = TrainParams {
            threads,
            ..TrainParams::new(&[7])
        };
        fit_one(&x, &y, 16, &params)
    };
    let one = fit(1);
    assert_eq!(bits(&fit(1)), bits(&one));
    assert_eq!(bits(&fit(3)), bits(&one));
    assert_eq!(bits(&fit(0)), bits(&one));
    assert_eq!(fit(3).choices, one.choices);
}

#[test]
fn pairs_hold_each_layers_summaries_and_the_embeddings_in_order() {
    let mut pairs = Pairs::new(2, &[5, 1]).unwrap();
    assert!(pairs.is_empty());
    pairs
        .push(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[0.1, 0.2])
        .unwrap();
    pairs
        .push(&[9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], &[0.3, 0.4])
        .unwrap();

    assert_eq!((pairs.len(), pairs.dim(), pairs.layers()), (2, 2, &[5, 1][..]));
    assert_eq!(pairs.states(5).unwrap(), [1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0]);
    assert_eq!(pairs.states(1).unwrap(), [5.0, 6.0, 7.0, 8.0, 13.0, 14.0, 15.0, 16.0]);
    assert!(pairs.states(2).is_none());
    assert_eq!(pairs.embeddings(), [0.1, 0.2, 0.3, 0.4]);

    let tables = Pairs::from_tables(
        2,
        &[5, 1],
        vec![pairs.states(5).unwrap().to_vec(), pairs.states(1).unwrap().to_vec()],
        pairs.embeddings().to_vec(),
    );
    assert_eq!(tables.unwrap(), pairs);
}

#[test]
fn pairs_refuse_wrong_sizes_and_values_that_arent_finite() {
    let new = |dim: usize, layers: &[usize]| Pairs::new(dim, layers).err().map(|e| e.kind());
    let many: Vec<usize> = (0..4097).collect();

    let cases: [(usize, &[usize]); 5] = [(0, &[1]), (2, &[]), (2, &[3, 3]), ((1 << 20) + 1, &[1]), (2, &many)];

    for (dim, layers) in cases {
        assert_eq!(
            new(dim, layers),
            Some(ErrorKind::InvalidInput),
            "{dim}, {} layers",
            layers.len()
        );
    }

    assert_eq!(new(1 << 20, &[1]), None);
    assert_eq!(new(2, &many[..4096]), None);
    let mut pairs = Pairs::new(2, &[5, 1]).unwrap();

    for (states, embedding) in [
        (vec![1.0; 7], vec![0.1; 2]),
        (vec![1.0; 9], vec![0.1; 2]),
        (vec![1.0; 8], vec![0.1; 1]),
        (vec![1.0; 8], vec![0.1; 3]),
        (vec![1.0, 2.0, f32::NAN, 4.0, 5.0, 6.0, 7.0, 8.0], vec![0.1; 2]),
        (vec![1.0; 8], vec![0.1, f32::INFINITY]),
    ] {
        let found = pairs.push(&states, &embedding).err().map(|e| e.kind());
        assert_eq!(found, Some(ErrorKind::InvalidInput), "{states:?}, {embedding:?}");
    }

    assert!(pairs.is_empty());
    let tables = |states: Vec<Vec<f32>>, embeddings: Vec<f32>| {
        Pairs::from_tables(2, &[5, 1], states, embeddings)
            .err()
            .map(|e| e.kind())
    };
    assert_eq!(tables(vec![vec![1.0; 4], vec![1.0; 4]], vec![0.1; 2]), None);
    assert_eq!(tables(vec![vec![1.0; 4]], vec![0.1; 2]), Some(ErrorKind::InvalidInput));
    assert_eq!(
        tables(vec![vec![1.0; 4], vec![1.0; 3]], vec![0.1; 2]),
        Some(ErrorKind::InvalidInput)
    );
    assert_eq!(
        tables(vec![vec![1.0; 4], vec![1.0; 4]], vec![0.1; 3]),
        Some(ErrorKind::InvalidInput)
    );
    assert_eq!(
        tables(vec![vec![1.0; 4], vec![1.0, f32::NAN, 1.0, 1.0]], vec![0.1; 2]),
        Some(ErrorKind::InvalidInput)
    );
}

/// An encoder whose summary at each layer and whose embedding are fixed functions of the text.
struct Fake {
    dim: usize,
    layers: usize,
    /// A layer it never shows the hook.
    skip: Option<usize>,
    /// Its summaries are one value short.
    short: bool,
    /// It shows the hook each layer twice.
    twice: bool,
    calls: usize,
}

impl Fake {
    fn new(dim: usize, layers: usize) -> Fake {
        Fake {
            dim,
            layers,
            skip: None,
            short: false,
            twice: false,
            calls: 0,
        }
    }

    fn seed(text: &str) -> u64 {
        text.bytes()
            .fold(17, |h, b| h.wrapping_mul(31).wrapping_add(u64::from(b)))
    }

    fn summary(text: &str, layer: usize, dim: usize) -> Vec<f32> {
        Rng(Fake::seed(text) ^ ((layer as u64) << 40)).normals(2 * dim)
    }

    fn embedding(text: &str, dim: usize) -> Vec<f32> {
        Rng(!Fake::seed(text)).normals(dim)
    }
}

impl LayerEncoder for Fake {
    fn layers(&self) -> usize {
        self.layers
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn encode(&mut self, text: &str, hook: &mut dyn LayerHook) -> Result<Vec<f32>, Error> {
        self.calls += 1;

        for layer in 0..=self.layers {
            if self.skip != Some(layer) && hook.wants(layer) {
                let mut summary = Fake::summary(text, layer, self.dim);
                if self.short {
                    summary.pop();
                }
                hook.on_layer(layer, &summary);
                if self.twice {
                    hook.on_layer(layer, &summary);
                }
            }
        }

        Ok(Fake::embedding(text, self.dim))
    }
}

#[test]
fn encoding_a_text_adds_its_summaries_at_the_wanted_layers_and_its_embedding() {
    let mut encoder = Fake::new(3, 4);
    let mut pairs = Pairs::new(3, &[4, 0, 2]).unwrap();
    pairs.encode(&mut encoder, "how to save").unwrap();
    pairs.encode(&mut encoder, "cheap index funds").unwrap();

    for layer in [4, 0, 2] {
        let expected = [
            Fake::summary("how to save", layer, 3),
            Fake::summary("cheap index funds", layer, 3),
        ];
        assert_eq!(pairs.states(layer).unwrap(), expected.concat(), "layer {layer}");
    }

    let expected = [
        Fake::embedding("how to save", 3),
        Fake::embedding("cheap index funds", 3),
    ];
    assert_eq!(pairs.embeddings(), expected.concat());
    assert_eq!(encoder.calls, 2);
}

#[test]
fn an_encoder_that_skips_a_layer_repeats_one_or_doesnt_fit_the_pairs_gives_an_error() {
    let mut pairs = Pairs::new(3, &[1, 2]).unwrap();
    let cases = [
        (
            "skips a wanted layer",
            Fake {
                skip: Some(2),
                ..Fake::new(3, 4)
            },
            ErrorKind::Other,
        ),
        (
            "gives short summaries",
            Fake {
                short: true,
                ..Fake::new(3, 4)
            },
            ErrorKind::Other,
        ),
        (
            "gives a layer twice",
            Fake {
                twice: true,
                ..Fake::new(3, 4)
            },
            ErrorKind::Other,
        ),
        ("is of another size", Fake::new(4, 4), ErrorKind::InvalidInput),
        ("has too few layers", Fake::new(3, 1), ErrorKind::InvalidInput),
    ];

    for (what, mut encoder, kind) in cases {
        let found = pairs.encode(&mut encoder, "text").err().map(|e| e.kind());
        assert_eq!(found, Some(kind), "an encoder that {what}");
    }

    assert!(pairs.is_empty());
    let mut skips_another = Fake {
        skip: Some(3),
        ..Fake::new(3, 4)
    };
    pairs.encode(&mut skips_another, "text").unwrap();
    assert_eq!(pairs.len(), 1);
}

#[test]
fn fit_runs_the_encoder_on_every_text_then_fits_its_pairs() {
    let texts: Vec<String> = (0..40).map(|i| format!("text number {i}")).collect();
    let mut encoder = Fake::new(3, 4);
    let mut pairs = Pairs::new(3, &[2, 4]).unwrap();

    for text in &texts {
        pairs.encode(&mut encoder, text).unwrap();
    }

    let params = TrainParams::new(&[2, 4]);
    let mut fresh = Fake::new(3, 4);
    let fit = fit(&mut fresh, "org/model", &texts, &params).unwrap();
    assert_eq!(fresh.calls, 40);
    assert_eq!(fit, refit("org/model", &pairs, &params).unwrap());
    assert_eq!(fit.predictors.layers().collect::<Vec<_>>(), [2, 4]);
    assert_eq!(fit.choices.iter().map(|c| c.layer).collect::<Vec<_>>(), [2, 4]);
    assert_eq!(fit.predictors.model(), "org/model");
}

#[test]
fn bad_settings_give_invalid_input_before_the_model_runs() {
    let texts = vec!["a text".to_string(); 10];
    let base = TrainParams::new(&[2]);
    let cases = [
        ("no layer", TrainParams::new(&[])),
        ("a layer twice", TrainParams::new(&[2, 2])),
        ("a layer past the last", TrainParams::new(&[5])),
        (
            "no penalty",
            TrainParams {
                lambdas: vec![],
                ..TrainParams::new(&[2])
            },
        ),
        (
            "a penalty of 0",
            TrainParams {
                lambdas: vec![0.0, 0.1],
                ..TrainParams::new(&[2])
            },
        ),
        (
            "a penalty that isn't a number",
            TrainParams {
                lambdas: vec![f64::NAN],
                ..TrainParams::new(&[2])
            },
        ),
        (
            "an infinite penalty",
            TrainParams {
                lambdas: vec![f64::INFINITY],
                ..TrainParams::new(&[2])
            },
        ),
        (
            "a split of 0",
            TrainParams {
                split: 0.0,
                ..TrainParams::new(&[2])
            },
        ),
        (
            "a split of 1",
            TrainParams {
                split: 1.0,
                ..TrainParams::new(&[2])
            },
        ),
        (
            "a split that isn't a number",
            TrainParams {
                split: f64::NAN,
                ..TrainParams::new(&[2])
            },
        ),
    ];

    for (what, params) in cases {
        let mut encoder = Fake::new(3, 4);
        let found = fit(&mut encoder, "m", &texts, &params).err().map(|e| e.kind());
        assert_eq!((found, encoder.calls), (Some(ErrorKind::InvalidInput), 0), "{what}");
    }

    let mut encoder = Fake::new(3, 4);
    assert!(fit(&mut encoder, "m", &texts, &base).is_ok());
}

#[test]
fn refit_needs_pairs_that_hold_its_layers_and_enough_of_them_to_hold_some_out() {
    let (x, y) = synthetic(10, 2, 0.1, 6);
    let pairs = Pairs::from_tables(2, &[7], vec![x.clone()], y.clone()).unwrap();
    let refit_with = |pairs: &Pairs, params: &TrainParams| refit("m", pairs, params).err().map(|e| e.kind());
    assert_eq!(
        refit_with(&pairs, &TrainParams::new(&[8])),
        Some(ErrorKind::InvalidInput)
    );
    assert_eq!(
        refit_with(&Pairs::new(2, &[7]).unwrap(), &TrainParams::new(&[7])),
        Some(ErrorKind::InvalidInput)
    );
    assert_eq!(refit_with(&pairs, &TrainParams::new(&[7])), None);

    // 0.85 of 1 pair is 0 pairs to fit on, and 0.05 of 10 is none either.
    let one = Pairs::from_tables(2, &[7], vec![x[..4].to_vec()], y[..2].to_vec()).unwrap();
    assert_eq!(refit_with(&one, &TrainParams::new(&[7])), Some(ErrorKind::InvalidInput));
    let small = TrainParams {
        split: 0.05,
        ..TrainParams::new(&[7])
    };
    assert_eq!(refit_with(&pairs, &small), Some(ErrorKind::InvalidInput));
    let tiny = TrainParams {
        split: 0.1,
        ..TrainParams::new(&[7])
    };
    assert_eq!(refit_with(&pairs, &tiny), None);
}

/// Seeded pairs for a log of `layers` layers of a model with `dim` dimensions.
fn log_rows(count: usize, dim: usize, layers: usize, seed: u64) -> Vec<(Vec<f32>, Vec<f32>)> {
    let mut rng = Rng(seed);
    (0..count)
        .map(|_| (rng.normals(layers * 2 * dim), rng.normals(dim)))
        .collect()
}

fn pairs_of(rows: &[(Vec<f32>, Vec<f32>)], dim: usize, layers: &[usize]) -> Pairs {
    let mut pairs = Pairs::new(dim, layers).unwrap();
    rows.iter().for_each(|(s, e)| pairs.push(s, e).unwrap());
    pairs
}

#[test]
fn a_log_reads_back_every_pair_it_was_given_across_opens() {
    let scratch = Scratch::new("log-round-trip");
    let path = scratch.path("queries.log");
    let rows = log_rows(5, 3, 2, 1);
    let mut log = QueryLog::open(&path, 3, &[4, 9]).unwrap();

    for (states, embedding) in &rows[..3] {
        log.append(states, embedding).unwrap();
    }

    log.sync().unwrap();
    drop(log);
    assert_eq!(QueryLog::read(&path).unwrap(), pairs_of(&rows[..3], 3, &[4, 9]));
    let mut log = QueryLog::open(&path, 3, &[4, 9]).unwrap();

    for (states, embedding) in &rows[3..] {
        log.append(states, embedding).unwrap();
    }

    drop(log);
    assert_eq!(QueryLog::read(&path).unwrap(), pairs_of(&rows, 3, &[4, 9]));
    // A header of 8 + 4 + 4 + 4 + 2·4 + 4 bytes, then 5 records of 4(2·3·2 + 3) + 4 bytes.
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 32 + 5 * 64);

    let empty = scratch.path("empty.log");
    drop(QueryLog::open(&empty, 3, &[4, 9]).unwrap());
    assert!(QueryLog::read(&empty).unwrap().is_empty());
}

#[test]
fn a_record_cut_short_is_left_out_and_dropped_before_the_next_append() {
    let scratch = Scratch::new("log-torn");
    let path = scratch.path("queries.log");
    let rows = log_rows(4, 2, 1, 2);
    let mut log = QueryLog::open(&path, 2, &[3]).unwrap();

    for (states, embedding) in &rows[..3] {
        log.append(states, embedding).unwrap();
    }

    drop(log);
    let len = std::fs::metadata(&path).unwrap().len();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(len - 10)
        .unwrap();
    assert_eq!(QueryLog::read(&path).unwrap(), pairs_of(&rows[..2], 2, &[3]));
    let mut log = QueryLog::open(&path, 2, &[3]).unwrap();
    log.append(&rows[3].0, &rows[3].1).unwrap();
    drop(log);
    let expected = [rows[0].clone(), rows[1].clone(), rows[3].clone()];
    assert_eq!(QueryLog::read(&path).unwrap(), pairs_of(&expected, 2, &[3]));
}

#[test]
fn a_log_of_other_settings_gives_invalid_input_and_a_damaged_one_invalid_data() {
    let scratch = Scratch::new("log-damaged");
    let path = scratch.path("queries.log");
    let rows = log_rows(2, 3, 2, 3);
    let mut log = QueryLog::open(&path, 3, &[4, 9]).unwrap();
    rows.iter().for_each(|(s, e)| log.append(s, e).unwrap());

    for (what, states, embedding) in [
        ("short summaries", vec![0.5; 11], vec![0.5; 3]),
        ("a long embedding", vec![0.5; 12], vec![0.5; 4]),
        ("a value that isn't a number", vec![0.5; 12], vec![0.5, f32::NAN, 0.5]),
    ] {
        let found = log.append(&states, &embedding).err().map(|e| e.kind());
        assert_eq!(found, Some(ErrorKind::InvalidInput), "{what}");
    }

    drop(log);
    let bytes = std::fs::read(&path).unwrap();

    for (what, dim, layers) in [
        ("another size", 4, &[4, 9][..]),
        ("other layers", 3, &[9, 4]),
        ("fewer layers", 3, &[4]),
    ] {
        let found = QueryLog::open(&path, dim, layers).err().map(|e| e.kind());
        assert_eq!(found, Some(ErrorKind::InvalidInput), "{what}");
    }

    let other = scratch.path("not-made.log");
    let found = QueryLog::open(&other, 0, &[4]).err().map(|e| e.kind());
    assert_eq!(found, Some(ErrorKind::InvalidInput));
    assert!(!other.exists());

    let flip = |at: usize| {
        let mut b = bytes.clone();
        b[at] ^= 1;
        b
    };
    let mut newer = bytes.clone();
    newer[8] = 2;
    let crc = crc32fast::hash(&newer[..28]);
    newer[28..32].copy_from_slice(&crc.to_le_bytes());
    let cases: [(&str, Vec<u8>, ErrorKind); 6] = [
        (
            "not a log",
            b"a text file, not a log of pairs".to_vec(),
            ErrorKind::InvalidData,
        ),
        ("a header cut short", bytes[..20].to_vec(), ErrorKind::InvalidData),
        ("a flipped size", flip(13), ErrorKind::InvalidData),
        ("a flipped layer", flip(24), ErrorKind::InvalidData),
        ("a flipped header checksum", flip(30), ErrorKind::InvalidData),
        ("a newer format", newer, ErrorKind::Unsupported),
    ];

    for (what, bytes, kind) in cases {
        std::fs::write(&path, &bytes).unwrap();
        let opened = QueryLog::open(&path, 3, &[4, 9]).err().map(|e| e.kind());
        let read = QueryLog::read(&path).err().map(|e| e.kind());
        assert_eq!((opened, read), (Some(kind), Some(kind)), "{what}");
        assert_eq!(
            std::fs::read(&path).unwrap(),
            bytes,
            "{what}: the open changed the file"
        );
    }

    // A damaged record fails the read, but the open checks only the header.
    std::fs::write(&path, flip(32 + 64 + 5)).unwrap();
    assert_eq!(QueryLog::read(&path).unwrap_err().kind(), ErrorKind::InvalidData);
    assert!(QueryLog::open(&path, 3, &[4, 9]).is_ok());
    let missing = QueryLog::read(&scratch.path("missing.log")).unwrap_err();
    assert_eq!(missing.kind(), ErrorKind::NotFound);
}

/// `count` documents, each of `words` words "d<doc>w<word>".
fn documents(count: usize, words: usize) -> Vec<String> {
    (0..count)
        .map(|d| (0..words).map(|w| format!("d{d}w{w}")).collect::<Vec<_>>().join(" "))
        .collect()
}

#[test]
fn openings_take_the_first_6_to_24_words_of_different_documents() {
    let mut docs = documents(1000, 30);
    docs[3] = "   ".to_string();
    docs[7] = String::new();
    docs[11] = "only three words".to_string();
    let texts = openings(&docs, 900, 7).unwrap();
    assert_eq!(texts.len(), 900);
    let mut seen = std::collections::HashSet::new();
    let (mut fewest, mut most) = (usize::MAX, 0);

    for text in &texts {
        let words: Vec<&str> = text.split(' ').collect();
        let doc = &docs[words[0]
            .strip_prefix('d')
            .map_or(11, |w| w.split('w').next().unwrap().parse().unwrap())];
        let doc_words: Vec<&str> = doc.split_whitespace().collect();
        assert_eq!(words, doc_words[..words.len()], "{text}");
        assert!(seen.insert(doc.clone()), "{text} repeats a document");
        if doc_words.len() >= 24 {
            fewest = fewest.min(words.len());
            most = most.max(words.len());
        }
    }

    assert_eq!((fewest, most), (6, 24));
    assert!(!seen.contains("   ") && !seen.contains(""));
    assert_eq!(openings(&docs, 900, 7).unwrap(), texts);
    assert_ne!(openings(&docs, 900, 8).unwrap(), texts);
    assert_eq!(openings(&docs, 998, 1).unwrap().len(), 998);
    let found = openings(&docs, 999, 1).err().map(|e| e.kind());
    assert_eq!(found, Some(ErrorKind::InvalidInput));
}

/// The rows of the `.fbin` file at `path`, i.e. its width and its values. The file holds its row count and width as
/// little-endian i32, then the rows as little-endian f32 values.
fn read_fbin(path: &Path) -> (usize, Vec<f32>) {
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .read_to_end(&mut bytes)
        .unwrap();
    let width = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let values = bytes[8..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| f32::from_le_bytes(b))
        .collect();
    (width, values)
}

/// The number after `"key": ` in `text`.
fn number_after(text: &str, key: &str) -> f64 {
    let at = text.find(&format!("\"{key}\": ")).unwrap_or_else(|| panic!("no {key}")) + key.len() + 4;
    let end = text[at..].find([',', '\n', '}']).unwrap() + at;
    text[at..end].trim().parse().unwrap()
}

fn data_dir() -> PathBuf {
    std::env::var_os("SPECWAVE_DATA")
        .map_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../data"), PathBuf::from)
}

/// Fits each stored FiQA training set again from its stored layer states, and compares the guesses for the test
/// queries with the stored ones. The fit on document openings picks its own penalty, which must be the stored one,
/// with the same held-out cosines. The fit on real queries picked its penalty after a shuffle that only the stored
/// numbers record, so it gets the stored penalty. The data folder is `$SPECWAVE_DATA`, else this repository's `data`.
#[test]
#[ignore = "reads the stored FiQA layer states and predictions, run by hand"]
fn the_fit_of_the_stored_fiqa_states_gives_the_stored_predictions() {
    let spec = data_dir().join("spec").join("fiqa");
    let states = spec.join("states");

    for set in ["openings", "queries"] {
        let (_, y) = read_fbin(&states.join(format!("{set}.final.fbin")));
        for layer in [0, 3, 6, 8, 9, 10] {
            let (width, x) = read_fbin(&states.join(format!("{set}.L{layer}.fbin")));
            let (_, z) = read_fbin(&states.join(format!("test.L{layer}.fbin")));
            let (dim, stored) = read_fbin(&spec.join("ehat").join(format!("{set}.L{layer}.fbin")));
            let info = std::fs::read_to_string(spec.join("ehat").join(format!("{set}.L{layer}.json"))).unwrap();
            let lambda = number_after(&info, "lam");
            assert_eq!(width, 2 * dim);
            let params = match set {
                "openings" => TrainParams::new(&[layer]),
                _ => TrainParams {
                    lambdas: vec![lambda],
                    ..TrainParams::new(&[layer])
                },
            };
            let pairs = Pairs::from_tables(dim, &[layer], vec![x], y.clone()).unwrap();
            let fit = refit("Snowflake/snowflake-arctic-embed-m-v1.5", &pairs, &params).unwrap();
            let choice = &fit.choices[0];
            assert_eq!(choice.lambda, lambda, "{set} L{layer}");
            let held: &str = &info[info.find("\"val_cos\"").unwrap_or(0)..];
            for (&l, &found) in params.lambdas.iter().zip(&choice.held_out) {
                let expected = number_after(held, &l.to_string());
                assert!(
                    (found - expected).abs() <= 1e-9,
                    "{set} L{layer} λ {l}: {found} against {expected}"
                );
            }
            let ridge = fit.predictors.get(layer).unwrap();
            let mut guess = vec![0f32; dim];
            let (mut worst, mut sum) = (f64::INFINITY, 0.0);
            for (summary, stored) in z.chunks_exact(width).zip(stored.chunks_exact(dim)) {
                ridge.predict(summary, &mut guess).unwrap();
                let c = cosine(&guess, stored);
                (worst, sum) = (worst.min(c), sum + c);
            }
            let n = (stored.len() / dim) as f64;
            assert!(
                worst >= 0.9999,
                "{set} L{layer}: cosine {worst} with the stored guesses"
            );
            eprintln!(
                "{set} L{layer}: λ {lambda}, 1 - cosine with the stored guesses {:.1e} at worst and {:.1e} on average",
                1.0 - worst,
                1.0 - sum / n
            );
        }
    }

    // At full size the fit splits its matrix products across threads, and the bits must not change.
    let (_, y) = read_fbin(&states.join("openings.final.fbin"));
    let (_, x) = read_fbin(&states.join("openings.L10.fbin"));
    let pairs = Pairs::from_tables(768, &[10], vec![x], y).unwrap();
    let with = |threads: usize| {
        let params = TrainParams {
            threads,
            ..TrainParams::new(&[10])
        };
        bits(&refit("m", &pairs, &params).unwrap())
    };
    assert_eq!(with(1), with(4));
}
