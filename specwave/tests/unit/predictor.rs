//! Unit tests for the predictors and their file.

use std::io::ErrorKind;

use super::*;
use crate::testutil::{Rng, Scratch};

/// A predictor with seeded random weights for a model of `dim` dimensions.
fn random_ridge(layer: usize, dim: usize, seed: u64) -> Ridge {
    let mut rng = Rng(seed);
    let weights = rng.normals(2 * dim * dim).iter().map(|w| w * 0.1).collect();
    Ridge::new(layer, dim, weights, rng.normals(dim), 0.01).unwrap()
}

/// W′·x + b′ in f64, scaled to length 1.
fn reference(ridge: &Ridge, x: &[f32]) -> Vec<f64> {
    let width = 2 * ridge.dim();
    let mut out: Vec<f64> = ridge
        .weights()
        .chunks_exact(width)
        .zip(ridge.bias())
        .map(|(row, &b)| {
            row.iter()
                .zip(x)
                .map(|(&w, &v)| f64::from(w) * f64::from(v))
                .sum::<f64>()
                + f64::from(b)
        })
        .collect();
    let norm = out.iter().map(|v| v * v).sum::<f64>().sqrt();
    out.iter_mut().for_each(|v| *v /= norm);
    out
}

#[test]
fn a_guess_is_the_weights_times_the_summary_plus_the_bias_scaled_to_length_1() {
    // 2d is a multiple of 16 for d = 8 and 64 only, so the other sizes check the products past the last group.
    for dim in [1, 5, 8, 37, 64] {
        let ridge = random_ridge(3, dim, dim as u64);
        let x = Rng(100 + dim as u64).normals(2 * dim);
        let mut guess = vec![0f32; dim];
        ridge.predict(&x, &mut guess).unwrap();

        for (i, (&g, &e)) in guess.iter().zip(&reference(&ridge, &x)).enumerate() {
            assert!((f64::from(g) - e).abs() <= 1e-6, "d {dim}, value {i}: {g} against {e}");
        }

        let norm = guess.iter().map(|&g| f64::from(g).powi(2)).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() <= 1e-6, "d {dim}: length {norm}");
    }
}

#[test]
fn the_dot_product_adds_every_product_at_any_length() {
    for len in 0..=40 {
        let (a, b) = (Rng(len as u64).normals(len), Rng(1000 + len as u64).normals(len));
        let exact: f64 = a.iter().zip(&b).map(|(&x, &y)| f64::from(x) * f64::from(y)).sum();
        let scale: f64 = a.iter().zip(&b).map(|(&x, &y)| f64::from(x * y).abs()).sum();
        let found = f64::from(dot(&a, &b));
        assert!(
            (found - exact).abs() <= 1e-6 * (1.0 + scale),
            "length {len}: {found} against {exact}"
        );
    }
}

#[test]
fn a_guess_needs_2d_values_in_and_d_out_and_must_have_a_direction() {
    let ridge = random_ridge(0, 4, 1);
    let mut out = vec![0f32; 4];

    for (x_len, out_len) in [(7, 4), (9, 4), (8, 3), (8, 5), (0, 0)] {
        let found = ridge.predict(&vec![0.5; x_len], &mut vec![0.0; out_len]);
        assert_eq!(
            found.unwrap_err().kind(),
            ErrorKind::InvalidInput,
            "{x_len} in, {out_len} out"
        );
    }

    let mut nan = vec![0.5f32; 8];
    nan[3] = f32::NAN;
    assert_eq!(
        ridge.predict(&nan, &mut out).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    let zero = Ridge::new(0, 4, vec![0.0; 32], vec![0.0; 4], 0.0).unwrap();
    assert_eq!(
        zero.predict(&[1.0; 8], &mut out).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    let huge = Ridge::new(0, 4, vec![f32::MAX; 32], vec![0.0; 4], 0.0).unwrap();
    assert_eq!(
        huge.predict(&[f32::MAX; 8], &mut out).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    assert!(ridge.predict(&[0.5; 8], &mut out).is_ok());
}

#[test]
fn a_predictor_needs_d_rows_of_2d_weights_d_finite_biases_and_a_penalty_of_at_least_0() {
    let make = |w: usize, b: usize, penalty: f64| Ridge::new(1, 3, vec![0.1; w], vec![0.2; b], penalty);
    let ridge = make(18, 3, 0.0).unwrap();
    assert_eq!((ridge.layer(), ridge.dim(), ridge.penalty()), (1, 3, 0.0));
    assert_eq!((ridge.weights().len(), ridge.bias().len()), (18, 3));

    for (w, b, penalty) in [
        (17, 3, 0.1),
        (19, 3, 0.1),
        (18, 2, 0.1),
        (18, 4, 0.1),
        (18, 3, -0.1),
        (18, 3, f64::NAN),
        (18, 3, f64::INFINITY),
    ] {
        let found = make(w, b, penalty).err().map(|e| e.kind());
        assert_eq!(
            found,
            Some(ErrorKind::InvalidInput),
            "{w} weights, {b} biases, penalty {penalty}"
        );
    }

    let mut weights = vec![0.1; 18];
    weights[5] = f32::INFINITY;
    let found = Ridge::new(1, 3, weights, vec![0.2; 3], 0.1).err().map(|e| e.kind());
    assert_eq!(found, Some(ErrorKind::InvalidInput), "an infinite weight");
    let found = Ridge::new(1, 3, vec![0.1; 18], vec![0.2, f32::NAN, 0.2], 0.1)
        .err()
        .map(|e| e.kind());
    assert_eq!(found, Some(ErrorKind::InvalidInput), "a bias that isn't a number");
    let found = Ridge::new(1, 0, Vec::new(), Vec::new(), 0.1).err().map(|e| e.kind());
    assert_eq!(found, Some(ErrorKind::InvalidInput), "no dimensions");
}

#[test]
fn predictors_sort_by_layer_and_refuse_none_two_for_one_layer_or_mixed_sizes() {
    let predictors = Predictors::new(
        "org/model",
        vec![random_ridge(9, 4, 1), random_ridge(0, 4, 2), random_ridge(3, 4, 3)],
    )
    .unwrap();
    assert_eq!(predictors.layers().collect::<Vec<_>>(), [0, 3, 9]);
    assert_eq!(predictors.get(0), Some(&random_ridge(0, 4, 2)));
    assert_eq!(predictors.get(3), Some(&random_ridge(3, 4, 3)));
    assert_eq!(predictors.get(9), Some(&random_ridge(9, 4, 1)));
    assert!(predictors.get(4).is_none());
    assert_eq!((predictors.model(), predictors.dim()), ("org/model", 4));

    for (what, ridges) in [
        ("none", vec![]),
        ("two for one layer", vec![random_ridge(2, 4, 1), random_ridge(2, 4, 2)]),
        ("mixed sizes", vec![random_ridge(1, 4, 1), random_ridge(2, 5, 2)]),
    ] {
        let found = Predictors::new("m", ridges).err().map(|e| e.kind());
        assert_eq!(found, Some(ErrorKind::InvalidInput), "{what}");
    }
}

/// The bits of every value of `predictors`, layer by layer.
fn bits(predictors: &Predictors) -> Vec<(usize, u64, Vec<u32>)> {
    predictors
        .layers()
        .map(|l| {
            let r = predictors.get(l).unwrap();
            let values = r.weights().iter().chain(r.bias()).map(|x| x.to_bits()).collect();
            (l, r.penalty().to_bits(), values)
        })
        .collect()
}

#[test]
fn saved_predictors_read_back_bit_for_bit() {
    let scratch = Scratch::new("predictors-round-trip");
    let path = scratch.path("models/Snowflake/arctic/predictors.bin");
    let predictors = Predictors::new(
        "Snowflake/snowflake-arctic-embed-m-v1.5",
        vec![random_ridge(10, 6, 1), random_ridge(8, 6, 2)],
    )
    .unwrap();
    predictors.save(&path).unwrap();
    let read = Predictors::load(&path).unwrap();
    assert_eq!(read, predictors);
    assert_eq!(bits(&read), bits(&predictors));
    assert_eq!(scratch.names("models/Snowflake/arctic"), ["predictors.bin"]);

    let x = Rng(5).normals(12);
    let (mut a, mut b) = (vec![0f32; 6], vec![0f32; 6]);
    predictors.get(8).unwrap().predict(&x, &mut a).unwrap();
    read.get(8).unwrap().predict(&x, &mut b).unwrap();
    assert_eq!(a, b);

    // Saving again replaces the file, and its size follows the layout.
    let other = Predictors::new("other", vec![random_ridge(1, 2, 3)]).unwrap();
    other.save(&path).unwrap();
    assert_eq!(Predictors::load(&path).unwrap(), other);
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 24 + 5 + 12 + 4 * (8 + 2) + 4);
    assert_eq!(scratch.names("models/Snowflake/arctic"), ["predictors.bin"]);
}

/// `bytes` with its last 4 bytes set to the checksum of the rest.
fn with_checksum(mut bytes: Vec<u8>) -> Vec<u8> {
    let n = bytes.len() - 4;
    let crc = crc32fast::hash(&bytes[..n]);
    bytes[n..].copy_from_slice(&crc.to_le_bytes());
    bytes
}

#[test]
fn a_damaged_predictors_file_gives_invalid_data_and_another_format_unsupported() {
    let scratch = Scratch::new("predictors-damaged");
    let good = scratch.path("good.bin");
    Predictors::new("m", vec![random_ridge(2, 3, 1), random_ridge(5, 3, 2)])
        .unwrap()
        .save(&good)
        .unwrap();
    let bytes = std::fs::read(&good).unwrap();
    let path = scratch.path("bad.bin");
    let load = |bytes: &[u8]| {
        std::fs::write(&path, bytes).unwrap();
        Predictors::load(&path).err().map(|e| e.kind())
    };
    let set = |at: usize, value: &[u8]| {
        let mut b = bytes.clone();
        b[at..at + value.len()].copy_from_slice(value);
        b
    };
    // The first predictor starts after the header and the 1-byte model id, at byte 25, and its values at byte 37. The
    // second starts after the first one's 12 + 4·21 bytes.
    let second = 24 + 1 + 12 + 4 * 21;
    let cases: [(&str, Vec<u8>, ErrorKind); 13] = [
        ("empty", Vec::new(), ErrorKind::InvalidData),
        ("another mark", set(0, b"SWRIDGF"), ErrorKind::InvalidData),
        ("cut in the header", bytes[..10].to_vec(), ErrorKind::InvalidData),
        (
            "one byte short",
            bytes[..bytes.len() - 1].to_vec(),
            ErrorKind::InvalidData,
        ),
        ("a byte too many", [&bytes[..], &[0]].concat(), ErrorKind::InvalidData),
        ("a flipped weight", set(60, &[bytes[60] ^ 1]), ErrorKind::InvalidData),
        (
            "a flipped checksum",
            set(bytes.len() - 1, &[bytes[bytes.len() - 1] ^ 1]),
            ErrorKind::InvalidData,
        ),
        (
            "a size that overflows",
            set(12, &u32::MAX.to_le_bytes()),
            ErrorKind::InvalidData,
        ),
        ("a newer format", set(8, &2u32.to_le_bytes()), ErrorKind::Unsupported),
        ("format 0", set(8, &0u32.to_le_bytes()), ErrorKind::Unsupported),
        (
            "one layer twice",
            with_checksum(set(second, &2u32.to_le_bytes())),
            ErrorKind::InvalidData,
        ),
        (
            "a model id that isn't UTF-8",
            with_checksum(set(24, &[0xff])),
            ErrorKind::InvalidData,
        ),
        (
            "a weight that isn't a number",
            with_checksum(set(41, &f32::NAN.to_le_bytes())),
            ErrorKind::InvalidData,
        ),
    ];

    assert_eq!(load(&bytes), None);

    for (what, bytes, kind) in cases {
        assert_eq!(load(&bytes), Some(kind), "{what}");
    }

    let missing = Predictors::load(&scratch.path("missing.bin")).unwrap_err();
    assert_eq!(missing.kind(), ErrorKind::NotFound);
}

#[test]
fn the_predictors_of_a_model_live_in_a_folder_named_by_its_id() {
    let models = Path::new("/data/models");
    let arctic = path(models, "Snowflake/snowflake-arctic-embed-m-v1.5").unwrap();
    assert_eq!(
        arctic,
        Path::new("/data/models/Snowflake/snowflake-arctic-embed-m-v1.5/predictors.bin")
    );
    assert_eq!(
        path(models, "local").unwrap(),
        Path::new("/data/models/local/predictors.bin")
    );

    for model in ["", "..", "../escape", "a/../b", "/absolute", "./here"] {
        let found = path(models, model).err().map(|e| e.kind());
        assert_eq!(found, Some(ErrorKind::InvalidInput), "{model:?}");
    }
}
