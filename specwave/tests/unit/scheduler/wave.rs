//! Unit tests for the wave model's fit and its cache files.

use super::*;
use crate::testutil::Scratch;

fn volume(uuid: &str) -> Volume {
    Volume {
        uuid: uuid.to_string(),
        drive: "APPLE SSD AP0512Z".to_string(),
        chip: "Apple M3 Pro".to_string(),
        os_build: "25G241".to_string(),
    }
}

const UUID: &str = "2A8CF686-E3D1-4548-BFFF-59918B6B966D";

#[test]
fn the_default_takes_a_quarter_millisecond_then_reads_at_3_4_gb_per_second() {
    assert_eq!(WaveModel::default(), WaveModel::DEFAULT);
    assert!((WaveModel::DEFAULT.read_ms(3.4e6) - 1.25).abs() < 1e-12);
    assert!((WaveModel::DEFAULT.read_ms(16384.0) - (0.25 + 16384.0 / 3.4e6)).abs() < 1e-12);
    assert_eq!(WaveModel::DEFAULT.read_ms(0.0), 0.0);
}

#[test]
fn a_fit_through_points_on_a_line_finds_the_line() {
    let line = WaveModel {
        a_ms: 0.3,
        b_ms_per_mb: 0.25,
    };
    let samples: Vec<(f64, f64)> = [16, 32, 64, 128, 256, 512]
        .iter()
        .flat_map(|&pages| {
            let bytes = pages as f64 * 16384.0;
            [(bytes, line.read_ms(bytes)), (bytes, line.read_ms(bytes))]
        })
        .collect();
    let fit = WaveModel::fit(&samples).unwrap();

    assert!((fit.a_ms - 0.3).abs() < 1e-9, "{fit:?}");
    assert!((fit.b_ms_per_mb - 0.25).abs() < 1e-9, "{fit:?}");

    // Noise around the line cancels out in a least-squares fit when it is symmetric.
    let noisy: Vec<(f64, f64)> = samples
        .iter()
        .enumerate()
        .map(|(i, &(bytes, t))| (bytes, t + if i % 2 == 0 { 0.05 } else { -0.05 }))
        .collect();
    let fit = WaveModel::fit(&noisy).unwrap();
    assert!(
        (fit.a_ms - 0.3).abs() < 1e-9 && (fit.b_ms_per_mb - 0.25).abs() < 1e-9,
        "{fit:?}"
    );
}

#[test]
fn a_fit_keeps_a_and_b_at_zero_or_above() {
    // On t = 2·MB − 0.5 the best line with a at 0 runs through the origin.
    let falling_start = [(1e6, 1.5), (2e6, 3.5), (3e6, 5.5)];
    let fit = WaveModel::fit(&falling_start).unwrap();
    let through = (1.5 + 2.0 * 3.5 + 3.0 * 5.5) / (1.0 + 4.0 + 9.0);
    assert_eq!(fit.a_ms, 0.0);
    assert!((fit.b_ms_per_mb - through).abs() < 1e-12, "{fit:?}");

    // On a falling line the best line with b at 0 is flat at the mean time.
    let falling = [(1e6, 3.0), (2e6, 2.0), (3e6, 1.0)];
    let fit = WaveModel::fit(&falling).unwrap();
    assert_eq!(fit.b_ms_per_mb, 0.0);
    assert!((fit.a_ms - 2.0).abs() < 1e-12, "{fit:?}");

    // A small negative slope by least squares, near a flat line with a large start, stays flat rather than passing
    // through the origin.
    let flat = [(1e6, 2.1), (2e6, 2.0), (3e6, 2.05), (4e6, 2.0)];
    let fit = WaveModel::fit(&flat).unwrap();
    assert_eq!(fit.b_ms_per_mb, 0.0);
    assert!((fit.a_ms - 2.0375).abs() < 1e-12, "{fit:?}");
}

#[test]
fn a_fit_needs_waves_of_two_sizes_with_good_numbers() {
    let bad: [&[(f64, f64)]; 6] = [
        &[],
        &[(16384.0, 0.3)],
        &[(16384.0, 0.3), (16384.0, 0.4)],
        &[(16384.0, 0.3), (32768.0, f64::NAN)],
        &[(16384.0, 0.3), (f64::INFINITY, 0.4)],
        &[(16384.0, 0.3), (32768.0, -0.4)],
    ];

    for samples in bad {
        assert_eq!(
            WaveModel::fit(samples).unwrap_err().kind(),
            ErrorKind::InvalidInput,
            "{samples:?}"
        );
    }

    assert!(WaveModel::fit(&[(0.0, 0.3), (16384.0, 0.4)]).is_ok());
}

#[test]
fn a_saved_fit_belongs_to_its_volume_drive_chip_and_build_only() {
    let dir = Scratch::new("wave-cache");
    let model = WaveModel {
        a_ms: 0.31,
        b_ms_per_mb: 0.27,
    };
    let path = model
        .save(
            &dir.path("support"),
            &volume(UUID),
            &[(262144.0, 0.4), (524288.0, 0.45)],
        )
        .unwrap();

    assert_eq!(path, dir.path("support").join(format!("wave-{UUID}.json")));
    assert_eq!(cache_path(&dir.path("support"), &volume(UUID)).unwrap(), path);
    assert_eq!(dir.names("support"), [format!("wave-{UUID}.json")]);
    assert_eq!(
        WaveModel::cached(&dir.path("support"), &volume(UUID)).unwrap(),
        Some(model)
    );

    let saved: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(saved["samples"], json!([[262144.0, 0.4], [524288.0, 0.45]]));
    assert_eq!(saved["format"], json!(1));

    let others = [
        Volume {
            drive: "APPLE SSD AP0256Z".to_string(),
            ..volume(UUID)
        },
        Volume {
            chip: "Apple M2".to_string(),
            ..volume(UUID)
        },
        Volume {
            os_build: "25H5".to_string(),
            ..volume(UUID)
        },
        volume("C7EFD3BE-4B34-4558-A11C-BC38127B8789"),
    ];

    for other in others {
        assert_eq!(
            WaveModel::cached(&dir.path("support"), &other).unwrap(),
            None,
            "{other:?}"
        );
    }

    assert_eq!(WaveModel::cached(&dir.path("nowhere"), &volume(UUID)).unwrap(), None);

    let newer = WaveModel {
        a_ms: 0.2,
        b_ms_per_mb: 0.3,
    };
    newer.save(&dir.path("support"), &volume(UUID), &[]).unwrap();
    assert_eq!(
        WaveModel::cached(&dir.path("support"), &volume(UUID)).unwrap(),
        Some(newer)
    );
}

#[test]
fn a_damaged_or_newer_cache_file_gives_an_error() {
    let dir = Scratch::new("wave-damaged");
    let path = cache_path(&dir.path(""), &volume(UUID)).unwrap();
    let good = json!({
        "format": 1, "volume": UUID, "drive": "APPLE SSD AP0512Z", "chip": "Apple M3 Pro", "os_build": "25G241",
        "a_ms": 0.3, "b_ms_per_mb": 0.25,
    });
    let changed = |key: &str, value: Value| {
        let mut v = good.clone();
        match value {
            Value::Null => v.as_object_mut().unwrap().remove(key),
            value => v.as_object_mut().unwrap().insert(key.to_string(), value),
        };
        serde_json::to_vec(&v).unwrap()
    };
    let cases = [
        (b"not json".to_vec(), ErrorKind::InvalidData),
        (changed("format", Value::Null), ErrorKind::InvalidData),
        (changed("format", json!(2)), ErrorKind::Unsupported),
        (changed("chip", Value::Null), ErrorKind::InvalidData),
        (changed("drive", json!(7)), ErrorKind::InvalidData),
        (changed("a_ms", Value::Null), ErrorKind::InvalidData),
        (changed("a_ms", json!(-0.1)), ErrorKind::InvalidData),
        (changed("b_ms_per_mb", json!("fast")), ErrorKind::InvalidData),
    ];

    for (bytes, kind) in cases {
        std::fs::write(&path, &bytes).unwrap();
        let found = WaveModel::cached(&dir.path(""), &volume(UUID)).unwrap_err();
        assert_eq!(found.kind(), kind, "{}", String::from_utf8_lossy(&bytes));
        assert!(found.to_string().contains(&path.display().to_string()), "{found}");
    }

    std::fs::write(&path, serde_json::to_vec(&good).unwrap()).unwrap();
    let expected = WaveModel {
        a_ms: 0.3,
        b_ms_per_mb: 0.25,
    };
    assert_eq!(WaveModel::cached(&dir.path(""), &volume(UUID)).unwrap(), Some(expected));
}

#[test]
fn only_plain_uuids_and_good_numbers_are_cached() {
    let dir = Scratch::new("wave-plain");
    let model = WaveModel::DEFAULT;

    for uuid in ["", "../x", "a/b", "C7EFD3BE 4B34"] {
        let kind = |e: Error| e.kind();
        assert_eq!(
            cache_path(&dir.path(""), &volume(uuid)).map_err(kind),
            Err(ErrorKind::InvalidInput)
        );
        assert_eq!(
            WaveModel::cached(&dir.path(""), &volume(uuid)).map_err(kind),
            Err(ErrorKind::InvalidInput)
        );
        assert_eq!(
            model.save(&dir.path(""), &volume(uuid), &[]).map_err(kind),
            Err(ErrorKind::InvalidInput)
        );
    }

    let bad = [
        (
            WaveModel {
                a_ms: f64::NAN,
                ..model
            },
            vec![],
        ),
        (
            WaveModel {
                b_ms_per_mb: -1.0,
                ..model
            },
            vec![],
        ),
        (model, vec![(16384.0, f64::INFINITY)]),
        (model, vec![(-1.0, 0.3)]),
    ];

    for (model, samples) in bad {
        let found = model.save(&dir.path(""), &volume(UUID), &samples).unwrap_err();
        assert_eq!(found.kind(), ErrorKind::InvalidInput, "{model:?} {samples:?}");
    }

    assert!(dir.names("").is_empty());
}

#[cfg(target_os = "macos")]
#[test]
fn the_fit_for_a_paths_volume_comes_from_the_cache_or_else_is_the_default() {
    let dir = Scratch::new("wave-volume");
    std::fs::write(dir.path("index"), b"x").unwrap();
    let here = Volume::of(&dir.path("index")).unwrap();
    let model = WaveModel {
        a_ms: 0.4,
        b_ms_per_mb: 0.2,
    };

    assert_eq!(
        WaveModel::for_volume_in(&dir.path("support"), &dir.path("index")),
        WaveModel::DEFAULT
    );
    model.save(&dir.path("support"), &here, &[]).unwrap();
    assert_eq!(
        WaveModel::for_volume_in(&dir.path("support"), &dir.path("index")),
        model
    );
    assert_eq!(
        WaveModel::for_volume_in(&dir.path("support"), &dir.path("missing")),
        WaveModel::DEFAULT
    );

    std::fs::write(cache_path(&dir.path("support"), &here).unwrap(), b"{").unwrap();
    assert_eq!(
        WaveModel::for_volume_in(&dir.path("support"), &dir.path("index")),
        WaveModel::DEFAULT
    );
}

#[test]
fn the_support_folder_follows_each_systems_convention() {
    let Some(dir) = support_dir() else {
        assert!(!cfg!(any(target_os = "macos", target_os = "linux")) || std::env::var_os("HOME").is_none());
        return;
    };

    assert!(dir.is_absolute());

    if cfg!(target_os = "macos") {
        assert!(
            dir.ends_with("Library/Application Support/SpecWave"),
            "{}",
            dir.display()
        );
    } else {
        assert!(dir.ends_with("specwave"), "{}", dir.display());
    }
}
