//! Unit tests for the wave calibration's plan, its summaries, its command line and, on macOS, a short run on a small
//! scratch file.

use std::path::PathBuf;

use super::*;
#[cfg(target_os = "macos")]
use crate::testutil::cold;
use crate::testutil::scratch;

#[test]
fn a_plan_runs_rounds_of_every_size_with_distinct_sorted_pages() {
    let waves = plan(&[16, 32, 512], 3, 1000, 7).unwrap();
    let sizes: Vec<usize> = waves.iter().map(Vec::len).collect();

    assert_eq!(sizes, [16, 32, 512, 16, 32, 512, 16, 32, 512]);

    for wave in &waves {
        assert!(wave.windows(2).all(|w| w[0] < w[1]), "{wave:?}");
        assert!(wave.iter().all(|&p| p < 1000));
    }

    assert_eq!(waves, plan(&[16, 32, 512], 3, 1000, 7).unwrap());
    assert_ne!(waves, plan(&[16, 32, 512], 3, 1000, 8).unwrap());
    assert_ne!(waves[0], waves[3]);

    // A wave as large as the file reads all of it.
    assert_eq!(plan(&[50], 1, 50, 1).unwrap(), [(0..50).collect::<Vec<usize>>()]);

    let pages: std::collections::HashSet<usize> = waves.iter().flatten().copied().collect();
    assert!(pages.iter().any(|&p| p >= 900) && pages.iter().any(|&p| p < 100));

    for (sizes, trials, pages) in [
        (&[][..], 3, 1000),
        (&[16][..], 0, 1000),
        (&[0][..], 3, 1000),
        (&[16][..], 3, 15),
    ] {
        let found = plan(sizes, trials, pages, 7).unwrap_err().kind();
        assert_eq!(found, ErrorKind::InvalidInput, "{sizes:?} {trials} {pages}");
    }
}

#[test]
fn neighboring_pages_make_one_read() {
    assert_eq!(ranges(&[1, 2, 3, 7, 9, 10]), [(1, 3), (7, 1), (9, 2)]);
    assert_eq!(ranges(&[5]), [(5, 1)]);
    assert_eq!(ranges(&[0, 2, 4]), [(0, 1), (2, 1), (4, 1)]);
    assert!(ranges(&[]).is_empty());
}

fn wave(pages: usize, ms: f64) -> Wave {
    Wave {
        pages,
        ranges: pages,
        bytes: (pages * PAGE) as u64,
        ms,
    }
}

#[test]
fn each_size_gets_the_median_and_the_90th_percentile_of_its_waves() {
    let waves = [
        wave(32, 10.0),
        wave(16, 5.0),
        wave(16, 1.0),
        wave(32, 20.0),
        wave(16, 4.0),
        wave(16, 2.0),
        wave(16, 3.0),
    ];
    let sizes = summarize(&waves);

    assert_eq!(
        sizes,
        [
            Size {
                pages: 16,
                bytes: 16 * 16384,
                waves: 5,
                median_ms: 3.0,
                p90_ms: 5.0,
            },
            Size {
                pages: 32,
                bytes: 32 * 16384,
                waves: 2,
                median_ms: 10.0,
                p90_ms: 20.0,
            },
        ]
    );
    assert!(summarize(&[]).is_empty());
}

#[test]
fn the_fit_runs_through_each_sizes_median() {
    let line = WaveModel {
        a_ms: 0.2,
        b_ms_per_mb: 0.3,
    };
    let waves: Vec<Wave> = [16, 64, 256]
        .iter()
        .flat_map(|&pages| {
            let t = line.read_ms((pages * PAGE) as f64);
            [wave(pages, t - 0.1), wave(pages, t), wave(pages, t + 5.0)]
        })
        .collect();
    let sizes = summarize(&waves);
    let fitted = fit(&sizes).unwrap();

    assert!(
        (fitted.a_ms - 0.2).abs() < 1e-9 && (fitted.b_ms_per_mb - 0.3).abs() < 1e-9,
        "{fitted:?}"
    );
    assert_eq!(
        points(&sizes)[1],
        ((64 * PAGE) as f64, line.read_ms((64 * PAGE) as f64))
    );
    assert_eq!(fit(&sizes[..1]).unwrap_err().kind(), ErrorKind::InvalidInput);
}

fn parse(args: &[&str], support: Option<&str>) -> Result<Settings, Error> {
    let args: Vec<String> = args.iter().map(|a| a.to_string()).collect();
    Settings::parse(&args, &PathBuf::from("/repo"), support.map(PathBuf::from))
}

#[test]
fn the_command_line_takes_each_known_name_once_with_good_values() {
    let defaults = parse(&[], Some("/support")).unwrap();
    assert_eq!(
        defaults,
        Settings {
            file: PathBuf::from("/repo/data/wavecal.bin"),
            size_mib: 256,
            sizes: vec![16, 32, 64, 128, 256, 512],
            trials: 30,
            busy_ms: 150,
            seed: 1,
            cache: Some(PathBuf::from("/support")),
            out: PathBuf::from("/repo/results/wavecal.jsonl"),
            keep: false,
        }
    );

    let given = [
        "--file",
        "/x/scratch.bin",
        "--size-mib",
        "64",
        "--sizes",
        "16, 64",
        "--trials",
        "5",
        "--busy-ms",
        "0",
        "--seed",
        "9",
        "--cache",
        "/c",
        "--out",
        "/o.jsonl",
        "--keep",
    ];
    assert_eq!(
        parse(&given, None).unwrap(),
        Settings {
            file: PathBuf::from("/x/scratch.bin"),
            size_mib: 64,
            sizes: vec![16, 64],
            trials: 5,
            busy_ms: 0,
            seed: 9,
            cache: Some(PathBuf::from("/c")),
            out: PathBuf::from("/o.jsonl"),
            keep: true,
        }
    );
    assert_eq!(parse(&["--cache", "none"], None).unwrap().cache, None);

    let bad: [&[&str]; 12] = [
        &["--file"],
        &["--files", "x"],
        &["file", "x"],
        &["--trials", "5", "--trials", "6"],
        &["--keep", "--keep"],
        &["--trials", "0"],
        &["--trials", "many"],
        &["--size-mib", "0"],
        &["--sizes", "16,x"],
        &["--sizes", "16,0"],
        &["--size-mib", "1", "--sizes", "128"],
        &["--busy-ms", "-1"],
    ];

    for args in bad {
        let found = parse(args, Some("/support")).unwrap_err().kind();
        assert_eq!(found, ErrorKind::InvalidInput, "{args:?}");
    }

    assert_eq!(parse(&[], None).unwrap_err().kind(), ErrorKind::InvalidInput);
    assert!(parse(&["--size-mib", "1", "--sizes", "64"], Some("/s")).is_ok());
}

#[test]
fn a_scratch_file_holds_random_bytes_of_its_size_and_is_never_written_over() {
    let dir = scratch("wavecal-make");
    let (a, b, c) = (dir.join("a.bin"), dir.join("b.bin"), dir.join("c.bin"));
    make(&a, (1 << 20) + 24, 3).unwrap();
    make(&b, (1 << 20) + 24, 3).unwrap();
    make(&c, (1 << 20) + 24, 4).unwrap();
    let bytes = std::fs::read(&a).unwrap();

    assert_eq!(bytes.len(), (1 << 20) + 24);
    assert_eq!(bytes, std::fs::read(&b).unwrap());
    assert_ne!(bytes, std::fs::read(&c).unwrap());
    assert!(bytes.chunks(4096).all(|page| page.iter().any(|&x| x != 0)));
    assert_eq!(make(&a, 100, 3).unwrap_err().kind(), ErrorKind::AlreadyExists);
    assert_eq!(std::fs::read(&a).unwrap(), bytes);
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(target_os = "macos")]
#[test]
fn waves_read_their_pages_through_the_cache_check_and_report_their_sizes() {
    let dir = scratch("wavecal-run");
    let file = dir.join("scratch.bin");
    make(&file, 256 * PAGE, 5).unwrap();
    let waves = plan(&[1, 4, 16], 2, 256, 3).unwrap();
    let timed = run(&file, &waves, std::time::Duration::ZERO).unwrap();

    // Another process may read a new file's pages back into the cache, eg. a security scanner, which leaves a wave out.
    assert_eq!(timed.len(), waves.len());
    assert!(
        timed.iter().filter(|w| w.is_some()).count() >= waves.len() - 1,
        "{timed:?}"
    );

    for (wave, pages) in timed.iter().zip(&waves) {
        let Some(wave) = wave else {
            continue;
        };
        assert_eq!(wave.pages, pages.len());
        assert_eq!(wave.ranges, ranges(pages).len());
        assert_eq!(wave.bytes, (pages.len() * PAGE) as u64);
        assert!(wave.ms > 0.0 && wave.ms.is_finite());
    }

    let past = vec![vec![255, 256]];
    assert_eq!(
        run(&file, &past, std::time::Duration::ZERO).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(target_os = "macos")]
#[test]
fn the_cache_check_sees_pages_that_a_read_left_cached_and_each_wave_evicts_them_first() {
    let _cold = cold();
    let dir = scratch("wavecal-cache");
    let file = dir.join("scratch.bin");
    make(&file, 64 * PAGE, 6).unwrap();
    let handle = std::fs::File::open(&file).unwrap();
    let map = Mapped::of(&handle).unwrap();
    let every: Vec<usize> = (0..64).collect();

    let _ = std::fs::read(&file).unwrap();
    assert!(cached(&map, &[0, 10, 63]).unwrap());
    map.evict().unwrap();
    assert!(!cached(&map, &every).unwrap());

    // A read of the whole file leaves every page cached, and the waves still find their pages cold.
    let _ = std::fs::read(&file).unwrap();
    assert!(cached(&map, &every).unwrap());
    let timed = run(&file, &plan(&[4], 3, 64, 1).unwrap(), std::time::Duration::ZERO).unwrap();
    assert!(timed.iter().filter(|w| w.is_some()).count() >= 2, "{timed:?}");
    let _ = std::fs::remove_dir_all(&dir);
}
