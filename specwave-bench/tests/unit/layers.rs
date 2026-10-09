//! Unit tests for the layer timing's command line and its summary of a run.

use std::path::PathBuf;

use super::*;

fn parse(args: &[&str]) -> Result<Args, Error> {
    Args::parse(args.iter().map(|a| a.to_string()))
}

#[test]
fn only_known_names_given_once_with_good_values_parse() {
    assert!(parse(&[]).is_ok());
    let args = parse(&["--model", "minilm", "--runs", "20", "--device", "cpu", "--threads", "4"]).unwrap();
    assert_eq!(args.str("model"), Some("minilm"));
    assert_eq!(args.get::<usize>("runs", 50).unwrap(), 20);
    assert_eq!(args.get::<usize>("warmup", 20).unwrap(), 20);
    assert!(parse(&["--device", "metal", "--threads", "6", "--run-id", "r1"]).is_ok());

    let bad: [&[&str]; 11] = [
        &["--run", "20"],
        &["--runs", "0"],
        &["--runs", "20", "--runs", "30"],
        &["--runs", "many"],
        &["--warmup", "-1"],
        &["--device", "gpu", "--threads", "4"],
        &["--device", "cpu", "--threads", "0"],
        &["--threads", "4"],
        &["--runs"],
        &["runs", "20"],
        &["--model", "arctic", "--model", "minilm"],
    ];

    for args in bad {
        let found = parse(args).err().map(|e| e.kind());
        assert_eq!(found, Some(ErrorKind::InvalidInput), "{args:?}");
    }
}

#[test]
fn a_missing_results_file_has_an_empty_summary_and_a_run_shows_each_setup() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target/specwave-bench-tests")
        .join(format!("layers-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let out = dir.join("layers.jsonl");
    assert_eq!(summary(&out, "r1").unwrap(), "");

    let line = |run: &str, device: &str, layer: u64, t: f64| {
        Json::new()
            .str("bench", "layers")
            .str("run", run)
            .str("model", "m")
            .str("device", device)
            .int("threads", 5)
            .int("layer", layer)
            .num("t_ms", t)
            .num("headroom_ms", 2.0 - t)
            .line()
    };
    let lines = [
        line("r1", "cpu", 0, 0.5),
        line("r1", "cpu", 1, 2.0),
        line("r1", "metal", 0, 0.25),
        line("r0", "cpu", 0, 9.0),
    ];
    append(&out, &lines).unwrap();
    let text = summary(&out, "r1").unwrap();
    assert!(text.contains("m: T_l and T_L-T_l in ms, medians"), "{text}");
    assert!(text.contains("cpu, 5 threads") && text.contains("metal"), "{text}");
    assert!(
        text.contains("0.500     1.500") && text.contains("0.250     1.750"),
        "{text}"
    );
    assert!(!text.contains("9.000"), "{text}");
    std::fs::remove_dir_all(&dir).unwrap();
}
