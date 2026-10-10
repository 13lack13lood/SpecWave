//! Unit tests for the training binary's command line, its guessing hook and its sums of cosines.

use specwave::predictor::Ridge;

use super::*;

fn parse(args: &[&str]) -> Result<Settings, Error> {
    Settings::parse(args.iter().map(|a| a.to_string()))
}

#[test]
fn a_model_gives_the_default_layers_and_every_other_setting_has_a_default() {
    let here = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let s = parse(&["--model", "arctic", "--texts", "t.jsonl"]).unwrap();
    let expected = Settings {
        repo: "Snowflake/snowflake-arctic-embed-m-v1.5".to_string(),
        texts: Some(PathBuf::from("t.jsonl")),
        layers: vec![0, 3, 6, 8, 9, 10],
        lambdas: vec![1e-3, 1e-2, 1e-1],
        split: 0.85,
        max_tokens: None,
        models: here.join("models"),
        out: here.join("results/train.jsonl"),
        guess: None,
    };
    assert_eq!(s, expected);
    assert_eq!(parse(&["--model", "arctic"]).unwrap().texts, None);

    let s = parse(&[
        "--model",
        "org/other",
        "--layers",
        "2, 5",
        "--lambdas",
        "0.5",
        "--split",
        "0.9",
        "--max-tokens",
        "128",
        "--models",
        "m",
        "--out",
        "o.jsonl",
        "--queries",
        "q.jsonl",
        "--guesses",
        "g",
        "--label",
        "rust.openings",
        "--compare",
        "openings",
        "--embeddings",
        "e.fbin",
    ])
    .unwrap();
    let expected = Settings {
        repo: "org/other".to_string(),
        texts: None,
        layers: vec![2, 5],
        lambdas: vec![0.5],
        split: 0.9,
        max_tokens: Some(128),
        models: PathBuf::from("m"),
        out: PathBuf::from("o.jsonl"),
        guess: Some(Guess {
            queries: PathBuf::from("q.jsonl"),
            dir: PathBuf::from("g"),
            label: "rust.openings".to_string(),
            compare: Some("openings".to_string()),
            embeddings: Some(PathBuf::from("e.fbin")),
        }),
    };
    assert_eq!(s, expected);
}

#[test]
fn bad_or_incomplete_command_lines_give_invalid_input() {
    let bad: [&[&str]; 18] = [
        &[],
        &["--texts", "t.jsonl"],
        &["--model", "org/other"],
        // A model without a short name needs its layers.
        &["--model", "minilm"],
        &["--model", "arctic", "--layer", "3"],
        &["--model", "arctic", "--model", "arctic"],
        &["--model", "arctic", "--layers"],
        &["--model", "arctic", "--layers", "3,,6"],
        &["--model", "arctic", "--layers", "-1"],
        &["--model", "arctic", "--lambdas", "0.1,0"],
        &["--model", "arctic", "--lambdas", "nan"],
        &["--model", "arctic", "--split", "1"],
        &["--model", "arctic", "--split", "0"],
        &["--model", "arctic", "--max-tokens", "many"],
        &["--model", "arctic", "--queries", "q", "--guesses", "g"],
        &["--model", "arctic", "--guesses", "g", "--label", "l"],
        &["--model", "arctic", "--compare", "openings"],
        &["model", "arctic"],
    ];

    for args in bad {
        let found = parse(args).err().map(|e| e.kind());
        assert_eq!(found, Some(ErrorKind::InvalidInput), "{args:?}");
    }

    let embeddings_alone = parse(&["--model", "arctic", "--embeddings", "e.fbin"])
        .err()
        .map(|e| e.kind());
    assert_eq!(embeddings_alone, Some(ErrorKind::InvalidInput));
}

/// Predictors of 2 dimensions at layers 1 and 3. Layer 1 guesses the summary's first half and layer 3 its second.
fn halves() -> Predictors {
    let first = Ridge::new(1, 2, vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![0.0; 2], 0.1).unwrap();
    let second = Ridge::new(3, 2, vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0], vec![0.0; 2], 0.01).unwrap();
    Predictors::new("m", vec![first, second]).unwrap()
}

#[test]
fn the_guesser_wants_the_layers_with_a_predictor_and_guesses_at_each() {
    let predictors = halves();
    let layers: Vec<usize> = predictors.layers().collect();
    let mut hook = Guesser::new(&predictors, &layers);
    assert_eq!(
        (0..5).map(|l| hook.wants(l)).collect::<Vec<_>>(),
        [false, true, false, true, false]
    );

    hook.on_layer(1, &[3.0, 4.0, 0.0, 9.0]);
    hook.on_layer(2, &[1.0, 1.0, 1.0, 1.0]);
    assert_eq!(hook.done, [true, false]);
    hook.on_layer(3, &[1.0, 1.0, 0.0, -2.0]);
    assert_eq!(hook.done, [true, true]);
    assert_eq!(hook.guesses, [vec![0.6, 0.8], vec![0.0, -1.0]]);
    assert!(hook.error.is_none());

    hook.on_layer(3, &[1.0, 1.0]);
    assert_eq!(hook.error.map(|e| e.kind()), Some(ErrorKind::InvalidInput));
}

#[test]
fn the_cosines_of_matching_rows_give_their_least_their_10th_percentile_and_their_mean() {
    // Rows at angles 0, 60 and 90 degrees from (1, 0), and 10 more at 0.
    let mut a = vec![1.0, 0.0, 0.5, 0.75f32.sqrt(), 0.0, 1.0];
    let mut b = vec![2.0, 0.0, 1.0, 0.0, 3.0, 0.0];
    a.extend([1.0, 0.0].repeat(10));
    b.extend([1.0, 0.0].repeat(10));
    let c = Cosines::of(&a, &b, 2);
    assert_eq!(c.min, 0.0);
    assert!((c.p10 - 0.5).abs() < 1e-7, "{c:?}");
    assert!((c.mean - 11.5 / 13.0).abs() < 1e-7, "{c:?}");

    let json = c.add(Json::new(), "cos").line();
    assert!(json.starts_with("{\"cos_min\":0,\"cos_p10\":0.5"), "{json}");
    assert!(json.contains("\"cos_mean\":"), "{json}");
}

#[test]
fn texts_come_from_the_text_field_of_each_line() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../target/specwave-bench-tests/train-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("texts.jsonl");
    std::fs::write(
        &path,
        "{\"row\": 0, \"text\": \"cheap index funds\"}\n{\"text\": \"how to save\", \"id\": \"7\"}\n",
    )
    .unwrap();
    assert_eq!(read_texts(&path).unwrap(), ["cheap index funds", "how to save"]);

    for (what, contents) in [
        ("no text", "{\"text\": \"a\"}\n{\"id\": \"7\"}\n"),
        ("not JSON", "{\"text\": \"a\"}\nhello\n"),
    ] {
        std::fs::write(&path, contents).unwrap();
        let error = read_texts(&path).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidData, "{what}");
        assert!(error.to_string().contains("line 2"), "{what}: {error}");
    }

    assert_eq!(
        read_texts(&dir.join("missing.jsonl")).unwrap_err().kind(),
        ErrorKind::NotFound
    );
    std::fs::remove_dir_all(&dir).unwrap();
}
