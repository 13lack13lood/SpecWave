//! Unit tests for the models list.

use std::io::ErrorKind;

use super::*;

#[test]
fn the_list_parses_and_names_each_model_and_repository_once() {
    let models = registered().unwrap();
    let names: Vec<&str> = models.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "arctic",
            "minilm",
            "bge-small",
            "bge-base",
            "e5-base",
            "bge-large",
            "mxbai-large",
            "nomic",
            "qwen3-0.6b"
        ]
    );

    for model in models {
        assert_eq!(find(&model.name).unwrap(), Some(model));
        assert_eq!(find(&model.repo).unwrap(), Some(model));
        assert_eq!(repo(&model.name).unwrap(), model.repo);
        assert!(model.layers.windows(2).all(|w| w[0] < w[1]), "{}", model.name);
    }
}

#[test]
fn arctic_keeps_its_layers_and_takes_its_prompts_from_its_snapshot() {
    let arctic = find("arctic").unwrap().unwrap();
    assert_eq!(arctic.repo, "Snowflake/snowflake-arctic-embed-m-v1.5");
    assert_eq!(arctic.layers, [0, 3, 6, 8, 9, 10]);
    assert_eq!(
        (&arctic.query_prompt, &arctic.passage_prompt, &arctic.prompts_from),
        (&None, &None, &None)
    );
}

#[test]
fn listed_prompts_come_with_the_file_that_gives_them() {
    let e5 = find("intfloat/e5-base-v2").unwrap().unwrap();
    assert_eq!(e5.query_prompt.as_deref(), Some("query: "));
    assert_eq!(e5.passage_prompt.as_deref(), Some("passage: "));
    assert_eq!(e5.prompts_from.as_deref(), Some("README.md"));

    for name in ["bge-small", "bge-base", "bge-large"] {
        let bge = find(name).unwrap().unwrap();
        assert_eq!(
            bge.query_prompt.as_deref(),
            Some("Represent this sentence for searching relevant passages: ")
        );
        assert_eq!(bge.passage_prompt.as_deref(), Some(""));
    }

    // mxbai-embed-large-v1's own settings hold its prompts, as Qwen3-Embedding-0.6B's do.
    assert_eq!(find("mxbai-large").unwrap().unwrap().prompts_from, None);
    assert_eq!(find("qwen3-0.6b").unwrap().unwrap().prompts_from, None);

    let nomic = find("nomic-ai/nomic-embed-text-v1.5").unwrap().unwrap();
    assert_eq!(
        (nomic.query_prompt.as_deref(), nomic.passage_prompt.as_deref()),
        (Some("search_query: "), Some("search_document: "))
    );
    assert_eq!(nomic.prompts_from.as_deref(), Some("README.md"));
}

#[test]
fn nomic_and_qwen3_keep_their_layers_that_may_fire() {
    assert_eq!(find("nomic").unwrap().unwrap().layers, [0, 3, 6, 7, 8, 9, 10, 11]);
    assert_eq!(
        find("Qwen/Qwen3-Embedding-0.6B").unwrap().unwrap().layers,
        [0, 7, 14, 19, 21, 23, 24, 25, 26, 27]
    );
}

#[test]
fn a_name_not_in_the_list_stays_as_it_is() {
    assert_eq!(find("org/other").unwrap(), None);
    assert_eq!(repo("org/other").unwrap(), "org/other");
    assert_eq!(repo("").unwrap(), "");
}

#[test]
fn damaged_lists_give_invalid_data() {
    let entry = r#"{"name": "a", "repo": "org/a", "layers": [0, 2]}"#;
    let cases = [
        "{}".to_string(),
        "[3]".to_string(),
        "not json".to_string(),
        format!("[{entry}, {entry}]"),
        r#"[{"name": "a", "repo": "org/a", "layers": [0, 2], "extra": 1}]"#.to_string(),
        r#"[{"repo": "org/a", "layers": [0]}]"#.to_string(),
        r#"[{"name": "", "repo": "org/a", "layers": [0]}]"#.to_string(),
        r#"[{"name": "a", "repo": "a", "layers": [0]}]"#.to_string(),
        r#"[{"name": "a", "repo": "org--x/a", "layers": [0]}]"#.to_string(),
        r#"[{"name": "a", "repo": "org/a", "layers": []}]"#.to_string(),
        r#"[{"name": "a", "repo": "org/a", "layers": [2, 2]}]"#.to_string(),
        r#"[{"name": "a", "repo": "org/a", "layers": [3, 1]}]"#.to_string(),
        r#"[{"name": "a", "repo": "org/a", "layers": [-1]}]"#.to_string(),
        r#"[{"name": "a", "repo": "org/a", "layers": [0], "query_prompt": "q: "}]"#.to_string(),
        r#"[{"name": "a", "repo": "org/a", "layers": [0], "prompts_from": "README.md"}]"#.to_string(),
        r#"[{"name": "a", "repo": "org/a", "layers": [0], "query_prompt": 3, "prompts_from": "README.md"}]"#
            .to_string(),
        format!(r#"[{entry}, {{"name": "b", "repo": "org/a", "layers": [0]}}]"#),
    ];

    for case in &cases {
        let found = parse(case).err().map(|e| e.kind());
        assert_eq!(found, Some(ErrorKind::InvalidData), "{case}");
    }

    let fine = format!(
        r#"[{entry}, {{"name": "b", "repo": "org/b", "layers": [1], "passage_prompt": "p: ", "prompts_from": "x"}}]"#
    );
    let models = parse(&fine).unwrap();
    assert_eq!(models.len(), 2);
    assert_eq!(models[1].query_prompt, None);
    assert_eq!(models[1].passage_prompt.as_deref(), Some("p: "));
}
