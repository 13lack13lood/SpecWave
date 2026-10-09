//! Unit tests for the result files.

use std::path::PathBuf;

use super::*;

#[test]
fn a_line_holds_its_fields_in_order_as_json() {
    let line = Json::new()
        .str("a", "x\"y\\z\n\u{1}")
        .num("b", 0.5)
        .num("c", f64::NAN)
        .int("d", 7)
        .opt("e", None)
        .opt("f", Some(3))
        .bool("g", true)
        .nums("h", &[1.0, f64::INFINITY, -0.25])
        .ints("i", &[1, 2])
        .ints("j", &[])
        .line();
    let expected =
        r#"{"a":"x\"y\\z\n\u0001","b":0.5,"c":null,"d":7,"e":null,"f":3,"g":true,"h":[1,null,-0.25],"i":[1,2],"j":[]}"#;
    assert_eq!(line, expected);
    assert_eq!(Json::new().line(), "{}");
}

#[test]
fn append_adds_lines_at_the_end_and_makes_the_folder() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target/specwave-bench-tests")
        .join(format!("results-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let path = dir.join("nested/out.jsonl");

    append(&path, &["a".to_string()]).unwrap();
    append(&path, &["b".to_string(), "c".to_string()]).unwrap();
    append(&path, &[]).unwrap();
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "a\nb\nc\n");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn provenance_names_the_build_of_both_repositories_and_the_machine() {
    let line: serde_json::Value = serde_json::from_str(&provenance(Json::new()).line()).unwrap();
    let field = |key: &str| {
        line[key]
            .as_str()
            .unwrap_or_else(|| panic!("{key} in {line}"))
            .to_string()
    };
    let label = |commit: String, changes: String| match changes.is_empty() {
        true => commit,
        false => format!("{commit}+{changes}"),
    };
    let (commit, changes) = (field("commit"), field("changes"));
    let (coldivf_commit, coldivf_changes) = (field("coldivf_commit"), field("coldivf_changes"));

    assert_eq!(field("build"), BUILD);
    assert_eq!(line["dirty"], !changes.is_empty());
    assert_eq!(line["coldivf_dirty"], !coldivf_changes.is_empty());
    assert!(!field("machine").is_empty());

    // This checkout and the ColdIVF one next to it are both git repositories.
    for hash in [&commit, &coldivf_commit] {
        assert!(
            hash.len() == 12 && hash.bytes().all(|b| b.is_ascii_hexdigit()),
            "{hash}"
        );
    }

    assert!(names_its_code(BUILD), "{BUILD}");
    assert_eq!(
        BUILD,
        format!(
            "specwave {} coldivf {}",
            label(commit, changes),
            label(coldivf_commit, coldivf_changes)
        )
    );
}

#[test]
fn each_repository_is_dirty_with_its_own_changes_alone() {
    let clean = Repo {
        commit: "1d2a8be0c3f1",
        changes: "",
    };
    let changed = Repo {
        commit: "d90edb1c2d3e",
        changes: "0f3a5c7e9b1d2f40",
    };
    let line = |specwave: &Repo, coldivf: &Repo| -> serde_json::Value {
        serde_json::from_str(&code(Json::new(), "b", specwave, coldivf).line()).unwrap()
    };

    let v = line(&clean, &changed);
    let expected = serde_json::json!({
        "build": "b", "commit": "1d2a8be0c3f1", "dirty": false, "changes": "", "coldivf_commit": "d90edb1c2d3e",
        "coldivf_dirty": true, "coldivf_changes": "0f3a5c7e9b1d2f40"
    });
    assert_eq!(v, expected);
    let v = line(&changed, &clean);
    assert_eq!(
        (v["dirty"].as_bool(), v["coldivf_dirty"].as_bool()),
        (Some(true), Some(false))
    );
    assert_eq!(
        (v["changes"].as_str(), v["coldivf_changes"].as_str()),
        (Some("0f3a5c7e9b1d2f40"), Some(""))
    );
}

#[test]
fn a_build_names_its_code_unless_a_repository_is_unknown() {
    assert!(names_its_code(
        "specwave 1d2a8be0c3f1+0f3a5c7e9b1d2f40 coldivf d90edb1c2d3e"
    ));
    assert!(!names_its_code("specwave unknown coldivf d90edb1c2d3e"));
    assert!(!names_its_code("specwave 1d2a8be0c3f1 coldivf unknown"));
}
