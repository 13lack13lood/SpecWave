//! Unit tests for the accuracy rule over coverage results.

use serde_json::json;

use super::*;

/// The values of one cell, i.e. the beta asked for and the one read, the multiple, and the gap and both rerank
/// coverages, each as (mean, half the interval's width).
struct Values {
    beta: f64,
    read: f64,
    mult: u64,
    gap: (f64, f64),
    rr: (f64, f64),
    rr_normal: (f64, f64),
}

fn values(beta: f64, mult: u64, gap: f64, rr: f64, rr_normal: f64) -> Values {
    Values {
        beta,
        read: beta,
        mult,
        gap: (gap, 0.001),
        rr: (rr, 0.005),
        rr_normal: (rr_normal, 0.005),
    }
}

/// A grid line of `spec_coverage` for the guess `guess` of fiqa's `sph` clustering at nprobe 6 and R 29.
fn line(guess: &str, seed: u64, v: &Values) -> String {
    let layer = guess.rsplit_once(".L").and_then(|(_, l)| l.parse::<u64>().ok());
    let mut line = json!({
        "record": "grid", "build": "b1", "dataset": "fiqa", "clustering": "sph", "seed": seed, "guess": guess,
        "layer": layer, "queries": 1548, "nprobe": 6, "beta": v.beta, "beta_effective": v.read, "r_label": "29",
        "rerank": 29, "mult": v.mult,
    });
    for (name, (mean, half)) in [
        ("recall_gap", v.gap),
        ("rerank_coverage", v.rr),
        ("rerank_coverage_normal", v.rr_normal),
    ] {
        line[name] = mean.into();
        line[format!("{name}_lo")] = (mean - half).into();
        line[format!("{name}_hi")] = (mean + half).into();
    }
    line.to_string()
}

fn cells_of(lines: &[String]) -> Vec<Cell> {
    read(lines).unwrap()
}

#[test]
fn cells_join_their_seeds_and_a_later_line_replaces_an_earlier_one() {
    let lines = vec![
        "{\"record\":\"run\"}".to_string(),
        "not json".to_string(),
        line("openings.L6", 7, &values(0.25, 2, -0.004, 0.91, 0.88)),
        line("openings.L6", 11, &values(0.25, 2, -0.002, 0.93, 0.86)),
        line("openings.L6", 23, &values(0.25, 2, 0.5, 0.5, 0.5)),
        line("openings.L6", 23, &values(0.25, 2, 0.0, 0.92, 0.87)),
    ];
    let cells = cells_of(&lines);
    assert_eq!(cells.len(), 1);
    let c = &cells[0];
    assert_eq!(
        (c.seeds.clone(), c.builds.clone()),
        (vec![7, 11, 23], vec!["b1".to_string()])
    );
    assert_eq!(
        (c.dataset.as_str(), c.guess.as_str(), c.layer, c.nprobe),
        ("fiqa", "openings.L6", Some(6), 6)
    );
    let close = |a: f64, b: f64| (a - b).abs() < 1e-12;
    assert!(
        close(c.gap.mean, -0.002) && close(c.gap.lo, -0.005) && close(c.gap.hi, 0.001),
        "{c:?}"
    );
    assert!(close(c.coverage[0].mean, 0.92) && close(c.coverage[0].lo, 0.905) && close(c.coverage[0].hi, 0.935));
    assert!(close(c.coverage[1].mean, 0.87) && close(c.coverage[1].lo, 0.855) && close(c.coverage[1].hi, 0.885));
    assert_eq!((c.rerank, c.queries), (29, 1548));

    let mut old: serde_json::Value = serde_json::from_str(&lines[2]).unwrap();
    old.as_object_mut().unwrap().remove("rerank_coverage_normal_lo");
    let found = read(&[old.to_string()]).err().map(|e| e.kind());
    assert_eq!(found, Some(ErrorKind::InvalidData));
}

#[test]
fn lines_over_other_queries_or_at_another_r_for_the_same_label_are_refused() {
    let first = line("openings.L6", 7, &values(0.25, 2, -0.004, 0.91, 0.88));
    // A line of `guess` with `seed`, with the fields of `changes` set to new values.
    let changed = |guess: &str, seed: u64, changes: &[(&str, serde_json::Value)]| {
        let mut v: serde_json::Value =
            serde_json::from_str(&line(guess, seed, &values(0.5, 4, 0.0, 0.9, 0.9))).unwrap();
        changes.iter().for_each(|(field, value)| v[*field] = value.clone());
        v.to_string()
    };
    let kind = |second: String| read(&[first.clone(), second]).err().map(|e| e.kind());

    // A run on 50 of the queries, or one at R 31 under the label of R 29, for any guess, seed and cell.
    for (guess, seed) in [("openings.L6", 7), ("openings.L6", 11), ("queries.L8", 23)] {
        for change in [("queries", 50.into()), ("rerank", 31.into())] {
            assert_eq!(
                kind(changed(guess, seed, std::slice::from_ref(&change))),
                Some(ErrorKind::InvalidData),
                "{guess} {seed} {change:?}"
            );
        }
    }

    // Another dataset has queries of its own, and another clustering, nprobe or R label an R of its own.
    let others: [&[(&str, serde_json::Value)]; 4] = [
        &[("dataset", "quora".into()), ("queries", 1000.into())],
        &[("clustering", "l2".into()), ("rerank", 31.into())],
        &[("nprobe", 9.into()), ("rerank", 31.into())],
        &[("r_label", "r384".into()), ("rerank", 31.into())],
    ];

    for change in others {
        assert_eq!(kind(changed("openings.L6", 7, change)), None, "{change:?}");
    }
}

#[test]
fn a_test_passes_when_its_whole_interval_reaches_the_threshold() {
    let i = |lo: f64, hi: f64| Interval {
        mean: (lo + hi) / 2.0,
        lo,
        hi,
    };
    assert_eq!(test(&i(0.9, 0.95), 0.9), Verdict::Pass);
    assert_eq!(
        test(&i(0.9 - 1e-12, 0.95), 0.9),
        Verdict::Pass,
        "within the rounding slack"
    );
    assert_eq!(test(&i(0.85, 0.9), 0.9), Verdict::Gray);
    assert_eq!(test(&i(0.85, 0.899), 0.9), Verdict::Fail);
    assert_eq!(test(&i(-0.0051, -0.0049), -0.005), Verdict::Gray);
    assert_eq!(test(&i(-0.0049, 0.01), -0.005), Verdict::Pass);
    assert_eq!(test(&i(-0.02, -0.0051), -0.005), Verdict::Fail);
}

#[test]
fn a_cell_passes_when_both_tests_pass_under_each_definition_alone() {
    let rule = Rule::default();
    let judge = |v: Values| {
        let c = &cells_of(&[line("openings.L6", 7, &v)])[0];
        [rule.judge(c, 0), rule.judge(c, 1)]
    };
    assert_eq!(judge(values(0.25, 2, 0.0, 0.95, 0.95)), [Verdict::Pass; 2]);
    assert_eq!(judge(values(0.25, 2, -0.0045, 0.95, 0.95)), [Verdict::Gray; 2]);
    assert_eq!(judge(values(0.25, 2, -0.01, 0.95, 0.95)), [Verdict::Fail; 2]);
    assert_eq!(judge(values(0.25, 2, 0.0, 0.95, 0.85)), [Verdict::Pass, Verdict::Fail]);
    assert_eq!(judge(values(0.25, 2, 0.0, 0.902, 0.95)), [Verdict::Gray, Verdict::Pass]);
    assert_eq!(
        judge(values(0.25, 2, -0.01, 0.902, 0.95)),
        [Verdict::Fail, Verdict::Fail]
    );
}

#[test]
fn a_group_passes_at_its_smallest_beta_then_multiple_within_the_limits() {
    let rule = Rule::default();
    let lines = [
        line("openings.L6", 7, &values(0.0, 2, -0.02, 0.99, 0.99)),
        line("openings.L6", 7, &values(0.25, 2, 0.0, 0.85, 0.85)),
        line("openings.L6", 7, &values(0.25, 3, 0.0, 0.95, 0.85)),
        line("openings.L6", 7, &values(0.25, 4, 0.0, 0.97, 0.902)),
        line("openings.L6", 7, &values(0.5, 2, 0.01, 0.96, 0.95)),
        line("openings.L6", 7, &values(1.0, 2, 0.02, 0.99, 0.99)),
        line("openings.L6", 7, &values(0.25, 5, 0.0, 0.99, 0.99)),
    ];
    let cells = cells_of(&lines);
    let group: Vec<&Cell> = cells.iter().collect();
    let chosen = |rule: &Rule, d: usize| {
        let (verdict, c) = rule.decide(&group, d);
        (verdict, c.map(|c| (c.beta, c.mult)))
    };
    assert_eq!(chosen(&rule, 0), (Verdict::Pass, Some((0.25, 3))));
    assert_eq!(chosen(&rule, 1), (Verdict::Pass, Some((0.5, 2))));

    // With beta at most 0.25 the normal-search coverage is gray at best, and with multiples up to 5 it passes.
    let narrow = Rule {
        max_beta_otherwise: 0.25,
        ..Rule::default()
    };
    assert_eq!(chosen(&narrow, 1), (Verdict::Gray, Some((0.25, 4))));
    let five = Rule {
        max_mult: 5,
        ..narrow.clone()
    };
    assert_eq!(chosen(&five, 1), (Verdict::Pass, Some((0.25, 5))));

    // Without beta 0.25 to 0.5 nothing passes recall, so the best gap decides. A dataset's own limit overrides the
    // others'.
    let none = Rule {
        max_beta: HashMap::from([("fiqa".to_string(), 0.1)]),
        ..Rule::default()
    };
    assert_eq!(chosen(&none, 0), (Verdict::Fail, Some((0.0, 2))));

    // The limit holds the beta the guess reads, not the one asked for.
    let mut read_more = values(0.25, 3, 0.0, 0.95, 0.95);
    read_more.read = 0.333;
    let mut read_less = values(0.5, 2, 0.0, 0.95, 0.95);
    read_less.read = 0.444;
    let cells = cells_of(&[line("queries.L6", 7, &read_more), line("queries.L6", 7, &read_less)]);
    let group: Vec<&Cell> = cells.iter().collect();
    let limited = Rule {
        max_beta_otherwise: 0.45,
        ..Rule::default()
    };
    assert_eq!(limited.decide(&group, 0).1.map(|c| c.beta), Some(0.25));
    let tighter = Rule {
        max_beta_otherwise: 0.3,
        ..Rule::default()
    };
    assert_eq!(tighter.decide(&group, 0), (Verdict::Fail, None));

    // When recall passes but no multiple covers enough rows, the largest multiple at that beta decides.
    let cells = cells_of(&[
        line("queries.L3", 7, &values(0.25, 2, 0.0, 0.7, 0.7)),
        line("queries.L3", 7, &values(0.25, 4, 0.0, 0.8, 0.8)),
        line("queries.L3", 7, &values(0.5, 4, 0.0, 0.79, 0.79)),
    ]);
    let group: Vec<&Cell> = cells.iter().collect();
    assert_eq!(rule.decide(&group, 0).1.map(|c| (c.beta, c.mult)), Some((0.25, 4)));
    assert_eq!(rule.smallest_mult(&group, 0.25, 0), (None, None));
}

#[test]
fn the_smallest_multiple_counts_by_mean_and_by_interval() {
    let rule = Rule::default();
    let cells = cells_of(&[
        line("openings.L5", 7, &values(0.25, 2, 0.0, 0.85, 0.80)),
        line("openings.L5", 7, &values(0.25, 4, 0.0, 0.902, 0.88)),
        line("openings.L5", 7, &values(0.25, 6, 0.0, 0.93, 0.901)),
        line("openings.L5", 7, &values(0.25, 8, 0.0, 0.95, 0.92)),
    ]);
    let group: Vec<&Cell> = cells.iter().collect();
    assert_eq!(rule.smallest_mult(&group, 0.25, 0), (Some(4), Some(6)));
    assert_eq!(rule.smallest_mult(&group, 0.25, 1), (Some(6), Some(8)));
    assert_eq!(rule.smallest_mult(&group, 0.5, 0), (None, None));

    // The beta comes from the recall test, which holds at any multiple, eg. in a file of multiples above 4 alone.
    let cells = cells_of(&[
        line("openings.L5", 7, &values(0.0, 5, -0.01, 0.95, 0.95)),
        line("openings.L5", 7, &values(0.25, 5, -0.0045, 0.95, 0.95)),
        line("openings.L5", 7, &values(0.5, 6, 0.0, 0.95, 0.95)),
        line("openings.L5", 7, &values(1.0, 6, 0.01, 0.95, 0.95)),
    ]);
    let group: Vec<&Cell> = cells.iter().collect();
    assert_eq!(rule.recall_beta(&group), Some(0.5));
    let gray = cells_of(&[line("openings.L5", 7, &values(0.25, 5, -0.0045, 0.95, 0.95))]);
    assert_eq!(rule.recall_beta(&gray.iter().collect::<Vec<_>>()), Some(0.25));
    let wide = Rule {
        max_beta_otherwise: 1.0,
        ..Rule::default()
    };
    assert_eq!(wide.recall_beta(&group), Some(0.5));
    let narrow = Rule {
        max_beta_otherwise: 0.25,
        ..Rule::default()
    };
    assert_eq!(
        narrow.recall_beta(&group),
        Some(0.25),
        "gray at 0.25 when 0.5 isn't allowed"
    );
}

#[test]
fn the_report_shows_each_definition_the_first_passing_layer_and_the_smallest_multiple() {
    let mut lines = Vec::new();

    for seed in [7, 11] {
        lines.push(line("openings.L3", seed, &values(0.25, 2, -0.02, 0.8, 0.7)));
        lines.push(line("openings.L6", seed, &values(0.25, 2, 0.0, 0.902, 0.85)));
        lines.push(line("openings.L6", seed, &values(0.25, 4, 0.0, 0.903, 0.89)));
        lines.push(line("openings.L8", seed, &values(0.25, 2, 0.0, 0.902, 0.89)));
        lines.push(line("openings.L8", seed, &values(0.25, 4, 0.0, 0.95, 0.903)));
        lines.push(line("openings.L9", seed, &values(0.25, 2, 0.01, 0.97, 0.95)));
    }

    let text = report(&cells_of(&lines), &Rule::default());

    for part in [
        "Seeds: 7, 11. Builds: b1.",
        "## fiqa, R 29, rerank_coverage\n",
        "## fiqa, R 29, rerank_coverage_normal\n",
        "| Guess | sph 6 |",
        "| openings.L3 | FAIL gap -0.0200 [-0.0210, -0.0190], rr 0.800 [0.795, 0.805] β0.25 m2 |",
        "| openings.L3 | FAIL gap -0.0200 [-0.0210, -0.0190], rr 0.700 [0.695, 0.705] β0.25 m2 |",
        "| openings.L6 | GRAY rr 0.902 [0.897, 0.907] β0.25 m2 |",
        "| openings.L8 | PASS β0.25 m4 (GRAY at β0.25 m2) |",
        "| openings.L6 | FAIL rr 0.890 [0.885, 0.895] β0.25 m4 |",
        "| openings.L8 | GRAY rr 0.903 [0.898, 0.908] β0.25 m4 |",
        "| openings.L9 | PASS β0.25 m2 |",
        "| fiqa | openings | sph | 29 | 6 | L8 (L6) | L9 (L8) |",
        "| fiqa | 29 | openings.L6 | sph | 6 | 0.25 | m2 / none | none / none |",
        "| fiqa | 29 | openings.L8 | sph | 6 | 0.25 | m2 / m4 | m4 / none |",
        "| fiqa | 29 | openings.L3 | sph | 6 | none | | |",
    ] {
        assert!(text.contains(part), "{part:?} in\n{text}");
    }
}

#[test]
fn a_fit_is_the_label_without_its_layer() {
    assert_eq!(fit_of("openings.L9"), "openings");
    assert_eq!(fit_of("queries.L10"), "queries");
    assert_eq!(fit_of("minilm.openings"), "minilm.openings");
    assert_eq!(fit_of("a.L3.v2"), "a.v2");
    assert_eq!(fit_of("L"), "L");
}

#[test]
fn settings_name_the_files_and_the_rule() {
    let parse = |args: &[&str]| Settings::parse(&args.iter().map(|a| a.to_string()).collect::<Vec<_>>());
    let s = parse(&["a.jsonl", "b.jsonl"]).unwrap();
    assert_eq!(
        (s.files, s.rule, s.out),
        (
            vec!["a.jsonl".to_string(), "b.jsonl".to_string()],
            Rule::default(),
            None
        )
    );

    let s = parse(&[
        "a",
        "--max-beta",
        "quora=1,fiqa=0.5,0.25",
        "--max-mult",
        "16",
        "--min-gap",
        "-0.01",
        "--coverage",
        "0.95",
        "--out",
        "v.md",
    ])
    .unwrap();
    let rule = Rule {
        min_gap: -0.01,
        min_coverage: 0.95,
        max_mult: 16,
        max_beta: HashMap::from([("quora".to_string(), 1.0), ("fiqa".to_string(), 0.5)]),
        max_beta_otherwise: 0.25,
    };
    assert_eq!((s.rule, s.out), (rule, Some("v.md".to_string())));

    let bad: [&[&str]; 8] = [
        &[],
        &["a", "--max-beta", "x"],
        &["a", "--max-beta", "fiqa=-1"],
        &["a", "--max-mult", "-1"],
        &["a", "--coverage", "nan"],
        &["a", "--min-gap"],
        &["a", "--min-gap", "0", "--min-gap", "0"],
        &["a", "--beta", "0.5"],
    ];

    for args in bad {
        assert_eq!(
            parse(args).err().map(|e| e.kind()),
            Some(ErrorKind::InvalidInput),
            "{args:?}"
        );
    }
}
