//! Unit tests for the vector files and their names.

use super::*;

/// A fresh folder in the build folder for one test's files.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../target/specwave-bench-tests")
        .join(format!("data-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// The bytes of a table file whose header says `count` rows of `width`, followed by `values`.
fn table(count: i32, width: i32, values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend(count.to_le_bytes());
    bytes.extend(width.to_le_bytes());
    values.iter().for_each(|v| bytes.extend(v.to_le_bytes()));
    bytes
}

#[test]
fn vectors_and_ids_read_back_by_row() {
    let dir = scratch("rows");
    let floats: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();
    let bits: Vec<u32> = floats.iter().map(|f| f.to_bits()).collect();
    fs::write(dir.join("v.fbin"), table(4, 3, &bits)).unwrap();
    fs::write(dir.join("i.ibin"), table(2, 2, &[4, 5, 6, 7])).unwrap();
    fs::write(dir.join("empty.fbin"), table(0, 5, &[])).unwrap();

    let vectors = Fbin::open(&dir.join("v.fbin")).unwrap();
    assert_eq!((vectors.len(), vectors.dim(), vectors.is_empty()), (4, 3, false));
    assert_eq!(vectors.row(3), &[4.5, 5.0, 5.5]);
    assert_eq!(vectors.values(), &floats[..]);
    assert_eq!((vectors.count(), VectorSource::dim(&vectors)), (4, 3));
    assert_eq!(vectors.rows(1, 2), &floats[3..9]);
    let ids = Ibin::open(&dir.join("i.ibin")).unwrap();
    assert_eq!((ids.len(), ids.width(), ids.row(1)), (2, 2, &[6, 7][..]));
    assert_eq!(ids.values(), &[4, 5, 6, 7]);
    let empty = Fbin::open(&dir.join("empty.fbin")).unwrap();
    assert!(empty.is_empty() && empty.values().is_empty());
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_damaged_file_fails_with_invalid_data_and_a_missing_one_with_not_found() {
    let dir = scratch("damaged");
    let path = dir.join("bad.fbin");
    let cases = [
        ("empty", Vec::new()),
        ("shorter than its header", table(1, 1, &[])[..6].to_vec()),
        ("a negative count", table(-1, 2, &[])),
        ("a zero width", table(0, 0, &[])),
        ("too few values", table(2, 3, &[1, 2, 3, 4, 5])),
        ("too many values", table(1, 2, &[1, 2, 3])),
        ("a count whose size overflows", table(i32::MAX, i32::MAX, &[1])),
    ];

    for (what, bytes) in cases {
        fs::write(&path, bytes).unwrap();
        assert_eq!(
            Fbin::open(&path).err().map(|e| e.kind()),
            Some(ErrorKind::InvalidData),
            "{what}"
        );
        assert_eq!(
            Ibin::open(&path).err().map(|e| e.kind()),
            Some(ErrorKind::InvalidData),
            "{what}"
        );
    }

    let missing = Fbin::open(&dir.join("missing.fbin")).err().unwrap();
    assert_eq!(missing.kind(), ErrorKind::NotFound);
    assert!(missing.to_string().contains("missing.fbin"));
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn guess_files_are_found_by_name_and_ordered_by_layer() {
    let dir = scratch("guesses");
    let files = Files::new(&dir, "fiqa");
    assert!(files.guesses().unwrap().is_empty(), "no guess folder yet");

    let ehat = dir.join("spec/fiqa/ehat");
    fs::create_dir_all(&ehat).unwrap();
    fs::create_dir_all(dir.join("spec/quora/ehat")).unwrap();
    let names = [
        "openings.L9.fbin",
        "queries.L10.fbin",
        "minilm.openings.fbin",
        "openings.L0.fbin",
        "queries.L9.fbin",
        "openings.L9.json",
        "openings8000.json",
        ".fbin",
        "openings.L3.fbin.part",
    ];

    for name in names {
        fs::write(ehat.join(name), b"").unwrap();
    }

    fs::write(dir.join("spec/quora/ehat/openings.L3.fbin"), b"").unwrap();
    let found: Vec<(String, Option<usize>)> = files
        .guesses()
        .unwrap()
        .into_iter()
        .map(|g| (g.label, g.layer))
        .collect();
    let expected = [
        ("openings.L0", Some(0)),
        ("openings.L9", Some(9)),
        ("queries.L9", Some(9)),
        ("queries.L10", Some(10)),
        ("minilm.openings", None),
    ];
    assert_eq!(found, expected.map(|(l, n)| (l.to_string(), n)));
    assert_eq!(files.guesses().unwrap()[0].path, ehat.join("openings.L0.fbin"));
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn every_file_of_a_dataset_has_its_name() {
    let files = Files::new(Path::new("/d"), "quora");
    assert_eq!(files.corpus(), Path::new("/d/quora_corpus.fbin"));
    assert_eq!(files.queries(), Path::new("/d/quora_test.fbin"));
    assert_eq!(files.truth(), Path::new("/d/quora_test.gt.ibin"));
    assert_eq!(files.targets(), Path::new("/d/spec/quora/e.fbin"));
    assert_eq!(
        files.centroids("l2"),
        Path::new("/d/spec/quora/clusters/l2.centroids.fbin")
    );
    assert_eq!(
        files.assignment("sph"),
        Path::new("/d/spec/quora/clusters/sph.assign.ibin")
    );
    assert_eq!(files.index("l2"), Path::new("/d/coverage/quora_l2.coldivf"));
}

#[test]
fn cos_reads_the_l2_clusters_with_unit_centers_and_sph_scales_its_own() {
    assert_eq!(clusters_of("l2"), ("l2", false));
    assert_eq!(clusters_of("sph"), ("sph", true));
    assert_eq!(clusters_of("cos"), ("l2", true));
    assert_eq!(clusters_of("l2-ip"), ("l2-ip", false));
    assert_eq!(clusters_of("other"), ("other", false));

    let files = Files::new(Path::new("/d"), "fiqa");
    assert_eq!(
        files.centroids("cos"),
        Path::new("/d/spec/fiqa/clusters/l2.centroids.fbin")
    );
    assert_eq!(
        files.assignment("cos"),
        Path::new("/d/spec/fiqa/clusters/l2.assign.ibin")
    );
    assert_eq!(files.index("cos"), Path::new("/d/coverage/fiqa_cos.coldivf"));
}

/// Writes `values` as a table of `width` per row at `path`.
fn write_table<T: Copy>(path: &Path, width: usize, values: &[T], bits: impl Fn(T) -> u32) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let words: Vec<u32> = values.iter().map(|&v| bits(v)).collect();
    fs::write(path, table((values.len() / width) as i32, width as i32, &words)).unwrap();
}

/// A dataset of 6 documents of 3 values in `dir`, with `l2` and `sph` clusterings of 2 clusters each, whose centers
/// are `centers`.
fn cluster_files(dir: &Path, centers: &[f32]) -> (Files, Fbin) {
    let files = Files::new(dir, "tiny");
    let corpus: Vec<f32> = (0..18).map(|i| 1.0 + i as f32).collect();
    write_table(&files.corpus(), 3, &corpus, f32::to_bits);

    for kind in ["l2", "sph"] {
        write_table(&files.centroids(kind), 3, centers, f32::to_bits);
        write_table(&files.assignment(kind), 1, &[0u32, 1, 0, 1, 0, 1], |v| v);
    }

    let corpus = Fbin::open(&files.corpus()).unwrap();
    (files, corpus)
}

#[test]
fn clusters_keep_or_scale_their_centers_by_kind() {
    let dir = scratch("clusters");
    let centers = [3.0, 0.0, 4.0, 0.0, 2.0, 0.0];
    let (files, corpus) = cluster_files(&dir, &centers);
    let l2 = clusters(&files, "l2", &corpus).unwrap();
    assert_eq!(l2.dim, 3);
    assert_eq!(l2.centroids, centers);
    assert_eq!(l2.assignment, [0, 1, 0, 1, 0, 1]);
    assert_eq!(l2.parent_of, [0, 0]);
    assert_eq!(l2.parents, [1.5, 1.0, 2.0]);

    for kind in ["cos", "sph"] {
        let scaled = clusters(&files, kind, &corpus).unwrap();
        assert_eq!(scaled.centroids, [0.6, 0.0, 0.8, 0.0, 1.0, 0.0], "{kind}");
        assert_eq!(scaled.parents, [0.3, 0.5, 0.4], "{kind}");
        assert_eq!(scaled.assignment, l2.assignment, "{kind}");
    }

    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn cluster_files_that_dont_fit_the_documents_fail_with_invalid_data() {
    let dir = scratch("bad-clusters");
    let (files, corpus) = cluster_files(&dir, &[3.0, 0.0, 4.0, 0.0, 2.0, 0.0]);
    let kind_of = |kind: &str| clusters(&files, kind, &corpus).err().map(|e| e.kind());
    let assignment = files.assignment("l2");
    let cases: [(&str, Vec<u32>, usize); 3] = [
        ("a cluster past the last", vec![0, 1, 0, 1, 0, 2], 1),
        ("too few documents", vec![0, 1, 0, 1, 0], 1),
        ("two columns", vec![0, 1, 0, 1, 0, 1], 2),
    ];

    for (what, values, width) in cases {
        write_table(&assignment, width, &values, |v| v);
        assert_eq!(kind_of("l2"), Some(ErrorKind::InvalidData), "{what}");
        assert_eq!(kind_of("cos"), Some(ErrorKind::InvalidData), "{what}");
    }

    write_table(&assignment, 1, &[0u32, 1, 0, 1, 0, 1], |v| v);
    let centers = files.centroids("l2");
    let cases: [(&str, Vec<f32>, usize, Option<ErrorKind>); 5] = [
        (
            "centers of 2 values",
            vec![3.0, 4.0, 0.0, 2.0],
            2,
            Some(ErrorKind::InvalidData),
        ),
        ("no centers", vec![], 3, Some(ErrorKind::InvalidData)),
        (
            "a value that isn't a number",
            vec![3.0, 0.0, f32::NAN, 0.0, 2.0, 0.0],
            3,
            Some(ErrorKind::InvalidData),
        ),
        (
            "an infinite value",
            vec![3.0, 0.0, 4.0, 0.0, f32::INFINITY, 0.0],
            3,
            Some(ErrorKind::InvalidData),
        ),
        ("a zero center", vec![3.0, 0.0, 4.0, 0.0, 0.0, 0.0], 3, None),
    ];

    for (what, values, width, expected) in cases {
        write_table(&centers, width, &values, f32::to_bits);
        assert_eq!(kind_of("l2"), expected, "{what}");
    }

    // A zero center has no direction to scale to.
    assert_eq!(kind_of("cos"), Some(ErrorKind::InvalidData));
    fs::remove_file(&centers).unwrap();
    assert_eq!(kind_of("cos"), Some(ErrorKind::NotFound));
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_label_names_the_layer_of_its_first_layer_part() {
    assert_eq!(layer_of("openings.L9"), Some(9));
    assert_eq!(layer_of("queries.L10"), Some(10));
    assert_eq!(layer_of("queries_l10"), Some(10));
    assert_eq!(layer_of("openings.L3.v2"), Some(3));
    assert_eq!(layer_of("L0.L5"), Some(0));
    assert_eq!(layer_of("minilm.openings"), None);
    assert_eq!(layer_of("L"), None);
    assert_eq!(layer_of("L3a"), None);
    assert_eq!(layer_of("level"), None);
}
