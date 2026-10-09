//! Unit tests for the candle encoder, on tiny random snapshots. The tests that compare the real models with the cached
//! embeddings and with PyTorch's layer states are ignored, and run by hand.

use std::io::{BufRead, BufReader, ErrorKind, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use super::*;
use crate::encoder::testutil::{
    Kind, TempSnapshot, VOCAB, assert_close, tiny_config, tiny_snapshot, tiny_weights, values,
};

const PROMPT: &str = "search for: ";
const TEXT: &str = "Cheap index funds";
/// The ids of the prompt and `TEXT`, between the classification token [CLS] and the separator token [SEP].
const IDS: [u32; 8] = [2, 4, 5, 6, 7, 8, 9, 3];

/// A 2-layer model that pools the first token's state, with a query prompt.
const CLS_WITH_PROMPT: Kind = Kind {
    cls: true,
    prompt: Some(PROMPT),
    layers: 2,
};

/// A hook that keeps the summaries of the layers in `layers`.
struct Recorder {
    layers: Vec<usize>,
    seen: Vec<(usize, Vec<f32>)>,
}

impl Recorder {
    fn new(layers: &[usize]) -> Recorder {
        Recorder {
            layers: layers.to_vec(),
            seen: Vec::new(),
        }
    }
}

impl LayerHook for Recorder {
    fn wants(&self, layer: usize) -> bool {
        self.layers.contains(&layer)
    }

    fn on_layer(&mut self, layer: usize, pooled: &[f32]) {
        self.seen.push((layer, pooled.to_vec()));
    }
}

/// The model's states for `ids`, layer by layer, straight from the network.
fn states(encoder: &CandleEncoder, ids: &[u32]) -> Vec<Vec<f32>> {
    let input_ids = Tensor::new(ids, &Device::Cpu).unwrap().unsqueeze(0).unwrap();
    let token_type_ids = input_ids.zeros_like().unwrap();
    let mut states = Vec::new();
    encoder
        .model
        .forward_layers(&input_ids, &token_type_ids, None, encoder.layers, &mut |_, state| {
            states.push(values(state));
            Ok(())
        })
        .unwrap();
    states
}

/// concat(average over the tokens, first token) of a state of `tokens` × `dim` values, computed on the host.
fn host_summary(state: &[f32], tokens: usize, dim: usize) -> Vec<f32> {
    let mut mean = vec![0f64; dim];

    for token in state.chunks_exact(dim) {
        for (m, &x) in mean.iter_mut().zip(token) {
            *m += f64::from(x);
        }
    }

    let mut out: Vec<f32> = mean.iter().map(|m| (m / tokens as f64) as f32).collect();
    out.extend_from_slice(&state[..dim]);
    out
}

fn normalized(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt();
    v.iter().map(|&x| (f64::from(x) / norm) as f32).collect()
}

fn open(snapshot: &TempSnapshot) -> CandleEncoder {
    CandleEncoder::open(&snapshot.0, Backend::Cpu).unwrap()
}

#[test]
fn the_hook_gets_the_summary_of_the_embedding_output_and_of_each_layers_output() {
    let snapshot = tiny_snapshot("hook", &CLS_WITH_PROMPT);
    let mut encoder = open(&snapshot);
    let mut hook = Recorder::new(&[0, 1, 2]);
    let embedding = encoder.encode(TEXT, &mut hook).unwrap();

    let states = states(&encoder, &IDS);
    assert_eq!(states.len(), 3);
    assert_eq!(hook.seen.iter().map(|(layer, _)| *layer).collect::<Vec<_>>(), [0, 1, 2]);

    for (layer, pooled) in &hook.seen {
        let expected = host_summary(&states[*layer], IDS.len(), 16);
        assert_close(pooled, &expected, 1e-6, &format!("layer {layer}"));
    }

    assert_close(&embedding, &normalized(&states[2][..16]), 1e-6, "the embedding");
    assert_eq!(encoder.prompt(), PROMPT);
    assert_eq!(encoder.pooling(), Pooling::Cls);
    assert_eq!((encoder.layers(), encoder.dim()), (2, 16));
}

#[test]
fn only_wanted_layers_reach_the_hook() {
    let snapshot = tiny_snapshot("wanted", &CLS_WITH_PROMPT);
    let mut encoder = open(&snapshot);
    let mut all = Recorder::new(&[0, 1, 2]);
    let mut one = Recorder::new(&[1]);
    let first = encoder.encode(TEXT, &mut all).unwrap();
    let second = encoder.encode(TEXT, &mut one).unwrap();
    let third = encoder.encode(TEXT, &mut NoHook).unwrap();

    assert_eq!(one.seen, all.seen[1..2]);
    assert_eq!(first, second);
    assert_eq!(first, third);
}

#[test]
fn mean_pooling_averages_the_last_layer_without_a_prompt() {
    let snapshot = tiny_snapshot(
        "mean",
        &Kind {
            cls: false,
            prompt: None,
            layers: 2,
        },
    );
    let mut encoder = open(&snapshot);
    let embedding = encoder.encode(TEXT, &mut NoHook).unwrap();

    let ids = [2, 7, 8, 9, 3];
    let states = states(&encoder, &ids);
    let mean = &host_summary(&states[2], ids.len(), 16)[..16];
    assert_close(&embedding, &normalized(mean), 1e-6, "the embedding");
    assert_eq!(encoder.prompt(), "");
    assert_eq!(encoder.pooling(), Pooling::Mean);
}

#[test]
fn texts_are_neither_padded_nor_left_too_long() {
    let snapshot = tiny_snapshot("lengths", &CLS_WITH_PROMPT);
    let encoder = open(&snapshot);
    let long = "how to save ".repeat(10);

    assert_eq!(encoder.tokens(TEXT).unwrap(), IDS.len());
    assert_eq!(encoder.tokens(&long).unwrap(), 20);
}

#[test]
fn run_to_reaches_every_layer_and_no_further() {
    let snapshot = tiny_snapshot("run-to", &CLS_WITH_PROMPT);
    let encoder = open(&snapshot);

    for layer in 0..=2 {
        encoder.run_to(TEXT, layer).unwrap();
    }

    assert_eq!(encoder.run_to(TEXT, 3).unwrap_err().kind(), ErrorKind::InvalidInput);
}

#[test]
fn damaged_or_unsupported_snapshots_give_errors() {
    let cases: [(&str, &str, Option<&str>, ErrorKind); 6] = [
        ("no-tokenizer", "tokenizer.json", None, ErrorKind::NotFound),
        (
            "bad-tokenizer",
            "tokenizer.json",
            Some("{\"model\": 3}"),
            ErrorKind::InvalidData,
        ),
        (
            "bad-config",
            "config.json",
            Some("{\"hidden_size\": 16}"),
            ErrorKind::InvalidData,
        ),
        (
            "bad-weights",
            "model.safetensors",
            Some("not a tensor file"),
            ErrorKind::InvalidData,
        ),
        (
            "max-pooling",
            "1_Pooling/config.json",
            Some("{\"pooling_mode_max_tokens\": true}"),
            ErrorKind::Unsupported,
        ),
        (
            "dense",
            "modules.json",
            Some("[{\"type\": \"sentence_transformers.models.Dense\", \"path\": \"2_Dense\"}]"),
            ErrorKind::Unsupported,
        ),
    ];

    for (name, file, contents, kind_of_error) in cases {
        let snapshot = tiny_snapshot(name, &CLS_WITH_PROMPT);
        let path = snapshot.0.join(file);
        match contents {
            Some(contents) => std::fs::write(&path, contents).unwrap(),
            None => std::fs::remove_file(&path).unwrap(),
        }
        let error = CandleEncoder::open(&snapshot.0, Backend::Cpu).err().expect(name);
        assert_eq!(error.kind(), kind_of_error, "{name}: {error}");
    }
}

/// Sets `field` of the JSON (JavaScript Object Notation) file `file` in `snapshot` to `value`.
fn set_field(snapshot: &TempSnapshot, file: &str, field: &str, value: serde_json::Value) {
    let path = snapshot.0.join(file);
    let mut json: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    json[field] = value;
    std::fs::write(&path, json.to_string()).unwrap();
}

/// Changes the tokenizer file of `snapshot` with `change`.
fn edit_tokenizer(snapshot: &TempSnapshot, change: impl Fn(&mut serde_json::Value)) {
    let path = snapshot.0.join("tokenizer.json");
    let mut tokenizer: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    change(&mut tokenizer);
    std::fs::write(&path, tokenizer.to_string()).unwrap();
}

/// A tiny snapshot with one field changed. It holds the case's name, how the snapshot pools, the file and the field,
/// the field's new value, and the error it gives, if any.
type Changed<'a> = (
    &'a str,
    &'a Kind,
    &'a str,
    &'a str,
    serde_json::Value,
    Option<ErrorKind>,
);

#[test]
fn settings_that_dont_fit_the_model_give_errors_and_fitting_ones_open() {
    let mean_with_prompt = Kind {
        cls: false,
        prompt: Some(PROMPT),
        layers: 2,
    };
    let no_prompt = Kind {
        cls: false,
        prompt: None,
        layers: 2,
    };
    let cases: [Changed; 9] = [
        (
            "no-heads",
            &CLS_WITH_PROMPT,
            "config.json",
            "num_attention_heads",
            0.into(),
            Some(ErrorKind::InvalidData),
        ),
        (
            "uneven-heads",
            &CLS_WITH_PROMPT,
            "config.json",
            "num_attention_heads",
            3.into(),
            Some(ErrorKind::InvalidData),
        ),
        (
            "two-heads",
            &CLS_WITH_PROMPT,
            "config.json",
            "num_attention_heads",
            2.into(),
            None,
        ),
        (
            "no-layers",
            &CLS_WITH_PROMPT,
            "config.json",
            "num_hidden_layers",
            0.into(),
            Some(ErrorKind::InvalidData),
        ),
        (
            "no-room",
            &CLS_WITH_PROMPT,
            "sentence_bert_config.json",
            "max_seq_length",
            2.into(),
            Some(ErrorKind::InvalidData),
        ),
        (
            "one-word",
            &CLS_WITH_PROMPT,
            "sentence_bert_config.json",
            "max_seq_length",
            3.into(),
            None,
        ),
        (
            "mean-without-prompt",
            &mean_with_prompt,
            "1_Pooling/config.json",
            "include_prompt",
            false.into(),
            Some(ErrorKind::Unsupported),
        ),
        (
            "mean-without-a-prompt-to-leave-out",
            &no_prompt,
            "1_Pooling/config.json",
            "include_prompt",
            false.into(),
            None,
        ),
        // The first token's state is the same with or without the prompt's tokens in the pooling.
        (
            "first-token-without-prompt",
            &CLS_WITH_PROMPT,
            "1_Pooling/config.json",
            "include_prompt",
            false.into(),
            None,
        ),
    ];

    for (name, kind, file, field, value, expected) in cases {
        let snapshot = tiny_snapshot(name, kind);
        set_field(&snapshot, file, field, value);
        let found = CandleEncoder::open(&snapshot.0, Backend::Cpu).err().map(|e| e.kind());
        assert_eq!(found, expected, "{name}");
    }

    // A vocabulary one word short of the tokenizer's, with weights of that size, so only the vocabulary check fails.
    let snapshot = tiny_snapshot("small-vocabulary", &CLS_WITH_PROMPT);
    let config = Config {
        vocab_size: VOCAB.len() - 1,
        ..tiny_config(2)
    };
    set_field(&snapshot, "config.json", "vocab_size", config.vocab_size.into());
    candle_core::safetensors::save(&tiny_weights(&config, 7), snapshot.0.join("model.safetensors")).unwrap();
    let found = CandleEncoder::open(&snapshot.0, Backend::Cpu).err().map(|e| e.kind());
    assert_eq!(found, Some(ErrorKind::InvalidData), "small-vocabulary");

    // A tokenizer whose ids leave a gap, so its 13 words reach id 13, and one with a token added at id 13. Neither
    // fits the 13 rows of the model.
    let gap = tiny_snapshot("vocabulary-gap", &CLS_WITH_PROMPT);
    edit_tokenizer(&gap, |t| t["model"]["vocab"]["save"] = 13.into());
    let added = tiny_snapshot("added-token", &CLS_WITH_PROMPT);
    edit_tokenizer(&added, |t| {
        let token = serde_json::json!({
            "id": 13, "content": "[NEW]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false,
            "special": true
        });
        t["added_tokens"].as_array_mut().unwrap().push(token);
    });

    for (name, snapshot) in [("vocabulary-gap", &gap), ("added-token", &added)] {
        let found = CandleEncoder::open(&snapshot.0, Backend::Cpu).err().map(|e| e.kind());
        assert_eq!(found, Some(ErrorKind::InvalidData), "{name}");
    }
}

#[test]
fn texts_stop_at_the_last_position_the_model_has() {
    let snapshot = tiny_snapshot("positions", &CLS_WITH_PROMPT);
    set_field(&snapshot, "sentence_bert_config.json", "max_seq_length", 100.into());
    let mut encoder = open(&snapshot);
    let long = "how to save ".repeat(20);

    assert_eq!(encoder.tokens(&long).unwrap(), 32);
    assert_eq!(encoder.encode(&long, &mut NoHook).unwrap().len(), 16);
}

#[test]
fn a_snapshot_is_found_through_refs_main_or_as_the_only_one() {
    let hub = TempSnapshot(std::env::temp_dir().join(format!("specwave-{}-hub", std::process::id())));
    let model = hub.0.join("models--org--name");
    std::fs::create_dir_all(model.join("snapshots/aaa")).unwrap();

    assert_eq!(snapshot_in(&hub.0, "org/name").unwrap(), model.join("snapshots/aaa"));
    std::fs::create_dir_all(model.join("snapshots/bbb")).unwrap();
    assert_eq!(
        snapshot_in(&hub.0, "org/name").unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    std::fs::create_dir_all(model.join("refs")).unwrap();
    std::fs::write(model.join("refs/main"), "bbb\n").unwrap();
    assert_eq!(snapshot_in(&hub.0, "org/name").unwrap(), model.join("snapshots/bbb"));
    assert_eq!(
        snapshot_in(&hub.0, "org/other").unwrap_err().kind(),
        ErrorKind::NotFound
    );
}

/// The texts of a JSON lines file with a "text" field per line, with each line's "row" (else its line number).
fn read_queries(path: &Path, n: usize) -> Vec<(usize, String)> {
    let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut out = Vec::new();

    for (i, line) in BufReader::new(file).lines().take(n).enumerate() {
        let value: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
        let row = value["row"].as_u64().map_or(i, |r| r as usize);
        out.push((row, value["text"].as_str().unwrap().to_string()));
    }

    out
}

/// Row `row` of an `.fbin` file, which holds its row count and row width as little-endian 32-bit integers, then the
/// rows as little-endian f32 values.
fn read_row(path: &Path, row: usize) -> Vec<f32> {
    let mut file = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut head = [0u8; 8];
    file.read_exact(&mut head).unwrap();
    let dim = u32::from_le_bytes([head[4], head[5], head[6], head[7]]) as usize;
    let mut bytes = vec![0u8; dim * 4];
    file.seek(SeekFrom::Start(8 + (row * dim * 4) as u64)).unwrap();
    file.read_exact(&mut bytes).unwrap();
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| f32::from_le_bytes(b))
        .collect()
}

fn data_path(var: &str, file: &str) -> PathBuf {
    match std::env::var_os(var) {
        Some(path) => PathBuf::from(path),
        None => {
            let data = std::env::var_os("SPECWAVE_DATA")
                .map_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../data"), PathBuf::from);
            data.join(file)
        }
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(&p, &q)| f64::from(p) * f64::from(q)).sum::<f64>();
    dot(a, b) / (dot(a, a) * dot(b, b)).sqrt()
}

/// The least cosine that the real models' outputs must have with PyTorch's, and on Metal with the processor's.
const AGREE: f64 = 0.999999;

/// The backends of this machine.
fn backends() -> Vec<Backend> {
    match cfg!(target_os = "macos") {
        true => vec![Backend::Cpu, Backend::Metal],
        false => vec![Backend::Cpu],
    }
}

/// Encodes the first 100 queries on every backend and compares each with its row of the cached embeddings. On
/// Metal it also compares every layer's summary with the processor's, for the first 10 queries.
fn check_cached(repo: &str, queries: &Path, embeddings: &Path) {
    let queries = read_queries(queries, 100);
    let mut cpu_layers = Vec::new();
    assert_eq!(queries.len(), 100);

    for backend in backends() {
        let mut encoder = CandleEncoder::open(&snapshot_dir(repo).unwrap(), backend).unwrap();
        let every: Vec<usize> = (0..=encoder.layers()).collect();
        let (mut worst, mut sum, mut worst_layer) = (f64::INFINITY, 0.0, f64::INFINITY);
        for (i, (row, text)) in queries.iter().enumerate() {
            let wanted: &[usize] = if i < 10 { &every } else { &[] };
            let mut hook = Recorder::new(wanted);
            let found = encoder.encode(text, &mut hook).unwrap();
            let c = cosine(&found, &read_row(embeddings, *row));
            assert!(c >= AGREE, "{repo} on {backend:?}: query {row} has cosine {c}");
            (worst, sum) = (worst.min(c), sum + c);
            match backend {
                Backend::Cpu if i < 10 => cpu_layers.push(hook.seen),
                Backend::Metal if i < 10 => {
                    assert_eq!(hook.seen.len(), cpu_layers[i].len());
                    for ((layer, metal), (_, cpu)) in hook.seen.iter().zip(&cpu_layers[i]) {
                        let c = cosine(metal, cpu);
                        assert!(
                            c >= AGREE,
                            "{repo}: query {row} layer {layer} has Metal-to-CPU cosine {c}"
                        );
                        worst_layer = worst_layer.min(c);
                    }
                }
                _ => {}
            }
        }
        eprintln!(
            "{repo} on {backend:?}: 1 - cosine with the cached embeddings is {:.1e} at worst and {:.1e} on average",
            1.0 - worst,
            1.0 - sum / 100.0
        );
        if backend == Backend::Metal {
            eprintln!(
                "{repo}: 1 - cosine of Metal's layer summaries with the processor's is {:.1e} at worst",
                1.0 - worst_layer
            );
        }
    }
}

#[test]
#[ignore = "reads the real models and the cached embeddings, run by hand"]
fn arctic_matches_the_cached_fiqa_query_embeddings() {
    check_cached(
        "Snowflake/snowflake-arctic-embed-m-v1.5",
        &data_path("SPECWAVE_FIQA_QUERIES", "fiqa_queries.jsonl"),
        &data_path("SPECWAVE_FIQA_EMBEDDINGS", "fiqa_queries.fbin"),
    );
}

#[test]
#[ignore = "reads the real models and the cached embeddings, run by hand"]
fn minilm_matches_the_cached_quora_query_embeddings() {
    check_cached(
        "sentence-transformers/all-MiniLM-L6-v2",
        &data_path("SPECWAVE_QUORA_QUERIES", "quora_queries.jsonl"),
        &data_path("SPECWAVE_QUORA_EMBEDDINGS", "quora_queries.fbin"),
    );
}

/// Encodes the first 100 evaluation queries of `dataset` on every backend. It compares the summary of each layer that
/// PyTorch stored with PyTorch's, and the embedding with the query's row of `<dataset>_test.fbin`. The data folder is
/// `$SPECWAVE_DATA`, else this repository's `data`.
fn check_states(repo: &str, dataset: &str) {
    let data = data_path("SPECWAVE_DATA", "");
    let states = data.join("spec").join(dataset).join("states");
    let queries = read_queries(&data.join(format!("{dataset}_test.jsonl")), 100);
    let mut layers: Vec<usize> = std::fs::read_dir(&states)
        .unwrap_or_else(|e| panic!("{}: {e}", states.display()))
        .filter_map(|entry| {
            let name = entry.unwrap().file_name().into_string().ok()?;
            name.strip_prefix("test.L")?.strip_suffix(".fbin")?.parse().ok()
        })
        .collect();
    layers.sort_unstable();
    assert_eq!(queries.len(), 100);
    assert!(!layers.is_empty(), "{} holds no test.L<layer>.fbin", states.display());

    for backend in backends() {
        let mut encoder = CandleEncoder::open(&snapshot_dir(repo).unwrap(), backend).unwrap();
        let (mut worst, mut worst_final) = (vec![f64::INFINITY; layers.len()], f64::INFINITY);
        for (row, text) in &queries {
            let mut hook = Recorder::new(&layers);
            let found = encoder.encode(text, &mut hook).unwrap();
            assert_eq!(hook.seen.iter().map(|(l, _)| *l).collect::<Vec<_>>(), layers);
            for ((layer, pooled), worst) in hook.seen.iter().zip(&mut worst) {
                let c = cosine(pooled, &read_row(&states.join(format!("test.L{layer}.fbin")), *row));
                assert!(
                    c >= AGREE,
                    "{repo} on {backend:?}: query {row} layer {layer} has cosine {c}"
                );
                *worst = worst.min(c);
            }
            let c = cosine(&found, &read_row(&data.join(format!("{dataset}_test.fbin")), *row));
            assert!(c >= AGREE, "{repo} on {backend:?}: query {row} has cosine {c}");
            worst_final = worst_final.min(c);
        }
        let shown: Vec<String> = layers
            .iter()
            .zip(&worst)
            .map(|(l, w)| format!("L{l} {:.1e}", 1.0 - w))
            .collect();
        eprintln!(
            "{repo} on {backend:?}: 1 - cosine with PyTorch at worst, {}, final {:.1e}",
            shown.join(", "),
            1.0 - worst_final
        );
    }
}

#[test]
#[ignore = "reads the real models and PyTorch's stored layer states, run by hand"]
fn arctic_matches_pytorchs_layer_states_of_the_fiqa_test_queries() {
    check_states("Snowflake/snowflake-arctic-embed-m-v1.5", "fiqa");
}

#[test]
#[ignore = "reads the real models and PyTorch's stored layer states, run by hand"]
fn minilm_matches_pytorchs_layer_states_of_the_quora_test_queries() {
    check_states("sentence-transformers/all-MiniLM-L6-v2", "quora");
}
