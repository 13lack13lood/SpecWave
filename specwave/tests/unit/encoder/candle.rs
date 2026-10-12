//! Unit tests for the candle encoder, on tiny random snapshots. The tests that compare the real models with the cached
//! embeddings and with PyTorch's layer states are ignored, and run by hand.

use std::cell::RefCell;
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
    let input_ids = Tensor::new(ids, &encoder.device).unwrap().unsqueeze(0).unwrap();
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

/// A hook that notes each call it gets, eg. "wants 0" and "gets 0", and wants the layers in `wanted`.
struct Events {
    wanted: Vec<usize>,
    seen: RefCell<Vec<String>>,
}

impl LayerHook for Events {
    fn wants(&self, layer: usize) -> bool {
        self.seen.borrow_mut().push(format!("wants {layer}"));
        self.wanted.contains(&layer)
    }

    fn on_layer(&mut self, layer: usize, pooled: &[f32]) {
        self.seen
            .get_mut()
            .push(format!("gets {layer} with {} values", pooled.len()));
    }
}

#[test]
fn the_hook_is_asked_about_each_layer_once_in_order_and_gets_only_the_layers_it_wants() {
    let snapshot = tiny_snapshot("asked", &CLS_WITH_PROMPT);
    let mut encoder = open(&snapshot);
    let cases: [(&[usize], &[&str]); 4] = [
        (&[], &["wants 0", "wants 1", "wants 2"]),
        (&[1], &["wants 0", "wants 1", "gets 1 with 32 values", "wants 2"]),
        (
            &[0, 2],
            &[
                "wants 0",
                "gets 0 with 32 values",
                "wants 1",
                "wants 2",
                "gets 2 with 32 values",
            ],
        ),
        (&[2, 5], &["wants 0", "wants 1", "wants 2", "gets 2 with 32 values"]),
    ];

    for (wanted, expected) in cases {
        let mut hook = Events {
            wanted: wanted.to_vec(),
            seen: RefCell::default(),
        };
        encoder.encode(TEXT, &mut hook).unwrap();
        assert_eq!(*hook.seen.borrow(), expected, "{wanted:?}");
    }
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
fn the_token_limit_can_be_lowered_but_never_past_the_models_positions() {
    let snapshot = tiny_snapshot("max-tokens", &CLS_WITH_PROMPT);
    let mut encoder = open(&snapshot);
    let long = "how to save ".repeat(20);
    assert_eq!((encoder.max_tokens(), encoder.tokens(&long).unwrap()), (20, 20));

    // The prompt and the text keep 8 tokens between the classification token [CLS] and the separator [SEP].
    encoder.set_max_tokens(10).unwrap();
    assert_eq!((encoder.max_tokens(), encoder.tokens(&long).unwrap()), (10, 10));
    let ids = [2, 4, 5, 6, 10, 11, 12, 10, 11, 3];
    let embedding = encoder.encode(&long, &mut NoHook).unwrap();
    assert_close(
        &embedding,
        &normalized(&states(&encoder, &ids)[2][..16]),
        1e-6,
        "the cut text",
    );

    encoder.set_max_tokens(100).unwrap();
    assert_eq!((encoder.max_tokens(), encoder.tokens(&long).unwrap()), (32, 32));
    assert_eq!(encoder.encode(&long, &mut NoHook).unwrap().len(), 16);
    encoder.set_max_tokens(3).unwrap();
    assert_eq!(encoder.tokens(&long).unwrap(), 3);
    assert_eq!(encoder.set_max_tokens(2).unwrap_err().kind(), ErrorKind::InvalidInput);
    assert_eq!((encoder.max_tokens(), encoder.tokens(&long).unwrap()), (3, 3));
}

#[test]
fn only_metal_queues_its_work() {
    let snapshot = tiny_snapshot("queues", &CLS_WITH_PROMPT);
    assert!(!open(&snapshot).queues_work());

    match CandleEncoder::open(&snapshot.0, Backend::Metal) {
        Ok(encoder) => assert!(encoder.queues_work()),
        Err(e) => assert_eq!(e.kind(), ErrorKind::Unsupported),
    }
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

/// Writes `prompts` as the sentence-transformers prompts of `snapshot`.
fn set_prompts(snapshot: &Path, prompts: serde_json::Value) {
    let json = serde_json::json!({"prompts": prompts, "default_prompt_name": null});
    std::fs::write(snapshot.join("config_sentence_transformers.json"), json.to_string()).unwrap();
}

#[test]
fn a_passage_takes_the_passage_prompt_and_a_query_the_query_prompt() {
    let snapshot = tiny_snapshot("passage", &CLS_WITH_PROMPT);
    set_prompts(&snapshot.0, serde_json::json!({"query": PROMPT, "passage": "how to: "}));
    let mut encoder = open(&snapshot);
    let passage_ids = [2, 10, 11, 6, 7, 8, 9, 3];
    let mut hook = Recorder::new(&[1]);
    let passage = encoder.encode_passage(TEXT, &mut hook).unwrap();
    let query = encoder.encode(TEXT, &mut NoHook).unwrap();

    assert_eq!((encoder.prompt(), encoder.passage_prompt()), (PROMPT, "how to: "));
    assert_eq!(encoder.tokenize("how to: ", TEXT).unwrap().get_ids(), passage_ids);
    let states = states(&encoder, &passage_ids);
    assert_close(&passage, &normalized(&states[2][..16]), 1e-6, "the passage");
    assert_eq!(hook.seen.iter().map(|(layer, _)| *layer).collect::<Vec<_>>(), [1]);
    assert_close(&hook.seen[0].1, &host_summary(&states[1], 8, 16), 1e-6, "layer 1");
    assert_eq!(query, encoder.encode(TEXT, &mut NoHook).unwrap());
    assert_ne!(query, passage);
}

#[test]
fn the_passage_prompt_is_the_document_prompt_else_the_passage_one_else_the_corpus_one() {
    let cases = [
        (
            serde_json::json!({"query": PROMPT, "corpus": "save: ", "passage": "how to: ", "document": "to: "}),
            "to: ",
        ),
        (
            serde_json::json!({"query": PROMPT, "corpus": "save: ", "passage": "how to: "}),
            "how to: ",
        ),
        (serde_json::json!({"corpus": "save: "}), "save: "),
        (serde_json::json!({"query": PROMPT}), ""),
    ];

    for (i, (prompts, expected)) in cases.into_iter().enumerate() {
        let snapshot = tiny_snapshot(&format!("passage-names-{i}"), &CLS_WITH_PROMPT);
        set_prompts(&snapshot.0, prompts);
        assert_eq!(open(&snapshot).passage_prompt(), expected, "case {i}");
    }

    // An average over the tokens can't leave out a passage prompt either.
    let snapshot = tiny_snapshot(
        "mean-without-passage-prompt",
        &Kind {
            cls: false,
            prompt: None,
            layers: 2,
        },
    );
    set_prompts(&snapshot.0, serde_json::json!({"passage": "how to: "}));
    set_field(&snapshot, "1_Pooling/config.json", "include_prompt", false.into());
    let found = CandleEncoder::open(&snapshot.0, Backend::Cpu).err().map(|e| e.kind());
    assert_eq!(found, Some(ErrorKind::Unsupported));
}

/// Moves a tiny snapshot into `hub` the way the Hugging Face cache lays out a snapshot of `repo`, and returns its new
/// place.
fn in_hub(hub: &TempSnapshot, repo: &str, snapshot: TempSnapshot) -> PathBuf {
    let snapshots = hub
        .0
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots");
    std::fs::create_dir_all(&snapshots).unwrap();
    std::fs::rename(&snapshot.0, snapshots.join("c0")).unwrap();
    snapshots.join("c0")
}

#[test]
fn a_listed_model_in_the_cache_gets_the_prompts_its_settings_leave_out() {
    let hub = TempSnapshot(std::env::temp_dir().join(format!("specwave-{}-hub-listed", std::process::id())));
    let plain = Kind {
        cls: false,
        prompt: None,
        layers: 2,
    };
    let e5 = in_hub(&hub, "intfloat/e5-base-v2", tiny_snapshot("listed", &plain));
    let elsewhere = tiny_snapshot("listed-elsewhere", &plain);
    let listed = CandleEncoder::open(&e5, Backend::Cpu).unwrap();
    let other = open(&elsewhere);

    assert_eq!((listed.prompt(), listed.passage_prompt()), ("query: ", "passage: "));
    assert_eq!((other.prompt(), other.passage_prompt()), ("", ""));
    // "query" is the unknown token [UNK] to the tiny tokenizer, and ":" its own token.
    assert_eq!(
        listed.tokenize("query: ", TEXT).unwrap().get_ids(),
        [2, 1, 6, 7, 8, 9, 3]
    );
    assert_eq!((listed.tokens(TEXT).unwrap(), other.tokens(TEXT).unwrap()), (7, 5));
}

#[test]
fn a_listed_prompt_that_the_snapshot_gives_otherwise_gives_invalid_data() {
    let hub = TempSnapshot(std::env::temp_dir().join(format!("specwave-{}-hub-conflict", std::process::id())));
    let e5 = in_hub(&hub, "intfloat/e5-base-v2", tiny_snapshot("conflict", &CLS_WITH_PROMPT));
    let found = CandleEncoder::open(&e5, Backend::Cpu).err().map(|e| e.kind());
    assert_eq!(found, Some(ErrorKind::InvalidData));

    set_prompts(&e5, serde_json::json!({"query": "query: ", "document": "passage: "}));
    let encoder = CandleEncoder::open(&e5, Backend::Cpu).unwrap();
    assert_eq!((encoder.prompt(), encoder.passage_prompt()), ("query: ", "passage: "));

    set_prompts(&e5, serde_json::json!({"query": "query: ", "document": ""}));
    let found = CandleEncoder::open(&e5, Backend::Cpu).err().map(|e| e.kind());
    assert_eq!(found, Some(ErrorKind::InvalidData));
}

#[test]
fn a_snapshot_in_the_cache_names_its_repository() {
    let hub = Path::new("/hub");
    let cases = [
        (
            "models--BAAI--bge-small-en-v1.5/snapshots/abc",
            Some("BAAI/bge-small-en-v1.5"),
        ),
        ("models--bert-base-uncased/snapshots/abc", Some("bert-base-uncased")),
        ("models--org--name/refs/abc", None),
        ("org/name/snapshots/abc", None),
        ("snapshots/abc", None),
        ("abc", None),
    ];

    for (path, expected) in cases {
        assert_eq!(repo_of(&hub.join(path)).as_deref(), expected, "{path}");
    }

    assert_eq!(repo_of(Path::new("/")), None);
}

#[test]
fn do_lower_case_puts_a_lowercasing_step_in_front_unless_the_tokenizer_has_one() {
    let lowered = vec![2, 4, 5, 6, 7, 8, 9, 3];
    let lowercase = serde_json::json!({"type": "Lowercase"});
    let nfc = serde_json::json!({"type": "NFC"});
    let cases = [
        (
            "no-step",
            serde_json::Value::Null,
            false,
            vec![2, 4, 5, 6, 1, 1, 1, 3],
            None,
        ),
        (
            "no-step-lowered",
            serde_json::Value::Null,
            true,
            lowered.clone(),
            Some(serde_json::json!({"type": "Sequence", "normalizers": [lowercase]})),
        ),
        (
            "nfc-lowered",
            serde_json::json!({"type": "Sequence", "normalizers": [nfc]}),
            true,
            lowered.clone(),
            Some(serde_json::json!({"type": "Sequence", "normalizers": [lowercase, nfc]})),
        ),
        (
            "nfc-alone-lowered",
            nfc.clone(),
            true,
            lowered.clone(),
            Some(serde_json::json!({"type": "Sequence", "normalizers": [lowercase, nfc]})),
        ),
        (
            "lowercase-kept",
            lowercase.clone(),
            true,
            lowered.clone(),
            Some(lowercase.clone()),
        ),
        (
            "lowercase-alone",
            lowercase.clone(),
            false,
            lowered,
            Some(lowercase.clone()),
        ),
    ];

    for (name, normalizer, do_lower_case, ids, steps) in cases {
        let snapshot = tiny_snapshot(name, &CLS_WITH_PROMPT);
        edit_tokenizer(&snapshot, |t| t["normalizer"] = normalizer.clone());
        set_field(
            &snapshot,
            "sentence_bert_config.json",
            "do_lower_case",
            do_lower_case.into(),
        );
        let encoder = open(&snapshot);
        assert_eq!(
            encoder.tokenize(PROMPT, "CHEAP INDEX FUNDS").unwrap().get_ids(),
            ids,
            "{name}"
        );
        let found = encoder
            .tokenizer
            .get_normalizer()
            .map(|n| serde_json::to_value(n).unwrap());
        assert_eq!(found, steps, "{name}");
    }
}

#[test]
fn an_embedding_of_length_1_must_fit_the_models_similarity() {
    let cases = [
        ("dot-normalized", true, Some("dot"), None),
        ("dot-raw", false, Some("dot"), Some(ErrorKind::Unsupported)),
        ("euclidean-raw", false, Some("euclidean"), Some(ErrorKind::Unsupported)),
        ("cosine-raw", false, Some("cosine"), None),
        ("unnamed-raw", false, None, None),
    ];

    let module = |kind: &str, path: &str| serde_json::json!({"path": path, "type": format!("models.{kind}")});

    for (name, normalizes, similarity, expected) in cases {
        let snapshot = tiny_snapshot(name, &CLS_WITH_PROMPT);
        let mut modules = vec![module("Transformer", ""), module("Pooling", "1_Pooling")];
        if normalizes {
            modules.push(module("Normalize", "2_Normalize"));
        }
        std::fs::write(
            snapshot.0.join("modules.json"),
            serde_json::Value::from(modules).to_string(),
        )
        .unwrap();
        set_field(
            &snapshot,
            "config_sentence_transformers.json",
            "similarity_fn_name",
            similarity.into(),
        );
        let found = CandleEncoder::open(&snapshot.0, Backend::Cpu).err().map(|e| e.kind());
        assert_eq!(found, expected, "{name}");
    }
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

/// The least cosine that a real model's embeddings must have with sentence-transformers'.
const MATCH: f64 = 0.9999;

/// Every value of an `.fbin` file, with its row width.
fn read_fbin(path: &Path) -> (Vec<f32>, usize) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let width = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let values = bytes[8..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| f32::from_le_bytes(b))
        .collect();
    (values, width)
}

/// Every value of an `.ibin` file, which holds its row count and row width, then the rows as 32-bit integers.
fn read_ibin(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes[8..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| u32::from_le_bytes(b))
        .collect()
}

/// Checks the listed model `name` on every backend against the references that `encoder_refs.py` wrote in
/// `<data>/encoders/<name>/`, for its queries and its passages: the token ids, the embedding against
/// sentence-transformers' and each layer's summary against transformers' hidden states. For the queries it also
/// checks every token's state at every layer. The data folder is `$SPECWAVE_DATA`, else this repository's `data`.
fn check_references(name: &str) {
    let dir = data_path("SPECWAVE_DATA", "").join("encoders").join(name);
    let meta_file = dir.join("meta.json");
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&meta_file).unwrap_or_else(|e| panic!("{}: {e}", meta_file.display())))
            .unwrap();
    let model = models::find(name).unwrap().unwrap();
    let snapshot = snapshot_dir(&model.repo).unwrap();

    for backend in backends() {
        let mut encoder = CandleEncoder::open(&snapshot, backend).unwrap();
        let every: Vec<usize> = (0..=encoder.layers()).collect();
        let d = encoder.dim();
        let mut shown = Vec::new();
        assert_eq!(encoder.prompt(), meta["query_prompt"].as_str().unwrap());
        assert_eq!(encoder.passage_prompt(), meta["passage_prompt"].as_str().unwrap());
        assert_eq!(
            (encoder.layers() as u64, d as u64),
            (meta["layers"].as_u64().unwrap(), meta["dim"].as_u64().unwrap())
        );

        for set in ["queries", "passages"] {
            let file = |stem: &str| dir.join(format!("{set}.{stem}"));
            let texts = read_queries(&file("jsonl"), usize::MAX);
            let (ids, lengths) = (read_ibin(&file("ids.ibin")), read_ibin(&file("lengths.ibin")));
            let (st, width) = read_fbin(&file("st.fbin"));
            let summaries: Vec<Vec<f32>> = every
                .iter()
                .map(|l| read_fbin(&file(&format!("L{l}.fbin"))).0)
                .collect();
            let tokens: Vec<Vec<f32>> = match set {
                "queries" => every
                    .iter()
                    .map(|l| read_fbin(&file(&format!("tokens.L{l}.fbin"))).0)
                    .collect(),
                _ => Vec::new(),
            };
            let prompt = match set {
                "queries" => encoder.prompt(),
                _ => encoder.passage_prompt(),
            }
            .to_string();
            let (mut start, mut worst, mut worst_layer, mut worst_token, mut largest) =
                (0, f64::INFINITY, f64::INFINITY, f64::INFINITY, 0f32);
            assert_eq!((texts.len(), lengths.len(), width), (50, 50, d));

            for (i, (_, text)) in texts.iter().enumerate() {
                let own = &ids[start..start + lengths[i] as usize];
                assert_eq!(
                    encoder.tokenize(&prompt, text).unwrap().get_ids(),
                    own,
                    "{name} {set} {i}"
                );
                let mut hook = Recorder::new(&every);
                let found = match set {
                    "queries" => encoder.encode(text, &mut hook),
                    _ => encoder.encode_passage(text, &mut hook),
                }
                .unwrap();
                let c = cosine(&found, &st[i * d..(i + 1) * d]);
                assert!(c >= MATCH, "{name} on {backend:?}: {set} {i} has cosine {c}");
                worst = worst.min(c);
                assert_eq!(hook.seen.iter().map(|(l, _)| *l).collect::<Vec<_>>(), every);
                for (layer, pooled) in &hook.seen {
                    let c = cosine(pooled, &summaries[*layer][i * 2 * d..(i + 1) * 2 * d]);
                    assert!(
                        c >= AGREE,
                        "{name} on {backend:?}: {set} {i} layer {layer} has cosine {c}"
                    );
                    worst_layer = worst_layer.min(c);
                }
                if !tokens.is_empty() {
                    for (layer, state) in states(&encoder, own).iter().enumerate() {
                        let expected = &tokens[layer][start * d..(start + own.len()) * d];
                        for (f, e) in state.chunks_exact(d).zip(expected.chunks_exact(d)) {
                            let c = cosine(f, e);
                            assert!(
                                c >= AGREE,
                                "{name} on {backend:?}: query {i} layer {layer} has a token at {c}"
                            );
                            worst_token = worst_token.min(c);
                            largest = f.iter().zip(e).fold(largest, |m, (a, b)| m.max((a - b).abs()));
                        }
                    }
                }
                start += own.len();
            }

            assert_eq!(start, ids.len());
            shown.push(format!(
                "{set} {:.1e}, layer summaries {:.1e}",
                1.0 - worst,
                1.0 - worst_layer
            ));
            if !tokens.is_empty() {
                shown.push(format!(
                    "query tokens {:.1e}, largest difference {largest:.1e}",
                    1.0 - worst_token
                ));
            }
        }

        eprintln!(
            "{name} on {backend:?}: token ids equal, 1 - cosine at worst: {}",
            shown.join(", ")
        );
    }
}

#[test]
#[ignore = "reads the real snapshots, run by hand"]
fn listed_prompts_appear_in_the_files_they_come_from() {
    for model in models::registered().unwrap() {
        let Some(source) = &model.prompts_from else {
            continue;
        };
        let path = snapshot_dir(&model.repo).unwrap().join(source);
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        for prompt in [&model.query_prompt, &model.passage_prompt].into_iter().flatten() {
            assert!(
                text.contains(prompt.as_str()),
                "{}: {source} doesn't hold {prompt:?}",
                model.name
            );
        }
    }
}

#[test]
#[ignore = "reads the real models and encoder_refs.py's references, run by hand"]
fn bge_small_matches_sentence_transformers_and_transformers() {
    check_references("bge-small");
}

#[test]
#[ignore = "reads the real models and encoder_refs.py's references, run by hand"]
fn bge_base_matches_sentence_transformers_and_transformers() {
    check_references("bge-base");
}

#[test]
#[ignore = "reads the real models and encoder_refs.py's references, run by hand"]
fn e5_base_matches_sentence_transformers_and_transformers() {
    check_references("e5-base");
}

#[test]
#[ignore = "reads the real models and encoder_refs.py's references, run by hand"]
fn bge_large_matches_sentence_transformers_and_transformers() {
    check_references("bge-large");
}

#[test]
#[ignore = "reads the real models and encoder_refs.py's references, run by hand"]
fn mxbai_large_matches_sentence_transformers_and_transformers() {
    check_references("mxbai-large");
}

#[test]
#[ignore = "reads the real models and encoder_refs.py's references, run by hand"]
fn arctic_matches_sentence_transformers_and_transformers() {
    check_references("arctic");
}

#[test]
#[ignore = "reads the real models and encoder_refs.py's references, run by hand"]
fn minilm_matches_sentence_transformers_and_transformers() {
    check_references("minilm");
}
