//! A tiny random BERT (Bidirectional Encoder Representations from Transformers) model, its tokenizer, and snapshot
//! directories that hold them, for the encoder's unit tests.

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{Device, Tensor};

use super::bert::{Config, HiddenAct, PositionEmbeddingType};

/// A small seeded generator, so the random weights are the same in every run.
pub(crate) struct SplitMix64(u64);

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> SplitMix64 {
        SplitMix64(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A uniform value in [-1, 1).
    pub(crate) fn signed(&mut self) -> f32 {
        ((self.next_u64() >> 11) as f64 / (1u64 << 52) as f64 - 1.0) as f32
    }
}

/// The words the tiny tokenizer knows, at their ids. The first four are the special tokens.
pub(crate) const VOCAB: [&str; 13] = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "search", "for", ":", "cheap", "index", "funds", "how", "to", "save",
];

pub(crate) fn tiny_config(layers: usize) -> Config {
    Config {
        vocab_size: VOCAB.len(),
        hidden_size: 16,
        num_hidden_layers: layers,
        num_attention_heads: 4,
        intermediate_size: 32,
        hidden_act: HiddenAct::Gelu,
        max_position_embeddings: 32,
        type_vocab_size: 2,
        layer_norm_eps: 1e-12,
        position_embedding_type: PositionEmbeddingType::Absolute,
        model_type: Some("bert".to_string()),
    }
}

/// Random weights for every tensor a BERT model with `config` loads, under Hugging Face's names.
pub(crate) fn tiny_weights(config: &Config, seed: u64) -> HashMap<String, Tensor> {
    let mut rng = SplitMix64::new(seed);
    let mut weights = HashMap::new();
    let mut put = |name: String, shape: &[usize], center: f32, spread: f32| {
        let n = shape.iter().product();
        let values: Vec<f32> = (0..n).map(|_| center + spread * rng.signed()).collect();
        weights.insert(name, Tensor::from_vec(values, shape, &Device::Cpu).unwrap());
    };
    let (h, inner) = (config.hidden_size, config.intermediate_size);
    put(
        "embeddings.word_embeddings.weight".into(),
        &[config.vocab_size, h],
        0.0,
        1.0,
    );
    put(
        "embeddings.position_embeddings.weight".into(),
        &[config.max_position_embeddings, h],
        0.0,
        0.5,
    );
    put(
        "embeddings.token_type_embeddings.weight".into(),
        &[config.type_vocab_size, h],
        0.0,
        0.5,
    );
    put("embeddings.LayerNorm.weight".into(), &[h], 1.0, 0.2);
    put("embeddings.LayerNorm.bias".into(), &[h], 0.0, 0.2);

    for layer in 0..config.num_hidden_layers {
        let p = format!("encoder.layer.{layer}");
        for part in ["query", "key", "value"] {
            put(format!("{p}.attention.self.{part}.weight"), &[h, h], 0.0, 0.5);
            put(format!("{p}.attention.self.{part}.bias"), &[h], 0.0, 0.1);
        }
        put(format!("{p}.attention.output.dense.weight"), &[h, h], 0.0, 0.5);
        put(format!("{p}.attention.output.dense.bias"), &[h], 0.0, 0.1);
        put(format!("{p}.attention.output.LayerNorm.weight"), &[h], 1.0, 0.2);
        put(format!("{p}.attention.output.LayerNorm.bias"), &[h], 0.0, 0.2);
        put(format!("{p}.intermediate.dense.weight"), &[inner, h], 0.0, 0.5);
        put(format!("{p}.intermediate.dense.bias"), &[inner], 0.0, 0.1);
        put(format!("{p}.output.dense.weight"), &[h, inner], 0.0, 0.5);
        put(format!("{p}.output.dense.bias"), &[h], 0.0, 0.1);
        put(format!("{p}.output.LayerNorm.weight"), &[h], 1.0, 0.2);
        put(format!("{p}.output.LayerNorm.bias"), &[h], 0.0, 0.2);
    }

    weights
}

/// A word-level tokenizer over `VOCAB` that lowercases, splits words from punctuation, and wraps the text in the
/// classification token [CLS] and the separator token [SEP]. Like all-MiniLM-L6-v2's, it pads every text to a fixed
/// length, 24 tokens, which the encoder must undo.
pub(crate) fn tokenizer_json() -> String {
    let special = |id: usize| {
        serde_json::json!({
            "id": id, "content": VOCAB[id], "single_word": false, "lstrip": false, "rstrip": false,
            "normalized": false, "special": true
        })
    };
    let vocab: serde_json::Map<String, serde_json::Value> = VOCAB
        .iter()
        .enumerate()
        .map(|(id, word)| (word.to_string(), id.into()))
        .collect();
    let json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": {
            "strategy": {"Fixed": 24}, "direction": "Right", "pad_to_multiple_of": null, "pad_id": 0,
            "pad_type_id": 0, "pad_token": "[PAD]"
        },
        "added_tokens": [special(0), special(1), special(2), special(3)],
        "normalizer": {"type": "Lowercase"},
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 0}}
            ],
            "pair": [
                {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 1}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 1}}
            ],
            "special_tokens": {
                "[CLS]": {"id": "[CLS]", "ids": [2], "tokens": ["[CLS]"]},
                "[SEP]": {"id": "[SEP]", "ids": [3], "tokens": ["[SEP]"]}
            }
        },
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"}
    });
    json.to_string()
}

/// A snapshot directory in the system's temporary folder, deleted on drop. The name holds the process id, so test
/// runs don't clash.
pub(crate) struct TempSnapshot(pub(crate) PathBuf);

impl Drop for TempSnapshot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// How a test snapshot pools, and its query prompt.
pub(crate) struct Kind {
    pub(crate) cls: bool,
    pub(crate) prompt: Option<&'static str>,
    pub(crate) layers: usize,
}

/// A sentence-transformers snapshot of a tiny random model, laid out like the Hugging Face cache's.
pub(crate) fn tiny_snapshot(name: &str, kind: &Kind) -> TempSnapshot {
    let dir = std::env::temp_dir().join(format!("specwave-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("1_Pooling")).unwrap();
    let snapshot = TempSnapshot(dir);
    let config = tiny_config(kind.layers);
    let write = |file: &str, json: serde_json::Value| std::fs::write(snapshot.0.join(file), json.to_string()).unwrap();
    write(
        "config.json",
        serde_json::json!({
            "architectures": ["BertModel"], "model_type": "bert", "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size, "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads, "intermediate_size": config.intermediate_size,
            "hidden_act": "gelu", "hidden_dropout_prob": 0.1, "max_position_embeddings": config.max_position_embeddings,
            "type_vocab_size": config.type_vocab_size, "layer_norm_eps": config.layer_norm_eps, "pad_token_id": 0,
            "position_embedding_type": "absolute"
        }),
    );
    write(
        "modules.json",
        serde_json::json!([
            {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
            {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
            {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"}
        ]),
    );
    write(
        "1_Pooling/config.json",
        serde_json::json!({
            "word_embedding_dimension": config.hidden_size, "pooling_mode_cls_token": kind.cls,
            "pooling_mode_mean_tokens": !kind.cls, "pooling_mode_max_tokens": false,
            "pooling_mode_mean_sqrt_len_tokens": false
        }),
    );
    write(
        "sentence_bert_config.json",
        serde_json::json!({"max_seq_length": 20, "do_lower_case": false}),
    );

    if let Some(prompt) = kind.prompt {
        write(
            "config_sentence_transformers.json",
            serde_json::json!({"prompts": {"query": prompt}, "default_prompt_name": null}),
        );
    }

    std::fs::write(snapshot.0.join("tokenizer.json"), tokenizer_json()).unwrap();
    candle_core::safetensors::save(&tiny_weights(&config, 7), snapshot.0.join("model.safetensors")).unwrap();
    snapshot
}

/// The values of a tensor, flattened.
pub(crate) fn values(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1().unwrap()
}

pub(crate) fn assert_close(found: &[f32], expected: &[f32], tolerance: f32, what: &str) {
    assert_eq!(found.len(), expected.len(), "{what}: lengths");

    for (i, (&f, &e)) in found.iter().zip(expected).enumerate() {
        assert!(
            (f - e).abs() <= tolerance * (1.0 + e.abs()),
            "{what}: value {i} is {f}, expected {e}"
        );
    }
}
