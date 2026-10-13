//! Tiny random models of each network, BERT (Bidirectional Encoder Representations from Transformers), NomicBERT and
//! Qwen3, their tokenizer, and snapshot directories that hold them, for the encoder's unit tests.

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{Device, Tensor};

use super::bert::{Config, HiddenAct, PositionEmbeddingType};
use super::{nomic_bert, qwen3};

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
/// length, 24 tokens, which the encoder must undo. Without `cls` it ends each text with [SEP] alone, the way
/// Qwen3-Embedding's tokenizer ends it with its end-of-text token.
pub(crate) fn tokenizer_json(cls: bool) -> String {
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
    let token = |kind: &str, id: &str, type_id: u32| serde_json::json!({kind: {"id": id, "type_id": type_id}});
    let (mut single, mut pair) = (
        vec![token("Sequence", "A", 0), token("SpecialToken", "[SEP]", 0)],
        vec![
            token("Sequence", "A", 0),
            token("SpecialToken", "[SEP]", 0),
            token("Sequence", "B", 1),
            token("SpecialToken", "[SEP]", 1),
        ],
    );

    if cls {
        single.insert(0, token("SpecialToken", "[CLS]", 0));
        pair.insert(0, token("SpecialToken", "[CLS]", 0));
    }

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
            "single": single,
            "pair": pair,
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

/// The network of a test snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Net {
    Bert,
    NomicBert,
    Qwen3,
}

/// A sentence-transformers snapshot of a tiny random BERT model, laid out like the Hugging Face cache's.
pub(crate) fn tiny_snapshot(name: &str, kind: &Kind) -> TempSnapshot {
    let pooling = if kind.cls { "cls" } else { "mean" };
    tiny_net_snapshot(name, Net::Bert, kind.layers, pooling, kind.prompt)
}

/// A snapshot like `tiny_snapshot`'s of a tiny random model of `net`, with `layers` layers, that pools by `pooling`,
/// eg. "lasttoken", and puts `prompt` in front of each query. Qwen3's tokenizer ends each text with [SEP] alone, as
/// Qwen3-Embedding's ends it with its end-of-text token.
pub(crate) fn tiny_net_snapshot(
    name: &str,
    net: Net,
    layers: usize,
    pooling: &str,
    prompt: Option<&str>,
) -> TempSnapshot {
    let dir = std::env::temp_dir().join(format!("specwave-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("1_Pooling")).unwrap();
    let snapshot = TempSnapshot(dir);
    let write = |file: &str, json: serde_json::Value| std::fs::write(snapshot.0.join(file), json.to_string()).unwrap();
    let (config, weights, dim) = match net {
        Net::Bert => {
            let config = tiny_config(layers);
            let json = serde_json::json!({
                "architectures": ["BertModel"], "model_type": "bert", "vocab_size": config.vocab_size,
                "hidden_size": config.hidden_size, "num_hidden_layers": config.num_hidden_layers,
                "num_attention_heads": config.num_attention_heads, "intermediate_size": config.intermediate_size,
                "hidden_act": "gelu", "hidden_dropout_prob": 0.1,
                "max_position_embeddings": config.max_position_embeddings, "type_vocab_size": config.type_vocab_size,
                "layer_norm_eps": config.layer_norm_eps, "pad_token_id": 0, "position_embedding_type": "absolute"
            });
            (json, tiny_weights(&config, 7), config.hidden_size)
        }
        Net::NomicBert => {
            let c = tiny_nomic_config(layers);
            let json = serde_json::json!({
                "architectures": ["NomicBertModel"], "model_type": "nomic_bert", "vocab_size": c.vocab_size,
                "n_embd": c.n_embd, "n_head": c.n_head, "n_layer": c.n_layer, "n_inner": c.n_inner,
                "n_positions": c.n_positions, "max_position_embeddings": 16, "type_vocab_size": c.type_vocab_size,
                "layer_norm_epsilon": c.layer_norm_epsilon, "rotary_emb_fraction": c.rotary_emb_fraction,
                "rotary_emb_base": c.rotary_emb_base, "rotary_emb_interleaved": c.rotary_emb_interleaved,
                "rotary_scaling_factor": null, "rotary_emb_scale_base": null, "qkv_proj_bias": c.qkv_proj_bias,
                "mlp_fc1_bias": c.mlp_fc1_bias, "mlp_fc2_bias": c.mlp_fc2_bias,
                "activation_function": c.activation_function, "prenorm": c.prenorm, "causal": false
            });
            (json, tiny_nomic_weights(&c, 7), c.n_embd)
        }
        Net::Qwen3 => {
            let c = tiny_qwen3_config(layers);
            let json = serde_json::json!({
                "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "vocab_size": c.vocab_size,
                "hidden_size": c.hidden_size, "intermediate_size": c.intermediate_size,
                "num_hidden_layers": c.num_hidden_layers, "num_attention_heads": c.num_attention_heads,
                "num_key_value_heads": c.num_key_value_heads, "head_dim": c.head_dim,
                "attention_bias": c.attention_bias, "max_position_embeddings": c.max_position_embeddings,
                "rope_theta": c.rope_theta, "rms_norm_eps": c.rms_norm_eps, "hidden_act": "silu",
                "use_sliding_window": false, "sliding_window": null, "rope_scaling": null,
                "tie_word_embeddings": true
            });
            (json, tiny_qwen3_weights(&c, 7), c.hidden_size)
        }
    };
    write("config.json", config);
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
            "word_embedding_dimension": dim, "pooling_mode_cls_token": pooling == "cls",
            "pooling_mode_mean_tokens": pooling == "mean", "pooling_mode_max_tokens": false,
            "pooling_mode_mean_sqrt_len_tokens": false, "pooling_mode_lasttoken": pooling == "lasttoken"
        }),
    );
    write(
        "sentence_bert_config.json",
        serde_json::json!({"max_seq_length": 20, "do_lower_case": false}),
    );

    if let Some(prompt) = prompt {
        write(
            "config_sentence_transformers.json",
            serde_json::json!({"prompts": {"query": prompt}, "default_prompt_name": null}),
        );
    }

    std::fs::write(snapshot.0.join("tokenizer.json"), tokenizer_json(net != Net::Qwen3)).unwrap();
    candle_core::safetensors::save(&weights, snapshot.0.join("model.safetensors")).unwrap();
    snapshot
}

/// Random tensors drawn one after another from a seeded generator, for the tiny NomicBERT and Qwen3 models.
struct Draws {
    rng: SplitMix64,
    weights: HashMap<String, Tensor>,
}

impl Draws {
    fn new(seed: u64) -> Draws {
        Draws {
            rng: SplitMix64::new(seed),
            weights: HashMap::new(),
        }
    }

    /// A tensor of `shape` whose values are `center` plus a uniform draw in [-`spread`, `spread`).
    fn put(&mut self, name: impl Into<String>, shape: &[usize], center: f32, spread: f32) {
        let values: Vec<f32> = (0..shape.iter().product())
            .map(|_| center + spread * self.rng.signed())
            .collect();
        let tensor = Tensor::from_vec(values, shape, &Device::Cpu).unwrap();
        self.weights.insert(name.into(), tensor);
    }
}

/// A NomicBERT config with 16 values per token in 4 heads, so each head turns 2 pairs of values.
pub(crate) fn tiny_nomic_config(layers: usize) -> nomic_bert::Config {
    nomic_bert::Config {
        vocab_size: VOCAB.len(),
        n_embd: 16,
        n_head: 4,
        n_layer: layers,
        n_inner: 32,
        n_positions: 32,
        type_vocab_size: 2,
        layer_norm_epsilon: 1e-12,
        rotary_emb_fraction: 1.0,
        rotary_emb_base: 1000.0,
        rotary_emb_interleaved: false,
        qkv_proj_bias: false,
        mlp_fc1_bias: false,
        mlp_fc2_bias: false,
        activation_function: "swiglu".to_string(),
        prenorm: false,
        rotary_scaling_factor: None,
        rotary_emb_scale_base: None,
        model_type: Some("nomic_bert".to_string()),
    }
}

/// Random weights for every tensor a NomicBERT model with `config` loads, under nomic-embed-text-v1.5's names.
pub(crate) fn tiny_nomic_weights(config: &nomic_bert::Config, seed: u64) -> HashMap<String, Tensor> {
    let mut draws = Draws::new(seed);
    let (h, inner) = (config.n_embd, config.n_inner);
    draws.put("embeddings.word_embeddings.weight", &[config.vocab_size, h], 0.0, 1.0);
    draws.put(
        "embeddings.token_type_embeddings.weight",
        &[config.type_vocab_size, h],
        0.0,
        0.5,
    );
    draws.put("emb_ln.weight", &[h], 1.0, 0.2);
    draws.put("emb_ln.bias", &[h], 0.0, 0.2);

    for layer in 0..config.n_layer {
        let p = format!("encoder.layers.{layer}");
        draws.put(format!("{p}.attn.Wqkv.weight"), &[3 * h, h], 0.0, 0.5);
        draws.put(format!("{p}.attn.out_proj.weight"), &[h, h], 0.0, 0.5);
        draws.put(format!("{p}.mlp.fc11.weight"), &[inner, h], 0.0, 0.5);
        draws.put(format!("{p}.mlp.fc12.weight"), &[inner, h], 0.0, 0.5);
        draws.put(format!("{p}.mlp.fc2.weight"), &[h, inner], 0.0, 0.5);
        for norm in ["norm1", "norm2"] {
            draws.put(format!("{p}.{norm}.weight"), &[h], 1.0, 0.2);
            draws.put(format!("{p}.{norm}.bias"), &[h], 0.0, 0.2);
        }
    }

    draws.weights
}

/// A Qwen3 config with 16 values per token and 4 query heads of 8 values that share 2 key and value heads. So the
/// attention is wider than the state, as in Qwen3-Embedding-0.6B.
pub(crate) fn tiny_qwen3_config(layers: usize) -> qwen3::Config {
    qwen3::Config {
        vocab_size: VOCAB.len(),
        hidden_size: 16,
        intermediate_size: 32,
        num_hidden_layers: layers,
        num_attention_heads: 4,
        head_dim: 8,
        attention_bias: false,
        num_key_value_heads: 2,
        max_position_embeddings: 32,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        hidden_act: candle_nn::Activation::Silu,
        use_sliding_window: false,
        rope_scaling: None,
    }
}

/// Random weights for every tensor a Qwen3 model with `config` loads, under Qwen3-Embedding's names.
pub(crate) fn tiny_qwen3_weights(config: &qwen3::Config, seed: u64) -> HashMap<String, Tensor> {
    let mut draws = Draws::new(seed);
    let (h, inner, head) = (config.hidden_size, config.intermediate_size, config.head_dim);
    let (q, kv) = (config.num_attention_heads * head, config.num_key_value_heads * head);
    draws.put("embed_tokens.weight", &[config.vocab_size, h], 0.0, 1.0);

    for layer in 0..config.num_hidden_layers {
        let p = format!("layers.{layer}");
        draws.put(format!("{p}.input_layernorm.weight"), &[h], 1.0, 0.2);
        draws.put(format!("{p}.post_attention_layernorm.weight"), &[h], 1.0, 0.2);
        draws.put(format!("{p}.self_attn.q_proj.weight"), &[q, h], 0.0, 0.5);
        draws.put(format!("{p}.self_attn.k_proj.weight"), &[kv, h], 0.0, 0.5);
        draws.put(format!("{p}.self_attn.v_proj.weight"), &[kv, h], 0.0, 0.5);
        draws.put(format!("{p}.self_attn.o_proj.weight"), &[h, q], 0.0, 0.5);
        draws.put(format!("{p}.self_attn.q_norm.weight"), &[head], 1.0, 0.2);
        draws.put(format!("{p}.self_attn.k_norm.weight"), &[head], 1.0, 0.2);
        draws.put(format!("{p}.mlp.gate_proj.weight"), &[inner, h], 0.0, 0.5);
        draws.put(format!("{p}.mlp.up_proj.weight"), &[inner, h], 0.0, 0.5);
        draws.put(format!("{p}.mlp.down_proj.weight"), &[h, inner], 0.0, 0.5);
    }

    draws.put("norm.weight", &[h], 1.0, 0.2);
    draws.weights
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

/// The networks' math written out plainly in f64 on the host, one text at a time, to check the networks against.
pub(crate) mod host {
    use std::collections::HashMap;

    use candle_core::Tensor;

    /// Rows of values, eg. one per token.
    pub(crate) type Rows = Vec<Vec<f64>>;

    /// The tensor `name` of `weights` as rows of its last dimension.
    pub(crate) fn rows(weights: &HashMap<String, Tensor>, name: &str) -> Rows {
        let tensor = &weights[name];
        let width = *tensor.dims().last().unwrap();
        super::values(tensor)
            .chunks(width)
            .map(|row| row.iter().map(|&v| f64::from(v)).collect())
            .collect()
    }

    /// The tensor `name` of `weights` as one row.
    pub(crate) fn row(weights: &HashMap<String, Tensor>, name: &str) -> Vec<f64> {
        rows(weights, name).concat()
    }

    pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// x·wᵀ for each row x, where `w` has a row per output.
    pub(crate) fn linear(x: &Rows, w: &Rows) -> Rows {
        x.iter().map(|r| w.iter().map(|out| dot(r, out)).collect()).collect()
    }

    pub(crate) fn add(a: &Rows, b: &Rows) -> Rows {
        a.iter()
            .zip(b)
            .map(|(x, y)| x.iter().zip(y).map(|(p, q)| p + q).collect())
            .collect()
    }

    /// (x − mean)/√(variance + `eps`)·`weight` + `bias` for each row x.
    pub(crate) fn layer_norm(x: &Rows, weight: &[f64], bias: &[f64], eps: f64) -> Rows {
        x.iter()
            .map(|r| {
                let mean = r.iter().sum::<f64>() / r.len() as f64;
                let variance = r.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / r.len() as f64;
                let scale = 1.0 / (variance + eps).sqrt();
                r.iter()
                    .zip(weight.iter().zip(bias))
                    .map(|(v, (w, b))| (v - mean) * scale * w + b)
                    .collect()
            })
            .collect()
    }

    /// x/√(mean(x²) + `eps`)·`weight`.
    pub(crate) fn rms_norm(x: &[f64], weight: &[f64], eps: f64) -> Vec<f64> {
        let scale = 1.0 / (dot(x, x) / x.len() as f64 + eps).sqrt();
        x.iter().zip(weight).map(|(v, w)| v * scale * w).collect()
    }

    /// Turns each pair (x[i], x[i + d/2]) of one head's d values by the angle position·`base`^(−2i/d), the rotary
    /// embedding of GPT-NeoX, which Hugging Face writes with `rotate_half`.
    pub(crate) fn rotate(x: &[f64], position: usize, base: f64) -> Vec<f64> {
        let half = x.len() / 2;
        let mut out = x.to_vec();

        for i in 0..half {
            let angle = position as f64 * base.powf(-2.0 * i as f64 / x.len() as f64);
            let (a, b) = (x[i], x[i + half]);
            out[i] = a * angle.cos() - b * angle.sin();
            out[i + half] = b * angle.cos() + a * angle.sin();
        }

        out
    }

    /// One head's attention: each query row averages the value rows by softmax(q·k/√d) over the key rows it sees,
    /// all of them, or with `causal` only those up to its own position.
    pub(crate) fn attend(q: &Rows, k: &Rows, v: &Rows, causal: bool) -> Rows {
        let scale = 1.0 / (q[0].len() as f64).sqrt();
        q.iter()
            .enumerate()
            .map(|(i, qi)| {
                let seen = if causal { i + 1 } else { k.len() };
                let scores: Vec<f64> = k[..seen].iter().map(|kj| dot(qi, kj) * scale).collect();
                let top = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
                let total: f64 = weights.iter().sum();
                (0..v[0].len())
                    .map(|c| weights.iter().zip(v).map(|(w, vj)| w * vj[c]).sum::<f64>() / total)
                    .collect()
            })
            .collect()
    }

    pub(crate) fn silu(x: f64) -> f64 {
        x / (1.0 + (-x).exp())
    }

    pub(crate) fn flat(rows: &Rows) -> Vec<f32> {
        rows.iter().flatten().map(|&v| v as f32).collect()
    }
}
