// Copyright (c) the candle authors (https://github.com/huggingface/candle)
//
// Permission is hereby granted, free of charge, to any
// person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the
// Software without restriction, including without
// limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software
// is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice
// shall be included in all copies or substantial portions
// of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
// ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
// TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
// PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
// SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

//! The NomicBERT network of nomic-embed-text-v1.5, with a way to watch each layer.
//!
//! NomicBERT is BERT (Bidirectional Encoder Representations from Transformers) with rotary position embeddings, one
//! projection for the query, key and value, and a feed-forward network gated by SiLU (sigmoid linear unit), called
//! SwiGLU. This file is a copy of `src/models/nomic_bert.rs` from candle-transformers 0.11.0
//! (https://github.com/huggingface/candle, commit 31f35b1), written by the candle authors at Hugging Face. candle comes
//! under the MIT (Massachusetts Institute of Technology) license or the Apache License 2.0, at your option. This copy
//! uses the MIT license, whose text heads the file. It differs from the original in five ways.
//! - `NomicBertModel::forward_layers` runs the embeddings and the first n transformer layers, and shows each layer's
//!   output to a callback. The output of the embedding layer norm is layer 0 and transformer layer i's output is layer
//!   i + 1, as in Hugging Face's `hidden_states`.
//! - Without an attention mask it adds no mask, where the original adds a mask of zeros.
//! - Only the post-norm block is kept, the one nomic-embed-text-v1.5 has. `Config::unsupported` names the settings
//!   this copy can't run, eg. a rotary embedding over part of each head, which the original only fails on at the
//!   first text.
//! - The config has no defaults, so a field that a snapshot leaves out fails to load, and the token type ids are
//!   always given, as in `bert.rs`.
//! - The tracing spans, the pooling helpers and the config fields that inference never reads are gone, and the layers
//!   come straight from candle-nn.

use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder, embedding, layer_norm, linear_b};
use serde::Deserialize;

/// The fields of a Hugging Face NomicBERT `config.json` that inference reads.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub n_inner: usize,
    /// The most tokens a text may have. The rotary tables hold this many positions.
    pub n_positions: usize,
    pub type_vocab_size: usize,
    pub layer_norm_epsilon: f64,
    pub rotary_emb_fraction: f64,
    pub rotary_emb_base: f64,
    pub rotary_emb_interleaved: bool,
    pub qkv_proj_bias: bool,
    pub mlp_fc1_bias: bool,
    pub mlp_fc2_bias: bool,
    pub activation_function: String,
    pub prenorm: bool,
    /// Set for a rotary embedding that stretches past the trained length.
    #[serde(default)]
    pub rotary_scaling_factor: Option<f64>,
    /// Set for a rotary embedding that also scales each position.
    #[serde(default)]
    pub rotary_emb_scale_base: Option<f64>,
    pub model_type: Option<String>,
}

impl Config {
    fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    /// The setting of this config that this copy can't run, if any.
    pub fn unsupported(&self) -> Option<&'static str> {
        if self.activation_function != "swiglu" {
            Some("a NomicBERT feed-forward network other than SwiGLU")
        } else if self.rotary_emb_fraction != 1.0 {
            Some("a rotary embedding over part of each head")
        } else if self.rotary_scaling_factor.is_some() || self.rotary_emb_scale_base.is_some() {
            Some("a scaled rotary embedding")
        } else if self.prenorm {
            Some("a NomicBERT block that norms before attention")
        } else {
            None
        }
    }
}

// Precomputed cos/sin tables for rotary position embeddings.
// Shared across all attention layers since they use identical frequencies.
#[derive(Clone, Debug)]
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    interleaved: bool,
}

impl RotaryEmbedding {
    fn new(dim: usize, max_seq_len: usize, base: f64, interleaved: bool, device: &Device) -> Result<Self> {
        let half_dim = dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1f32 / (base as f32).powf(2.0 * i as f32 / dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
        let positions = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = positions.matmul(&inv_freq.unsqueeze(0)?)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        Ok(Self { cos, sin, interleaved })
    }

    /// Apply rotary embeddings to x of shape (batch, n_heads, seq_len, head_dim).
    /// Dispatches to interleaved (GPT-J) or non-interleaved (GPT-NeoX) style
    /// based on the model config.
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let cos = self.cos.to_dtype(x.dtype())?;
        let sin = self.sin.to_dtype(x.dtype())?;
        if self.interleaved {
            candle_nn::rotary_emb::rope_i(x, &cos, &sin)
        } else {
            candle_nn::rotary_emb::rope(x, &cos, &sin)
        }
    }
}

// Word embeddings + token type embeddings.
// No position embeddings since NomicBert uses rotary embeddings.
struct NomicBertEmbeddings {
    word_embeddings: Embedding,
    token_type_embeddings: Embedding,
}

impl NomicBertEmbeddings {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let word_embeddings = embedding(config.vocab_size, config.n_embd, vb.pp("word_embeddings"))?;
        let token_type_embeddings = embedding(config.type_vocab_size, config.n_embd, vb.pp("token_type_embeddings"))?;
        Ok(Self {
            word_embeddings,
            token_type_embeddings,
        })
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let embeddings = self.word_embeddings.forward(input_ids)?;
        embeddings + self.token_type_embeddings.forward(token_type_ids)?
    }
}

// Self-attention with fused QKV projection and rotary embeddings.
struct NomicBertAttention {
    wqkv: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    n_embd: usize,
}

impl NomicBertAttention {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let wqkv = linear_b(config.n_embd, 3 * config.n_embd, config.qkv_proj_bias, vb.pp("Wqkv"))?;
        let out_proj = linear_b(config.n_embd, config.n_embd, config.qkv_proj_bias, vb.pp("out_proj"))?;
        Ok(Self {
            wqkv,
            out_proj,
            num_heads: config.n_head,
            head_dim: config.head_dim(),
            n_embd: config.n_embd,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        rotary_emb: &RotaryEmbedding,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;

        let qkv = self.wqkv.forward(hidden_states)?;
        let q = qkv.narrow(D::Minus1, 0, self.n_embd)?;
        let k = qkv.narrow(D::Minus1, self.n_embd, self.n_embd)?;
        let v = qkv.narrow(D::Minus1, 2 * self.n_embd, self.n_embd)?;

        // Reshape to (batch, seq_len, num_heads, head_dim) then transpose
        // to (batch, num_heads, seq_len, head_dim) for attention + rope.
        let q = q
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = rotary_emb.apply(&q)?;
        let k = rotary_emb.apply(&k)?;

        let scale = (self.head_dim as f64).sqrt();
        let attn_scores = (q.matmul(&k.t()?)? / scale)?;
        let attn_scores = match attention_mask {
            Some(mask) => attn_scores.broadcast_add(mask)?,
            None => attn_scores,
        };
        let attn_probs = candle_nn::ops::softmax_last_dim(&attn_scores)?;

        let attn_output = attn_probs.matmul(&v.contiguous()?)?;
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?;
        let attn_output = attn_output.flatten_from(D::Minus2)?;

        self.out_proj.forward(&attn_output)
    }
}

// SwiGLU feed-forward network.
// Two parallel projections (fc11 for value, fc12 for gate with SiLU),
// element-wise multiply, then project back.
struct NomicBertSwiGLU {
    fc11: Linear,
    fc12: Linear,
    fc2: Linear,
}

impl NomicBertSwiGLU {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let fc11 = linear_b(config.n_embd, config.n_inner, config.mlp_fc1_bias, vb.pp("fc11"))?;
        let fc12 = linear_b(config.n_embd, config.n_inner, config.mlp_fc1_bias, vb.pp("fc12"))?;
        let fc2 = linear_b(config.n_inner, config.n_embd, config.mlp_fc2_bias, vb.pp("fc2"))?;
        Ok(Self { fc11, fc12, fc2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let y = self.fc11.forward(xs)?;
        let gate = self.fc12.forward(xs)?.silu()?;
        self.fc2.forward(&(y * gate)?)
    }
}

// Transformer block: attention → norm → MLP → norm (post-norm).
struct NomicBertBlock {
    attn: NomicBertAttention,
    mlp: NomicBertSwiGLU,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl NomicBertBlock {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let attn = NomicBertAttention::load(vb.pp("attn"), config)?;
        let mlp = NomicBertSwiGLU::load(vb.pp("mlp"), config)?;
        let norm1 = layer_norm(config.n_embd, config.layer_norm_epsilon, vb.pp("norm1"))?;
        let norm2 = layer_norm(config.n_embd, config.layer_norm_epsilon, vb.pp("norm2"))?;
        Ok(Self {
            attn,
            mlp,
            norm1,
            norm2,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        rotary_emb: &RotaryEmbedding,
    ) -> Result<Tensor> {
        let attn_out = self.attn.forward(hidden_states, attention_mask, rotary_emb)?;
        let hidden_states = self.norm1.forward(&(hidden_states + attn_out)?)?;
        let mlp_out = self.mlp.forward(&hidden_states)?;
        self.norm2.forward(&(hidden_states + mlp_out)?)
    }
}

struct NomicBertEncoder {
    layers: Vec<NomicBertBlock>,
    rotary_emb: RotaryEmbedding,
}

impl NomicBertEncoder {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let layers = (0..config.n_layer)
            .map(|i| NomicBertBlock::load(vb.pp(format!("layers.{i}")), config))
            .collect::<Result<Vec<_>>>()?;
        let rotary_emb = RotaryEmbedding::new(
            config.head_dim(),
            config.n_positions,
            config.rotary_emb_base,
            config.rotary_emb_interleaved,
            vb.device(),
        )?;
        Ok(Self { layers, rotary_emb })
    }
}

pub struct NomicBertModel {
    embeddings: NomicBertEmbeddings,
    emb_ln: LayerNorm,
    encoder: NomicBertEncoder,
}

impl NomicBertModel {
    /// Loads the weights under `embeddings.`, `emb_ln.` and `encoder.`, or under `<model_type>.` for checkpoints saved
    /// with a task head, eg. `nomic_bert.encoder.layers.0`.
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let load_inner = |vb: VarBuilder| -> Result<Self> {
            let embeddings = NomicBertEmbeddings::load(vb.pp("embeddings"), config)?;
            let emb_ln = layer_norm(config.n_embd, config.layer_norm_epsilon, vb.pp("emb_ln"))?;
            let encoder = NomicBertEncoder::load(vb.pp("encoder"), config)?;
            Ok(Self {
                embeddings,
                emb_ln,
                encoder,
            })
        };

        // Try without prefix, then with model_type prefix (e.g. "nomic_bert").
        load_inner(vb.clone()).or_else(|err| {
            if let Some(model_type) = &config.model_type {
                load_inner(vb.pp(model_type)).map_err(|_| err)
            } else {
                Err(err)
            }
        })
    }

    /// Runs the embeddings and then the first `layers` transformer layers, and returns the last state, of shape
    /// (batch, tokens, hidden size). `observe(l, state)` sees every state on the way, with l = 0 for the output of the
    /// embedding layer norm and l = i + 1 for the output of transformer layer i. `attention_mask` holds 1 for each real
    /// token and 0 for each padding token, with shape (batch, tokens). Without it every token counts.
    pub fn forward_layers(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        layers: usize,
        observe: &mut dyn FnMut(usize, &Tensor) -> Result<()>,
    ) -> Result<Tensor> {
        let hidden_states = self.embeddings.forward(input_ids, token_type_ids)?;
        let mut hidden_states = self.emb_ln.forward(&hidden_states)?;
        let attention_mask = match attention_mask {
            Some(mask) => Some(get_extended_attention_mask(mask, hidden_states.dtype())?),
            None => None,
        };
        observe(0, &hidden_states)?;

        for (i, layer) in self.encoder.layers.iter().take(layers).enumerate() {
            hidden_states = layer.forward(&hidden_states, attention_mask.as_ref(), &self.encoder.rotary_emb)?;
            observe(i + 1, &hidden_states)?;
        }

        Ok(hidden_states)
    }
}

/// Convert an attention mask from (batch, seq_len) with 1=attend/0=pad
/// to (batch, 1, 1, seq_len) with 0=attend/-1e4=pad, suitable for
/// adding to attention scores before softmax.
fn get_extended_attention_mask(attention_mask: &Tensor, dtype: DType) -> Result<Tensor> {
    let mask = attention_mask.unsqueeze(1)?.unsqueeze(1)?;
    let on_true = mask.zeros_like()?.to_dtype(dtype)?;
    let on_false = Tensor::new(-1e4f32, mask.device())?
        .to_dtype(dtype)?
        .broadcast_as(mask.shape())?;
    mask.where_cond(&on_true, &on_false)
}

#[cfg(test)]
#[path = "../../tests/unit/encoder/nomic_bert.rs"]
mod tests;
