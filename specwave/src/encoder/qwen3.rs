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

//! The Qwen3 network of Qwen3-Embedding, with a way to watch each layer.
//!
//! Qwen3 is a decoder, so each token attends only to itself and the tokens before it. Its attention shares each key
//! and value head among several query heads and norms each head's query and key. This file is a copy of
//! `src/models/qwen3.rs` from candle-transformers 0.11.0 (https://github.com/huggingface/candle, commit 31f35b1),
//! written by the candle authors at Hugging Face. candle comes under the MIT (Massachusetts Institute of Technology)
//! license or the Apache License 2.0, at your option. This copy uses the MIT license, whose text heads the file. It
//! differs from the original in five ways.
//! - `Model::forward_layers` runs the token embedding and the first n decoder layers, and shows each layer's output to
//!   a callback. The embedding output is layer 0 and decoder layer i's output is layer i + 1, except that the last
//!   layer comes after the final norm, as in Hugging Face's `hidden_states`. A run that stops earlier skips the norm.
//! - It runs each text from its start, so the key and value cache, the offsets and the language model head are gone,
//!   and the model needs no `&mut`.
//! - The weights load as Qwen3-Embedding stores them, under `embed_tokens.`, `layers.` and `norm.`, or under `model.`
//!   for checkpoints saved with the language model head.
//! - It runs in f32 only, so the processor's attention calls candle's f32 kernel straight. `Config::unsupported` names
//!   the settings this copy can't run, eg. a sliding window.
//! - The tracing spans, the CUDA (Compute Unified Device Architecture) flash attention and the config fields that
//!   inference never reads are gone, and the layers come straight from candle-nn.

use std::sync::Arc;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::attention::{AttnMask, flash_attn};
use candle_nn::{Activation, Embedding, Linear, RmsNorm, VarBuilder, embedding, linear_b, linear_no_bias, rms_norm};
use serde::Deserialize;

/// The fields of a Hugging Face Qwen3 `config.json` that inference reads.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub attention_bias: bool,
    pub num_key_value_heads: usize,
    /// The most tokens a text may have. The rotary tables hold this many positions.
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub hidden_act: Activation,
    #[serde(default)]
    pub use_sliding_window: bool,
    /// Set for a rotary embedding stretched past the trained length.
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
}

impl Config {
    /// The setting of this config that this copy can't run, if any.
    pub fn unsupported(&self) -> Option<&'static str> {
        if self.use_sliding_window {
            Some("attention over a sliding window")
        } else if self.rope_scaling.is_some() {
            Some("a scaled rotary embedding")
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Qwen3RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl Qwen3RotaryEmbedding {
    pub(crate) fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    /// Apply RoPE (q, k shape: B x H x L x D)
    pub(crate) fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Qwen3MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Qwen3MLP {
    pub(crate) fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for Qwen3MLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let lhs = x.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = x.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Qwen3Attention {
    // projections
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    // norms
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    // hyper params
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    // utils
    rotary_emb: Arc<Qwen3RotaryEmbedding>,
}

impl Qwen3Attention {
    pub(crate) fn new(cfg: &Config, rotary_emb: Arc<Qwen3RotaryEmbedding>, vb: VarBuilder) -> Result<Self> {
        if cfg.use_sliding_window {
            candle_core::bail!("sliding window is not supported")
        }

        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;

        let q_proj = linear_b(
            cfg.hidden_size,
            num_heads * head_dim,
            cfg.attention_bias,
            vb.pp("q_proj"),
        )?;
        let k_proj = linear_b(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            cfg.attention_bias,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_b(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            cfg.attention_bias,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_b(
            num_heads * head_dim,
            cfg.hidden_size,
            cfg.attention_bias,
            vb.pp("o_proj"),
        )?;

        let q_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        // Necessary because the hidden_size in the config isn't always accurate
        let hidden_size = head_dim * cfg.num_attention_heads;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size,
            rotary_emb,
        })
    }

    pub(crate) fn forward(&self, x: &Tensor, attn_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        // 1. Proj
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // 2. Reshape: (B, L, H, D) -> (B, H, L, D)
        let q = q.reshape((b, l, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b, l, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b, l, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;

        // 3. Per-head RMSNorm
        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;
        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b, self.num_heads, l, self.head_dim))?;
        let k = k_flat.reshape((b, self.num_kv_heads, l, self.head_dim))?;

        // 4. RoPE
        let (q, k) = self.rotary_emb.apply(&q, &k)?;

        // 5. Attention dispatch: the fused CPU kernel on the processor, matmul attention elsewhere
        if x.device().is_cpu() {
            return self.forward_cpu_flash_attn(&q, &k, &v, b, l);
        }

        self.forward_standard_attn(&q, &k, &v, attn_mask, b, l)
    }

    /// CPU flash attention - optimized fused kernel for CPU
    ///
    /// The `flash_attn` dispatcher in candle-nn automatically selects:
    /// - B=1: single-batch optimized kernels (direct slice access)
    /// - B>1: packed varlen path (avoids batch-dim stride overhead)
    fn forward_cpu_flash_attn(&self, q: &Tensor, k: &Tensor, v: &Tensor, b: usize, l: usize) -> Result<Tensor> {
        // CPU flash attention expects (B, S, H, D) format
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let ctx = flash_attn::<f32>(&q, &k, &v, scale, AttnMask::causal(), None, None)?;

        // Output from CPU flash attention is (B, H, S, D), transpose to (B, S, H, D)
        let ctx = ctx.transpose(1, 2)?;

        ctx.reshape((b, l, self.hidden_size))?.apply(&self.o_proj)
    }

    /// Standard matmul-based attention (works on any device)
    fn forward_standard_attn(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attn_mask: Option<&Tensor>,
        b: usize,
        l: usize,
    ) -> Result<Tensor> {
        // GQA repeat_kv
        let k = repeat_kv(k.clone(), self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v.clone(), self.num_kv_groups)?.contiguous()?;

        // Attention score
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = attn_mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?; // (B, H, L, D)

        // Output proj
        ctx.transpose(1, 2)?
            .reshape((b, l, self.hidden_size))?
            .apply(&self.o_proj)
    }
}

/// Repeats each key and value head `n_rep` times, so query head h reads key and value head h / `n_rep`. This is
/// `repeat_kv` from candle-transformers' `utils.rs`.
pub(crate) fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
        // Using cat is faster than a broadcast as it avoids going through a potentially
        // strided copy.
        // https://github.com/huggingface/candle/pull/2043
        Tensor::cat(&vec![&xs; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Qwen3Attention,
    mlp: Qwen3MLP,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl DecoderLayer {
    fn new(cfg: &Config, rotary: Arc<Qwen3RotaryEmbedding>, vb: VarBuilder) -> Result<Self> {
        let self_attn = Qwen3Attention::new(cfg, rotary, vb.pp("self_attn"))?;
        let mlp = Qwen3MLP::new(cfg, vb.pp("mlp"))?;
        let ln1 = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let ln2 = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?;
        Ok(Self {
            self_attn,
            mlp,
            ln1,
            ln2,
        })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.self_attn.forward(&h, mask)?;
        let x = (x + h)?;
        let h2 = self.ln2.forward(&x)?;
        let h2 = h2.apply(&self.mlp)?;
        x + h2
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
}

impl Model {
    /// Loads the weights under `embed_tokens.`, `layers.` and `norm.`, or under `model.` for checkpoints saved with the
    /// language model head, eg. `model.layers.0`.
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        Self::new(cfg, vb.clone()).or_else(|err| Self::new(cfg, vb.pp("model")).map_err(|_| err))
    }

    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        if vb.dtype() != DType::F32 {
            candle_core::bail!("this Qwen3 runs in f32 only, not {:?}", vb.dtype());
        }
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let rotary = Arc::new(Qwen3RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, rotary.clone(), vb_l.pp(i))?);
        }
        Ok(Self {
            embed_tokens,
            layers,
            norm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?,
        })
    }

    fn causal_mask(&self, b: usize, tgt: usize, device: &Device) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| (0..tgt).map(move |j| if j <= i { 0. } else { minf }))
            .collect();
        Tensor::from_slice(&mask, (b, 1, tgt, tgt), device)
    }

    /// Runs the token embedding and then the first `layers` decoder layers, and returns the last state, of shape
    /// (batch, tokens, hidden size). `observe(l, state)` sees every state on the way, with l = 0 for the embedding
    /// output and l = i + 1 for the output of decoder layer i. The last layer's output goes through the final norm
    /// before `observe` sees it, so a run of every layer returns the normed state that the embedding pools.
    pub fn forward_layers(
        &self,
        input_ids: &Tensor,
        layers: usize,
        observe: &mut dyn FnMut(usize, &Tensor) -> Result<()>,
    ) -> Result<Tensor> {
        let (b, l) = input_ids.dims2()?;
        let mut h = self.embed_tokens.forward(input_ids)?;

        // Build causal mask only for the standard attention fallback path.
        // The CPU flash kernel handles masking internally.
        let causal = if !input_ids.device().is_cpu() && l > 1 {
            Some(self.causal_mask(b, l, input_ids.device())?)
        } else {
            None
        };
        observe(0, &h)?;

        for (i, layer) in self.layers.iter().take(layers).enumerate() {
            h = layer.forward(&h, causal.as_ref())?;
            if i + 1 == self.layers.len() {
                h = self.norm.forward(&h)?;
            }
            observe(i + 1, &h)?;
        }

        Ok(h)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/encoder/qwen3.rs"]
mod tests;
