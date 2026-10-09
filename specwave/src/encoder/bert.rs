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

//! The BERT (Bidirectional Encoder Representations from Transformers) network, with a way to watch each layer.
//!
//! This file is a copy of `src/models/bert.rs` from candle-transformers 0.11.0 (https://github.com/huggingface/candle,
//! commit 31f35b1), written by the candle authors at Hugging Face. candle comes under the MIT (Massachusetts Institute
//! of Technology) license or the Apache License 2.0, at your option. This copy uses the MIT license, whose text heads
//! the file. It differs from the original in five ways.
//! - `BertModel::forward_layers` runs the embeddings and the first n transformer layers, and shows each layer's output
//!   to a callback. The embedding output is layer 0 and transformer layer i's output is layer i + 1.
//! - Without an attention mask it adds no mask, where the original adds a mask of zeros.
//! - Attention takes candle's fused softmax over the last dimension.
//! - On the processor (CPU), the exact GELU (Gaussian error linear unit) comes from whole-tensor ops, as `gelu_erf`
//!   says. Like candle's own, they run on one thread, but they vectorize, where candle's calls a scalar erf for each
//!   value.
//! - The tracing spans, the dropout that did nothing, the masked language model heads and the config fields that
//!   inference never reads are gone, and the layers come straight from candle-nn.

use std::f64::consts::{FRAC_1_SQRT_2, LN_2};

use candle_core::{D, DType, Module, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder, embedding, layer_norm, linear};
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HiddenAct {
    Gelu,
    GeluApproximate,
    Relu,
}

impl HiddenAct {
    fn forward(self, xs: &Tensor) -> Result<Tensor> {
        match self {
            // The exact form with the error function, as Hugging Face's "gelu". On the graphics processor candle's own
            // kernel is one pass over the tensor, where the ops of `gelu_erf` would each cost a kernel launch.
            HiddenAct::Gelu if xs.device().is_cpu() => gelu_erf(xs),
            HiddenAct::Gelu => xs.gelu_erf(),
            HiddenAct::GeluApproximate => xs.gelu(),
            HiddenAct::Relu => xs.relu(),
        }
    }
}

/// The coefficients c0 to c9 of the erfc approximation in [`gelu_erf`], from `erfcc` in Numerical Recipes in C, 2nd
/// edition, section 6.2.
const ERFC: [f64; 10] = [
    -1.26551223,
    1.00002368,
    0.37409196,
    0.09678418,
    -0.18628806,
    0.27886807,
    -1.13520398,
    1.48851587,
    -0.82215223,
    0.17087277,
];

/// The exact GELU of every value, x·Φ(x), where Φ is the standard normal distribution function, built from ops that
/// each run over the whole tensor. With a = |x| and z = a/√2, it is
///
/// ```text
/// x·Φ(x) = relu(x) − a·erfc(z)/2    erfc(z) = t·exp(−z² + c0 + c1·t + c2·t² + ... + c9·t⁹)    t = 1/(1 + z/2)
/// ```
///
/// where c0 to c9 are `ERFC`. This erfc has a relative error below 1.2e-7 for every z ≥ 0, so the result is accurate
/// in the negative tail too, where x·Φ(x) is tiny. The factor 1/2 joins the exponent as −ln 2.
fn gelu_erf(xs: &Tensor) -> Result<Tensor> {
    let a = xs.abs()?;
    let t = a.affine(0.5 * FRAC_1_SQRT_2, 1.0)?.recip()?;
    let mut p = t.affine(ERFC[9], ERFC[8])?;

    for &c in ERFC[1..8].iter().rev() {
        p = (p * &t)?.affine(1.0, c)?;
    }

    // −z² = −x²/2, so the exponent is p·t − x²/2 + c0 − ln 2.
    let rest = xs.sqr()?.affine(-0.5, ERFC[0] - LN_2)?;
    let half_erfc = (((p * &t)? + rest)?.exp()? * &t)?;
    xs.relu()? - (half_erfc * a)?
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PositionEmbeddingType {
    #[default]
    Absolute,
}

/// The fields of a Hugging Face BERT `config.json` that inference reads.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: HiddenAct,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub layer_norm_eps: f64,
    #[serde(default)]
    pub position_embedding_type: PositionEmbeddingType,
    pub model_type: Option<String>,
}

struct BertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
}

impl BertEmbeddings {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let word_embeddings = embedding(config.vocab_size, config.hidden_size, vb.pp("word_embeddings"))?;
        let position_embeddings = embedding(
            config.max_position_embeddings,
            config.hidden_size,
            vb.pp("position_embeddings"),
        )?;
        let token_type_embeddings = embedding(
            config.type_vocab_size,
            config.hidden_size,
            vb.pp("token_type_embeddings"),
        )?;
        let layer_norm = layer_norm(config.hidden_size, config.layer_norm_eps, vb.pp("LayerNorm"))?;
        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
        })
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let (_bsize, seq_len) = input_ids.dims2()?;
        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;
        let embeddings = (&input_embeddings + token_type_embeddings)?;
        let position_ids = (0..seq_len as u32).collect::<Vec<_>>();
        let position_ids = Tensor::new(&position_ids[..], input_ids.device())?;
        let embeddings = embeddings.broadcast_add(&self.position_embeddings.forward(&position_ids)?)?;
        self.layer_norm.forward(&embeddings)
    }
}

struct BertSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    num_attention_heads: usize,
    attention_head_size: usize,
}

impl BertSelfAttention {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;
        let hidden_size = config.hidden_size;
        let query = linear(hidden_size, all_head_size, vb.pp("query"))?;
        let value = linear(hidden_size, all_head_size, vb.pp("value"))?;
        let key = linear(hidden_size, all_head_size, vb.pp("key"))?;
        Ok(Self {
            query,
            key,
            value,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
        })
    }

    fn transpose_for_scores(&self, xs: &Tensor) -> Result<Tensor> {
        let mut new_x_shape = xs.dims().to_vec();
        new_x_shape.pop();
        new_x_shape.push(self.num_attention_heads);
        new_x_shape.push(self.attention_head_size);
        let xs = xs.reshape(new_x_shape.as_slice())?.transpose(1, 2)?;
        xs.contiguous()
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let query_layer = self.query.forward(hidden_states)?;
        let key_layer = self.key.forward(hidden_states)?;
        let value_layer = self.value.forward(hidden_states)?;
        let query_layer = self.transpose_for_scores(&query_layer)?;
        let key_layer = self.transpose_for_scores(&key_layer)?;
        let value_layer = self.transpose_for_scores(&value_layer)?;
        let attention_scores = query_layer.matmul(&key_layer.t()?)?;
        let attention_scores = (attention_scores / (self.attention_head_size as f64).sqrt())?;
        let attention_scores = match attention_mask {
            Some(mask) => attention_scores.broadcast_add(mask)?,
            None => attention_scores,
        };
        let attention_probs = candle_nn::ops::softmax_last_dim(&attention_scores)?;
        let context_layer = attention_probs.matmul(&value_layer)?;
        let context_layer = context_layer.transpose(1, 2)?.contiguous()?;
        context_layer.flatten_from(D::Minus2)
    }
}

struct BertSelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
}

impl BertSelfOutput {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(config.hidden_size, config.layer_norm_eps, vb.pp("LayerNorm"))?;
        Ok(Self { dense, layer_norm })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}

struct BertAttention {
    self_attention: BertSelfAttention,
    self_output: BertSelfOutput,
}

impl BertAttention {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let self_attention = BertSelfAttention::load(vb.pp("self"), config)?;
        let self_output = BertSelfOutput::load(vb.pp("output"), config)?;
        Ok(Self {
            self_attention,
            self_output,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let self_outputs = self.self_attention.forward(hidden_states, attention_mask)?;
        self.self_output.forward(&self_outputs, hidden_states)
    }
}

struct BertIntermediate {
    dense: Linear,
    intermediate_act: HiddenAct,
}

impl BertIntermediate {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = linear(config.hidden_size, config.intermediate_size, vb.pp("dense"))?;
        Ok(Self {
            dense,
            intermediate_act: config.hidden_act,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        self.intermediate_act.forward(&hidden_states)
    }
}

struct BertOutput {
    dense: Linear,
    layer_norm: LayerNorm,
}

impl BertOutput {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = linear(config.intermediate_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(config.hidden_size, config.layer_norm_eps, vb.pp("LayerNorm"))?;
        Ok(Self { dense, layer_norm })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}

struct BertLayer {
    attention: BertAttention,
    intermediate: BertIntermediate,
    output: BertOutput,
}

impl BertLayer {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let attention = BertAttention::load(vb.pp("attention"), config)?;
        let intermediate = BertIntermediate::load(vb.pp("intermediate"), config)?;
        let output = BertOutput::load(vb.pp("output"), config)?;
        Ok(Self {
            attention,
            intermediate,
            output,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let attention_output = self.attention.forward(hidden_states, attention_mask)?;
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        self.output.forward(&intermediate_output, &attention_output)
    }
}

struct BertEncoder {
    layers: Vec<BertLayer>,
}

impl BertEncoder {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|index| BertLayer::load(vb.pp(format!("layer.{index}")), config))
            .collect::<Result<Vec<_>>>()?;
        Ok(BertEncoder { layers })
    }
}

pub struct BertModel {
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
}

impl BertModel {
    /// Loads the weights under `embeddings.` and `encoder.`, or under `<model_type>.embeddings.` and
    /// `<model_type>.encoder.` for checkpoints saved with a task head, eg. `bert.encoder.layer.0`.
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let (embeddings, encoder) = match (
            BertEmbeddings::load(vb.pp("embeddings"), config),
            BertEncoder::load(vb.pp("encoder"), config),
        ) {
            (Ok(embeddings), Ok(encoder)) => (embeddings, encoder),
            (Err(err), _) | (_, Err(err)) => {
                let Some(model_type) = &config.model_type else {
                    return Err(err);
                };
                match (
                    BertEmbeddings::load(vb.pp(format!("{model_type}.embeddings")), config),
                    BertEncoder::load(vb.pp(format!("{model_type}.encoder")), config),
                ) {
                    (Ok(embeddings), Ok(encoder)) => (embeddings, encoder),
                    _ => return Err(err),
                }
            }
        };
        Ok(Self { embeddings, encoder })
    }

    /// Runs the embeddings and then the first `layers` transformer layers, and returns the last state, of shape
    /// (batch, tokens, hidden size). `observe(l, state)` sees every state on the way, with l = 0 for the embedding
    /// output and l = i + 1 for the output of transformer layer i. `attention_mask` holds 1 for each real token and
    /// 0 for each padding token, with shape (batch, tokens). Without it every token counts.
    pub fn forward_layers(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        layers: usize,
        observe: &mut dyn FnMut(usize, &Tensor) -> Result<()>,
    ) -> Result<Tensor> {
        let mut hidden_states = self.embeddings.forward(input_ids, token_type_ids)?;
        let attention_mask = match attention_mask {
            Some(mask) => Some(get_extended_attention_mask(mask, hidden_states.dtype())?),
            None => None,
        };
        observe(0, &hidden_states)?;

        for (i, layer) in self.encoder.layers.iter().take(layers).enumerate() {
            hidden_states = layer.forward(&hidden_states, attention_mask.as_ref())?;
            observe(i + 1, &hidden_states)?;
        }

        Ok(hidden_states)
    }
}

/// Turns a mask of ones and zeros into one that adds 0 to the attention scores of real tokens and the lowest f32 to
/// those of padding tokens, shaped to broadcast over the heads and the query tokens.
fn get_extended_attention_mask(attention_mask: &Tensor, dtype: DType) -> Result<Tensor> {
    let attention_mask = match attention_mask.rank() {
        3 => attention_mask.unsqueeze(1)?,
        2 => attention_mask.unsqueeze(1)?.unsqueeze(1)?,
        _ => candle_core::bail!("Wrong shape for input_ids or attention_mask"),
    };
    let attention_mask = attention_mask.to_dtype(dtype)?;
    (attention_mask.ones_like()? - &attention_mask)?.broadcast_mul(
        &Tensor::try_from(f32::MIN)?
            .to_device(attention_mask.device())?
            .to_dtype(dtype)?,
    )
}

#[cfg(test)]
#[path = "../../tests/unit/encoder/bert.rs"]
mod tests;
