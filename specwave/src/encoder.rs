//! The embedding model, with a hook at chosen layers.
//!
//! Layer l is the state after l transformer layers, as in Hugging Face's `hidden_states[l]`. Layer 0 is the
//! output of the embedding lookup. The first backend (feature `candle`) is `CandleEncoder`, which runs a copy of
//! candle-transformers' BERT (Bidirectional Encoder Representations from Transformers) network in `encoder/bert.rs`.
//! It calls the hook with layer 0 after the embedding lookup, and with layer i + 1 after transformer layer i. It
//! summarizes the layer on the device and copies back only 2d floats, where d is the embedding size. It does this
//! only for the layers the hook wants, so on the graphics processor (GPU, through Metal) a query waits for the
//! device once.

use std::io::Error;

#[cfg(feature = "candle")]
mod bert;
#[cfg(feature = "candle")]
mod candle;

#[cfg(feature = "candle")]
pub use self::candle::{Backend, CandleEncoder, Pooling, snapshot_dir};

/// Called on the encoder's thread. It must return quickly.
pub trait LayerHook {
    /// Whether `on_layer` wants this layer. The encoder asks once per layer, in order, as soon as the layer is done.
    /// On a device that queues its work, that is as soon as the layer's work is queued.
    fn wants(&self, layer: usize) -> bool;
    /// `pooled` is the layer's summary of 2d values, the average over the tokens followed by the first token's state.
    /// BERT-style models put a summary token there, the classification (CLS) token.
    fn on_layer(&mut self, layer: usize, pooled: &[f32]);
}

pub trait LayerEncoder {
    /// The number of transformer layers, L. Layers 0 to L can be observed.
    fn layers(&self) -> usize;
    fn dim(&self) -> usize;
    /// Runs the model on `text` and calls `hook` at the layers it wants. It returns the final normalized embedding.
    /// The arctic-embed-m model uses the CLS token with a query prefix, and MiniLM uses the average over the tokens.
    fn encode(&mut self, text: &str, hook: &mut dyn LayerHook) -> Result<Vec<f32>, Error>;
    /// Whether the device queues the model's work and runs it later, eg. the GPU through Metal, rather than running
    /// each step as it is issued, as the processor does. The hook then hears of a layer before the layer has run.
    fn queues_work(&self) -> bool {
        false
    }
}

/// A hook that wants no layer, for queries that don't speculate.
pub struct NoHook;

impl LayerHook for NoHook {
    fn wants(&self, _layer: usize) -> bool {
        false
    }

    fn on_layer(&mut self, _layer: usize, _pooled: &[f32]) {}
}

#[cfg(all(test, feature = "candle"))]
#[path = "../tests/unit/encoder/testutil.rs"]
mod testutil;
