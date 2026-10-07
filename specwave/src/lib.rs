//! SpecWave starts ColdIVF's reads while the embedding model is still running, so the disk reads hide under the
//! model's remaining work.
//!
//! ```text
//! text → index.prepare()    (starts reading the cluster centers)
//!      → layers 1 to l* → guess the embedding → worker.speculate(guess)    (the worker starts the reads)
//!      → layers l*+1 to L → true embedding ──────→ worker.finish(embedding) → hits
//! ```
//!
//! l* is the fire layer, the layer where SpecWave makes its guess and starts reading. L is the model's last layer.
//!
//! | Module | Role |
//! |---|---|
//! | [`encoder`] | The embedding model, with a hook at the layers the schedule asks for |
//! | [`predictor`] | Turns a layer's state into a guess of the final embedding |
//! | [`scheduler`] | Picks the fire layer from the model's timing and the disk's read speed |
//! | [`speculate`] | The worker thread that reads through ColdIVF's `route` and `fetch` |
//! | [`train`] | Fits the predictors, first without real queries, later from logged ones |
//!
//! Errors are [`std::io::Error`]. Model failures use `ErrorKind::Other`.

// Allows the unused arguments of the stubs. Remove it once the functions are implemented.
#![allow(unused_variables, dead_code, clippy::ptr_arg)]

pub mod encoder;
pub mod predictor;
pub mod scheduler;
pub mod speculate;
pub mod train;

use std::io::Error;

use coldivf::{Hit, Index};
use speculate::{SpecParams, Worker};

/// An encoder with its predictors, its schedule and its worker, ready to serve queries.
pub struct SpecWave<E: encoder::LayerEncoder> {
    encoder: E,
    predictors: predictor::Predictors,
    schedule: scheduler::Schedule,
    worker: Worker,
}

impl<E: encoder::LayerEncoder> SpecWave<E> {
    pub fn new(encoder: E, predictors: predictor::Predictors, schedule: scheduler::Schedule, worker: Worker) -> Self {
        SpecWave {
            encoder,
            predictors,
            schedule,
            worker,
        }
    }

    /// Embeds `text` and searches `index`, hiding the disk reads under the model's remaining layers. The hits go
    /// into `out`.
    ///
    /// It calls `index.prepare` first. If the index is already in memory, it skips the guess and searches with the
    /// final embedding. Otherwise, at the fire layer it guesses the embedding and calls `worker.speculate`. When the
    /// model finishes, it calls `worker.finish` with the true embedding. If no layer fired, it calls `worker.search`.
    pub fn search(&mut self, text: &str, index: &Index, params: &SpecParams, out: &mut Vec<Hit>) -> Result<(), Error> {
        todo!()
    }
}
