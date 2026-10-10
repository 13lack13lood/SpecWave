//! SpecWave starts ColdIVF's reads while the embedding model is still running, so the disk reads hide under the
//! model's remaining work.
//!
//! ```text
//! text → worker.prepare(index)    (the worker loads the cluster tables)
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

pub mod encoder;
pub mod predictor;
pub mod scheduler;
pub mod speculate;
pub mod train;

#[cfg(test)]
#[path = "../tests/unit/testutil.rs"]
mod testutil;

use std::io::{Error, ErrorKind};
use std::time::Instant;

use coldivf::{Hit, Index, SearchParams};
use encoder::{LayerEncoder, LayerHook};
use predictor::Predictors;
use scheduler::Schedule;
use speculate::{Ranked, Served, SpecParams, Worker};

/// How a search ran.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Ran {
    /// The index was warm, so no layer guessed, and the worker searched with the final embedding.
    Warm,
    /// No layer guessed, eg. the schedule never guesses or the fire layer's summary gave no guess, so the worker
    /// searched with the final embedding.
    #[default]
    Unguessed,
    /// The fire layer's guess read the clusters, and the final embedding ranked them.
    Guessed,
    /// The fire layer's guess went to the worker, but its reads failed, so the worker searched with the final
    /// embedding.
    Fallback,
}

/// What the last search did, and when.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Report {
    pub ran: Ran,
    /// The layer whose guess the worker read, if one did.
    pub fire_layer: Option<usize>,
    /// Whether the worker took a guess from layer 0 before the fire layer's guess came, or before the search with the
    /// final embedding when no layer fired.
    pub layer0: bool,
    /// When the search began, when the fire layer's guess went to the worker, when the model ended, and when the hits
    /// were ready.
    pub started: Option<Instant>,
    pub fired: Option<Instant>,
    pub encoded: Option<Instant>,
    pub done: Option<Instant>,
    /// What the worker did for the search's answer.
    pub served: Served,
}

/// An encoder with its predictors, its schedule and its worker, ready to serve queries.
pub struct SpecWave<E: LayerEncoder> {
    encoder: E,
    predictors: Predictors,
    schedule: Schedule,
    worker: Worker,
    /// The guess, d values, reused across queries.
    guess: Vec<f32>,
    /// The final embedding of the last search whose model ran to the end.
    embedding: Vec<f32>,
    last: Report,
}

impl<E: LayerEncoder> SpecWave<E> {
    /// Checks that the predictors fit the encoder and have the layers where the schedule guesses, i.e. the fire layer
    /// and layer 0 for a wave from layer 0. Anything else gives `InvalidInput`, and so does a schedule's β that isn't
    /// finite or is below 0. Guesses need a device that runs each step of the model as it is issued, eg. the processor
    /// (CPU), so an encoder that queues its work, eg. on the graphics processor (GPU) through Metal, gives
    /// `Unsupported` with a schedule that guesses. There the model's time can't be measured layer by layer, and the
    /// copy of a layer's summary waits for the work queued before it.
    pub fn new(encoder: E, predictors: Predictors, schedule: Schedule, worker: Worker) -> Result<SpecWave<E>, Error> {
        let guesses = schedule.fire.is_some() || schedule.layer0_beta.is_some();

        if guesses && encoder.queues_work() {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "SpecWave guesses only on a device that runs the model's work as it is issued",
            ));
        }

        if predictors.dim() != encoder.dim() {
            return Err(invalid(
                "the predictors' guesses must be as long as the encoder's embeddings",
            ));
        }

        let layers = schedule.fire.iter().map(|f| (f.settings.layer, f.settings.beta));
        let layer0 = schedule.layer0_beta.map(|beta| (0, beta));

        for (layer, beta) in layers.chain(layer0) {
            if layer > encoder.layers() || predictors.get(layer).is_none() {
                return Err(invalid(&format!(
                    "the schedule guesses at layer {layer}, which needs a predictor and a layer of the model"
                )));
            }
            if !(beta.is_finite() && beta >= 0.0) {
                return Err(invalid("the schedule's beta must be finite and at least 0"));
            }
        }

        let guess = vec![0.0; predictors.dim()];
        Ok(SpecWave {
            encoder,
            predictors,
            schedule,
            worker,
            guess,
            embedding: Vec::new(),
            last: Report::default(),
        })
    }

    /// Embeds `text` and searches `index` with `params`, hiding the disk reads under the model's remaining layers.
    /// The hits go into `out`. The schedule must have been chosen for searches with `params`.
    ///
    /// It calls `worker.prepare` first, which loads the cluster tables while the model runs. If the index is warm, it
    /// skips the guess and searches with the final embedding. Otherwise, at the fire layer it guesses the embedding
    /// and calls `worker.speculate`, and the worker starts reading. When the model finishes, it calls `worker.finish`
    /// with the true embedding. If no layer fired, it calls `worker.search`. Guess settings beyond the worker's limits
    /// give `InvalidInput` before the model runs. When the model fails, the worker drops the guess, and the error comes
    /// back.
    pub fn search(
        &mut self,
        text: &str,
        index: &Index,
        params: &SearchParams,
        out: &mut Vec<Hit>,
    ) -> Result<(), Error> {
        let started = Some(Instant::now());
        self.last = Report {
            started,
            ..Report::default()
        };
        self.worker.prepare(index, self.guess.len())?;
        let warm = index.is_warm();
        let fire = self
            .schedule
            .fire
            .filter(|_| !warm)
            .map(|f| (f.settings.layer, f.params(*params)));
        let layer0 = self
            .schedule
            .layer0_beta
            .filter(|_| fire.is_some_and(|(layer, _)| layer > 0))
            .map(|beta| SpecParams {
                base: *params,
                beta,
                rerank_mult: 0,
            });

        for guess in fire.iter().map(|(_, p)| p).chain(&layer0) {
            self.worker.check(guess)?;
        }

        let SpecWave {
            encoder,
            predictors,
            worker,
            guess,
            ..
        } = self;
        let mut hook = Speculator {
            worker,
            predictors,
            index,
            guess,
            fire,
            layer0,
            fired: None,
            layer0_taken: false,
            failure: None,
        };
        let encoded = encoder.encode(text, &mut hook);
        let (fired, layer0_taken, failure) = (hook.fired, hook.layer0_taken, hook.failure);
        let encoded_at = Some(Instant::now());
        let embedding = match (encoded, failure) {
            (Ok(embedding), None) => embedding,
            (Err(e), _) | (Ok(_), Some(e)) => {
                self.worker.cancel();
                return Err(e);
            }
        };
        let result = match fired {
            Some(_) => self.worker.finish(&embedding, out),
            None => self.worker.search(index, &embedding, params, out),
        };
        // Without a fire layer's guess, the search replaces a guess from layer 0 that the worker hasn't taken.
        let layer0_taken = layer0_taken && (fired.is_some() || !self.worker.replaced_guess());
        self.embedding = embedding;
        let served = self.worker.served();
        self.last = Report {
            ran: ran(warm, fired.is_some(), served.ranked),
            fire_layer: fired.and(fire.map(|(layer, _)| layer)),
            layer0: layer0_taken,
            started,
            fired,
            encoded: encoded_at,
            done: Some(Instant::now()),
            served,
        };
        result
    }

    /// What the last search did.
    pub fn last(&self) -> &Report {
        &self.last
    }

    /// The fire layer's guess in the last search, if it went to the worker.
    pub fn guess(&self) -> Option<&[f32]> {
        self.last.fire_layer.map(|_| &self.guess[..])
    }

    /// The final embedding of the last search whose model ran to the end, eg. for a log of queries to refit the
    /// predictors with. It is empty before the first.
    pub fn embedding(&self) -> &[f32] {
        &self.embedding
    }

    pub fn encoder(&self) -> &E {
        &self.encoder
    }

    /// The encoder, eg. for calls that warm it up between searches.
    pub fn encoder_mut(&mut self) -> &mut E {
        &mut self.encoder
    }

    pub fn schedule(&self) -> &Schedule {
        &self.schedule
    }

    pub fn worker(&self) -> &Worker {
        &self.worker
    }
}

/// How a search ran, from whether the index was warm, whether the fire layer's guess went to the worker, and what the
/// worker's answer ranked.
fn ran(warm: bool, fired: bool, ranked: Ranked) -> Ran {
    match (warm, fired, ranked) {
        (true, _, _) => Ran::Warm,
        (false, false, _) => Ran::Unguessed,
        (false, true, Ranked::Fallback) => Ran::Fallback,
        (false, true, _) => Ran::Guessed,
    }
}

/// The hook that guesses the embedding at the fire layer, and at layer 0 for a wave from there, and hands each guess
/// to the worker.
struct Speculator<'a> {
    worker: &'a mut Worker,
    predictors: &'a Predictors,
    index: &'a Index,
    guess: &'a mut [f32],
    /// The fire layer and the worker's settings there.
    fire: Option<(usize, SpecParams)>,
    /// The settings of a wave from layer 0, which reads clusters only.
    layer0: Option<SpecParams>,
    /// When the fire layer's guess went to the worker.
    fired: Option<Instant>,
    /// Whether a guess from layer 0 went to the worker and no later guess replaced it before the worker took it.
    layer0_taken: bool,
    /// Why the worker refused a guess.
    failure: Option<Error>,
}

impl LayerHook for Speculator<'_> {
    fn wants(&self, layer: usize) -> bool {
        self.fire.is_some_and(|(fire, _)| fire == layer) || (layer == 0 && self.layer0.is_some())
    }

    fn on_layer(&mut self, layer: usize, pooled: &[f32]) {
        let fire = self.fire.filter(|&(fire, _)| fire == layer);
        let due = fire.map(|(_, params)| params).or(self.layer0.filter(|_| layer == 0));
        let (Some(params), Some(ridge)) = (due, self.predictors.get(layer)) else {
            return;
        };

        // A summary that gives no guess leaves the search to the final embedding.
        if ridge.predict(pooled, self.guess).is_err() {
            return;
        }

        let at = Instant::now();

        match (self.worker.speculate(self.index, self.guess, &params), fire.is_some()) {
            (Ok(()), true) => {
                self.fired = Some(at);
                self.layer0_taken &= !self.worker.replaced_guess();
            }
            (Ok(()), false) => self.layer0_taken = true,
            (Err(e), _) => self.failure = Some(e),
        }
    }
}

fn invalid(message: &str) -> Error {
    Error::new(ErrorKind::InvalidInput, message.to_string())
}

#[cfg(test)]
#[path = "../tests/unit/lib.rs"]
mod tests;
