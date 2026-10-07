//! Speculation, built on ColdIVF's generic `route`, `fetch` and `Fetched`. ColdIVF knows nothing about SpecWave.
//!
//! ```text
//! at the fire layer:  guess → index.route(guess, (1 + beta)·nprobe) → index.fetch(clusters)    (on the worker)
//!                     fetched.prefetch_rows(guess, rerank_mult·rerank)                         (likely rescoring rows)
//! when the model ends: fetched.search(embedding, k, rerank) → hits
//! ```
//!
//! The worker never reads a cluster the guess missed, because the extra clusters already make up for those. It is
//! one parked thread at the user-interactive Quality of Service (QoS) level, so the model's thread never waits on
//! the disk.

use std::io::Error;

use coldivf::{Hit, Index, Limits, SearchParams};

#[derive(Clone, Copy, Debug)]
pub struct SpecParams {
    pub base: SearchParams,
    /// Extra clusters to read, as a fraction of nprobe. Start at 0.5 for a 12-layer model and 1.0 for a 6-layer one,
    /// until calibration picks a value.
    pub beta: f32,
    /// How many candidates' precise copies to prefetch, as a multiple of `rerank`, 2 to 4.
    pub rerank_mult: u8,
}

/// The thread that runs the guessed half of a query. It has its own `SearchCtx`, sized for (1 + beta)·nprobe
/// clusters.
pub struct Worker {
    // thread handle, request slot, Mutex + Condvar handoff
}

impl Worker {
    pub fn new(limits: Limits) -> Worker {
        todo!()
    }

    /// Hands a guessed embedding to the worker and returns right away. The worker picks the clusters, starts
    /// reading them and prefetches the likely rescoring rows.
    pub fn speculate(&mut self, index: &Index, predicted: &[f32], params: &SpecParams) -> Result<(), Error> {
        todo!()
    }

    /// Finishes with the true embedding. It ranks only the clusters already read, and copies the hits into `out`.
    pub fn finish(&mut self, embedding: &[f32], out: &mut Vec<Hit>) -> Result<(), Error> {
        todo!()
    }

    /// Runs a normal `index.search` on the worker's context, for when no layer fired.
    pub fn search(
        &mut self,
        index: &Index,
        embedding: &[f32],
        params: &SearchParams,
        out: &mut Vec<Hit>,
    ) -> Result<(), Error> {
        todo!()
    }
}
