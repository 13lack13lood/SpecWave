//! Picks the fire layer, where SpecWave guesses the embedding and starts the reads. It is the latest layer l whose
//! remaining model time still covers the reads that a guess there starts,
//!
//! ```text
//! T_L − T_l ≥ 1.1·(max(0, t_tables − T_l) + t_route + t_clusters(⌊(1 + β)·nprobe⌋) + t_scan + t_rows(m·rerank))
//! ```
//!
//! where T_l is the model time when layer l is done and L is the model's last layer. t_tables is the worker's load of
//! the cluster tables, which starts when the text arrives, so the part of it still running at T_l delays the guess.
//! t_route is the guess's route, which scores every cluster center. t_clusters(n) is the read of n clusters, t_scan the
//! scan of their vectors, and t_rows(n) the read of the precise copies of n candidates. nprobe is the number of
//! clusters that a normal search reads, and rerank the number of candidates it rescores. β is the share of extra
//! clusters that a guess at layer l reads, and m·rerank the number of candidates whose precise copies it prefetches.
//! Each layer has its own β and m, the smallest that keep a search's recall and rerank coverage there, measured
//! offline. The factor 1.1 is a 10% margin.
//!
//! A later layer guesses better but leaves less time to hide the reads. When no layer leaves enough time, eg. on a
//! slow external drive, the reads can't all hide. The guess then still fires, at the layer whose reads run the least
//! past the model's end, and a guess from layer 0 also starts a read when the text arrives. That read uses
//! β = max(1, the calibrated β), and it is skipped when layer 0's guesses read less than 70% of the clusters that the
//! final embedding reads. It helps only where a fetch returns while its reads run, eg. with the read hints of macOS
//! and Linux. Where the fetch waits for every read, eg. on a Mac drive that refuses read hints, the read from layer 0
//! holds the worker until every cluster is in, and the fire layer's guess then reads them again.

mod volume;
mod wave;

pub use volume::Volume;
pub use wave::{WaveModel, cache_path, support_dir};

use std::cell::Cell;
use std::io::{Error, ErrorKind};
use std::time::{Duration, Instant};

use coldivf::{Limits, SearchParams};

use crate::encoder::{LayerEncoder, LayerHook};
use crate::speculate::{SpecParams, wide_nprobe};

/// The margin that the rule adds to the reads' time, as a share of it.
pub const MARGIN: f64 = 0.1;

/// The encoder-only calls that time the model when SpecWave opens.
pub const PROFILE_CALLS: usize = 20;

/// The least share of the final embedding's clusters that layer 0's guesses must read for its wave to start.
pub const LAYER0_COVERAGE: f64 = 0.7;

/// The least β of the wave from layer 0.
pub const LAYER0_BETA: f64 = 1.0;

/// The model time when each layer is done, T_l for layer l, in milliseconds (ms) from the start of a call. Layer 0 is
/// the embedding output, and the last layer is the model's last, L.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerProfile {
    /// (l, T_l), in increasing order of l.
    times: Vec<(usize, f64)>,
}

impl LayerProfile {
    /// The profile of `times`, pairs of a layer and its T_l, eg. ones measured elsewhere. It may skip layers, but its
    /// last layer must be the model's last. No times, layers out of order or named twice, or times that aren't finite,
    /// are below 0 or are below an earlier layer's give `InvalidInput`.
    pub fn new(times: Vec<(usize, f64)>) -> Result<LayerProfile, Error> {
        if times.is_empty() {
            return Err(invalid("a profile needs at least one layer"));
        }

        if !times.iter().all(|&(_, t)| t.is_finite() && t >= 0.0) {
            return Err(invalid("a profile's times must be finite and at least 0"));
        }

        if !times.windows(2).all(|w| w[0].0 < w[1].0 && w[0].1 <= w[1].1) {
            return Err(invalid(
                "a profile's layers must increase, and no time may be below an earlier layer's",
            ));
        }

        Ok(LayerProfile { times })
    }

    /// Times `calls` runs of `encoder`, each on the next text of `texts`, and keeps each layer's median by nearest
    /// rank. The calls never touch an index. The hook notes the time whenever the encoder asks about a layer, which it
    /// does as soon as the layer is done, and it wants no layer, so no state is copied back. The first calls after a
    /// load run slower, and the median leaves them out while they are fewer than half. No texts or no calls give
    /// `InvalidInput`, and an encoder that doesn't ask about each layer once per call gives `ErrorKind::Other`. A
    /// device that queues its work, eg. the graphics processor (GPU) through Metal, gives `Unsupported`, since the hook
    /// would note when each layer's work was queued rather than when it ran.
    pub fn measure<E: LayerEncoder + ?Sized, S: AsRef<str>>(
        encoder: &mut E,
        texts: &[S],
        calls: usize,
    ) -> Result<LayerProfile, Error> {
        if encoder.queues_work() {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "a device that queues its work can't be timed layer by layer from the hook",
            ));
        }

        if texts.is_empty() || calls == 0 {
            return Err(invalid("timing the model needs at least one text and one call"));
        }

        let layers = encoder.layers() + 1;
        let mut samples: Vec<Vec<f64>> = (0..layers).map(|_| Vec::with_capacity(calls)).collect();
        let mut hook = Stopwatch::new(layers);

        for i in 0..calls {
            hook.start();
            encoder.encode(texts[i % texts.len()].as_ref(), &mut hook)?;
            let times = hook
                .times()
                .ok_or_else(|| Error::other("the encoder didn't ask about each layer once"))?;
            for (s, t) in samples.iter_mut().zip(times) {
                s.push(t);
            }
        }

        LayerProfile::new(samples.iter().enumerate().map(|(l, s)| (l, median(s))).collect())
    }

    /// L, the model's last layer.
    pub fn last_layer(&self) -> usize {
        self.times[self.times.len() - 1].0
    }

    /// T_L, the model time when the last layer is done.
    pub fn total_ms(&self) -> f64 {
        self.times[self.times.len() - 1].1
    }

    /// T_l, if the profile has layer l.
    pub fn at(&self, layer: usize) -> Option<f64> {
        self.times
            .binary_search_by_key(&layer, |&(l, _)| l)
            .ok()
            .map(|i| self.times[i].1)
    }

    /// T_L − T_l, the model time still to come after layer l, if the profile has layer l.
    pub fn left_after(&self, layer: usize) -> Option<f64> {
        self.at(layer).map(|t| self.total_ms() - t)
    }

    /// Every (l, T_l), in increasing order of l.
    pub fn times(&self) -> &[(usize, f64)] {
        &self.times
    }
}

/// A hook that wants no layer and notes when the encoder asks about each one.
struct Stopwatch {
    start: Instant,
    /// The time from `start` to the encoder's question about each layer, or `None` before it asked.
    at: Vec<Cell<Option<Duration>>>,
    /// Whether the encoder asked about a layer twice, or about one past the last.
    bad: Cell<bool>,
}

impl Stopwatch {
    fn new(layers: usize) -> Stopwatch {
        Stopwatch {
            start: Instant::now(),
            at: (0..layers).map(|_| Cell::new(None)).collect(),
            bad: Cell::new(false),
        }
    }

    /// Clears the notes and starts the clock.
    fn start(&mut self) {
        self.at.iter().for_each(|at| at.set(None));
        self.bad.set(false);
        self.start = Instant::now();
    }

    /// Each layer's time in ms, if the encoder asked about each layer once.
    fn times(&self) -> Option<Vec<f64>> {
        if self.bad.get() {
            return None;
        }

        self.at
            .iter()
            .map(|at| at.get().map(|d| d.as_secs_f64() * 1e3))
            .collect()
    }
}

impl LayerHook for Stopwatch {
    fn wants(&self, layer: usize) -> bool {
        let now = self.start.elapsed();

        match self.at.get(layer) {
            Some(at) if at.get().is_none() => at.set(Some(now)),
            _ => self.bad.set(true),
        }

        false
    }

    fn on_layer(&mut self, _layer: usize, _pooled: &[f32]) {}
}

/// The median of `values` by nearest rank, i.e. the smallest value that at least half of them are at or below.
fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len().div_ceil(2).max(1) - 1]
}

/// The read and scan times that the rule adds up, in ms.
pub trait ReadTimes {
    /// t_tables, the load of the cluster tables, which starts when the text arrives.
    fn tables_ms(&self) -> f64;
    /// t_route + t_clusters(clusters) + t_scan + t_rows(rows), i.e. routing the guess, reading `clusters` clusters,
    /// scanning their vectors and reading the precise copies of `rows` candidates.
    fn chain_ms(&self, clusters: usize, rows: usize) -> f64;
}

/// Read times from a wave model and the sizes of one index's reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReadModel {
    pub wave: WaveModel,
    /// The bytes of the cluster tables.
    pub tables_bytes: f64,
    /// The time that a route takes, in ms, which scores every cluster center.
    pub route_ms: f64,
    /// The bytes that reading one cluster takes on average, rounded out to whole pages.
    pub cluster_bytes: f64,
    /// The bytes that reading one candidate's precise copy takes on average. That is a whole read page, eg. 16 KiB on
    /// macOS, unless candidates share pages.
    pub row_bytes: f64,
    /// The time that scanning one cluster's vectors takes on average, in ms.
    pub scan_ms_per_cluster: f64,
}

impl ReadTimes for ReadModel {
    fn tables_ms(&self) -> f64 {
        self.wave.read_ms(self.tables_bytes)
    }

    fn chain_ms(&self, clusters: usize, rows: usize) -> f64 {
        let clusters = clusters as f64;
        self.route_ms
            + self.wave.read_ms(clusters * self.cluster_bytes)
            + clusters * self.scan_ms_per_cluster
            + self.wave.read_ms(rows as f64 * self.row_bytes)
    }
}

/// What a guess at one layer reads. Its β and m are the smallest, in that order, whose searches keep their recall and
/// reach the rerank coverage target at that layer, as measured offline on held-out queries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerSettings {
    pub layer: usize,
    /// β, the extra clusters to read, as a share of nprobe.
    pub beta: f64,
    /// m, the candidates whose precise copies to prefetch, as a multiple of `rerank`.
    pub rerank_mult: u8,
}

/// The guess from layer 0, the embedding output, which is ready about as soon as the text arrives.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Layer0 {
    /// The calibrated β of layer 0's guesses, or 0 for none. Its wave reads with at least β = 1.
    pub beta: f64,
    /// The share of the final embedding's clusters that layer 0's guesses read at that wave's β, measured offline.
    pub coverage: f64,
}

/// A guess at one layer, with what it reads and the time that needs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fire {
    pub settings: LayerSettings,
    /// ⌊(1 + β)·nprobe⌋, the clusters it reads.
    pub clusters: usize,
    /// m·rerank, the candidates whose precise copies it prefetches.
    pub rows: usize,
    /// T_L − T_l, the model time left after the layer.
    pub left_ms: f64,
    /// The right side of the rule, the time its reads need with the margin.
    pub needed_ms: f64,
}

impl Fire {
    /// Whether the model time left after the layer covers its reads, margin included.
    pub fn fits(&self) -> bool {
        self.left_ms >= self.needed_ms
    }

    /// The worker's settings for this guess, for searches with `base`.
    pub fn params(&self, base: SearchParams) -> SpecParams {
        SpecParams {
            base,
            beta: self.settings.beta,
            rerank_mult: self.settings.rerank_mult,
        }
    }
}

/// Where SpecWave guesses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Schedule {
    /// The fire layer, or `None` when no layer may guess, so the search runs after the embedding.
    pub fire: Option<Fire>,
    /// The β of a read from layer 0's guess when the text arrives, which helps on a drive too slow for the fire
    /// layer's time.
    pub layer0_beta: Option<f64>,
}

impl Schedule {
    /// A schedule that never guesses.
    pub const NONE: Schedule = Schedule {
        fire: None,
        layer0_beta: None,
    };

    /// The limits of a worker that serves searches with `base` under this schedule, i.e. the search's own settings,
    /// the clusters and rows of the fire layer's guess, and the clusters of a wave from layer 0.
    pub fn limits(&self, base: &SearchParams) -> Limits {
        let fire = self.fire.map(|f| f.params(*base));
        let layer0 = self.layer0_beta.map_or(0, |beta| wide_nprobe(base.nprobe, beta));
        Limits {
            max_k: base.k,
            max_nprobe: fire.map_or(0, |p| p.guess_nprobe()).max(layer0).max(base.nprobe),
            max_rerank: fire.map_or(0, |p| p.prefetch_rows()).max(base.rerank),
        }
    }
}

/// The guess at the layer of `settings` for searches with `base`, i.e. what it reads and the two sides of the rule
/// there. A layer that `profile` lacks or that isn't before its last layer, a β that isn't finite or is below 0, m = 0,
/// a `base` that reads no clusters or rescores no candidates, and read times that aren't finite or are below 0 give
/// `InvalidInput`.
pub fn evaluate(
    profile: &LayerProfile,
    reads: &dyn ReadTimes,
    base: &SearchParams,
    settings: &LayerSettings,
) -> Result<Fire, Error> {
    let layer = settings.layer;
    let at = profile
        .at(layer)
        .filter(|_| layer < profile.last_layer())
        .ok_or_else(|| invalid(&format!("the profile has no layer {layer} before its last")))?;

    if !(settings.beta.is_finite() && settings.beta >= 0.0) || settings.rerank_mult == 0 {
        return Err(invalid(
            "a layer's beta must be finite and at least 0, and its multiple at least 1",
        ));
    }

    if base.nprobe == 0 || base.rerank == 0 {
        return Err(invalid(
            "a search must read at least one cluster and rescore at least one candidate",
        ));
    }

    let params = SpecParams {
        base: *base,
        beta: settings.beta,
        rerank_mult: settings.rerank_mult,
    };
    let (clusters, rows) = (params.guess_nprobe(), params.prefetch_rows());
    let (tables, chain) = (reads.tables_ms(), reads.chain_ms(clusters, rows));

    if !(tables.is_finite() && chain.is_finite() && tables >= 0.0 && chain >= 0.0) {
        return Err(invalid("read times must be finite and at least 0"));
    }

    Ok(Fire {
        settings: *settings,
        clusters,
        rows,
        left_ms: profile.total_ms() - at,
        needed_ms: (1.0 + MARGIN) * ((tables - at).max(0.0) + chain),
    })
}

/// Picks where to guess for searches with `base`, among the layers that may guess, `layers`. It is the latest layer
/// whose reads fit by the rule. When none fits, it is the layer whose reads run the least past the model's end, the
/// latest of equal ones, and then `layer0`, when given, adds a wave from layer 0 unless that layer fires anyway or its
/// coverage is below 70%. Leave `layer0` out where a fetch waits for its reads, as the module says. No layers give a
/// schedule that never guesses. A layer named twice, or a `layer0` with a β or
/// coverage out of range, gives `InvalidInput`, and so does anything that [`evaluate`] refuses.
pub fn choose(
    profile: &LayerProfile,
    reads: &dyn ReadTimes,
    base: &SearchParams,
    layers: &[LayerSettings],
    layer0: Option<Layer0>,
) -> Result<Schedule, Error> {
    if layers
        .iter()
        .enumerate()
        .any(|(i, s)| layers[..i].iter().any(|t| t.layer == s.layer))
    {
        return Err(invalid("each layer may have one set of settings at most"));
    }

    if layer0.is_some_and(|l| !(l.beta.is_finite() && l.beta >= 0.0 && (0.0..=1.0).contains(&l.coverage))) {
        return Err(invalid(
            "layer 0's beta must be finite and at least 0, and its coverage from 0 to 1",
        ));
    }

    let mut fires = layers
        .iter()
        .map(|s| evaluate(profile, reads, base, s))
        .collect::<Result<Vec<Fire>, Error>>()?;
    fires.sort_by_key(|f| f.settings.layer);

    if let Some(fire) = fires.iter().rev().find(|f| f.fits()) {
        return Ok(Schedule {
            fire: Some(*fire),
            layer0_beta: None,
        });
    }

    // Iterator::min_by keeps the first of equal values, so the latest layer wins a tie.
    let Some(least) = fires
        .iter()
        .rev()
        .min_by(|a, b| (a.needed_ms - a.left_ms).total_cmp(&(b.needed_ms - b.left_ms)))
    else {
        return Ok(Schedule::NONE);
    };

    let layer0_beta = layer0
        .filter(|l| least.settings.layer != 0 && l.coverage >= LAYER0_COVERAGE)
        .map(|l| l.beta.max(LAYER0_BETA));
    Ok(Schedule {
        fire: Some(*least),
        layer0_beta,
    })
}

fn invalid(message: &str) -> Error {
    Error::new(ErrorKind::InvalidInput, message.to_string())
}

#[cfg(test)]
#[path = "../tests/unit/scheduler.rs"]
mod tests;
