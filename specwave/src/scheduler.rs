//! Picks the fire layer, the latest layer l whose remaining model time still covers all the reads.
//!
//! ```text
//! T_L − T_l ≥ max(0, t_centers − T_l) + t_read((1 + beta)·nprobe) + t_scan + t_read_rows(rerank_mult·rerank) + 10%
//! ```
//!
//! T_l is the model time when layer l finishes, and L is the last layer. t_centers is the time to read the cluster
//! centers, a read that starts when the text arrives. t_read(n) is the time to read n clusters, t_scan is the time
//! to score their vectors, and t_read_rows(n) is the time to read n precise copies. The 10% is a safety margin.
//!
//! A later layer guesses better but leaves less time to hide the reads. When no layer has enough time left (eg. on
//! a slow USB drive), it also starts a read from layer 0 when the text arrives. That read uses beta = max(1.0, the
//! calibrated value), and it is skipped if layer 0 finds less than 70% of the clusters the true embedding needs.

use std::path::Path;

use crate::speculate::SpecParams;

/// The model time when each layer finishes, measured at open with calls that never touch the index. On the GPU
/// (Metal), each layer is timed in its own run that waits for the GPU at that layer.
pub struct LayerProfile {
    pub cumulative_ms: Vec<f32>,
}

/// A cold read takes t = a + b·bytes, where a is the fixed cost of a read and b is the time per byte. The model is
/// fitted once per disk volume, over pages known not to be in memory, and cached. Until it is fitted, a = 0.25 ms
/// and b = 1 / 3.4 GB/s.
pub struct WaveModel {
    pub a_ms: f32,
    pub b_ms_per_mb: f32,
}

impl WaveModel {
    /// The cached fit for the volume that holds `index_path`, or the default.
    pub fn for_volume(index_path: &Path) -> WaveModel {
        todo!()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Schedule {
    /// `None` means no layer leaves enough time, so the search runs after the embedding.
    pub fire_layer: Option<usize>,
    /// Also starts a read from layer 0 when the text arrives, for slow drives.
    pub layer0_wave: bool,
}

pub fn choose(profile: &LayerProfile, wave: &WaveModel, params: &SpecParams) -> Schedule {
    todo!()
}
