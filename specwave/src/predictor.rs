//! Predictors that turn a layer's summary into a guess of the final embedding.
//!
//! A layer's summary has 2d values, where d is the embedding size. Each predictor is a ridge regression, i.e. least
//! squares with a small penalty, fitted for one layer. The guess is W′·summary + b′, one matrix-vector product. At
//! 768 dimensions the matrix is 768 × 1536 f32 values (~4.7 MB), and one guess takes ~20 to 50 microseconds.
//!
//! Training fits W on standardized inputs. Saving folds the standardization back in, so W′ = W·diag(1/σ) and
//! b′ = Ȳ − W′·μ_X, where μ_X and σ are the mean and standard deviation of each input and Ȳ is the mean embedding.
//!
//! Predictors are stored at `models/<model-id>/predictors.bin`, never inside the Hugging Face cache.

use std::io::Error;
use std::path::Path;

/// The predictor for one layer.
pub struct Ridge {
    pub layer: usize,
    // W′ (d × 2d), b′
}

impl Ridge {
    pub fn predict(&self, pooled: &[f32], e_hat: &mut [f32]) {
        todo!()
    }
}

/// One predictor per layer that may fire.
pub struct Predictors {
    by_layer: Vec<Ridge>,
}

impl Predictors {
    pub fn load(path: &Path) -> Result<Predictors, Error> {
        todo!()
    }

    pub fn save(&self, path: &Path) -> Result<(), Error> {
        todo!()
    }

    pub fn get(&self, layer: usize) -> Option<&Ridge> {
        todo!()
    }
}
