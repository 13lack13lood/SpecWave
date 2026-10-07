//! Fits the predictors without any real queries.
//!
//! It runs the model on the first 6 to 24 words of 20K documents, with the query prompt in front. For each one it
//! records the summary at each layer that may fire and the final embedding. It then fits one ridge regression per
//! layer in f64 by solving (XsᵀXs + λ·n·I)·Wᵀ = Xsᵀ(Y − Ȳ). Xs holds the standardized summaries and Y the final
//! embeddings, one pair per row. Ȳ is the mean of Y, n is the number of pairs and I is the identity. It picks the
//! penalty λ from {1e-3, 1e-2, 1e-1} on an 85/15 split, then refits on all pairs. Once real queries are logged, it
//! can refit from those.

use std::io::Error;

use crate::encoder::LayerEncoder;
use crate::predictor::Predictors;

pub struct TrainParams {
    /// Layers that may fire, eg. 0, 3, 6, 8, 9, 10 for arctic-embed-m and 0, 1, 3, 4, 5 for MiniLM. Layer l is the
    /// state after l transformer layers.
    pub layers: Vec<usize>,
    pub lambdas: Vec<f64>,
    pub holdout: f32,
}

pub fn fit<E: LayerEncoder>(encoder: &mut E, texts: &[String], params: &TrainParams) -> Result<Predictors, Error> {
    todo!()
}
