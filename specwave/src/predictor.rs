//! Predictors that turn a layer's summary into a guess of the final embedding.
//!
//! A layer's summary has 2d values, where d is the embedding size. Each predictor is a ridge regression, i.e. least
//! squares with a small penalty, fitted for one layer. The guess is W′·summary + b′, one matrix-vector product, scaled
//! to length 1 like the final embeddings. At 768 dimensions the matrix is 768 × 1536 f32 values (~4.7 MB), and one
//! guess takes ~20 to 50 microseconds.
//!
//! The fit standardizes its inputs, and a predictor holds that step folded into its weights, so W′ = W·diag(1/σ) and
//! b′ = Ȳ − W′·μ_X. W is the fitted d × 2d matrix, μ_X and σ are the mean and the standard deviation of each input,
//! and Ȳ is the mean embedding.
//!
//! Predictors are stored at `models/<model-id>/predictors.bin`, never inside the Hugging Face cache, eg.
//! `models/Snowflake/snowflake-arctic-embed-m-v1.5/predictors.bin`. The file holds these fields in order, with
//! numbers little-endian.
//!
//! | Field | Bytes |
//! |---|---|
//! | The mark `SWRIDGE` and a zero byte | 8 |
//! | The format version, 1 | 4 |
//! | d | 4 |
//! | The number of predictors | 4 |
//! | The length of the model id, then the id in UTF-8 | 4 plus the length |
//! | Per predictor, its layer, its penalty λ as f64, then W′ row by row and b′ as f32 | 12 + 4(2d² + d) |
//! | The CRC-32 (cyclic redundancy check) of every byte before it | 4 |

use std::fs::{self, File};
use std::io::{Error, ErrorKind, Write};
use std::path::{Component, Path, PathBuf};

const MAGIC: [u8; 8] = *b"SWRIDGE\0";
const VERSION: u32 = 1;

/// The bytes before the model id, i.e. the mark, the version, d, the count and the id's length.
const HEADER: usize = 8 + 4 + 4 + 4 + 4;

/// The number of running sums in a dot product, so the compiler can keep them in vector registers.
const LANES: usize = 16;

/// The predictor for one layer.
#[derive(Clone, Debug, PartialEq)]
pub struct Ridge {
    layer: usize,
    dim: usize,
    penalty: f64,
    /// W′, d rows of 2d values.
    weights: Vec<f32>,
    /// b′, d values.
    bias: Vec<f32>,
}

impl Ridge {
    /// The predictor for `layer` of a model with d = `dim` dimensions, from W′ (d rows of 2d values), b′ (d values)
    /// and the penalty λ it was fitted with. Wrong lengths, a value that isn't finite or a negative penalty give
    /// `InvalidInput`.
    pub fn new(layer: usize, dim: usize, weights: Vec<f32>, bias: Vec<f32>, penalty: f64) -> Result<Ridge, Error> {
        let fits = |n: usize| u32::try_from(n).is_ok();
        let values = dim.checked_mul(dim).and_then(|v| v.checked_mul(2));

        if dim == 0 || !fits(layer) || !fits(dim) || values != Some(weights.len()) {
            return Err(invalid("W′ must hold d rows of 2d values, for a d above 0"));
        }

        if bias.len() != dim {
            return Err(invalid("b′ must hold d values"));
        }

        if !(penalty.is_finite() && penalty >= 0.0) {
            return Err(invalid("the penalty must be finite and at least 0"));
        }

        if !weights.iter().chain(&bias).all(|x| x.is_finite()) {
            return Err(invalid("W′ and b′ must hold only finite values"));
        }

        Ok(Ridge {
            layer,
            dim,
            penalty,
            weights,
            bias,
        })
    }

    pub fn layer(&self) -> usize {
        self.layer
    }

    /// d, the size of a guess. A summary has 2d values.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// λ, the penalty of the fit.
    pub fn penalty(&self) -> f64 {
        self.penalty
    }

    /// W′, d rows of 2d values.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// b′, d values.
    pub fn bias(&self) -> &[f32] {
        &self.bias
    }

    /// Writes the guess for the layer summary `pooled` (2d values) into `e_hat` (d values), i.e. W′·pooled + b′
    /// scaled to length 1. Wrong lengths give `InvalidInput`, and so does a summary whose guess has no direction, eg.
    /// one with a value that isn't finite.
    pub fn predict(&self, pooled: &[f32], e_hat: &mut [f32]) -> Result<(), Error> {
        let width = 2 * self.dim;

        if pooled.len() != width || e_hat.len() != self.dim {
            return Err(invalid(&format!(
                "a guess takes a summary of {width} values and gives {} values",
                self.dim
            )));
        }

        for ((out, row), &b) in e_hat.iter_mut().zip(self.weights.chunks_exact(width)).zip(&self.bias) {
            *out = dot(row, pooled) + b;
        }

        let norm = e_hat.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt();

        if !(norm.is_finite() && norm > 0.0) {
            return Err(invalid("the summary gives a guess with no direction"));
        }

        e_hat.iter_mut().for_each(|x| *x = (f64::from(*x) / norm) as f32);
        Ok(())
    }
}

/// The dot product of `w` and `x`, which have the same length. It keeps `LANES` running sums, each over every
/// `LANES`-th product, then adds them in order and adds the products past the last whole group.
fn dot(w: &[f32], x: &[f32]) -> f32 {
    let (w_groups, w_rest) = w.as_chunks::<LANES>();
    let (x_groups, x_rest) = x.as_chunks::<LANES>();
    let mut sums = [0f32; LANES];

    for (w, x) in w_groups.iter().zip(x_groups) {
        for ((s, &a), &b) in sums.iter_mut().zip(w).zip(x) {
            *s += a * b;
        }
    }

    let rest: f32 = w_rest.iter().zip(x_rest).map(|(&a, &b)| a * b).sum();
    sums.iter().sum::<f32>() + rest
}

/// One predictor per layer that may fire, for one model.
#[derive(Clone, Debug, PartialEq)]
pub struct Predictors {
    model: String,
    dim: usize,
    /// In increasing order of layer.
    by_layer: Vec<Ridge>,
}

impl Predictors {
    /// The predictors of `model`, eg. "Snowflake/snowflake-arctic-embed-m-v1.5", one per layer, in any order. None,
    /// two for one layer, or predictors of different sizes give `InvalidInput`.
    pub fn new(model: &str, mut by_layer: Vec<Ridge>) -> Result<Predictors, Error> {
        by_layer.sort_by_key(|r| r.layer);
        let dim = by_layer.first().map_or(0, |r| r.dim);

        if by_layer.is_empty() {
            return Err(invalid("predictors need at least one layer"));
        }

        if u32::try_from(by_layer.len()).is_err() || u32::try_from(model.len()).is_err() {
            return Err(invalid("too many predictors, or too long a model id"));
        }

        if by_layer.windows(2).any(|w| w[0].layer == w[1].layer) {
            return Err(invalid("each layer has one predictor at most"));
        }

        if by_layer.iter().any(|r| r.dim != dim) {
            return Err(invalid("every predictor must give guesses of the same size"));
        }

        Ok(Predictors {
            model: model.to_string(),
            dim,
            by_layer,
        })
    }

    /// The model the predictors belong to, eg. "Snowflake/snowflake-arctic-embed-m-v1.5".
    pub fn model(&self) -> &str {
        &self.model
    }

    /// d, the size of a guess.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The layers with a predictor, in increasing order.
    pub fn layers(&self) -> impl Iterator<Item = usize> + '_ {
        self.by_layer.iter().map(|r| r.layer)
    }

    pub fn get(&self, layer: usize) -> Option<&Ridge> {
        self.by_layer
            .binary_search_by_key(&layer, |r| r.layer)
            .ok()
            .map(|i| &self.by_layer[i])
    }

    /// Reads the predictors in the file at `path`. A file that isn't a whole predictors file, or whose checksum
    /// doesn't match, gives `InvalidData`, and one of a newer format gives `Unsupported`.
    pub fn load(path: &Path) -> Result<Predictors, Error> {
        let bytes = fs::read(path).map_err(|e| Error::new(e.kind(), format!("{}: {e}", path.display())))?;
        decode(&bytes).map_err(|e| Error::new(e.kind(), format!("{}: {e}", path.display())))
    }

    /// Writes the predictors to `path`, making its folder first. It writes a temporary file next to it and renames
    /// it over `path`, so a crash never leaves a partial file under that name.
    pub fn save(&self, path: &Path) -> Result<(), Error> {
        let name = path
            .file_name()
            .ok_or_else(|| invalid("the predictors file needs a file name"))?;

        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            fs::create_dir_all(dir)?;
        }

        let mut temporary = name.to_os_string();
        temporary.push(format!(".tmp-{}", std::process::id()));
        let temporary = path.with_file_name(temporary);
        let written = File::create(&temporary).and_then(|mut file| {
            file.write_all(&self.encode())?;
            file.sync_all()
        });
        let result = written.and_then(|()| fs::rename(&temporary, path));

        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }

        result
    }

    fn encode(&self) -> Vec<u8> {
        let per = 12 + 4 * (self.dim * 2 * self.dim + self.dim);
        let mut out = Vec::with_capacity(HEADER + self.model.len() + self.by_layer.len() * per + 4);
        // `new` and `Ridge::new` checked that every count and size fits 32 bits.
        let u32_of = |n: usize| (n as u32).to_le_bytes();
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&u32_of(self.dim));
        out.extend_from_slice(&u32_of(self.by_layer.len()));
        out.extend_from_slice(&u32_of(self.model.len()));
        out.extend_from_slice(self.model.as_bytes());

        for ridge in &self.by_layer {
            out.extend_from_slice(&u32_of(ridge.layer));
            out.extend_from_slice(&ridge.penalty.to_le_bytes());
            for x in ridge.weights.iter().chain(&ridge.bias) {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }

        let crc = crc32fast::hash(&out);
        out.extend_from_slice(&crc.to_le_bytes());
        out
    }
}

/// The predictors in `bytes`, the contents of a predictors file.
fn decode(bytes: &[u8]) -> Result<Predictors, Error> {
    let u32_at = |at: usize| bytes.get(at..at + 4).map(|b| u32::from_le_bytes(le(b)) as usize);

    if bytes.get(..8) != Some(&MAGIC[..]) {
        return Err(damaged("isn't a predictors file"));
    }

    let (Some(version), Some(dim), Some(count), Some(id_len)) = (u32_at(8), u32_at(12), u32_at(16), u32_at(20)) else {
        return Err(damaged("is cut short"));
    };

    if version != VERSION as usize {
        return Err(Error::new(
            ErrorKind::Unsupported,
            format!("is in predictors format {version}, and this build reads only format {VERSION}"),
        ));
    }

    // The bytes of one predictor, i.e. its layer, its penalty and its 2d² + d values.
    let per = dim
        .checked_mul(2 * dim)
        .and_then(|w| w.checked_add(dim))
        .and_then(|v| v.checked_mul(4))
        .and_then(|b| b.checked_add(12));
    let total = per
        .and_then(|p| p.checked_mul(count))
        .and_then(|b| b.checked_add(HEADER + 4))
        .and_then(|b| b.checked_add(id_len));

    let Some(per) = per.filter(|_| total == Some(bytes.len())) else {
        return Err(damaged("doesn't hold as many bytes as its header says"));
    };

    let (body, crc) = bytes.split_at(bytes.len() - 4);

    if crc32fast::hash(body) != u32::from_le_bytes(le(crc)) {
        return Err(damaged("doesn't match its checksum"));
    }

    let (id, records) = body[HEADER..].split_at(id_len);
    let model = std::str::from_utf8(id).map_err(|_| damaged("has a model id that isn't UTF-8"))?;
    let mut by_layer = Vec::with_capacity(count);

    for record in records.chunks_exact(per) {
        let (head, values) = record.split_at(12);
        let layer = u32::from_le_bytes(le(head)) as usize;
        let penalty = f64::from_le_bytes(le(&head[4..]));
        let mut weights: Vec<f32> = values
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_le_bytes(b))
            .collect();
        let bias = weights.split_off(2 * dim * dim);
        let ridge = Ridge::new(layer, dim, weights, bias, penalty).map_err(|e| damaged(&e.to_string()))?;
        by_layer.push(ridge);
    }

    Predictors::new(model, by_layer).map_err(|e| damaged(&e.to_string()))
}

/// The first `N` bytes of `bytes`, which holds at least that many.
fn le<const N: usize>(bytes: &[u8]) -> [u8; N] {
    let mut out = [0u8; N];
    out.copy_from_slice(&bytes[..N]);
    out
}

/// Where the predictors of the model `model`, eg. "Snowflake/snowflake-arctic-embed-m-v1.5", are kept in the folder
/// `models`, i.e. `<models>/<model>/predictors.bin`. A model id that isn't a relative path of plain names, eg. one
/// with "..", gives `InvalidInput`.
pub fn path(models: &Path, model: &str) -> Result<PathBuf, Error> {
    let id = Path::new(model);

    if model.is_empty() || !id.components().all(|c| matches!(c, Component::Normal(_))) {
        return Err(invalid(&format!(
            "the model id {model:?} isn't a relative path of plain names"
        )));
    }

    Ok(models.join(id).join("predictors.bin"))
}

fn invalid(message: &str) -> Error {
    Error::new(ErrorKind::InvalidInput, message.to_string())
}

fn damaged(message: &str) -> Error {
    Error::new(ErrorKind::InvalidData, message.to_string())
}

#[cfg(test)]
#[path = "../tests/unit/predictor.rs"]
mod tests;
