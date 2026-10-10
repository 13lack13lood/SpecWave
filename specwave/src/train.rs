//! Fits the predictors, first without any real queries.
//!
//! A fit needs pairs of a text's layer summaries and its final embedding. Before any query is logged, the texts are
//! [`openings`], the first 6 to 24 words of 20K documents, which the model reads with its query prompt in front. For
//! each layer that may fire, the fit then solves one ridge regression in f64 with a Cholesky factorization,
//!
//! ```text
//! (Xsᵀ·Xs + λ·n·I)·Wᵀ = Xsᵀ·(Y − Ȳ)
//! ```
//!
//! where Xs holds the standardized summaries and Y the final embeddings, one pair per row, and Ȳ is the mean of Y. n is
//! the number of pairs, I is the identity and W is the predictor's d × 2d matrix. Each summary value is standardized
//! with the mean and the standard deviation of its column, plus 1e-6. The penalty λ is the one of 1e-3, 1e-2 and 1e-1
//! whose fit on the first 85% of the pairs best predicts the last 15%, by the mean cosine, and the final fit uses every
//! pair. Once real queries are logged in a [`QueryLog`], [`refit`] fits them the same way.

use std::fs::{File, OpenOptions};
use std::io::{BufReader, Error, ErrorKind, Read, Write};
use std::ops::Range;
use std::path::Path;

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt;
use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatMut, MatRef, Par};

use crate::encoder::{LayerEncoder, LayerHook};
use crate::predictor::{Predictors, Ridge};

/// The pairs whose standardized summaries the fit holds in f64 at once.
const BLOCK: usize = 1024;

/// Added to each summary value's standard deviation, so a value that never changes doesn't divide by 0.
const SPREAD_FLOOR: f64 = 1e-6;

/// The largest embedding size, and the most layers, that pairs and logs may have.
const MAX_DIM: usize = 1 << 20;
const MAX_LAYERS: usize = 4096;

const LOG_MAGIC: [u8; 8] = *b"SWPAIRS\0";
const LOG_VERSION: u32 = 1;

pub struct TrainParams {
    /// Layers that may fire, eg. 0, 3, 6, 8, 9, 10 for arctic-embed-m. Layer l is the state after l transformer layers.
    pub layers: Vec<usize>,
    /// The penalties λ to choose from. A single one is used as it is, with no held-out fit.
    pub lambdas: Vec<f64>,
    /// The share of the pairs, in their order, that each penalty is first fitted on. The rest pick the penalty.
    pub split: f64,
    /// The threads of the matrix products, or 0 for every thread of the current rayon pool.
    pub threads: usize,
}

impl TrainParams {
    /// The fit described above at `layers`, i.e. the penalties 1e-3, 1e-2 and 1e-1 picked on an 85/15 split, on every
    /// thread.
    pub fn new(layers: &[usize]) -> TrainParams {
        TrainParams {
            layers: layers.to_vec(),
            lambdas: vec![1e-3, 1e-2, 1e-1],
            split: 0.85,
            threads: 0,
        }
    }

    fn check(&self) -> Result<(), Error> {
        if self.layers.is_empty() || !distinct(&self.layers) {
            return Err(invalid("the fit needs at least one layer, each named once"));
        }

        if self.lambdas.is_empty() || !self.lambdas.iter().all(|&l| l.is_finite() && l > 0.0) {
            return Err(invalid("the fit needs at least one penalty, each finite and above 0"));
        }

        if self.lambdas.len() > 1 && !(self.split > 0.0 && self.split < 1.0) {
            return Err(invalid("the split must be a share above 0 and below 1"));
        }

        Ok(())
    }

    fn par(&self) -> Par {
        match self.threads {
            1 => Par::Seq,
            n => Par::rayon(n),
        }
    }
}

/// Training pairs. Each holds one text's summaries at the chosen layers and its final embedding.
#[derive(Clone, Debug, PartialEq)]
pub struct Pairs {
    dim: usize,
    layers: Vec<usize>,
    /// One table per layer, with 2d values per pair.
    states: Vec<Vec<f32>>,
    /// d values per pair.
    embeddings: Vec<f32>,
}

impl Pairs {
    /// No pairs yet, for the summaries at `layers`, each named once, of a model with `dim` dimensions. A size of 0 or
    /// above 2^20, no layers, more than 4096 of them, or a layer named twice gives `InvalidInput`.
    pub fn new(dim: usize, layers: &[usize]) -> Result<Pairs, Error> {
        let sized = (1..=MAX_DIM).contains(&dim) && (1..=MAX_LAYERS).contains(&layers.len());

        if !sized || !distinct(layers) || layers.iter().any(|&l| u32::try_from(l).is_err()) {
            return Err(invalid(
                "pairs need a size from 1 to 2^20 and 1 to 4096 layers, each named once",
            ));
        }

        Ok(Pairs {
            dim,
            layers: layers.to_vec(),
            states: vec![Vec::new(); layers.len()],
            embeddings: Vec::new(),
        })
    }

    /// The pairs of whole tables, eg. of summaries computed elsewhere. `states` holds one table per layer of `layers`,
    /// in that order, with 2d values per pair, and `embeddings` holds d values per pair. Tables of other lengths, or a
    /// value that isn't finite, give `InvalidInput`.
    pub fn from_tables(
        dim: usize,
        layers: &[usize],
        states: Vec<Vec<f32>>,
        embeddings: Vec<f32>,
    ) -> Result<Pairs, Error> {
        let mut pairs = Pairs::new(dim, layers)?;
        let n = embeddings.len() / dim;

        if states.len() != layers.len() || n * dim != embeddings.len() || states.iter().any(|t| t.len() != n * 2 * dim)
        {
            return Err(invalid("each table must hold one row per pair"));
        }

        if !states.iter().flatten().chain(&embeddings).all(|x| x.is_finite()) {
            return Err(invalid("the pairs must hold only finite values"));
        }

        pairs.states = states;
        pairs.embeddings = embeddings;
        Ok(pairs)
    }

    /// d, the size of an embedding. A summary has 2d values.
    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn layers(&self) -> &[usize] {
        &self.layers
    }

    pub fn len(&self) -> usize {
        self.embeddings.len() / self.dim
    }

    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// The summaries at `layer`, 2d values per pair, if the pairs hold that layer.
    pub fn states(&self, layer: usize) -> Option<&[f32]> {
        let i = self.layers.iter().position(|&l| l == layer)?;
        Some(&self.states[i])
    }

    /// The final embeddings, d values per pair.
    pub fn embeddings(&self) -> &[f32] {
        &self.embeddings
    }

    /// Adds one pair. `states` holds the summary at each layer, 2d values each, in the order of the layers. Wrong
    /// lengths, or a value that isn't finite, give `InvalidInput`.
    pub fn push(&mut self, states: &[f32], embedding: &[f32]) -> Result<(), Error> {
        let width = 2 * self.dim;

        if states.len() != self.layers.len() * width || embedding.len() != self.dim {
            return Err(invalid(&format!(
                "a pair holds {} summaries of {width} values and an embedding of {} values",
                self.layers.len(),
                self.dim
            )));
        }

        if !states.iter().chain(embedding).all(|x| x.is_finite()) {
            return Err(invalid("the pairs must hold only finite values"));
        }

        for (table, state) in self.states.iter_mut().zip(states.chunks_exact(width)) {
            table.extend_from_slice(state);
        }

        self.embeddings.extend_from_slice(embedding);
        Ok(())
    }

    /// Runs `encoder` on `text` and adds its summaries and its final embedding. An encoder of another size, or one
    /// without all of the layers, gives `InvalidInput`, and one that skips a wanted layer gives `ErrorKind::Other`.
    pub fn encode<E: LayerEncoder + ?Sized>(&mut self, encoder: &mut E, text: &str) -> Result<(), Error> {
        if encoder.dim() != self.dim || self.layers.iter().any(|&l| l > encoder.layers()) {
            return Err(invalid("the encoder has another size or fewer layers than the pairs"));
        }

        let width = 2 * self.dim;
        let mut hook = Recorder {
            layers: &self.layers,
            width,
            row: vec![0.0; self.layers.len() * width],
            seen: vec![false; self.layers.len()],
            bad: false,
        };
        let embedding = encoder.encode(text, &mut hook)?;

        if hook.bad || !hook.seen.iter().all(|&s| s) || embedding.len() != self.dim {
            return Err(Error::other(
                "the encoder didn't give one summary of 2d values at each wanted layer and an embedding of d values",
            ));
        }

        let row = hook.row;
        self.push(&row, &embedding)
    }
}

/// A hook that copies the summaries of the wanted layers into one row, in the order of the layers.
struct Recorder<'a> {
    layers: &'a [usize],
    width: usize,
    row: Vec<f32>,
    seen: Vec<bool>,
    /// Whether the encoder gave a summary of another size, or one layer twice.
    bad: bool,
}

impl LayerHook for Recorder<'_> {
    fn wants(&self, layer: usize) -> bool {
        self.layers.contains(&layer)
    }

    fn on_layer(&mut self, layer: usize, pooled: &[f32]) {
        let Some(i) = self.layers.iter().position(|&l| l == layer) else {
            return;
        };

        if pooled.len() != self.width || self.seen[i] {
            self.bad = true;
            return;
        }

        self.row[i * self.width..(i + 1) * self.width].copy_from_slice(pooled);
        self.seen[i] = true;
    }
}

/// What a fit chose at one layer.
#[derive(Clone, Debug, PartialEq)]
pub struct Choice {
    pub layer: usize,
    /// The penalty of the final fit.
    pub lambda: f64,
    /// Each penalty's mean cosine between its guesses and the held-out final embeddings, in the order of the
    /// penalties. It is empty when there was only one penalty.
    pub held_out: Vec<f64>,
}

/// The predictors of a fit, and what it chose at each layer, in the order of the layers.
#[derive(Clone, Debug, PartialEq)]
pub struct Fit {
    pub predictors: Predictors,
    pub choices: Vec<Choice>,
}

/// Runs `encoder` on every text and fits one predictor per layer of `params`, for the model id `model`, eg.
/// "Snowflake/snowflake-arctic-embed-m-v1.5". It checks the settings before it runs the model, and bad ones give
/// `InvalidInput`.
pub fn fit<E: LayerEncoder + ?Sized, S: AsRef<str>>(
    encoder: &mut E,
    model: &str,
    texts: &[S],
    params: &TrainParams,
) -> Result<Fit, Error> {
    params.check()?;
    let mut pairs = Pairs::new(encoder.dim(), &params.layers)?;

    for text in texts {
        pairs.encode(encoder, text.as_ref())?;
    }

    refit(model, &pairs, params)
}

/// Fits one predictor per layer of `params` from `pairs`, eg. the pairs of a query log, for the model id `model`.
/// Settings that don't fit the pairs give `InvalidInput`, eg. a layer the pairs don't hold, or too few pairs to hold
/// some out.
pub fn refit(model: &str, pairs: &Pairs, params: &TrainParams) -> Result<Fit, Error> {
    params.check()?;

    if pairs.is_empty() {
        return Err(invalid("the fit needs at least one pair"));
    }

    let par = params.par();
    let mut ridges = Vec::with_capacity(params.layers.len());
    let mut choices = Vec::with_capacity(params.layers.len());

    for &layer in &params.layers {
        let x = pairs
            .states(layer)
            .ok_or_else(|| invalid(&format!("the pairs hold no summaries at layer {layer}")))?;
        let table = Table {
            x,
            y: pairs.embeddings(),
            width: 2 * pairs.dim(),
            dim: pairs.dim(),
        };
        let (ridge, choice) = fit_layer(&table, layer, params, par)?;
        ridges.push(ridge);
        choices.push(choice);
    }

    Ok(Fit {
        predictors: Predictors::new(model, ridges)?,
        choices,
    })
}

/// One layer's summaries and the final embeddings, a row per pair.
struct Table<'a> {
    x: &'a [f32],
    y: &'a [f32],
    width: usize,
    dim: usize,
}

impl Table<'_> {
    fn len(&self) -> usize {
        self.y.len() / self.dim
    }
}

/// Picks the penalty on the split when there are several, then fits on every pair.
fn fit_layer(table: &Table, layer: usize, params: &TrainParams, par: Par) -> Result<(Ridge, Choice), Error> {
    let n = table.len();
    let (lambda, held_out) = match params.lambdas[..] {
        [lambda] => (lambda, Vec::new()),
        _ => {
            let cut = (params.split * n as f64) as usize;
            if cut == 0 || cut >= n {
                return Err(invalid("too few pairs to hold some out"));
            }
            let normal = Normal::new(table, 0..cut, par);
            let mut held_out = Vec::with_capacity(params.lambdas.len());
            for &lambda in &params.lambdas {
                let w = normal.solve(lambda, par)?;
                held_out.push(normal.cosine(table, cut..n, &w, par));
            }
            (params.lambdas[best(&held_out)], held_out)
        }
    };
    let normal = Normal::new(table, 0..n, par);
    let w = normal.solve(lambda, par)?;
    let ridge = normal.fold(layer, &w, lambda)?;
    Ok((
        ridge,
        Choice {
            layer,
            lambda,
            held_out,
        },
    ))
}

/// The position of the highest held-out cosine, the first of equal ones.
fn best(held_out: &[f64]) -> usize {
    (1..held_out.len()).fold(0, |best, i| if held_out[i] > held_out[best] { i } else { best })
}

/// The normal equations of a fit over some of the pairs, and the standardization they use.
struct Normal {
    /// n, the number of pairs.
    rows: usize,
    /// Each summary value's mean.
    mean: Vec<f64>,
    /// Each summary value's standard deviation, plus `SPREAD_FLOOR`.
    scale: Vec<f64>,
    /// Ȳ, the mean final embedding.
    ybar: Vec<f64>,
    /// Xsᵀ·Xs, 2d × 2d.
    gram: Mat<f64>,
    /// Xsᵀ·(Y − Ȳ), 2d × d.
    cross: Mat<f64>,
}

impl Normal {
    fn new(t: &Table, rows: Range<usize>, par: Par) -> Normal {
        let mean = means(t.x, t.width, rows.clone());
        let ybar = means(t.y, t.dim, rows.clone());
        let mut scale = vec![0f64; t.width];

        for row in t.x[rows.start * t.width..rows.end * t.width].chunks_exact(t.width) {
            for ((s, &v), &m) in scale.iter_mut().zip(row).zip(&mean) {
                let d = f64::from(v) - m;
                *s += d * d;
            }
        }

        let n = rows.len() as f64;
        scale.iter_mut().for_each(|s| *s = (*s / n).sqrt() + SPREAD_FLOOR);
        let mut gram = Mat::<f64>::zeros(t.width, t.width);
        let mut cross = Mat::<f64>::zeros(t.width, t.dim);
        let mut xs = vec![0f64; BLOCK * t.width];
        let mut yc = vec![0f64; BLOCK * t.dim];

        for (i, start) in rows.clone().step_by(BLOCK).enumerate() {
            let m = BLOCK.min(rows.end - start);
            let xs = standardize(t, start..start + m, &mean, &scale, &mut xs);
            let yc = &mut yc[..m * t.dim];
            let y = &t.y[start * t.dim..(start + m) * t.dim];
            for ((c, &v), &b) in yc.iter_mut().zip(y).zip(ybar.iter().cycle()) {
                *c = f64::from(v) - b;
            }
            let yc = MatRef::from_row_major_slice(yc, m, t.dim);
            let accum = if i == 0 { Accum::Replace } else { Accum::Add };
            matmul(gram.as_mut(), accum, xs.transpose(), xs, 1.0, par);
            matmul(cross.as_mut(), accum, xs.transpose(), yc, 1.0, par);
        }

        Normal {
            rows: rows.len(),
            mean,
            scale,
            ybar,
            gram,
            cross,
        }
    }

    /// Wᵀ for the penalty `lambda`, 2d × d, from the Cholesky factor of Xsᵀ·Xs + λ·n·I.
    fn solve(&self, lambda: f64, par: Par) -> Result<Mat<f64>, Error> {
        let width = self.gram.nrows();
        let ridge = lambda * self.rows as f64;
        // The factorization reads only the lower triangle, so the upper one stays 0.
        let mut factor = Mat::<f64>::zeros(width, width);

        for j in 0..width {
            for i in j..width {
                factor[(i, j)] = self.gram[(i, j)];
            }
            factor[(j, j)] += ridge;
        }

        let scratch = llt::factor::cholesky_in_place_scratch::<f64>(width, par, Default::default());
        llt::factor::cholesky_in_place(
            factor.as_mut(),
            Default::default(),
            par,
            MemStack::new(&mut MemBuffer::new(scratch)),
            Default::default(),
        )
        .map_err(|_| Error::other("the penalized normal equations aren't positive definite"))?;

        // The factorization may leave values above the diagonal, which the solve would read as part of the factor.
        for j in 1..width {
            for i in 0..j {
                factor[(i, j)] = 0.0;
            }
        }

        let mut w = self.cross.clone();
        let scratch = llt::solve::solve_in_place_scratch::<f64>(width, w.ncols(), par);
        llt::solve::solve_in_place(
            factor.as_ref(),
            w.as_mut(),
            par,
            MemStack::new(&mut MemBuffer::new(scratch)),
        );
        Ok(w)
    }

    /// The mean cosine between the guesses of `w`, i.e. Wᵀ, for the pairs in `rows` and their final embeddings.
    fn cosine(&self, t: &Table, rows: Range<usize>, w: &Mat<f64>, par: Par) -> f64 {
        let mut xs = vec![0f64; BLOCK * t.width];
        let mut guesses = vec![0f64; BLOCK * t.dim];
        let mut sum = 0f64;

        for start in rows.clone().step_by(BLOCK) {
            let m = BLOCK.min(rows.end - start);
            let xs = standardize(t, start..start + m, &self.mean, &self.scale, &mut xs);
            let guesses = &mut guesses[..m * t.dim];
            let out = MatMut::from_row_major_slice_mut(&mut *guesses, m, t.dim);
            matmul(out, Accum::Replace, xs, w.as_ref(), 1.0, par);
            for (r, guess) in guesses.chunks_exact_mut(t.dim).enumerate() {
                guess.iter_mut().zip(&self.ybar).for_each(|(g, &b)| *g += b);
                let truth = &t.y[(start + r) * t.dim..(start + r + 1) * t.dim];
                sum += cosine(guess, truth);
            }
        }

        sum / rows.len() as f64
    }

    /// The predictor of `w`, i.e. Wᵀ, with the standardization folded in, so W′ = W·diag(1/σ) and b′ = Ȳ − W′·μ_X, as
    /// f32 values. b′ comes from W′ after its rounding, so W′·x + b′ stays W′·(x − μ_X) + Ȳ where x is large and
    /// varies little.
    fn fold(&self, layer: usize, w: &Mat<f64>, lambda: f64) -> Result<Ridge, Error> {
        let (width, dim) = (w.nrows(), w.ncols());
        let mut weights = vec![0f32; dim * width];
        let mut bias = vec![0f32; dim];

        for (j, (row, b)) in weights.chunks_exact_mut(width).zip(&mut bias).enumerate() {
            let mut shift = 0f64;
            for (i, out) in row.iter_mut().enumerate() {
                *out = (w[(i, j)] / self.scale[i]) as f32;
                shift += f64::from(*out) * self.mean[i];
            }
            *b = (self.ybar[j] - shift) as f32;
        }

        Ridge::new(layer, dim, weights, bias, lambda)
            .map_err(|_| Error::other("the fit gave a predictor with values that aren't finite"))
    }
}

/// The mean of each column of `table`, rows of `width` values, over `rows`.
fn means(table: &[f32], width: usize, rows: Range<usize>) -> Vec<f64> {
    let mut sum = vec![0f64; width];

    for row in table[rows.start * width..rows.end * width].chunks_exact(width) {
        sum.iter_mut().zip(row).for_each(|(s, &v)| *s += f64::from(v));
    }

    let n = rows.len() as f64;
    sum.iter_mut().for_each(|s| *s /= n);
    sum
}

/// The summaries of the pairs in `rows`, standardized into `buffer`, as a matrix of one row per pair.
fn standardize<'a>(
    t: &Table,
    rows: Range<usize>,
    mean: &[f64],
    scale: &[f64],
    buffer: &'a mut [f64],
) -> MatRef<'a, f64> {
    let m = rows.len();
    let x = &t.x[rows.start * t.width..rows.end * t.width];
    let out = &mut buffer[..m * t.width];

    for (o, x) in out.chunks_exact_mut(t.width).zip(x.chunks_exact(t.width)) {
        for (((o, &v), &mu), &s) in o.iter_mut().zip(x).zip(mean).zip(scale) {
            *o = (f64::from(v) - mu) / s;
        }
    }

    MatRef::from_row_major_slice(out, m, t.width)
}

fn cosine(a: &[f64], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);

    for (&x, &y) in a.iter().zip(b) {
        let y = f64::from(y);
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }

    ab / (aa * bb).sqrt()
}

/// Training texts that need no real queries, i.e. the first 6 to 24 words of `count` different documents, with the
/// number of words drawn evenly for each. Documents without words are never drawn. The same documents, count and seed
/// give the same texts, in the order they were drawn. Fewer than `count` documents with words give `InvalidInput`.
pub fn openings<S: AsRef<str>>(documents: &[S], count: usize, seed: u64) -> Result<Vec<String>, Error> {
    let mut pool: Vec<usize> = (0..documents.len())
        .filter(|&i| documents[i].as_ref().split_whitespace().next().is_some())
        .collect();

    if pool.len() < count {
        return Err(invalid(&format!(
            "only {} documents have words, and {count} openings were asked for",
            pool.len()
        )));
    }

    let mut rng = SplitMix64(seed);
    let mut texts = Vec::with_capacity(count);

    for i in 0..count {
        let j = i + rng.below(pool.len() - i);
        pool.swap(i, j);
        let words = 6 + rng.below(19);
        let opening: Vec<&str> = documents[pool[i]].as_ref().split_whitespace().take(words).collect();
        texts.push(opening.join(" "));
    }

    Ok(texts)
}

/// A small seeded generator (SplitMix64).
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A value in 0..n, for n above 0, from the top bits of the product of a random 64-bit value and n.
    fn below(&mut self, n: usize) -> usize {
        ((u128::from(self.next()) * n as u128) >> 64) as usize
    }
}

/// A log of training pairs, eg. of real queries, that grows one pair at a time and that [`refit`] fits from, through
/// [`QueryLog::read`]. Each record holds one query's summaries at the log's layers and its final embedding, with a
/// checksum of its own. A record cut short at the end, eg. by a crash, is dropped when the log is opened or read. The
/// file holds these fields in order, with numbers little-endian.
///
/// | Field | Bytes |
/// |---|---|
/// | The mark `SWPAIRS` and a zero byte | 8 |
/// | The format version, 1 | 4 |
/// | d | 4 |
/// | The number of layers k, then the layers | 4 + 4k |
/// | The CRC-32 (cyclic redundancy check) of the header's bytes before it | 4 |
/// | Per pair, its summaries at the layers in order and its final embedding, then their CRC-32 | 4(2dk + d) + 4 |
pub struct QueryLog {
    file: File,
    dim: usize,
    layers: Vec<usize>,
    /// One record's bytes, kept between appends.
    record: Vec<u8>,
}

impl QueryLog {
    /// Opens the log at `path` for appending, and makes it if it doesn't exist, for the summaries at `layers` of a
    /// model with `dim` dimensions. A log of other layers or another size gives `InvalidInput`, and a file that isn't
    /// a whole log header gives `InvalidData`.
    pub fn open(path: &Path, dim: usize, layers: &[usize]) -> Result<QueryLog, Error> {
        Pairs::new(dim, layers)?;
        let mut file = OpenOptions::new().read(true).append(true).create(true).open(path)?;
        let len = file.metadata()?.len();
        let header = log_header(dim, layers);

        if len == 0 {
            file.write_all(&header)?;
        } else {
            let found =
                read_log_header(&mut file).map_err(|e| Error::new(e.kind(), format!("{}: {e}", path.display())))?;
            if found != (dim, layers.to_vec()) {
                return Err(invalid(&format!(
                    "{} logs other layers or another size",
                    path.display()
                )));
            }
            let record = record_len(dim, layers.len()) as u64;
            let start = header.len() as u64;
            let whole = start + (len - start) / record * record;
            if whole != len {
                file.set_len(whole)?;
            }
        }

        Ok(QueryLog {
            file,
            dim,
            layers: layers.to_vec(),
            record: Vec::with_capacity(record_len(dim, layers.len())),
        })
    }

    /// Appends one pair, i.e. a query's summaries at the log's layers, 2d values each in the order of the layers, and
    /// its final embedding. Wrong lengths, or a value that isn't finite, give `InvalidInput`. The record is durable
    /// once [`QueryLog::sync`] returns.
    pub fn append(&mut self, states: &[f32], embedding: &[f32]) -> Result<(), Error> {
        let width = 2 * self.dim;

        if states.len() != self.layers.len() * width || embedding.len() != self.dim {
            return Err(invalid(&format!(
                "a record holds {} summaries of {width} values and an embedding of {} values",
                self.layers.len(),
                self.dim
            )));
        }

        if !states.iter().chain(embedding).all(|x| x.is_finite()) {
            return Err(invalid("a record must hold only finite values"));
        }

        self.record.clear();

        for x in states.iter().chain(embedding) {
            self.record.extend_from_slice(&x.to_le_bytes());
        }

        let crc = crc32fast::hash(&self.record);
        self.record.extend_from_slice(&crc.to_le_bytes());
        self.file.write_all(&self.record)
    }

    /// Waits until every appended pair is on the disk.
    pub fn sync(&mut self) -> Result<(), Error> {
        self.file.sync_data()
    }

    /// Every whole pair of the log at `path`. A record cut short at the end is left out, and a header or a record
    /// that doesn't match its checksum gives `InvalidData`.
    pub fn read(path: &Path) -> Result<Pairs, Error> {
        let at = |e: Error| Error::new(e.kind(), format!("{}: {e}", path.display()));
        let mut reader = BufReader::with_capacity(1 << 20, File::open(path).map_err(at)?);
        let (dim, layers) = read_log_header(&mut reader).map_err(at)?;
        let mut pairs = Pairs::new(dim, &layers)?;
        let mut record = vec![0u8; record_len(dim, layers.len())];
        let mut values = vec![0f32; layers.len() * 2 * dim + dim];

        loop {
            match reader.read_exact(&mut record) {
                Ok(()) => {}
                Err(e) if e.kind() == ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(at(e)),
            }
            let (body, crc) = record.split_at(record.len() - 4);
            if crc32fast::hash(body) != u32::from_le_bytes([crc[0], crc[1], crc[2], crc[3]]) {
                return Err(at(damaged(&format!(
                    "the record of pair {} doesn't match its checksum",
                    pairs.len()
                ))));
            }
            for (v, b) in values.iter_mut().zip(body.as_chunks::<4>().0) {
                *v = f32::from_le_bytes(*b);
            }
            let (states, embedding) = values.split_at(layers.len() * 2 * dim);
            pairs.push(states, embedding).map_err(|e| at(damaged(&e.to_string())))?;
        }

        Ok(pairs)
    }
}

/// The header of a log of the summaries at `layers` of a model with `dim` dimensions. The layers and the size were
/// checked to fit 32 bits.
fn log_header(dim: usize, layers: &[usize]) -> Vec<u8> {
    let mut out = Vec::with_capacity(24 + 4 * layers.len());
    out.extend_from_slice(&LOG_MAGIC);
    out.extend_from_slice(&LOG_VERSION.to_le_bytes());

    for n in [dim, layers.len()].iter().chain(layers) {
        out.extend_from_slice(&(*n as u32).to_le_bytes());
    }

    let crc = crc32fast::hash(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    out
}

/// The size and the layers that the log header at the start of `reader` names.
fn read_log_header(reader: &mut impl Read) -> Result<(usize, Vec<usize>), Error> {
    let cut_short = |e: Error| match e.kind() {
        ErrorKind::UnexpectedEof => damaged("is cut short inside its header"),
        _ => e,
    };
    let mut head = [0u8; 20];
    reader.read_exact(&mut head).map_err(cut_short)?;

    if head[..8] != LOG_MAGIC {
        return Err(damaged("isn't a log of training pairs"));
    }

    let word = |b: &[u8]| u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    let (version, dim, count) = (word(&head[8..]), word(&head[12..]) as usize, word(&head[16..]) as usize);

    if version != LOG_VERSION {
        return Err(Error::new(
            ErrorKind::Unsupported,
            format!("is a log of format {version}, and this build reads only format {LOG_VERSION}"),
        ));
    }

    if count > MAX_LAYERS || dim > MAX_DIM {
        return Err(damaged("names too many layers, or too large a size, in its header"));
    }

    let mut rest = vec![0u8; 4 * count + 4];
    reader.read_exact(&mut rest).map_err(cut_short)?;
    let (layer_bytes, crc) = rest.split_at(4 * count);
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&head);
    hasher.update(layer_bytes);

    if hasher.finalize() != word(crc) {
        return Err(damaged("has a header that doesn't match its checksum"));
    }

    let layers: Vec<usize> = layer_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| word(b) as usize)
        .collect();

    if dim == 0 || layers.is_empty() || !distinct(&layers) {
        return Err(damaged("has a header with no size, no layers, or a layer named twice"));
    }

    Ok((dim, layers))
}

/// The bytes of one log record for `layers` layers of a model with `dim` dimensions.
fn record_len(dim: usize, layers: usize) -> usize {
    4 * (layers * 2 * dim + dim) + 4
}

/// Whether no layer appears twice.
fn distinct(layers: &[usize]) -> bool {
    layers.iter().enumerate().all(|(i, l)| !layers[..i].contains(l))
}

fn invalid(message: &str) -> Error {
    Error::new(ErrorKind::InvalidInput, message.to_string())
}

fn damaged(message: &str) -> Error {
    Error::new(ErrorKind::InvalidData, message.to_string())
}

#[cfg(test)]
#[path = "../tests/unit/train.rs"]
mod tests;
