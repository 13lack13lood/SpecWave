//! Synthetic guesses of a query's final embedding, for an index whose queries have no model states, eg. Cohere's 1M
//! embeddings. A guess is the final embedding moved by noise and scaled to length 1,
//!
//! ```text
//! ê = unit(e + s·(√a·n_iso + √(1 − a)·n_data))
//! ```
//!
//! where e is the final embedding, of length 1, n_iso is Gaussian noise of the same size in every direction, n_data is
//! Gaussian noise with the covariance of the corpus, both with an expected squared length of 1, a is the isotropic
//! share and s the noise's scale. At the same cosine with e, noise shaped like the data moves a guess toward other
//! documents more than isotropic noise does, so the guess misses more of e's clusters. The scale sets the guesses'
//! cosine with e, and the share then sets their posting coverage, the share of e's own clusters that a guess's wider
//! read covers.
//!
//! n_data is the sum of the differences of m random documents from the corpus mean, each weighted by a standard normal
//! value and divided by √(m·T), where T is the documents' mean squared distance from the mean. Its covariance is the
//! corpus's, divided by T.
//!
//! The calibration fits both, per layer, on a dataset whose guesses came from a model. For each share it finds the
//! scale that gives the model's guesses' mean cosine, then it keeps the share whose posting coverage at the layer's β
//! comes nearest the model's. Every share and scale use the same noise draws, so the comparison is between settings,
//! not between draws.

use std::io::{Error, ErrorKind};

use coldivf::{Index, LeafId, Limits, SearchCtx, SearchParams};
use specwave::speculate::{SpecParams, wide_nprobe};

use crate::data::Fbin;

/// The rows a pass over the corpus reads at a time.
const CHUNK: usize = 20_000;

/// A guess's noise.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Noise {
    /// a, the isotropic share, from 0 to 1.
    pub share: f64,
    /// s, the noise's scale.
    pub scale: f64,
}

/// Noise shaped like a corpus.
pub struct DataNoise<'a> {
    corpus: &'a Fbin,
    /// The corpus mean, and 1/√(m·T).
    mean: Vec<f64>,
    weight: f64,
    /// m, the documents that one draw adds up.
    documents: usize,
}

impl<'a> DataNoise<'a> {
    /// The noise of `corpus`, from `documents` documents per draw. It reads the corpus once, a chunk at a time. An
    /// empty corpus, one whose documents are all the same, or no documents per draw give `InvalidInput`.
    pub fn new(corpus: &'a Fbin, documents: usize) -> Result<DataNoise<'a>, Error> {
        let (n, dim) = (corpus.len(), corpus.dim());

        if n == 0 || documents == 0 {
            return Err(invalid("data noise needs documents"));
        }

        let (mut sum, mut squares) = (vec![0f64; dim], 0f64);

        for start in (0..n).step_by(CHUNK) {
            let rows = &corpus.values()[start * dim..(start + CHUNK).min(n) * dim];
            for row in rows.chunks_exact(dim) {
                sum.iter_mut().zip(row).for_each(|(s, &x)| *s += f64::from(x));
                squares += row.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>();
            }
        }

        let mean: Vec<f64> = sum.iter().map(|s| s / n as f64).collect();
        let spread = squares / n as f64 - mean.iter().map(|m| m * m).sum::<f64>();

        if !(spread.is_finite() && spread > 1e-12) {
            return Err(invalid("the corpus's documents don't spread around their mean"));
        }

        Ok(DataNoise {
            corpus,
            mean,
            weight: 1.0 / (documents as f64 * spread).sqrt(),
            documents,
        })
    }

    pub fn dim(&self) -> usize {
        self.mean.len()
    }

    /// One draw, into `out`.
    pub fn draw(&self, rng: &mut Rng, out: &mut [f64]) {
        out.fill(0.0);

        for _ in 0..self.documents {
            let (row, g) = (self.corpus.row(rng.below(self.corpus.len())), rng.normal());
            for ((o, &x), m) in out.iter_mut().zip(row).zip(&self.mean) {
                *o += g * (f64::from(x) - m);
            }
        }

        out.iter_mut().for_each(|o| *o *= self.weight);
    }
}

/// Each query's noise, both kinds, drawn once, with the dot products that a guess's cosine needs.
pub struct Draws {
    dim: usize,
    /// One row of `dim` values per query.
    iso: Vec<f64>,
    data: Vec<f64>,
    /// Per query, ⟨e, n_iso⟩, ⟨e, n_data⟩, ‖n_iso‖², ‖n_data‖² and ⟨n_iso, n_data⟩.
    dots: Vec<[f64; 5]>,
}

impl Draws {
    /// The noise of each query of `queries`, rows of final embeddings, with the data noise `data` and the seed
    /// `seed`. Query i's noise depends only on the seed and i, so a subset of queries keeps its draws. Rows of
    /// another size than the noise's give `InvalidInput`.
    pub fn new(queries: &[f32], data: &DataNoise, seed: u64) -> Result<Draws, Error> {
        let dim = data.dim();

        if dim == 0 || !queries.len().is_multiple_of(dim) {
            return Err(invalid("the queries must be whole rows of the corpus's size"));
        }

        let n = queries.len() / dim;
        let (mut iso, mut noise) = (vec![0f64; n * dim], vec![0f64; n * dim]);
        let mut dots = Vec::with_capacity(n);
        let unit = 1.0 / (dim as f64).sqrt();

        for (q, e) in queries.chunks_exact(dim).enumerate() {
            let mut rng = Rng(seed ^ (q as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
            let (iso, noise) = (&mut iso[q * dim..(q + 1) * dim], &mut noise[q * dim..(q + 1) * dim]);
            iso.iter_mut().for_each(|x| *x = rng.normal() * unit);
            data.draw(&mut rng, noise);
            let e = unit_f64(e);
            let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
            dots.push([
                dot(&e, iso),
                dot(&e, noise),
                dot(iso, iso),
                dot(noise, noise),
                dot(iso, noise),
            ]);
        }

        Ok(Draws {
            dim,
            iso,
            data: noise,
            dots,
        })
    }

    pub fn len(&self) -> usize {
        self.dots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dots.is_empty()
    }

    /// The cosine of query `q`'s guess with its final embedding under `noise`.
    pub fn cosine(&self, q: usize, noise: &Noise) -> f64 {
        let [ei, ed, ii, dd, id] = self.dots[q];
        let (a, b) = (noise.share.sqrt(), (1.0 - noise.share).sqrt());
        let (p, m) = (a * ei + b * ed, a * a * ii + b * b * dd + 2.0 * a * b * id);
        let s = noise.scale;
        (1.0 + s * p) / (1.0 + 2.0 * s * p + s * s * m).sqrt()
    }

    /// The mean cosine of the guesses with their final embeddings under `noise`.
    pub fn mean_cosine(&self, noise: &Noise) -> f64 {
        (0..self.len()).map(|q| self.cosine(q, noise)).sum::<f64>() / self.len().max(1) as f64
    }

    /// Writes query `q`'s guess, from its final embedding `e`, into `out`.
    pub fn guess(&self, q: usize, e: &[f32], noise: &Noise, out: &mut [f32]) {
        let (a, b) = (noise.share.sqrt(), (1.0 - noise.share).sqrt());
        let (iso, data) = (&self.iso[q * self.dim..], &self.data[q * self.dim..]);
        let e = unit_f64(e);
        let moved: Vec<f64> = (0..self.dim)
            .map(|i| e[i] + noise.scale * (a * iso[i] + b * data[i]))
            .collect();
        let norm = moved.iter().map(|x| x * x).sum::<f64>().sqrt();
        out.iter_mut().zip(&moved).for_each(|(o, x)| *o = (x / norm) as f32);
    }

    /// The scale at which the guesses with the isotropic share `share` have the mean cosine `cosine`, found by
    /// halving an interval. A cosine that isn't above 0 and below 1 gives `InvalidInput`.
    pub fn scale_for(&self, share: f64, cosine: f64) -> Result<f64, Error> {
        if !(cosine > 0.0 && cosine < 1.0) || self.is_empty() {
            return Err(invalid("a target cosine is above 0 and below 1, for some queries"));
        }

        let at = |scale| self.mean_cosine(&Noise { share, scale });
        let mut hi = 1.0;

        while at(hi) > cosine {
            hi *= 2.0;
            if hi > 1e6 {
                return Err(invalid("no scale of the noise gets the cosine that low"));
            }
        }

        let mut lo = 0.0;

        for _ in 0..100 {
            let mid = 0.5 * (lo + hi);
            match at(mid) > cosine {
                true => lo = mid,
                false => hi = mid,
            }
        }

        Ok(0.5 * (lo + hi))
    }
}

/// The mean posting coverage of the guesses under `noise` on `index`, i.e. the share of the `nprobe` clusters nearest
/// each final embedding that are among the ⌊(1 + beta)·nprobe⌋ nearest its guess. `queries` holds the final
/// embeddings, one row per draw.
pub fn posting_coverage(
    index: &Index,
    queries: &[f32],
    draws: &Draws,
    noise: &Noise,
    nprobe: usize,
    beta: f64,
) -> Result<f64, Error> {
    let (dim, wide) = (draws.dim, wide_nprobe(nprobe, beta));
    let limits = Limits {
        max_k: 1,
        max_nprobe: wide.max(nprobe),
        max_rerank: 1,
    };
    let mut ctx = SearchCtx::new(limits);
    let mut guess = vec![0f32; dim];
    let mut total = 0.0;

    for (q, e) in queries.chunks_exact(dim).enumerate().take(draws.len()) {
        draws.guess(q, e, noise, &mut guess);
        total += coverage_of(index, &mut ctx, e, &guess, nprobe, wide)?;
    }

    Ok(total / draws.len().max(1) as f64)
}

/// The posting coverage of one guess on `index`, i.e. the share of the `nprobe` clusters nearest the final embedding
/// `e` that are among the `wide` nearest `guess`. `ctx` must allow both.
pub fn coverage_of(
    index: &Index,
    ctx: &mut SearchCtx,
    e: &[f32],
    guess: &[f32],
    nprobe: usize,
    wide: usize,
) -> Result<f64, Error> {
    let (mut own, mut read): (Vec<LeafId>, Vec<LeafId>) = (Vec::new(), Vec::new());
    index.route(e, nprobe, None, ctx, &mut own)?;
    index.route(guess, wide, None, ctx, &mut read)?;
    let covered = own.iter().filter(|l| read.contains(l)).count();
    Ok(covered as f64 / own.len().max(1) as f64)
}

/// The rerank coverage of one guess on `index` with the worker's settings `params`, i.e. the share of the rows that a
/// normal search of the final embedding `e` rescores that are among the rows the guess prefetches. Those are the rows
/// of the guess's best rerank_mult·rerank candidates by their 1-bit estimate, over the ⌊(1 + beta)·nprobe⌋ clusters
/// nearest it. `ctx` must allow both searches.
pub fn rerank_coverage_of(
    index: &Index,
    ctx: &mut SearchCtx,
    e: &[f32],
    guess: &[f32],
    params: &SpecParams,
) -> Result<f64, Error> {
    let (base, rows) = (&params.base, params.prefetch_rows());
    let normal = SearchParams {
        k: base.rerank,
        ..*base
    };
    let rescored: Vec<u64> = index.search(e, &normal, ctx)?.iter().map(|h| h.id).collect();
    let mut leaves = Vec::with_capacity(params.guess_nprobe());
    index.route(guess, params.guess_nprobe(), base.reach, ctx, &mut leaves)?;
    let prefetched: Vec<u64> = index
        .fetch(&leaves, ctx)?
        .search(guess, rows, rows)?
        .iter()
        .map(|h| h.id)
        .collect();
    let covered = rescored.iter().filter(|id| prefetched.contains(id)).count();
    Ok(covered as f64 / rescored.len().max(1) as f64)
}

/// A layer's calibrated noise, with what it was fitted to and what it gives.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fitted {
    pub noise: Noise,
    /// The posting coverage and the mean cosine that the noise gives.
    pub coverage: f64,
    pub cosine: f64,
    /// The posting coverage with isotropic noise alone, and with data noise alone, at the same cosine.
    pub isotropic: f64,
    pub shaped: f64,
}

/// The noise among the isotropic shares `shares` whose guesses come nearest the posting coverage `coverage` at
/// `nprobe` and `beta` on `index`, each at the scale that gives the mean cosine `cosine`. No shares, or a share out of
/// 0 to 1, give `InvalidInput`.
pub fn calibrate(
    index: &Index,
    queries: &[f32],
    draws: &Draws,
    target: (f64, f64),
    nprobe: usize,
    beta: f64,
    shares: &[f64],
) -> Result<Fitted, Error> {
    let (cosine, coverage) = target;

    if shares.is_empty() || shares.iter().any(|s| !(0.0..=1.0).contains(s)) {
        return Err(invalid("the shares to try are from 0 to 1"));
    }

    let at = |share: f64| -> Result<(Noise, f64), Error> {
        let noise = Noise {
            share,
            scale: draws.scale_for(share, cosine)?,
        };
        Ok((noise, posting_coverage(index, queries, draws, &noise, nprobe, beta)?))
    };
    let mut best: Option<(Noise, f64)> = None;

    for &share in shares {
        let (noise, got) = at(share)?;
        if best.is_none_or(|(_, b)| (got - coverage).abs() < (b - coverage).abs()) {
            best = Some((noise, got));
        }
    }

    let (noise, got) = best.ok_or_else(|| invalid("no share to try"))?;
    Ok(Fitted {
        noise,
        coverage: got,
        cosine: draws.mean_cosine(&noise),
        isotropic: at(1.0)?.1,
        shaped: at(0.0)?.1,
    })
}

/// The shares that the calibration tries by default, 0 to 1 in steps of 0.05.
pub fn shares() -> Vec<f64> {
    (0..=20).map(|i| i as f64 / 20.0).collect()
}

/// The targets of one layer, from coverage results, i.e. the guesses' mean cosine with their final embeddings and
/// their mean posting coverage at the layer's β, over every seed of the grid's cells of `fit` on `dataset` with
/// `clustering` at `nprobe` and the label `r_label` of R, the number of candidates that a normal search rescores. A
/// layer without such cells has none.
pub fn targets(lines: &[String], key: &Key, layer: usize, beta: f64) -> Option<(f64, f64)> {
    let cells: Vec<serde_json::Value> = lines
        .iter()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|v| {
            v["record"] == "grid"
                && v["dataset"] == key.dataset.as_str()
                && v["clustering"] == key.clustering.as_str()
                && v["layer"].as_u64() == Some(layer as u64)
                && v["guess"]
                    .as_str()
                    .is_some_and(|g| crate::verdict::fit_of(g) == key.fit)
                && v["nprobe"].as_u64() == Some(key.nprobe as u64)
                && v["r_label"] == key.r_label.as_str()
        })
        .collect();
    let mean = |values: Vec<f64>| match values.is_empty() {
        true => None,
        false => Some(values.iter().sum::<f64>() / values.len() as f64),
    };
    let cosine = mean(cells.iter().filter_map(|v| v["cos"].as_f64()).collect())?;
    let at_beta = cells.iter().filter(|v| v["beta"].as_f64() == Some(beta));
    let coverage = mean(at_beta.filter_map(|v| v["posting_coverage"].as_f64()).collect())?;
    Some((cosine, coverage))
}

/// The cells of the coverage grid that a calibration reads.
#[derive(Clone, Debug, PartialEq)]
pub struct Key {
    pub dataset: String,
    pub fit: String,
    pub clustering: String,
    pub nprobe: usize,
    pub r_label: String,
}

/// The command line of the `synthetic` binary.
#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    /// Fits each layer's noise on a dataset with real guesses, and appends a line for each to `out`.
    Calibrate {
        index: String,
        queries: String,
        corpus: String,
        coverage: Vec<String>,
        key: Key,
        rerank: usize,
        /// The layers to fit, or every layer that passes.
        layers: Option<Vec<usize>>,
        documents: usize,
        seed: u64,
        out: String,
    },
    /// Writes each fitted layer's guesses for the final embeddings in `queries` as `<dir>/<label>.L<layer>.fbin`.
    Guesses {
        noise: String,
        queries: String,
        corpus: String,
        dir: String,
        label: String,
        documents: usize,
        seed: u64,
    },
}

/// Every argument name.
const NAMES: [&str; 16] = [
    "index",
    "queries",
    "corpus",
    "coverage",
    "dataset",
    "fit",
    "clustering",
    "nprobe",
    "rerank",
    "layers",
    "documents",
    "seed",
    "out",
    "noise",
    "dir",
    "label",
];

impl Command {
    /// Reads `args`, the arguments after the program's name, the first of which is `calibrate` or `guesses`. `repo`
    /// is this repository's folder, which holds the default results file. A name that isn't known or is given twice, a
    /// missing or bad value, or another command gives `InvalidInput`.
    pub fn parse(args: &[String], repo: &std::path::Path) -> Result<Command, Error> {
        let (command, rest) = args.split_first().ok_or_else(|| invalid("give calibrate or guesses"))?;
        let mut values: Vec<(&str, &str)> = Vec::new();
        let mut it = rest.iter();

        while let Some(arg) = it.next() {
            let name = arg
                .strip_prefix("--")
                .filter(|n| NAMES.contains(n))
                .ok_or_else(|| invalid(&format!("unknown argument {arg}")))?;
            let value = it.next().ok_or_else(|| invalid(&format!("--{name} needs a value")))?;
            if values.iter().any(|(n, _)| *n == name) {
                return Err(invalid(&format!("--{name} is given twice")));
            }
            values.push((name, value));
        }

        let get = |name: &str| values.iter().find(|(n, _)| *n == name).map(|&(_, v)| v.to_string());
        let need = |name: &str| get(name).ok_or_else(|| invalid(&format!("--{name} is required")));
        let number = |name: &str, default: u64| -> Result<u64, Error> {
            get(name).map_or(Ok(default), |v| {
                v.trim()
                    .parse()
                    .map_err(|_| invalid(&format!("--{name} has a bad value: {v}")))
            })
        };
        let documents = number("documents", 64)? as usize;

        if documents == 0 {
            return Err(invalid("--documents is at least 1"));
        }

        match command.as_str() {
            "calibrate" => {
                let layers = get("layers")
                    .map(|v| {
                        v.split(',')
                            .map(|l| {
                                l.trim()
                                    .parse()
                                    .map_err(|_| invalid(&format!("--layers has a bad value: {v}")))
                            })
                            .collect::<Result<Vec<usize>, Error>>()
                    })
                    .transpose()?;
                let rerank = number("rerank", 29)? as usize;
                Ok(Command::Calibrate {
                    index: need("index")?,
                    queries: need("queries")?,
                    corpus: need("corpus")?,
                    coverage: need("coverage")?.split(',').map(str::to_string).collect(),
                    key: Key {
                        dataset: get("dataset").unwrap_or_else(|| "fiqa".to_string()),
                        fit: get("fit").unwrap_or_else(|| "openings".to_string()),
                        clustering: get("clustering").unwrap_or_else(|| "l2".to_string()),
                        nprobe: number("nprobe", 9)? as usize,
                        r_label: rerank.to_string(),
                    },
                    rerank,
                    layers,
                    documents,
                    seed: number("seed", 1)?,
                    out: get("out").unwrap_or_else(|| repo.join("results/synthetic.jsonl").display().to_string()),
                })
            }
            "guesses" => Ok(Command::Guesses {
                noise: need("noise")?,
                queries: need("queries")?,
                corpus: need("corpus")?,
                dir: need("dir")?,
                label: need("label")?,
                documents,
                seed: number("seed", 1)?,
            }),
            other => Err(invalid(&format!("the command is calibrate or guesses, not {other}"))),
        }
    }
}

/// Each fitted layer's noise in `lines`, the result lines of a calibration, as (layer, noise). When a layer has
/// several lines, the last one counts.
pub fn fitted(lines: &[String]) -> Vec<(usize, Noise)> {
    let mut by_layer: std::collections::BTreeMap<usize, Noise> = std::collections::BTreeMap::new();

    for line in lines {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if v["bench"] != "synthetic" || v["record"] != "noise" {
            continue;
        }
        if let (Some(layer), Some(share), Some(scale)) = (v["layer"].as_u64(), v["share"].as_f64(), v["scale"].as_f64())
        {
            by_layer.insert(layer as usize, Noise { share, scale });
        }
    }

    by_layer.into_iter().collect()
}

/// `e` as f64, scaled to length 1.
fn unit_f64(e: &[f32]) -> Vec<f64> {
    let norm = e.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt().max(1e-30);
    e.iter().map(|&x| f64::from(x) / norm).collect()
}

/// A small seeded generator (SplitMix64), with standard normal values by the Box-Muller transform.
pub struct Rng(pub u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A value in 0..n, for n above 0.
    fn below(&mut self, n: usize) -> usize {
        ((u128::from(self.next()) * n as u128) >> 64) as usize
    }

    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u1, u2) = (1.0 - self.uniform(), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn invalid(message: &str) -> Error {
    Error::new(ErrorKind::InvalidInput, message.to_string())
}

#[cfg(test)]
#[path = "../tests/unit/synthetic.rs"]
mod tests;
