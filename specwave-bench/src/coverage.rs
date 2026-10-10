//! Coverage of guessed embeddings, measured with ColdIVF's own codes.
//!
//! A guess ê of a query's final embedding e reads the wide = ⌊(1 + β)·nprobe⌋ clusters nearest to it, the set L̂,
//! where nprobe is the number of clusters a normal search reads and β is the share of extra clusters. Rounding down
//! keeps the extra clusters within β, and each cell also records the effective β, wide/nprobe − 1. When e arrives,
//! the search ranks all of L̂ with e and rescores the best R candidates with their 8-bit rows, and it reads no other
//! cluster. Before that, it prefetches the rows of the m·R best candidates for ê, where m is the prefetch multiple.
//! Each cell of the grid reports:
//! - posting coverage, the share of e's own nprobe clusters inside L̂
//! - recall@10 of that ranking, next to a normal search's at the same nprobe and R, and the gap between them
//! - rerank coverage (`rerank_coverage`), the share of the R rows that e rescores over L̂ that are among the m·R
//!   prefetched rows. The rows it misses are the ones the search reads once the model ends.
//! - rerank coverage of the normal search (`rerank_coverage_normal`), the share of the R rows that a normal search of e
//!   rescores over its own nprobe clusters that are among the m·R prefetched rows
//!
//! The gap and both rerank coverages come with their standard error over the queries and a 95% bootstrap interval.
//! Every cell resamples the same draws of the queries, and the gap pairs each query's two searches.
//!
//! The file index routes queries and reads any set of clusters. The in-memory index holds the same codes, and it runs
//! the searches that pick their own clusters, which return what the file would.

use std::cmp::Ordering;
use std::io::{Error, ErrorKind};
use std::ops::Range;
use std::path::Path;

use coldivf::{
    Anchor, BuildParams, Builder, Clusters, Hit, InMemoryIndex, Index, LeafId, LeafOrder, Limits, Metric, SearchCtx,
    SearchParams, VectorSource,
};

/// Recall counts the true `K` nearest neighbors among the first `K` hits.
pub const K: usize = 10;

/// The number of resamples of the queries behind each bootstrap interval.
pub const RESAMPLES: usize = 1000;

/// The seed of the resamples' draws.
pub const RESAMPLE_SEED: u64 = 1;

/// One dataset's index over one clustering, as a file and in memory. Both hold the same codes.
pub struct Indexes {
    pub file: Index,
    pub memory: InMemoryIndex,
    pub nlist: usize,
    /// The vector count of every cluster, largest first.
    sizes: Vec<usize>,
}

impl Indexes {
    /// Builds both from `src` and `clusters` with `threads` threads, and writes the file to `path`. Both use cosine,
    /// keep the clusters in id order, so a cluster's position in the file is its row in `clusters`, and rotate the
    /// vectors with the seed `seed`.
    pub fn build(
        src: &dyn VectorSource,
        clusters: &Clusters,
        path: &Path,
        threads: usize,
        seed: u64,
    ) -> Result<Indexes, Error> {
        // The clusters come from `clusters`, so the k-means settings go unused.
        let params = BuildParams {
            metric: Metric::Cosine,
            vectors_per_list: 160,
            parents: 1,
            order: LeafOrder::Id,
            page_size: 4096,
            threads,
            seed,
            anchor: Anchor::Cluster,
        };
        Builder::new(params.clone()).build_with_clusters(src, clusters, path)?;
        let file = Index::open(path)?;
        let memory = InMemoryIndex::build(&params, src, clusters)?;
        // The builds checked that every assignment names a cluster.
        let nlist = clusters.centroids.len() / clusters.dim;
        let mut sizes = vec![0; nlist];
        clusters.assignment.iter().for_each(|&c| sizes[c as usize] += 1);
        sizes.sort_unstable_by(|a, b| b.cmp(a));
        Ok(Indexes {
            file,
            memory,
            nlist,
            sizes,
        })
    }

    /// The rescoring depth that takes every vector of any `nprobe` clusters, i.e. the sizes of the `nprobe` largest
    /// added up, and at least `K`.
    pub fn ceiling_depth(&self, nprobe: usize) -> usize {
        self.sizes.iter().take(nprobe).sum::<usize>().max(K)
    }
}

/// Query embeddings and their true nearest neighbors.
#[derive(Clone, Copy)]
pub struct Queries<'a> {
    pub dim: usize,
    /// One embedding per query, one after another.
    pub vectors: &'a [f32],
    /// Each query's true nearest neighbors by input row, nearest first, `width` per query.
    pub truth: &'a [u32],
    pub width: usize,
}

impl<'a> Queries<'a> {
    /// Checks that `vectors` and `truth` hold the same number of queries, with at least `K` true neighbors each.
    pub fn new(dim: usize, vectors: &'a [f32], truth: &'a [u32], width: usize) -> Result<Queries<'a>, Error> {
        if dim == 0 || width < K || !vectors.len().is_multiple_of(dim) || !truth.len().is_multiple_of(width) {
            return Err(invalid(
                "the queries need whole rows and at least 10 true neighbors each",
            ));
        }

        if vectors.len() / dim != truth.len() / width {
            return Err(invalid("the queries and their true neighbors differ in number"));
        }

        Ok(Queries {
            dim,
            vectors,
            truth,
            width,
        })
    }

    pub fn len(&self) -> usize {
        self.vectors.len() / self.dim
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    fn vector(&self, q: usize) -> &'a [f32] {
        &self.vectors[q * self.dim..(q + 1) * self.dim]
    }

    /// The true `K` nearest neighbors of query `q`.
    fn truth(&self, q: usize) -> &'a [u32] {
        &self.truth[q * self.width..q * self.width + K]
    }
}

/// A rescoring depth R, with its label for the results, eg. "29" or "r384".
#[derive(Clone, Debug, PartialEq)]
pub struct Depth {
    pub label: String,
    pub rerank: usize,
}

/// A normal search's clusters, as a share of all clusters and as nprobe, and the rescoring depths to try with them.
#[derive(Clone, Debug, PartialEq)]
pub struct Base {
    pub fraction: f64,
    pub nprobe: usize,
    pub depths: Vec<Depth>,
}

/// The settings to measure. Every base is tried with every beta and every prefetch multiple.
#[derive(Clone, Debug, PartialEq)]
pub struct Grid {
    pub bases: Vec<Base>,
    pub betas: Vec<f64>,
    pub mults: Vec<usize>,
}

impl Grid {
    fn check(&self, nlist: usize) -> Result<(), Error> {
        if self.bases.is_empty() || self.betas.is_empty() || self.mults.is_empty() {
            return Err(invalid("the grid needs a base, a beta and a prefetch multiple"));
        }

        if self
            .bases
            .iter()
            .any(|b| b.nprobe == 0 || b.nprobe > nlist || b.depths.is_empty())
        {
            return Err(invalid("every base needs 1 to nlist clusters and a rescoring depth"));
        }

        if self.bases.iter().flat_map(|b| &b.depths).any(|d| d.rerank < K) {
            return Err(invalid("every rescoring depth must be at least 10"));
        }

        if self.mults.contains(&0) || self.betas.iter().any(|b| !(b.is_finite() && *b >= 0.0)) {
            return Err(invalid(
                "the multiples must be at least 1 and the betas finite and at least 0",
            ));
        }

        Ok(())
    }

    /// The most clusters that a guess reads.
    fn widest(&self, nlist: usize) -> usize {
        let wide = |b: &Base| self.betas.iter().map(|&beta| wide_nprobe(b.nprobe, beta, nlist)).max();
        self.bases.iter().filter_map(wide).max().unwrap_or(1)
    }

    fn deepest(&self) -> usize {
        self.bases
            .iter()
            .flat_map(|b| &b.depths)
            .map(|d| d.rerank)
            .max()
            .unwrap_or(K)
    }

    /// The most rows that a guess prefetches.
    fn most_prefetched(&self) -> usize {
        self.deepest() * self.mults.iter().copied().max().unwrap_or(1)
    }
}

/// The nprobe that reads `fraction` of `nlist` clusters, rounded half away from zero, from 1 to `nlist`.
pub fn base_nprobe(fraction: f64, nlist: usize) -> usize {
    ((fraction * nlist as f64).round() as usize).clamp(1, nlist.max(1))
}

/// The clusters that a guess reads for a base `nprobe`, ⌊(1 + beta)·nprobe⌋ as SpecWave's worker reads them, at most
/// `nlist`.
pub fn wide_nprobe(nprobe: usize, beta: f64, nlist: usize) -> usize {
    specwave::speculate::wide_nprobe(nprobe, beta).min(nlist)
}

/// The share of extra clusters that reading `wide` clusters adds to a base `nprobe`, wide/nprobe − 1.
pub fn effective_beta(nprobe: usize, wide: usize) -> f64 {
    wide as f64 / nprobe.max(1) as f64 - 1.0
}

/// The rescoring depths that a sweep tries, every depth from 10 to 100, then coarser steps up to 500.
pub fn sweep_depths() -> Vec<usize> {
    (10..=100)
        .chain((110..=300).step_by(10))
        .chain((350..=500).step_by(50))
        .collect()
}

/// Recall@10 of normal searches at `nprobe` for each rescoring depth of `reranks`, over every query.
pub fn sweep(
    indexes: &Indexes,
    queries: &Queries,
    nprobe: usize,
    reranks: &[usize],
    threads: usize,
) -> Result<Vec<f64>, Error> {
    if nprobe == 0 || nprobe > indexes.nlist || reranks.iter().any(|&r| r < K) {
        return Err(invalid("a sweep needs 1 to nlist clusters and depths of at least 10"));
    }

    let limits = Limits {
        max_k: K,
        max_nprobe: nprobe,
        max_rerank: reranks.iter().copied().max().unwrap_or(K),
    };
    let counts = parallel(queries.len(), threads, |range| {
        let mut ctx = SearchCtx::new(limits);
        let mut out = Vec::with_capacity(range.len());

        for q in range {
            let mut found = Vec::with_capacity(reranks.len());
            for &rerank in reranks {
                let params = SearchParams {
                    k: K,
                    nprobe,
                    rerank,
                    reach: None,
                };
                let hits = indexes.memory.search(queries.vector(q), &params, &mut ctx)?;
                found.push(true_hits(hits, queries.truth(q)));
            }
            out.push(found);
        }

        Ok(out)
    })?;
    let total = (queries.len() * K).max(1) as f64;

    Ok((0..reranks.len())
        .map(|i| counts.iter().map(|found| found[i]).sum::<usize>() as f64 / total)
        .collect())
}

/// Recall@10 when every vector of the `nprobe` clusters is rescored, the most that a normal search there finds.
pub fn ceiling(indexes: &Indexes, queries: &Queries, nprobe: usize, threads: usize) -> Result<f64, Error> {
    let all = indexes.ceiling_depth(nprobe);
    Ok(sweep(indexes, queries, nprobe, &[all], threads)?[0])
}

/// The smallest of the increasing depths `reranks` from which every recall in `recalls` is at least `target`.
pub fn depth_for(reranks: &[usize], recalls: &[f64], target: f64) -> Option<usize> {
    let mut depth = None;

    for (&rerank, &recall) in reranks.iter().zip(recalls).rev() {
        if recall < target {
            break;
        }
        depth = Some(rerank);
    }

    depth
}

/// What normal searches of the true embeddings read and find, for each query and base.
pub struct Exact {
    /// For each query, one entry per base.
    queries: Vec<Vec<Normal>>,
}

/// One normal search setting's results for one query.
struct Normal {
    /// The clusters it reads, in id order.
    clusters: Vec<LeafId>,
    /// One entry per rescoring depth of the base.
    depths: Vec<Rescored>,
}

struct Rescored {
    /// How many of the true 10 nearest neighbors it returns.
    found: usize,
    /// The ids of the vectors it rescores, in order.
    ids: Vec<u64>,
}

/// Runs every base's normal searches of the true embeddings on `threads` threads.
pub fn exact(indexes: &Indexes, queries: &Queries, grid: &Grid, threads: usize) -> Result<Exact, Error> {
    grid.check(indexes.nlist)?;

    let limits = Limits {
        max_k: grid.deepest(),
        max_nprobe: grid.widest(indexes.nlist),
        max_rerank: grid.deepest(),
    };
    let normal = parallel(queries.len(), threads, |range| {
        let mut ctx = SearchCtx::new(limits);
        let mut leaves = Vec::with_capacity(limits.max_nprobe);
        let mut out = Vec::with_capacity(range.len());

        for q in range {
            let e = queries.vector(q);
            let mut bases = Vec::with_capacity(grid.bases.len());
            for base in &grid.bases {
                indexes.file.route(e, base.nprobe, None, &mut ctx, &mut leaves)?;
                let mut clusters = leaves.clone();
                clusters.sort_unstable();
                let mut depths = Vec::with_capacity(base.depths.len());
                for depth in &base.depths {
                    let params = SearchParams {
                        k: depth.rerank,
                        nprobe: base.nprobe,
                        rerank: depth.rerank,
                        reach: None,
                    };
                    let hits = indexes.memory.search(e, &params, &mut ctx)?;
                    let mut ids = Vec::with_capacity(hits.len());
                    sorted_ids(hits, &mut ids);
                    depths.push(Rescored {
                        found: true_hits(hits, queries.truth(q)),
                        ids,
                    });
                }
                bases.push(Normal { clusters, depths });
            }
            out.push(bases);
        }

        Ok(out)
    })?;

    Ok(Exact { queries: normal })
}

/// A mean over the queries, with its standard error and its 95% bootstrap interval, from `lo` to `hi`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Estimate {
    pub mean: f64,
    pub se: f64,
    pub lo: f64,
    pub hi: f64,
}

/// Resamples of the queries, with repeats, for bootstrap intervals. Every estimate uses the same draws, so the cells
/// are compared on the same resamples.
pub struct Resamples {
    queries: usize,
    /// The query numbers of every resample, `queries` of them per resample, one resample after another.
    draws: Vec<u32>,
}

impl Resamples {
    /// `count` resamples of `queries` queries, drawn with the seed `seed`.
    pub fn new(queries: usize, count: usize, seed: u64) -> Resamples {
        let mut rng = SplitMix64(seed);
        let draws = match queries {
            0 => Vec::new(),
            _ => (0..queries * count)
                .map(|_| (rng.next_u64() % queries as u64) as u32)
                .collect(),
        };
        Resamples { queries, draws }
    }

    /// The mean of `values`, one per query, with its standard error and the 2.5th and 97.5th percentiles of the
    /// resamples' means. The error needs at least 2 queries and the interval a resample, or they are NaN (not a
    /// number).
    pub fn estimate(&self, values: &[f64]) -> Result<Estimate, Error> {
        let n = values.len();

        if n != self.queries {
            return Err(invalid("the values and the resamples differ in queries"));
        }

        let mean = mean(values);
        let se = match n {
            0 | 1 => f64::NAN,
            _ => (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / ((n - 1) * n) as f64).sqrt(),
        };
        let mut means: Vec<f64> = self
            .draws
            .chunks_exact(n.max(1))
            .map(|draw| draw.iter().map(|&q| values[q as usize]).sum::<f64>() / n as f64)
            .collect();
        Ok(Estimate {
            mean: match n {
                0 => f64::NAN,
                _ => mean,
            },
            se,
            lo: percentile(&mut means, 2.5),
            hi: percentile(&mut means, 97.5),
        })
    }
}

/// One cell of the grid, over every query.
#[derive(Clone, Debug, PartialEq)]
pub struct Cell {
    /// The base's position in the grid.
    pub base: usize,
    pub beta: f64,
    /// The clusters the guess reads, ⌊(1 + beta)·nprobe⌋, at most nlist.
    pub wide: usize,
    /// The share of extra clusters that the guess reads, wide/nprobe − 1, at most `beta`.
    pub beta_effective: f64,
    /// The rescoring depth's position in its base.
    pub depth: usize,
    pub mult: usize,
    /// The mean share of the true embedding's own clusters that the guess reads.
    pub posting_coverage: f64,
    /// The 10th percentile of that share.
    pub posting_coverage_p10: f64,
    /// The share of queries whose own clusters the guess reads all of.
    pub fully_covered: f64,
    /// The mean count of the true embedding's own clusters that the guess misses, which a normal search would read.
    pub missed_mean: f64,
    pub missed_p99: f64,
    /// Recall@10 of a normal search at the base's nprobe and R.
    pub recall_normal: f64,
    /// Recall@10 when the true embedding ranks every vector of the clusters the guess reads, and only those, at R.
    pub recall_guess: f64,
    /// `recall_guess` minus `recall_normal`, from each query's pair of searches.
    pub recall_gap: Estimate,
    /// The share of the rows that the true embedding rescores, out of the clusters the guess reads, that the guess
    /// prefetched.
    pub rerank_coverage: Estimate,
    /// The 10th percentile of that share.
    pub rerank_coverage_p10: f64,
    /// The mean count of those rows that the guess didn't prefetch, which are read once the model ends.
    pub residual_mean: f64,
    pub residual_p99: f64,
    /// The share of the rows that a normal search rescores, out of its own clusters, that the guess prefetched.
    pub rerank_coverage_normal: Estimate,
    /// The 10th percentile of that share.
    pub rerank_coverage_normal_p10: f64,
}

/// One guess's results for one query, base and beta.
struct Guessed {
    /// How many of the true embedding's own clusters the guess reads, out of `own`.
    covered: usize,
    own: usize,
    /// One entry per rescoring depth of the base.
    depths: Vec<GuessedDepth>,
}

struct GuessedDepth {
    /// How many of the true 10 nearest neighbors the true embedding finds in the clusters the guess reads.
    found: usize,
    /// The rows it rescores there.
    rescored: usize,
    /// For each multiple, how many of those rows the guess prefetched, and how many of a normal search's.
    prefetched: Vec<(usize, usize)>,
}

/// What one thread needs to measure a query, reused across queries.
struct Scratch {
    /// For the file index, sized for the widest read and the deepest rescore.
    reads: SearchCtx,
    /// For the in-memory index, sized for the most prefetched rows.
    prefetch: SearchCtx,
    leaves: Vec<LeafId>,
    wide: Vec<LeafId>,
    ids: Vec<u64>,
    prefetched: Vec<u64>,
}

/// One guess per query, measured against the normal searches in `exact`.
struct Measure<'a> {
    indexes: &'a Indexes,
    queries: &'a Queries<'a>,
    guesses: &'a [f32],
    exact: &'a Exact,
    grid: &'a Grid,
}

impl Measure<'_> {
    fn query(&self, q: usize, s: &mut Scratch) -> Result<Vec<Guessed>, Error> {
        let (e, guess) = (
            self.queries.vector(q),
            &self.guesses[q * self.queries.dim..(q + 1) * self.queries.dim],
        );
        let nlist = self.indexes.nlist;
        let mut out = Vec::with_capacity(self.grid.bases.len() * self.grid.betas.len());

        for (base, normal) in self.grid.bases.iter().zip(&self.exact.queries[q]) {
            for &beta in &self.grid.betas {
                let wide = wide_nprobe(base.nprobe, beta, nlist);
                self.indexes
                    .file
                    .route(guess, wide, None, &mut s.reads, &mut s.leaves)?;
                s.wide.clone_from(&s.leaves);
                s.wide.sort_unstable();
                let mut depths = Vec::with_capacity(base.depths.len());
                for (depth, rescored) in base.depths.iter().zip(&normal.depths) {
                    let r = depth.rerank;
                    let hits = self.indexes.file.fetch(&s.leaves, &mut s.reads)?.search(e, r, r)?;
                    let found = true_hits(hits, self.queries.truth(q));
                    sorted_ids(hits, &mut s.ids);
                    let mut prefetched = Vec::with_capacity(self.grid.mults.len());
                    for &m in &self.grid.mults {
                        let params = SearchParams {
                            k: m * r,
                            nprobe: wide,
                            rerank: m * r,
                            reach: None,
                        };
                        sorted_ids(
                            self.indexes.memory.search(guess, &params, &mut s.prefetch)?,
                            &mut s.prefetched,
                        );
                        prefetched.push((common(&s.ids, &s.prefetched), common(&rescored.ids, &s.prefetched)));
                    }
                    depths.push(GuessedDepth {
                        found,
                        rescored: s.ids.len(),
                        prefetched,
                    });
                }
                out.push(Guessed {
                    covered: common(&normal.clusters, &s.wide),
                    own: normal.clusters.len(),
                    depths,
                });
            }
        }

        Ok(out)
    }
}

/// Measures one guess per query, `guesses` holding them one after another, against the normal searches in `exact`,
/// which must come from the same queries and grid. The intervals come from `resamples`, which must draw from those
/// queries too. It runs on `threads` threads and returns the cells in grid order, i.e. by base, then beta, then
/// rescoring depth, then multiple.
pub fn speculate(
    indexes: &Indexes,
    queries: &Queries,
    guesses: &[f32],
    exact: &Exact,
    grid: &Grid,
    resamples: &Resamples,
    threads: usize,
) -> Result<Vec<Cell>, Error> {
    grid.check(indexes.nlist)?;

    if guesses.len() != queries.vectors.len() {
        return Err(invalid("there must be one guess per query"));
    }

    let shape = |n: &[Normal]| {
        n.len() == grid.bases.len() && n.iter().zip(&grid.bases).all(|(n, b)| n.depths.len() == b.depths.len())
    };

    if exact.queries.len() != queries.len() || !exact.queries.iter().all(|n| shape(n)) {
        return Err(invalid("the normal searches come from other queries or another grid"));
    }

    let widest = grid.widest(indexes.nlist);
    let reads = Limits {
        max_k: grid.deepest(),
        max_nprobe: widest,
        max_rerank: grid.deepest(),
    };
    let prefetch = Limits {
        max_k: grid.most_prefetched(),
        max_nprobe: widest,
        max_rerank: grid.most_prefetched(),
    };
    let measure = Measure {
        indexes,
        queries,
        guesses,
        exact,
        grid,
    };
    let guessed = parallel(queries.len(), threads, |range| {
        let mut s = Scratch {
            reads: SearchCtx::new(reads),
            prefetch: SearchCtx::new(prefetch),
            leaves: Vec::with_capacity(widest),
            wide: Vec::with_capacity(widest),
            ids: Vec::with_capacity(reads.max_rerank),
            prefetched: Vec::with_capacity(prefetch.max_rerank),
        };
        range.map(|q| measure.query(q, &mut s)).collect()
    })?;
    let results = Results {
        grid,
        nlist: indexes.nlist,
        exact,
        guessed: &guessed,
        resamples,
    };

    results.cells(threads)
}

/// The per-query results of one guess, to average into the grid's cells.
struct Results<'a> {
    grid: &'a Grid,
    nlist: usize,
    exact: &'a Exact,
    guessed: &'a [Vec<Guessed>],
    resamples: &'a Resamples,
}

impl Results<'_> {
    /// Every cell, with the pairs of a base and a beta shared out over `threads` threads.
    fn cells(&self, threads: usize) -> Result<Vec<Cell>, Error> {
        let slots: Vec<(usize, f64)> = (0..self.grid.bases.len())
            .flat_map(|b| self.grid.betas.iter().map(move |&beta| (b, beta)))
            .collect();
        let cells = parallel(slots.len(), threads, |range| {
            range.map(|slot| self.slot(slot, slots[slot])).collect()
        })?;
        Ok(cells.into_iter().flatten().collect())
    }

    /// The cells of base `b` and `beta`, the pair at position `slot` of each query's results.
    fn slot(&self, slot: usize, (b, beta): (usize, f64)) -> Result<Vec<Cell>, Error> {
        let (grid, exact, resamples) = (self.grid, self.exact, self.resamples);
        let base = &grid.bases[b];
        let n = self.guessed.len().max(1) as f64;
        let at: Vec<&Guessed> = self.guessed.iter().map(|g| &g[slot]).collect();
        let mut coverage: Vec<f64> = at.iter().map(|g| share(g.covered, g.own)).collect();
        let mut missed: Vec<f64> = at.iter().map(|g| (g.own - g.covered) as f64).collect();
        let fully = at.iter().filter(|g| g.covered == g.own).count() as f64 / n;
        let (coverage_mean, missed_mean) = (mean(&coverage), mean(&missed));
        let (coverage_p10, missed_p99) = (percentile(&mut coverage, 10.0), percentile(&mut missed, 99.0));
        let wide = wide_nprobe(base.nprobe, beta, self.nlist);
        let mut cells = Vec::with_capacity(base.depths.len() * grid.mults.len());

        for d in 0..base.depths.len() {
            let normal_found: Vec<usize> = exact.queries.iter().map(|q| q[b].depths[d].found).collect();
            let guess_found: Vec<usize> = at.iter().map(|g| g.depths[d].found).collect();
            let gaps: Vec<f64> = guess_found
                .iter()
                .zip(&normal_found)
                .map(|(&g, &f)| (g as f64 - f as f64) / K as f64)
                .collect();
            let recall_gap = resamples.estimate(&gaps)?;
            for (i, &mult) in grid.mults.iter().enumerate() {
                // Per query, the rows the true embedding rescores, and how many of them and of a normal search's the
                // guess prefetched.
                let rows: Vec<(usize, (usize, usize))> = at
                    .iter()
                    .map(|g| (g.depths[d].rescored, g.depths[d].prefetched[i]))
                    .collect();
                let mut rerank: Vec<f64> = rows.iter().map(|&(r, (p, _))| share(p, r)).collect();
                let mut residual: Vec<f64> = rows.iter().map(|&(r, (p, _))| (r - p) as f64).collect();
                let mut normal: Vec<f64> = (exact.queries.iter().zip(&rows))
                    .map(|(q, &(_, (_, p)))| share(p, q[b].depths[d].ids.len()))
                    .collect();
                // The estimates come before the percentiles, which sort the values out of query order.
                let (rerank_coverage, rerank_coverage_normal) =
                    (resamples.estimate(&rerank)?, resamples.estimate(&normal)?);
                cells.push(Cell {
                    base: b,
                    beta,
                    wide,
                    beta_effective: effective_beta(base.nprobe, wide),
                    depth: d,
                    mult,
                    posting_coverage: coverage_mean,
                    posting_coverage_p10: coverage_p10,
                    fully_covered: fully,
                    missed_mean,
                    missed_p99,
                    recall_normal: normal_found.iter().sum::<usize>() as f64 / (n * K as f64),
                    recall_guess: guess_found.iter().sum::<usize>() as f64 / (n * K as f64),
                    recall_gap,
                    rerank_coverage,
                    rerank_coverage_p10: percentile(&mut rerank, 10.0),
                    residual_mean: mean(&residual),
                    residual_p99: percentile(&mut residual, 99.0),
                    rerank_coverage_normal,
                    rerank_coverage_normal_p10: percentile(&mut normal, 10.0),
                });
            }
        }

        Ok(cells)
    }
}

/// The mean cosine similarity of the matching rows of `a` and `b`, each `dim` values long.
pub fn mean_cosine(a: &[f32], b: &[f32], dim: usize) -> f64 {
    let n = a.len() / dim.max(1);
    let total: f64 = a
        .chunks_exact(dim.max(1))
        .zip(b.chunks_exact(dim.max(1)))
        .map(|(x, y)| {
            let (mut xy, mut xx, mut yy) = (0f64, 0f64, 0f64);
            for (&p, &q) in x.iter().zip(y) {
                let (p, q) = (f64::from(p), f64::from(q));
                xy += p * q;
                xx += p * p;
                yy += q * q;
            }
            xy / (xx * yy).sqrt()
        })
        .sum();

    total / n.max(1) as f64
}

/// The first row of `vectors`, each `dim` values long, that a cosine search can't take, i.e. one with a value that
/// isn't finite or one that is all zeros.
pub fn bad_row(vectors: &[f32], dim: usize) -> Option<usize> {
    vectors
        .chunks_exact(dim.max(1))
        .position(|row| row.iter().any(|x| !x.is_finite()) || row.iter().all(|&x| x == 0.0))
}

/// How many of the first `K` hits are among the true neighbors `truth`.
fn true_hits(hits: &[Hit], truth: &[u32]) -> usize {
    hits.iter()
        .take(K)
        .filter(|h| truth.iter().any(|&t| u64::from(t) == h.id))
        .count()
}

fn sorted_ids(hits: &[Hit], out: &mut Vec<u64>) {
    out.clear();
    out.extend(hits.iter().map(|h| h.id));
    out.sort_unstable();
}

/// The count of values in both `a` and `b`, which are sorted and hold no repeats.
fn common<T: Ord>(a: &[T], b: &[T]) -> usize {
    let (mut i, mut j, mut n) = (0, 0, 0);

    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => (i, j, n) = (i + 1, j + 1, n + 1),
        }
    }

    n
}

/// `part` out of `whole`, or 1 when `whole` is 0, since nothing is then missing.
fn share(part: usize, whole: usize) -> f64 {
    match whole {
        0 => 1.0,
        _ => part as f64 / whole as f64,
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

/// The nearest-rank percentile `p`, from 0 to 100, of `values`, i.e. the smallest value that at least `p` percent of
/// them are at or below. It sorts `values`.
fn percentile(values: &mut [f64], p: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }

    values.sort_unstable_by(f64::total_cmp);
    let rank = (p / 100.0 * values.len() as f64).ceil() as usize;
    values[rank.clamp(1, values.len()) - 1]
}

/// Runs `work` over contiguous ranges of `0..n` on up to `threads` threads, and returns its results in order.
fn parallel<T: Send>(
    n: usize,
    threads: usize,
    work: impl Fn(Range<usize>) -> Result<Vec<T>, Error> + Sync,
) -> Result<Vec<T>, Error> {
    if n == 0 {
        return Ok(Vec::new());
    }

    let size = n.div_ceil(threads.clamp(1, n));

    std::thread::scope(|scope| {
        let work = &work;
        let handles: Vec<_> = (0..n)
            .step_by(size)
            .map(|start| scope.spawn(move || work(start..(start + size).min(n))))
            .collect();
        let mut out = Vec::with_capacity(n);

        for handle in handles {
            out.extend(
                handle
                    .join()
                    .map_err(|_| Error::other("a measuring thread panicked"))??,
            );
        }

        Ok(out)
    })
}

/// A small seeded generator (SplitMix64), so the resamples are the same in every run and on every platform.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

fn invalid(message: &str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

#[cfg(test)]
#[path = "../tests/unit/coverage.rs"]
mod tests;
