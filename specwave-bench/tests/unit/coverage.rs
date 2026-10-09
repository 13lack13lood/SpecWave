//! Unit tests for the coverage measurement, on small random data in clusters.

use std::collections::HashSet;
use std::path::PathBuf;

use super::*;
use crate::testutil::{Rng, Shape, Synthetic, perturbed, scratch, synthetic};

const DIM: usize = 128;
const NLIST: usize = 24;
const N: usize = 3840;
const QUERIES: usize = 40;
/// The true neighbors per query, more than the 10 that recall counts.
const WIDTH: usize = 20;

const SMALL: Shape = Shape {
    dim: DIM,
    n: N,
    nlist: NLIST,
    queries: QUERIES,
    width: WIDTH,
};

/// Where the index file of the test `name` goes, in a folder of its own, since the tests run at the same time.
fn index_path(name: &str) -> PathBuf {
    scratch(&format!("coverage-{name}")).join("index.coldivf")
}

/// Deletes the folder of the index file at `path`.
fn remove(path: PathBuf) {
    std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
}

/// The synthetic data's indexes with the rotation seed `seed`, with the file at a path of its own.
fn indexes(data: &Synthetic, name: &str, seed: u64) -> (Indexes, PathBuf) {
    let path = index_path(name);
    let indexes = Indexes::build(&data.rows, &data.clusters, &path, 1, seed).unwrap();
    (indexes, path)
}

fn queries(data: &Synthetic) -> Queries<'_> {
    Queries::new(DIM, &data.queries, &data.truth, WIDTH).unwrap()
}

/// Row `q` of `rows`, `dim` values long.
fn row(rows: &[f32], dim: usize, q: usize) -> &[f32] {
    &rows[q * dim..(q + 1) * dim]
}

fn resamples(queries: usize) -> Resamples {
    Resamples::new(queries, RESAMPLES, RESAMPLE_SEED)
}

fn grid(nprobes: &[usize], reranks: &[usize], betas: &[f64], mults: &[usize]) -> Grid {
    Grid {
        bases: nprobes
            .iter()
            .map(|&nprobe| Base {
                fraction: nprobe as f64 / NLIST as f64,
                nprobe,
                depths: reranks
                    .iter()
                    .map(|&r| Depth {
                        label: r.to_string(),
                        rerank: r,
                    })
                    .collect(),
            })
            .collect(),
        betas: betas.to_vec(),
        mults: mults.to_vec(),
    }
}

/// Recall@10 of normal searches of every query at `nprobe` and `rerank`, counted here against the first 10 of each
/// query's `WIDTH` true neighbors.
fn recall_here(indexes: &Indexes, data: &Synthetic, nprobe: usize, rerank: usize) -> f64 {
    let limits = Limits {
        max_k: K,
        max_nprobe: nprobe,
        max_rerank: rerank,
    };
    let mut ctx = SearchCtx::new(limits);
    let mut found = 0;

    for q in 0..QUERIES {
        let params = SearchParams {
            k: K,
            nprobe,
            rerank,
            reach: None,
        };
        let hits = indexes
            .memory
            .search(row(&data.queries, DIM, q), &params, &mut ctx)
            .unwrap();
        let truth = &data.truth[q * WIDTH..q * WIDTH + K];
        found += hits.iter().filter(|h| truth.contains(&(h.id as u32))).count();
    }

    found as f64 / (QUERIES * K) as f64
}

#[test]
fn nprobe_rounds_the_share_of_clusters_and_the_guess_rounds_its_clusters_down() {
    assert_eq!(base_nprobe(0.0256, 360), 9);
    assert_eq!(base_nprobe(0.0307, 360), 11);
    assert_eq!(base_nprobe(0.017, 360), 6);
    assert_eq!(base_nprobe(0.0256, 3268), 84);
    assert_eq!(base_nprobe(0.0307, 3268), 100);
    assert_eq!(base_nprobe(0.017, 3268), 56);
    assert_eq!(base_nprobe(0.0001, 100), 1);
    assert_eq!(base_nprobe(2.0, 100), 100);
    assert_eq!(wide_nprobe(9, 0.0, 360), 9);
    assert_eq!(wide_nprobe(9, 0.5, 360), 13);
    assert_eq!(wide_nprobe(11, 0.5, 360), 16);
    assert_eq!(wide_nprobe(11, 0.25, 360), 13);
    assert_eq!(wide_nprobe(6, 0.25, 360), 7);
    assert_eq!(wide_nprobe(56, 0.25, 3268), 70);
    assert_eq!(wide_nprobe(20, 0.15, 360), 23);
    assert_eq!(wide_nprobe(100, 1.0, 3268), 200);
    assert_eq!(wide_nprobe(100, 1.0, 150), 150);
    assert_eq!(effective_beta(9, 13), 4.0 / 9.0);
    assert_eq!(effective_beta(84, 84), 0.0);

    // The guess reads as many clusters as it can without passing the beta asked for.
    for nprobe in 1..400 {
        for beta in [0.0, 0.1, 0.15, 0.25, 0.3, 0.5, 0.75, 1.0] {
            let wide = wide_nprobe(nprobe, beta, 10_000);
            let what = format!("nprobe {nprobe}, beta {beta}, {wide} clusters");
            assert!(effective_beta(nprobe, wide) <= beta + 1e-12, "{what}");
            assert!(effective_beta(nprobe, wide + 1) > beta + 1e-12, "{what}");
        }
    }
}

#[test]
fn depth_for_finds_the_depth_from_which_recall_stays_at_the_target() {
    let depths = [10, 20, 30, 40, 50];
    assert_eq!(depth_for(&depths, &[0.9, 0.94, 0.96, 0.95, 0.97], 0.95), Some(30));
    assert_eq!(depth_for(&depths, &[0.9, 0.96, 0.94, 0.97, 0.98], 0.95), Some(40));
    assert_eq!(depth_for(&depths, &[0.9, 0.91, 0.92, 0.93, 0.94], 0.95), None);
    assert_eq!(depth_for(&depths, &[0.96; 5], 0.95), Some(10));
    assert_eq!(depth_for(&[], &[], 0.95), None);

    let sweep = sweep_depths();
    assert!(sweep.windows(2).all(|w| w[0] < w[1]));
    assert_eq!((sweep[0], sweep[90], *sweep.last().unwrap()), (10, 100, 500));
}

#[test]
fn a_guess_equal_to_the_query_covers_everything_and_finds_what_a_normal_search_finds() {
    let data = synthetic(1, &SMALL);
    let (indexes, path) = indexes(&data, "equal", 7);
    let queries = queries(&data);
    // At nprobe 1 and beta 1, the second cluster holds rows that queries between two clusters rescore.
    let grid = grid(&[1, 2, 4], &[10, 25], &[0.0, 0.5, 1.0], &[1, 2, 4]);
    let exact = exact(&indexes, &queries, &grid, 2).unwrap();
    let cells = speculate(&indexes, &queries, &data.queries, &exact, &grid, &resamples(QUERIES), 3).unwrap();
    assert_eq!(cells.len(), 3 * 3 * 2 * 3);

    for c in &cells {
        let base = &grid.bases[c.base];
        let what = format!("{c:?}");
        assert_eq!(c.wide, wide_nprobe(base.nprobe, c.beta, NLIST), "{what}");
        assert_eq!(c.beta_effective, effective_beta(base.nprobe, c.wide), "{what}");
        assert!(c.beta_effective <= c.beta, "{what}");
        assert_eq!(
            (c.posting_coverage, c.posting_coverage_p10, c.fully_covered),
            (1.0, 1.0, 1.0),
            "{what}"
        );
        assert_eq!((c.missed_mean, c.missed_p99), (0.0, 0.0), "{what}");
        let full = Estimate {
            mean: 1.0,
            se: 0.0,
            lo: 1.0,
            hi: 1.0,
        };
        assert_eq!((c.rerank_coverage, c.rerank_coverage_p10), (full, 1.0), "{what}");
        assert_eq!((c.residual_mean, c.residual_p99), (0.0, 0.0), "{what}");
        assert!(
            (c.recall_gap.mean - (c.recall_guess - c.recall_normal)).abs() < 1e-12,
            "{what}"
        );
        if c.wide == base.nprobe {
            assert_eq!(c.recall_guess, c.recall_normal, "{what}");
            assert_eq!((c.recall_gap.lo, c.recall_gap.hi), (0.0, 0.0), "{what}");
            assert_eq!(
                (c.rerank_coverage_normal, c.rerank_coverage_normal_p10),
                (full, 1.0),
                "{what}"
            );
        }
        let rerank = base.depths[c.depth].rerank;
        assert_eq!(
            c.recall_normal,
            recall_here(&indexes, &data, base.nprobe, rerank),
            "{what}"
        );
    }

    // The equalities above mean something only if the searches find neighbors, more of them with a deeper rescore, and
    // a guess's extra clusters hold rows it rescores.
    let recall = |depth: usize| {
        cells
            .iter()
            .find(|c| c.depth == depth && c.base == 1)
            .map(|c| c.recall_normal)
            .unwrap()
    };
    assert!(
        recall(0) > 0.2 && recall(1) > recall(0),
        "{} then {}",
        recall(0),
        recall(1)
    );
    assert!(cells.iter().any(|c| c.recall_guess > c.recall_normal));
    remove(path);
}

#[test]
fn the_in_memory_index_ranks_what_the_file_index_routes_to_and_fetches() {
    let data = synthetic(2, &SMALL);
    let (indexes, path) = indexes(&data, "fetch", 7);
    let mut sizes = [0usize; NLIST];
    data.clusters.assignment.iter().for_each(|&c| sizes[c as usize] += 1);
    let mut rng = Rng(3);
    let limits = Limits {
        max_k: 200,
        max_nprobe: NLIST,
        max_rerank: 200,
    };
    let (mut file, mut memory) = (SearchCtx::new(limits), SearchCtx::new(limits));
    let mut leaves = Vec::new();

    for i in 0..30 {
        let q = match i % 2 {
            0 => row(&data.queries, DIM, i % QUERIES).to_vec(),
            _ => rng.direction(DIM),
        };
        let (wide, n) = (1 + i % NLIST, 10 + 6 * i);
        indexes.file.route(&q, wide, None, &mut file, &mut leaves).unwrap();
        assert_eq!(leaves.len(), wide);
        let fetched: Vec<(u64, f32)> = indexes
            .file
            .fetch(&leaves, &mut file)
            .unwrap()
            .search(&q, n, n)
            .unwrap()
            .iter()
            .map(|h| (h.id, h.score))
            .collect();
        let params = SearchParams {
            k: n,
            nprobe: wide,
            rerank: n,
            reach: None,
        };
        let routed: Vec<(u64, f32)> = indexes
            .memory
            .search(&q, &params, &mut memory)
            .unwrap()
            .iter()
            .map(|h| (h.id, h.score))
            .collect();
        assert_eq!(fetched, routed, "query {i}, {wide} clusters, {n} rows");
        let in_clusters: usize = leaves.iter().map(|&l| sizes[l as usize]).sum();
        assert_eq!(fetched.len(), n.min(in_clusters), "query {i}");
    }

    remove(path);
}

#[test]
fn both_rerank_coverages_match_a_count_made_here() {
    let data = synthetic(8, &SMALL);
    let (indexes, path) = indexes(&data, "count", 7);
    let queries = queries(&data);
    let guesses = perturbed(&data.queries, DIM, 9, 0.5);
    let grid = grid(&[1, 3], &[10, 20], &[0.0, 1.0], &[1, 3]);
    let exact = exact(&indexes, &queries, &grid, 2).unwrap();
    let cells = speculate(&indexes, &queries, &guesses, &exact, &grid, &resamples(QUERIES), 2).unwrap();
    let limits = Limits {
        max_k: 60,
        max_nprobe: NLIST,
        max_rerank: 60,
    };
    let (mut ctx, mut leaves) = (SearchCtx::new(limits), Vec::new());
    let ids = |hits: &[Hit]| -> HashSet<u64> { hits.iter().map(|h| h.id).collect() };

    for c in &cells {
        let base = &grid.bases[c.base];
        let r = base.depths[c.depth].rerank;
        let (mut rerank, mut normal, mut residual, mut covered) = (0.0, 0.0, 0.0, 0.0);
        for q in 0..QUERIES {
            let (e, guess) = (row(&data.queries, DIM, q), row(&guesses, DIM, q));
            let params = SearchParams {
                k: r,
                nprobe: base.nprobe,
                rerank: r,
                reach: None,
            };
            let normal_rows = ids(indexes.memory.search(e, &params, &mut ctx).unwrap());
            indexes.file.route(e, base.nprobe, None, &mut ctx, &mut leaves).unwrap();
            let own: HashSet<LeafId> = leaves.iter().copied().collect();
            // The guess's clusters, then the rows the true embedding rescores there, and the m·R rows that the guess
            // ranks best there by their 1-bit codes, which is what a prefetch reads.
            indexes.file.route(guess, c.wide, None, &mut ctx, &mut leaves).unwrap();
            covered += leaves.iter().filter(|l| own.contains(l)).count() as f64 / own.len() as f64;
            let rescored = ids(indexes.file.fetch(&leaves, &mut ctx).unwrap().search(e, r, r).unwrap());
            let m = c.mult * r;
            let prefetched = ids(indexes
                .file
                .fetch(&leaves, &mut ctx)
                .unwrap()
                .search(guess, m, m)
                .unwrap());
            let shared = rescored.intersection(&prefetched).count();
            rerank += shared as f64 / rescored.len() as f64;
            residual += (rescored.len() - shared) as f64;
            normal += normal_rows.intersection(&prefetched).count() as f64 / normal_rows.len() as f64;
        }
        let n = QUERIES as f64;
        let what = format!("{c:?}");
        assert!((c.rerank_coverage.mean - rerank / n).abs() < 1e-12, "{what}");
        assert!((c.rerank_coverage_normal.mean - normal / n).abs() < 1e-12, "{what}");
        assert!((c.residual_mean - residual / n).abs() < 1e-12, "{what}");
        assert!((c.posting_coverage - covered / n).abs() < 1e-12, "{what}");
    }

    // The counts are worth something only if the guesses miss rows and the two coverages differ.
    assert!(cells.iter().any(|c| c.rerank_coverage.mean < 0.95));
    assert!(
        cells
            .iter()
            .any(|c| (c.rerank_coverage.mean - c.rerank_coverage_normal.mean).abs() > 0.02)
    );
    remove(path);
}

#[test]
fn the_rows_that_a_search_reads_after_the_prefetch_are_the_residual() {
    // A row of 2040 values takes more than half of a 4096-byte page, so each page holds one row. The portable backend
    // reads whole 4096-byte pages, so the bytes of a search's last read wave count the rows it needed then.
    let shape = Shape {
        dim: 2040,
        n: 480,
        nlist: 6,
        queries: 12,
        width: K,
    };
    let data = synthetic(10, &shape);
    let path = index_path("residual");
    let indexes = Indexes::build(&data.rows, &data.clusters, &path, 2, 7).unwrap();
    let queries = Queries::new(shape.dim, &data.queries, &data.truth, K).unwrap();
    let guesses = perturbed(&data.queries, shape.dim, 11, 0.6);
    let grid = grid(&[2], &[10, 20], &[0.0, 1.0], &[1, 2]);
    let exact = exact(&indexes, &queries, &grid, 1).unwrap();
    let cells = speculate(
        &indexes,
        &queries,
        &guesses,
        &exact,
        &grid,
        &Resamples::new(12, 10, 1),
        1,
    )
    .unwrap();
    let file = Index::open_with(&path, coldivf::Backend::Portable).unwrap();
    let limits = Limits {
        max_k: 40,
        max_nprobe: shape.nlist,
        max_rerank: 40,
    };
    let (mut ctx, mut leaves) = (SearchCtx::new(limits), Vec::new());
    let pairs = |hits: &[Hit]| -> Vec<(u64, f32)> { hits.iter().map(|h| (h.id, h.score)).collect() };

    for c in &cells {
        let r = grid.bases[0].depths[c.depth].rerank;
        let mut late = 0;
        for q in 0..shape.queries {
            let (e, guess) = (row(&data.queries, shape.dim, q), row(&guesses, shape.dim, q));
            file.route(guess, c.wide, None, &mut ctx, &mut leaves).unwrap();
            // With no prefetch, the last wave reads the page of every rescored row.
            let plain = pairs(file.fetch(&leaves, &mut ctx).unwrap().search(e, r, r).unwrap());
            assert_eq!(ctx.trace().row_bytes, plain.len() * 4096, "query {q}");
            let mut fetched = file.fetch(&leaves, &mut ctx).unwrap();
            fetched.prefetch_rows(guess, c.mult * r).unwrap();
            let hits = pairs(fetched.search(e, r, r).unwrap());
            assert_eq!(hits, plain, "the prefetch changes no hit, query {q}");
            late += ctx.trace().row_bytes / 4096;
        }
        assert_eq!(late as f64, c.residual_mean * shape.queries as f64, "{c:?}");
    }

    assert!(
        cells.iter().any(|c| c.residual_mean > 1.0),
        "the guesses must miss some rows"
    );
    remove(path);
}

#[test]
fn a_far_guess_misses_clusters_and_rows() {
    let data = synthetic(4, &SMALL);
    let (indexes, path) = indexes(&data, "far", 7);
    let queries = queries(&data);
    let mut rng = Rng(5);
    let guesses: Vec<f32> = (0..QUERIES).flat_map(|_| rng.direction(DIM)).collect();
    let grid = grid(&[3], &[20], &[0.0, 1.0], &[2]);
    let exact = exact(&indexes, &queries, &grid, 2).unwrap();
    let cells = speculate(&indexes, &queries, &guesses, &exact, &grid, &resamples(QUERIES), 2).unwrap();
    assert_eq!(cells.len(), 2);

    for c in &cells {
        let what = format!("{c:?}");
        assert!(
            c.posting_coverage < 0.5 && c.posting_coverage_p10 <= c.posting_coverage,
            "{what}"
        );
        assert!(
            c.fully_covered < 0.5 && c.missed_mean > 1.0 && c.missed_p99 <= 3.0,
            "{what}"
        );
        assert!(c.recall_guess < c.recall_normal, "{what}");
        assert!(c.recall_gap.hi < 0.0 && c.recall_gap.lo <= c.recall_gap.mean, "{what}");
        assert!(
            c.rerank_coverage.mean < 0.9 && c.rerank_coverage_normal.mean < 0.5,
            "{what}"
        );
        for e in [c.recall_gap, c.rerank_coverage, c.rerank_coverage_normal] {
            assert!(e.lo <= e.mean && e.mean <= e.hi && e.se > 0.0, "{what}");
        }
        assert!(c.residual_mean > 2.0 && c.residual_p99 <= 20.0, "{what}");
    }

    assert!(cells[1].posting_coverage > cells[0].posting_coverage);
    remove(path);
}

#[test]
fn the_ceiling_rescores_every_vector_of_the_clusters() {
    let data = synthetic(6, &SMALL);
    let (indexes, path) = indexes(&data, "ceiling", 7);
    let queries = queries(&data);
    let mut sizes = [0usize; NLIST];
    data.clusters.assignment.iter().for_each(|&c| sizes[c as usize] += 1);
    sizes.sort_unstable_by(|a, b| b.cmp(a));

    for nprobe in 1..=NLIST {
        let largest: usize = sizes[..nprobe].iter().sum();
        assert_eq!(indexes.ceiling_depth(nprobe), largest, "nprobe {nprobe}");
    }

    for nprobe in [1, 3, NLIST] {
        let ceiling = ceiling(&indexes, &queries, nprobe, 2).unwrap();
        let all = sweep(&indexes, &queries, nprobe, &[10, N], 2).unwrap();
        assert_eq!(ceiling, all[1], "nprobe {nprobe}");
        assert!(all[0] <= ceiling + 0.05, "nprobe {nprobe}");
    }

    let full = ceiling(&indexes, &queries, NLIST, 2).unwrap();
    assert!(
        full > 0.95,
        "rescoring every vector with 8-bit rows should find nearly all of the true 10: {full}"
    );
    remove(path);
}

#[test]
fn the_rotation_seed_changes_the_one_bit_ranking_and_the_same_seed_repeats_it() {
    let data = synthetic(12, &SMALL);
    let (a, path_a) = indexes(&data, "seed-a", 7);
    let (b, path_b) = indexes(&data, "seed-b", 11);
    let (again, path_again) = indexes(&data, "seed-again", 7);
    let limits = Limits {
        max_k: K,
        max_nprobe: 3,
        max_rerank: K,
    };
    let mut ctx = SearchCtx::new(limits);
    let params = SearchParams {
        k: K,
        nprobe: 3,
        rerank: K,
        reach: None,
    };
    let mut search = |ix: &Indexes, q: usize| -> Vec<u64> {
        let hits = ix.memory.search(row(&data.queries, DIM, q), &params, &mut ctx).unwrap();
        hits.iter().map(|h| h.id).collect()
    };
    let mut differ = 0;

    for q in 0..QUERIES {
        let first = search(&a, q);
        assert_eq!(search(&again, q), first, "query {q}");
        differ += usize::from(search(&b, q) != first);
    }

    assert!(differ > 0, "another rotation must rank some rows otherwise");

    for path in [path_a, path_b, path_again] {
        remove(path);
    }
}

#[test]
fn intervals_resample_the_queries() {
    let four = Resamples::new(4, RESAMPLES, 1);
    let e = four.estimate(&[0.0, 1.0, 0.0, 1.0]).unwrap();
    assert_eq!(e.mean, 0.5);
    // The sample's standard deviation is √(1/3), over √4.
    assert!((e.se - (1.0f64 / 3.0).sqrt() / 2.0).abs() < 1e-12, "{e:?}");
    assert!(0.0 <= e.lo && e.lo < 0.5 && 0.5 < e.hi && e.hi <= 1.0, "{e:?}");
    let same = four.estimate(&[0.25; 4]).unwrap();
    assert_eq!((same.mean, same.se, same.lo, same.hi), (0.25, 0.0, 0.25, 0.25));

    // With many queries the interval is close to 1.96 standard errors on each side of the mean.
    let values: Vec<f64> = (0..400).map(|i| (i * 37 % 101) as f64 / 100.0).collect();
    let many = Resamples::new(400, RESAMPLES, 1);
    let e = many.estimate(&values).unwrap();
    let ratio = (e.hi - e.lo) / (2.0 * 1.96 * e.se);
    assert!((0.85..1.15).contains(&ratio), "{e:?}, ratio {ratio}");
    assert!((e.mean - (e.lo + e.hi) / 2.0).abs() < 0.2 * (e.hi - e.lo), "{e:?}");
    assert_eq!(many.estimate(&values).unwrap(), e, "the same draws");
    assert_ne!(
        Resamples::new(400, RESAMPLES, 2).estimate(&values).unwrap(),
        e,
        "other draws"
    );

    // Every resample counts. Of 2 resamples, the interval runs from the lower mean to the higher one.
    let two = Resamples::new(3, 2, 2);
    let three = [0.0, 1.0, 4.0];
    let means: Vec<f64> = two
        .draws
        .chunks(3)
        .map(|draw| draw.iter().map(|&q| three[q as usize]).sum::<f64>() / 3.0)
        .collect();
    let e = two.estimate(&three).unwrap();
    assert_eq!(means.len(), 2);
    assert!(means[0] != means[1], "{means:?}");
    assert_eq!((e.lo, e.hi), (means[0].min(means[1]), means[0].max(means[1])));

    assert_eq!(
        many.estimate(&values[1..]).err().map(|e| e.kind()),
        Some(ErrorKind::InvalidInput)
    );
    let none = Resamples::new(0, RESAMPLES, 1).estimate(&[]).unwrap();
    assert!(none.mean.is_nan() && none.se.is_nan() && none.lo.is_nan() && none.hi.is_nan());
    assert!(Resamples::new(1, RESAMPLES, 1).estimate(&[0.5]).unwrap().se.is_nan());
}

/// The results of one query at one base, beta and depth, with one multiple.
fn guessed(covered: usize, own: usize, found: usize, rescored: usize, prefetched: (usize, usize)) -> Vec<Guessed> {
    vec![Guessed {
        covered,
        own,
        depths: vec![GuessedDepth {
            found,
            rescored,
            prefetched: vec![prefetched],
        }],
    }]
}

/// A normal search of one query at one base and depth, which rescores the rows 0 to `rows` - 1.
fn normal(clusters: &[LeafId], found: usize, rows: u64) -> Vec<Normal> {
    vec![Normal {
        clusters: clusters.to_vec(),
        depths: vec![Rescored {
            found,
            ids: (0..rows).collect(),
        }],
    }]
}

#[test]
fn cells_average_the_queries() {
    // Shares and means that binary fractions hold exactly.
    let grid = grid(&[4], &[16], &[0.5], &[2]);
    let exact = Exact {
        queries: vec![normal(&[1, 2, 3, 4], 10, 16), normal(&[5, 6, 7, 8], 5, 16)],
    };
    let guessed = [guessed(4, 4, 8, 16, (16, 12)), guessed(2, 4, 4, 16, (8, 4))];
    let resamples = resamples(2);
    let results = Results {
        grid: &grid,
        nlist: NLIST,
        exact: &exact,
        guessed: &guessed,
        resamples: &resamples,
    };
    let estimate = |values: &[f64]| resamples.estimate(values).unwrap();
    let expected = Cell {
        base: 0,
        beta: 0.5,
        wide: 6,
        beta_effective: 0.5,
        depth: 0,
        mult: 2,
        posting_coverage: 0.75,
        posting_coverage_p10: 0.5,
        fully_covered: 0.5,
        missed_mean: 1.0,
        missed_p99: 2.0,
        recall_normal: 0.75,
        recall_guess: 0.6,
        recall_gap: estimate(&[-0.2, -0.1]),
        rerank_coverage: estimate(&[1.0, 0.5]),
        rerank_coverage_p10: 0.5,
        residual_mean: 4.0,
        residual_p99: 8.0,
        rerank_coverage_normal: estimate(&[0.75, 0.25]),
        rerank_coverage_normal_p10: 0.25,
    };
    assert_eq!(results.cells(2).unwrap(), [expected]);
}

#[test]
fn cells_take_percentiles_in_value_order_and_intervals_in_query_order() {
    // Query q of the 20 has the rank 7·q mod 20, so the values aren't in query order. It covers rank + 1 of its 20
    // clusters, and the true embedding finds rank mod 11 of the true 10 there, where a normal search finds 5. The guess
    // prefetched rank of the 20 rows that the true embedding rescores, and rank / 2, rounded down, of the rows that a
    // normal search rescores. A normal search of an even q rescores only 16 rows, as when its clusters hold fewer than
    // R vectors.
    let rank = |q: usize| 7 * q % 20;
    let rows = |q: usize| if q.is_multiple_of(2) { 16 } else { 20 };
    let grid = grid(&[4], &[20], &[0.5], &[2]);
    let exact = Exact {
        queries: (0..20).map(|q| normal(&[1, 2, 3, 4], 5, rows(q) as u64)).collect(),
    };
    let guessed: Vec<Vec<Guessed>> = (0..20)
        .map(|q| guessed(rank(q) + 1, 20, rank(q) % 11, 20, (rank(q), rank(q) / 2)))
        .collect();
    let resamples = resamples(20);
    let results = Results {
        grid: &grid,
        nlist: NLIST,
        exact: &exact,
        guessed: &guessed,
        resamples: &resamples,
    };
    let c = &results.cells(1).unwrap()[0];
    let close = |a: f64, b: f64| (a - b).abs() < 1e-12;
    // By nearest rank, the 10th percentile of 20 values is the 2nd smallest and the 99th the largest.
    assert_eq!((c.posting_coverage_p10, c.missed_p99), (0.1, 19.0));
    assert_eq!((c.rerank_coverage_p10, c.residual_p99), (0.05, 20.0));
    assert_eq!(c.rerank_coverage_normal_p10, 0.0);
    assert_eq!((c.missed_mean, c.residual_mean), (9.5, 10.5));
    assert_eq!(
        c.fully_covered, 0.05,
        "only the query of rank 19 covers all 20 clusters"
    );
    assert!(close(c.posting_coverage, 0.525), "{c:?}");
    assert_eq!((c.recall_normal, c.recall_guess), (0.5, 0.455));

    // Each interval resamples the queries' values in query order, and sorting them first would give another.
    let in_order = |value: &dyn Fn(usize) -> f64| (0..20).map(value).collect::<Vec<f64>>();
    let normal_share = |q: usize| (rank(q) / 2) as f64 / rows(q) as f64;
    let expected = [
        (c.recall_gap, in_order(&|q| ((rank(q) % 11) as f64 - 5.0) / 10.0)),
        (c.rerank_coverage, in_order(&|q| rank(q) as f64 / 20.0)),
        (c.rerank_coverage_normal, in_order(&normal_share)),
    ];

    for (found, mut values) in expected {
        assert_eq!(found, resamples.estimate(&values).unwrap(), "{c:?}");
        values.sort_unstable_by(f64::total_cmp);
        assert_ne!(found, resamples.estimate(&values).unwrap(), "{c:?}");
    }

    let normal_mean = (0..20).map(normal_share).sum::<f64>() / 20.0;
    assert!(close(c.rerank_coverage.mean, 0.475), "{c:?}");
    assert!(close(c.rerank_coverage_normal.mean, normal_mean), "{c:?}");
}

#[test]
fn bad_arguments_fail_with_invalid_input() {
    let data = synthetic(7, &SMALL);
    let (indexes, path) = indexes(&data, "bad", 7);
    let kind = |e: Error| e.kind();
    assert_eq!(
        Queries::new(DIM, &data.queries, &data.truth[..QUERIES * 9], 9)
            .err()
            .map(kind),
        Some(ErrorKind::InvalidInput)
    );
    assert_eq!(
        Queries::new(DIM, &data.queries[1..], &data.truth, WIDTH)
            .err()
            .map(kind),
        Some(ErrorKind::InvalidInput)
    );
    assert_eq!(
        Queries::new(DIM, &data.queries[DIM..], &data.truth, WIDTH)
            .err()
            .map(kind),
        Some(ErrorKind::InvalidInput)
    );
    assert_eq!(
        Queries::new(0, &[], &[], K).err().map(kind),
        Some(ErrorKind::InvalidInput)
    );

    let queries = queries(&data);
    let good = grid(&[2], &[10], &[0.5], &[2]);
    let bad_grids = [
        grid(&[2], &[9], &[0.5], &[2]),
        grid(&[0], &[10], &[0.5], &[2]),
        grid(&[NLIST + 1], &[10], &[0.5], &[2]),
        grid(&[2], &[10], &[-0.5], &[2]),
        grid(&[2], &[10], &[f64::NAN], &[2]),
        grid(&[2], &[10], &[0.5], &[0]),
        grid(&[2], &[10], &[], &[2]),
        grid(&[], &[10], &[0.5], &[2]),
        grid(&[2], &[], &[0.5], &[2]),
    ];

    for g in &bad_grids {
        assert_eq!(
            exact(&indexes, &queries, g, 1).err().map(kind),
            Some(ErrorKind::InvalidInput),
            "{g:?}"
        );
    }

    let normal = exact(&indexes, &queries, &good, 1).unwrap();
    let short = &data.queries[..data.queries.len() - DIM];
    let other = grid(&[2], &[10, 20], &[0.5], &[2]);
    let fewer = Queries::new(DIM, &data.queries[DIM..], &data.truth[WIDTH..], WIDTH).unwrap();
    let draws = resamples(QUERIES);
    assert_eq!(
        speculate(&indexes, &queries, short, &normal, &good, &draws, 1)
            .err()
            .map(kind),
        Some(ErrorKind::InvalidInput)
    );
    assert_eq!(
        speculate(&indexes, &queries, &data.queries, &normal, &other, &draws, 1)
            .err()
            .map(kind),
        Some(ErrorKind::InvalidInput)
    );
    assert_eq!(
        speculate(&indexes, &fewer, &data.queries[DIM..], &normal, &good, &draws, 1)
            .err()
            .map(kind),
        Some(ErrorKind::InvalidInput)
    );
    assert_eq!(
        speculate(
            &indexes,
            &queries,
            &data.queries,
            &normal,
            &good,
            &resamples(QUERIES - 1),
            1
        )
        .err()
        .map(kind),
        Some(ErrorKind::InvalidInput)
    );
    assert_eq!(
        sweep(&indexes, &queries, 0, &[10], 1).err().map(kind),
        Some(ErrorKind::InvalidInput)
    );
    assert_eq!(
        sweep(&indexes, &queries, NLIST + 1, &[10], 1).err().map(kind),
        Some(ErrorKind::InvalidInput)
    );
    assert_eq!(
        sweep(&indexes, &queries, 2, &[5], 1).err().map(kind),
        Some(ErrorKind::InvalidInput)
    );
    remove(path);
}

#[test]
fn small_helpers_count_average_and_rank() {
    assert_eq!(common(&[1, 3, 5, 7], &[2, 3, 4, 7, 9]), 2);
    assert_eq!(common::<u32>(&[], &[1]), 0);
    assert_eq!(share(3, 4), 0.75);
    assert_eq!(share(0, 0), 1.0);
    assert_eq!(mean(&[1.0, 2.0, 6.0]), 3.0);
    assert_eq!(mean(&[]), 0.0);
    let mut values = [5.0, 1.0, 4.0, 2.0, 3.0];
    assert_eq!(percentile(&mut values, 10.0), 1.0);
    assert_eq!(percentile(&mut values, 50.0), 3.0);
    assert_eq!(percentile(&mut values, 99.0), 5.0);
    assert!(percentile(&mut [], 50.0).is_nan());

    let hits = [3, 9, 4, 1].map(|id| Hit { id, score: 0.0 });
    assert_eq!(true_hits(&hits, &[1, 2, 3]), 2);
    // Only the first 10 hits count, so the true neighbors 11 and 12 at places 11 and 12 don't.
    let twelve: Vec<Hit> = (1..=12).map(|id| Hit { id, score: 0.0 }).collect();
    assert_eq!(true_hits(&twelve, &[1, 2, 11, 12]), 2);
    let mut ids = Vec::new();
    sorted_ids(&hits, &mut ids);
    assert_eq!(ids, [1, 3, 4, 9]);

    let a = [1.0, 0.0, 0.0, 0.0, 2.0, 0.0];
    let b = [2.0, 0.0, 0.0, 1.0, 1.0, 0.0];
    assert!((mean_cosine(&a, &b, 3) - (1.0 + 0.5f64.sqrt()) / 2.0).abs() < 1e-12);
    assert_eq!(bad_row(&a, 3), None);
    assert_eq!(bad_row(&[1.0, 2.0, 0.0, 0.0], 2), Some(1));
    assert_eq!(bad_row(&[1.0, f32::NAN], 1), Some(1));
}

#[test]
fn parallel_keeps_the_order_and_passes_errors_on() {
    let squares = parallel(10, 3, |range| Ok(range.map(|i| i * i).collect())).unwrap();
    assert_eq!(squares, (0..10).map(|i| i * i).collect::<Vec<_>>());
    assert_eq!(parallel(5, 0, |range| Ok(range.collect())).unwrap(), [0, 1, 2, 3, 4]);
    assert!(parallel::<usize>(0, 4, |_| unreachable!()).unwrap().is_empty());
    let failed = parallel::<usize>(10, 4, |range| {
        if range.contains(&7) {
            Err(Error::other("seven"))
        } else {
            Ok(range.collect())
        }
    });
    assert_eq!(failed.err().map(|e| e.to_string()), Some("seven".to_string()));
}
