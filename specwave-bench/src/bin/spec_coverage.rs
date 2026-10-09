//! Measures how much of a search SpecWave's guesses already cover, with ColdIVF's own codes.
//!
//! For each dataset, clustering and rotation seed, it builds the index from the documents and their clusters, as a
//! file for routes and reads and in memory for normal searches. It first sweeps the rescoring depth R of normal
//! searches at each base nprobe, for the smallest R that reaches the target recall. On Quora, with 384 dimensions,
//! that depth is R384, which the first seed's sweeps give, and every dataset's grid tries it next to the depths of
//! `--reranks`, at the same share of clusters and with the same kind of clustering. Then for every guess file, base
//! nprobe, beta, R and prefetch multiple m, it reports posting coverage, recall@10 with no second read next to a normal
//! search's, and rerank coverage both over the guess's clusters and over a normal search's, as the coverage module
//! defines them. The recall gap and both rerank coverages come with their standard errors and 95% bootstrap
//! intervals.
//!
//! ```text
//! spec_coverage --out FILE [--data DIR] [--datasets quora,fiqa] [--kinds l2,cos,sph] [--guesses LABEL,...]
//!     [--queries N] [--fractions 0.017,0.0256,0.0307] [--betas 0,0.25,0.5,1] [--reranks 29,39] [--mults 2,3,4]
//!     [--target 0.95] [--r384 R|none] [--seeds 7,11,23] [--threads 4] [--redo]
//! ```
//!
//! The inputs are the files that the data module names, in the data folder `DIR`, which is this repository's `data`
//! by default. A kind of clustering is `l2`, from k-means with L2 (Euclidean) distance, `sph`, from spherical k-means,
//! or `cos`, the `l2` clusters with their centers scaled to length 1. The index routes a query to the clusters whose
//! centers are nearest to it by L2 distance. The centers of `sph` and `cos` are scaled to length 1, so that route
//! ranks them by cosine. Each seed of `--seeds` builds the indexes again with its own rotation. A dataset without
//! guess files still gets its sweeps. The guesses and the true neighbors must belong to the test queries. The data
//! step writes the dataset's `spec/<name>/e.fbin` last, once they do, so it must hold the same bytes as
//! `<name>_test.fbin`. Results are appended to `FILE` as JavaScript Object Notation (JSON) lines, each with the build
//! of this program. A guess that `FILE` already marks as done, by this build with the same grid, seed and input files,
//! is skipped, unless `--redo` is given or the build doesn't name its code, eg. when git couldn't read a repository.

use specwave_bench::spec_coverage::{Settings, run};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let command = std::env::args().collect::<Vec<_>>().join(" ");

    if let Err(e) = Settings::parse(&args).and_then(|s| run(&s, &command)) {
        eprintln!("spec_coverage: {e}");
        std::process::exit(1);
    }
}
