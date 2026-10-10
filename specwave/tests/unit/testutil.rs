//! Seeded random values, scratch folders, small index files and a time limit for the unit tests that need no model.

use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Duration;

use coldivf::{
    Anchor, BuildParams, Builder, Clusters, Hit, Index, LeafOrder, Limits, Metric, SearchCtx, SearchParams,
    VectorSource,
};

/// A small seeded generator (SplitMix64), so the tests need no dependency.
pub(crate) struct Rng(pub(crate) u64);

impl Rng {
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A uniform value in [0, 1).
    pub(crate) fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A standard normal value, by the Box-Muller transform.
    pub(crate) fn normal(&mut self) -> f64 {
        let (u1, u2) = (1.0 - self.uniform(), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// `n` standard normal values as f32.
    pub(crate) fn normals(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.normal() as f32).collect()
    }
}

/// A folder in the system's temporary folder, deleted on drop. Its name holds the process id and the test's name, so
/// neither test runs nor tests clash.
pub(crate) struct Scratch(PathBuf);

impl Scratch {
    pub(crate) fn new(name: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!("specwave-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }

    pub(crate) fn path(&self, file: &str) -> PathBuf {
        self.0.join(file)
    }

    /// The names in the folder `dir` inside the scratch folder, sorted.
    pub(crate) fn names(&self, dir: &str) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(self.0.join(dir))
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The cosine between `a` and `b`, in f64.
pub(crate) fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(&p, &q)| f64::from(p) * f64::from(q)).sum::<f64>();
    dot(a, b) / (dot(a, a) * dot(b, b)).sqrt()
}

/// A vector of length 1 in the direction of `v`.
pub(crate) fn unit(v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
    v.iter().map(|&x| (f64::from(x) / norm) as f32).collect()
}

/// The size of the vectors of a test index.
pub(crate) const DIM: usize = 128;

/// The clusters, their groups, the vectors and the queries of a test index.
const LISTS: usize = 24;
const GROUPS: usize = 6;
const ROWS: usize = 3000;
const QUERIES: usize = 24;

/// Rows held in memory, as a vector source.
struct Rows {
    values: Vec<f32>,
}

impl VectorSource for Rows {
    fn count(&self) -> usize {
        self.values.len() / DIM
    }

    fn dim(&self) -> usize {
        DIM
    }

    fn rows(&self, start: usize, rows: usize) -> &[f32] {
        &self.values[start * DIM..(start + rows) * DIM]
    }
}

/// A small index file of unit vectors in clusters, with its queries. The vectors sit around 24 directions, one cluster
/// each, in 6 groups of 4 nearby directions, so a vector's nearest neighbors spread over the clusters of its group. The
/// queries sit around the same directions, every second one between two of them, so its nearest vectors sit in two
/// groups. The file is in a scratch folder that goes when this drops, after the index.
pub(crate) struct Indexed {
    pub(crate) index: Index,
    queries: Vec<f32>,
    _scratch: Scratch,
}

/// The name of a test index's file in its scratch folder.
const FILE: &str = "index.coldivf";

impl Indexed {
    /// The index file, eg. to open it with another read backend.
    #[cfg(feature = "bench")]
    pub(crate) fn path(&self) -> PathBuf {
        self._scratch.path(FILE)
    }

    pub(crate) fn len(&self) -> usize {
        self.queries.len() / DIM
    }

    pub(crate) fn query(&self, q: usize) -> &[f32] {
        &self.queries[q * DIM..(q + 1) * DIM]
    }
}

/// Builds the test index `name` from the seed `seed`, with cosine distances and the cluster anchor.
pub(crate) fn indexed(name: &str, seed: u64) -> Indexed {
    let mut rng = Rng(seed);
    let groups: Vec<Vec<f32>> = (0..GROUPS).map(|_| unit(rng.normals(DIM))).collect();
    let directions: Vec<Vec<f32>> = (0..LISTS)
        .map(|c| {
            let spread = unit(rng.normals(DIM));
            unit(
                groups[c % GROUPS]
                    .iter()
                    .zip(&spread)
                    .map(|(g, s)| g + 0.5 * s)
                    .collect(),
            )
        })
        .collect();
    let around = |rng: &mut Rng, center: &[f32]| -> Vec<f32> {
        unit(center.iter().map(|&x| x + (0.08 * rng.normal()) as f32).collect())
    };
    let pick = |rng: &mut Rng| (rng.next_u64() % LISTS as u64) as usize;
    let (mut values, mut assignment) = (Vec::with_capacity(ROWS * DIM), Vec::with_capacity(ROWS));

    for _ in 0..ROWS {
        let c = pick(&mut rng);
        assignment.push(c as u32);
        values.extend(around(&mut rng, &directions[c]));
    }

    let mut queries = Vec::with_capacity(QUERIES * DIM);

    for q in 0..QUERIES {
        let a = pick(&mut rng);
        let b = (a + 1 + pick(&mut rng) % (LISTS - 1)) % LISTS;
        let center = match q % 2 {
            0 => directions[a].clone(),
            _ => unit(directions[a].iter().zip(&directions[b]).map(|(x, y)| x + y).collect()),
        };
        queries.extend(around(&mut rng, &center));
    }

    let mut centroids = vec![0f32; LISTS * DIM];
    let mut counts = vec![0f32; LISTS];

    for (&c, row) in assignment.iter().zip(values.as_chunks::<DIM>().0) {
        let center = &mut centroids[c as usize * DIM..(c as usize + 1) * DIM];
        center.iter_mut().zip(row).for_each(|(s, &x)| *s += x);
        counts[c as usize] += 1.0;
    }

    for (center, &count) in centroids.as_chunks_mut::<DIM>().0.iter_mut().zip(&counts) {
        center.iter_mut().for_each(|x| *x /= count.max(1.0));
    }

    let clusters = Clusters {
        dim: DIM,
        parents: centroids[..DIM].to_vec(),
        parent_of: vec![0; LISTS],
        centroids,
        assignment,
    };
    let params = BuildParams {
        metric: Metric::Cosine,
        vectors_per_list: ROWS / LISTS,
        parents: 1,
        order: LeafOrder::Chained,
        anchor: Anchor::Cluster,
        page_size: 4096,
        threads: 1,
        seed,
    };
    let scratch = Scratch::new(name);
    let path = scratch.path(FILE);
    Builder::new(params)
        .build_with_clusters(&Rows { values }, &clusters, &path)
        .unwrap();
    Indexed {
        index: Index::open(&path).unwrap(),
        queries,
        _scratch: scratch,
    }
}

/// The hits of a normal search of `query` with `params`, on a context of its own.
pub(crate) fn searched(index: &Index, query: &[f32], params: &SearchParams) -> Vec<Hit> {
    let limits = Limits {
        max_k: params.k,
        max_nprobe: params.nprobe,
        max_rerank: params.rerank,
    };
    let mut ctx = SearchCtx::new(limits);
    index.search(query, params, &mut ctx).unwrap().to_vec()
}

/// Whether `a` and `b` hold the same ids with the same score bits, in the same order.
pub(crate) fn same(a: &[Hit], b: &[Hit]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.id == y.id && x.score.to_bits() == y.score.to_bits())
}

/// Runs `body` on a thread of its own, and fails if it takes longer than `limit`, eg. because a call waits forever.
pub(crate) fn within(limit: Duration, body: impl FnOnce() + Send + 'static) {
    let (done, finished) = mpsc::channel();
    let thread = std::thread::spawn(move || {
        body();
        let _ = done.send(());
    });

    match finished.recv_timeout(limit) {
        Ok(()) => thread.join().unwrap(),
        Err(mpsc::RecvTimeoutError::Disconnected) => std::panic::resume_unwind(thread.join().unwrap_err()),
        Err(mpsc::RecvTimeoutError::Timeout) => panic!("the calls took over {limit:?}, so one likely waits forever"),
    }
}
