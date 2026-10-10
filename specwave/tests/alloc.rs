//! Checks that the worker's path allocates nothing per query once its buffers have grown. The path is a prepare, a
//! guess and its finish, a guess replaced by a newer one, a guess that a search drops, and a cancel. It also checks
//! that on an index's first query the guess and its finish allocate nothing once the prepare has run, since the
//! prepare loads the index's cluster tables and makes its buffers while the model runs. A counting allocator counts
//! every allocation and every free in the process while a check runs, the worker thread's too, so this runs as its own
//! program.

use std::alloc::{GlobalAlloc, Layout, System};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use coldivf::{
    Anchor, BuildParams, Builder, Clusters, Hit, Index, LeafOrder, Limits, Metric, SearchParams, VectorSource,
};
use specwave::speculate::{SpecParams, Worker};

struct Counting;

static ON: AtomicBool = AtomicBool::new(false);
static COUNT: AtomicUsize = AtomicUsize::new(0);
static FREES: AtomicUsize = AtomicUsize::new(0);

fn note(counter: &AtomicUsize) {
    if ON.load(Ordering::Relaxed) {
        counter.fetch_add(1, Ordering::Relaxed);
    }
}

// SAFETY: every call goes straight to the system allocator with the caller's arguments. Counting touches only atomics.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note(&COUNT);
        // SAFETY: the caller keeps the promises `GlobalAlloc::alloc` asks of it.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note(&COUNT);
        // SAFETY: as for `alloc`.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note(&COUNT);
        // SAFETY: `ptr` came from this allocator with `layout`, as the caller promises.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        note(&FREES);
        // SAFETY: as for `realloc`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// How many allocations and frees `f` makes.
fn counts(f: impl FnOnce()) -> (usize, usize) {
    COUNT.store(0, Ordering::SeqCst);
    FREES.store(0, Ordering::SeqCst);
    ON.store(true, Ordering::SeqCst);
    f();
    ON.store(false, Ordering::SeqCst);
    (COUNT.load(Ordering::SeqCst), FREES.load(Ordering::SeqCst))
}

const DIM: usize = 128;
const LISTS: usize = 16;
const ROWS: usize = 2000;

/// Rows held in memory.
struct Rows(Vec<f32>);

impl VectorSource for Rows {
    fn count(&self) -> usize {
        self.0.len() / DIM
    }

    fn dim(&self) -> usize {
        DIM
    }

    fn rows(&self, start: usize, rows: usize) -> &[f32] {
        &self.0[start * DIM..(start + rows) * DIM]
    }
}

/// Random values in [−1, 1), from a SplitMix64 stream.
fn random(state: &mut u64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

/// The folder of the index file, deleted when dropped, even when a check panics.
struct TempDir(std::path::PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// An index of random rows from the seed `seed`, each in the cluster of its row number modulo the cluster count,
/// around the clusters' means, in the file `name` of `dir`.
fn build(dir: &TempDir, name: &str, seed: u64) -> Index {
    let mut state = seed;
    let rows = Rows(random(&mut state, ROWS * DIM));
    let mut centroids = vec![0f32; LISTS * DIM];

    for (i, row) in rows.0.as_chunks::<DIM>().0.iter().enumerate() {
        let center = &mut centroids[(i % LISTS) * DIM..(i % LISTS + 1) * DIM];
        center
            .iter_mut()
            .zip(row)
            .for_each(|(c, &x)| *c += x / (ROWS / LISTS) as f32);
    }

    let clusters = Clusters {
        dim: DIM,
        parents: centroids[..DIM].to_vec(),
        parent_of: vec![0; LISTS],
        centroids,
        assignment: (0..ROWS).map(|i| (i % LISTS) as u32).collect(),
    };
    let params = BuildParams {
        metric: Metric::Cosine,
        vectors_per_list: ROWS / LISTS,
        parents: 1,
        order: LeafOrder::Chained,
        anchor: Anchor::Cluster,
        page_size: 4096,
        threads: 1,
        seed: 3,
    };
    let path = dir.0.join(name);
    Builder::new(params)
        .build_with_clusters(&rows, &clusters, &path)
        .unwrap();
    Index::open(&path).unwrap()
}

/// Every kind of request the worker serves, for the queries `a` and `b`, ending with answers, so the worker has
/// served the rest by the time it returns.
fn round(worker: &mut Worker, index: &Index, a: &[f32], b: &[f32], spec: &SpecParams, out: &mut Vec<Hit>) {
    worker.prepare(index, DIM).unwrap();
    worker.speculate(index, a, spec).unwrap();
    worker.finish(a, out).unwrap();
    worker.speculate(index, b, spec).unwrap();
    worker.speculate(index, a, spec).unwrap();
    worker.finish(b, out).unwrap();
    worker.speculate(index, a, spec).unwrap();
    worker.search(index, b, &spec.base, out).unwrap();
    worker.speculate(index, b, spec).unwrap();
    worker.cancel();
    worker.search(index, a, &spec.base, out).unwrap();
}

fn main() -> ExitCode {
    let dir = TempDir(std::env::temp_dir().join(format!("specwave-alloc-{}", std::process::id())));
    let _ = std::fs::remove_dir_all(&dir.0);
    std::fs::create_dir_all(&dir.0).unwrap();
    let index = build(&dir, "index.coldivf", 1);
    let base = SearchParams {
        k: 10,
        nprobe: 4,
        rerank: 20,
        reach: None,
    };
    let spec = SpecParams {
        base,
        beta: 0.5,
        rerank_mult: 2,
    };
    let limits = Limits {
        max_k: 10,
        max_nprobe: 6,
        max_rerank: 40,
    };
    let mut worker = Worker::new(limits).unwrap();
    let mut state = 7;
    let queries = random(&mut state, 8 * DIM);
    let query = |i: usize| &queries[(i % 8) * DIM..(i % 8 + 1) * DIM];
    let mut out = Vec::new();

    for i in 0..4 {
        round(&mut worker, &index, query(i), query(i + 1), &spec, &mut out);
    }

    let mut failed = false;

    for i in 0..8 {
        let (allocations, frees) = counts(|| round(&mut worker, &index, query(i), query(i + 3), &spec, &mut out));
        println!("round {i}: {allocations} allocations, {frees} frees");
        failed |= allocations > 0 || frees > 0;
    }

    // The text of an index's first query arrives, and the model runs long enough for the prepare to end.
    let first = build(&dir, "first.coldivf", 2);
    worker.prepare(&first, DIM).unwrap();
    std::thread::sleep(Duration::from_millis(500));
    let (allocations, frees) = counts(|| {
        worker.speculate(&first, query(5), &spec).unwrap();
        worker.finish(query(5), &mut out).unwrap();
    });
    println!("an index's first guess and finish: {allocations} allocations, {frees} frees");
    failed |= allocations > 0 || frees > 0;
    drop(worker);

    match failed {
        true => {
            eprintln!("the worker's path allocated");
            ExitCode::FAILURE
        }
        false => {
            println!("the worker's path allocates nothing");
            ExitCode::SUCCESS
        }
    }
}
