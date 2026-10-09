//! Small random data in clusters, and the files that hold it, for the unit tests.

use std::path::{Path, PathBuf};

use coldivf::{Clusters, VectorSource};

/// A small random number generator (SplitMix64), so the tests need no dependency.
pub(crate) struct Rng(pub(crate) u64);

impl Rng {
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub(crate) fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A standard normal value, by the Box-Muller transform.
    pub(crate) fn normal(&mut self) -> f64 {
        let (u1, u2) = (1.0 - self.uniform(), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// A random direction of `dim` values, as a vector of length 1.
    pub(crate) fn direction(&mut self, dim: usize) -> Vec<f32> {
        unit((0..dim).map(|_| self.normal() as f32).collect())
    }
}

pub(crate) fn unit(v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
    v.iter().map(|&x| (f64::from(x) / norm) as f32).collect()
}

pub(crate) fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| f64::from(x) * f64::from(y)).sum()
}

/// Rows in memory, as a vector source.
pub(crate) struct Rows {
    pub(crate) dim: usize,
    pub(crate) values: Vec<f32>,
}

impl VectorSource for Rows {
    fn count(&self) -> usize {
        self.values.len() / self.dim
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn rows(&self, start: usize, rows: usize) -> &[f32] {
        &self.values[start * self.dim..(start + rows) * self.dim]
    }
}

/// The size of a synthetic dataset.
pub(crate) struct Shape {
    pub(crate) dim: usize,
    /// The documents.
    pub(crate) n: usize,
    pub(crate) nlist: usize,
    pub(crate) queries: usize,
    /// The true neighbors kept per query.
    pub(crate) width: usize,
}

/// `n` unit vectors around `nlist` random directions, clustered by the direction each came from, with the mean of
/// each cluster as its center. The queries are drawn the same way, but every second one is drawn around the middle of
/// two directions, so its nearest vectors sit in two clusters. Each query keeps its `width` nearest vectors.
pub(crate) struct Synthetic {
    pub(crate) rows: Rows,
    pub(crate) clusters: Clusters,
    pub(crate) queries: Vec<f32>,
    pub(crate) truth: Vec<u32>,
}

pub(crate) fn synthetic(seed: u64, shape: &Shape) -> Synthetic {
    let (dim, nlist) = (shape.dim, shape.nlist);
    let mut rng = Rng(seed);
    let directions: Vec<Vec<f32>> = (0..nlist).map(|_| rng.direction(dim)).collect();
    let around = |rng: &mut Rng, center: &[f32]| -> Vec<f32> {
        unit(center.iter().map(|&x| x + (0.08 * rng.normal()) as f32).collect())
    };
    let pick = |rng: &mut Rng| (rng.next_u64() % nlist as u64) as usize;
    let mut assignment = Vec::with_capacity(shape.n);
    let mut values = Vec::with_capacity(shape.n * dim);

    for _ in 0..shape.n {
        let c = pick(&mut rng);
        assignment.push(c as u32);
        values.extend(around(&mut rng, &directions[c]));
    }

    let mut queries = Vec::with_capacity(shape.queries * dim);

    for q in 0..shape.queries {
        let a = pick(&mut rng);
        let center = match q % 2 {
            0 => directions[a].clone(),
            _ => {
                let b = (a + 1 + pick(&mut rng) % (nlist - 1)) % nlist;
                unit(directions[a].iter().zip(&directions[b]).map(|(x, y)| x + y).collect())
            }
        };
        queries.extend(around(&mut rng, &center));
    }

    let mut centroids = vec![0f32; nlist * dim];
    let mut counts = vec![0f32; nlist];

    for (&c, row) in assignment.iter().zip(values.chunks_exact(dim)) {
        let center = &mut centroids[c as usize * dim..(c as usize + 1) * dim];
        center.iter_mut().zip(row).for_each(|(s, &x)| *s += x);
        counts[c as usize] += 1.0;
    }

    for (center, &count) in centroids.chunks_exact_mut(dim).zip(&counts) {
        center.iter_mut().for_each(|x| *x /= count.max(1.0));
    }

    let truth = queries
        .chunks_exact(dim)
        .flat_map(|q| nearest(&values, dim, q, shape.width))
        .collect();
    let clusters = Clusters {
        dim,
        parents: centroids[..dim].to_vec(),
        parent_of: vec![0; nlist],
        centroids,
        assignment,
    };
    Synthetic {
        rows: Rows { dim, values },
        clusters,
        queries,
        truth,
    }
}

/// Each row of `rows`, `dim` values long, moved by `noise` times a random normal value over √dim per value, then
/// scaled to length 1, so the noise has a length of about `noise`.
pub(crate) fn perturbed(rows: &[f32], dim: usize, seed: u64, noise: f64) -> Vec<f32> {
    let mut rng = Rng(seed);
    let scale = noise / (dim as f64).sqrt();
    rows.chunks_exact(dim)
        .flat_map(|q| unit(q.iter().map(|&x| x + (scale * rng.normal()) as f32).collect()))
        .collect()
}

/// The `k` rows of `rows` nearest to `q` by cosine, nearest first, with ties to the lower row.
pub(crate) fn nearest(rows: &[f32], dim: usize, q: &[f32], k: usize) -> Vec<u32> {
    let mut scored: Vec<(f64, u32)> = rows
        .chunks_exact(dim)
        .enumerate()
        .map(|(i, r)| (-dot(r, q), i as u32))
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    scored.iter().take(k).map(|&(_, i)| i).collect()
}

/// A fresh folder for one test's files in the build folder, with the process id in its name.
pub(crate) fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../target/specwave-bench-tests")
        .join(format!("{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Writes `values` as an `.fbin` or `.ibin` file of rows of `width`, i.e. the row count and the width as
/// little-endian i32, then the values as 4 bytes each.
pub(crate) fn write_table<T: Copy>(path: &Path, width: usize, values: &[T], bits: impl Fn(T) -> u32) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut bytes = Vec::with_capacity(8 + 4 * values.len());
    bytes.extend(((values.len() / width) as i32).to_le_bytes());
    bytes.extend((width as i32).to_le_bytes());
    values.iter().for_each(|&v| bytes.extend(bits(v).to_le_bytes()));
    std::fs::write(path, bytes).unwrap();
}
