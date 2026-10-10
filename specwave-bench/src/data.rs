//! The vector files. Vectors come as `.fbin` files and ids as `.ibin` files. Each starts with its row count and row
//! width as little-endian i32, then holds the rows, 4 bytes per value. Both are mapped, so nothing loads until it is
//! read.
//!
//! A dataset's files are named after it, eg. `quora`, inside the data folder:
//!
//! | File | Rows |
//! |---|---|
//! | `<name>_corpus.fbin` | The documents' final embeddings |
//! | `<name>_test.fbin` | The test queries' final embeddings |
//! | `<name>_test.gt.ibin` | Each test query's true nearest documents by row, nearest first |
//! | `spec/<name>/e.fbin` | The final embeddings that the guesses were fitted to, the bytes of `<name>_test.fbin` |
//! | `spec/<name>/ehat/<label>.fbin` | A guess of each test query's final embedding, eg. `openings.L9` from layer 9 |
//! | `spec/<name>/clusters/<kind>.centroids.fbin` | The cluster centers of one clustering, eg. `l2` |
//! | `spec/<name>/clusters/<kind>.assign.ibin` | The cluster of each document |
//! | `coverage/<name>_<kind>.coldivf` | The index over one clustering, which the coverage measurement writes |
//!
//! The kind `cos` has no cluster files of its own. It reads those of `l2`, as [`clusters_of`] says.

use std::fs::{self, File};
use std::io::{Error, ErrorKind};
use std::path::{Path, PathBuf};

use coldivf::{Clusters, VectorSource};
use memmap2::Mmap;

// The values are read in place, so the machine must store numbers in the same byte order as the files.
const _: () = assert!(cfg!(target_endian = "little"), "the vector files are little-endian");

/// A 4-byte value that every bit pattern is valid for.
trait Word: bytemuck::AnyBitPattern {}

impl Word for f32 {}

impl Word for u32 {}

/// A mapped table of `count` rows of `width` values, after its 8-byte header.
struct Table {
    map: Mmap,
    count: usize,
    width: usize,
}

impl Table {
    fn open(path: &Path) -> Result<Table, Error> {
        let file = File::open(path).map_err(|e| Error::new(e.kind(), format!("{}: {e}", path.display())))?;

        // A file too short for its header isn't mapped at all.
        if file.metadata()?.len() < 8 {
            return Err(damaged(path, "is shorter than its header"));
        }

        // SAFETY: the files are written once and then only read, so nothing changes them under the map.
        let map = unsafe { Mmap::map(&file)? };
        let field = |at: usize| {
            map.get(at..at + 4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        };
        let (Some(count), Some(width)) = (field(0), field(4)) else {
            return Err(damaged(path, "is shorter than its header"));
        };

        if count < 0 || width <= 0 {
            return Err(damaged(path, "has a negative row count or a width that isn't positive"));
        }

        let (count, width) = (count as usize, width as usize);
        let len = count
            .checked_mul(width)
            .and_then(|values| values.checked_mul(4))
            .and_then(|bytes| bytes.checked_add(8));

        if len != Some(map.len()) {
            return Err(damaged(path, "doesn't hold as many rows as its header says"));
        }

        // The map starts on a page boundary, so this holds unless the system maps files some other way.
        if bytemuck::try_cast_slice::<u8, u32>(&map[8..]).is_err() {
            return Err(damaged(path, "doesn't hold its values at 4-byte boundaries"));
        }

        Ok(Table { map, count, width })
    }

    /// Every value after the header, row after row. `open` checked that they are whole 4-byte values at 4-byte
    /// boundaries, so the cast can't fail.
    fn values<T: Word>(&self) -> &[T] {
        bytemuck::cast_slice(&self.map[8..])
    }
}

fn damaged(path: &Path, what: &str) -> Error {
    Error::new(ErrorKind::InvalidData, format!("{} {what}", path.display()))
}

/// A mapped `.fbin` file of f32 vectors.
pub struct Fbin(Table);

impl Fbin {
    pub fn open(path: &Path) -> Result<Fbin, Error> {
        Table::open(path).map(Fbin)
    }

    pub fn len(&self) -> usize {
        self.0.count
    }

    pub fn is_empty(&self) -> bool {
        self.0.count == 0
    }

    pub fn dim(&self) -> usize {
        self.0.width
    }

    /// Every vector, one after another.
    pub fn values(&self) -> &[f32] {
        self.0.values()
    }

    pub fn row(&self, i: usize) -> &[f32] {
        &self.values()[i * self.dim()..(i + 1) * self.dim()]
    }
}

impl VectorSource for Fbin {
    fn count(&self) -> usize {
        self.len()
    }

    fn dim(&self) -> usize {
        self.0.width
    }

    fn rows(&self, start: usize, rows: usize) -> &[f32] {
        let dim = self.0.width;
        &self.values()[start * dim..(start + rows) * dim]
    }
}

/// Writes `values`, rows of `width` values, as an `.fbin` file at `path`. It writes `<path>.part` first and renames
/// it, so a crash never leaves a partial file under the final name. A width of 0, values that don't fill whole rows,
/// or more rows or a wider row than the header can hold give `InvalidInput`.
pub fn write_fbin(path: &Path, values: &[f32], width: usize) -> Result<(), Error> {
    let rows = values
        .len()
        .checked_div(width)
        .filter(|rows| rows * width == values.len());
    let (Some(Ok(count)), Ok(width)) = (rows.map(i32::try_from), i32::try_from(width)) else {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "an .fbin file holds whole rows, of a width above 0",
        ));
    };
    let mut bytes = Vec::with_capacity(8 + 4 * values.len());
    bytes.extend_from_slice(&count.to_le_bytes());
    bytes.extend_from_slice(&width.to_le_bytes());
    values.iter().for_each(|v| bytes.extend_from_slice(&v.to_le_bytes()));
    let mut part = path.as_os_str().to_os_string();
    part.push(".part");
    let part = PathBuf::from(part);
    fs::write(&part, &bytes).and_then(|()| fs::rename(&part, path))
}

/// A mapped `.ibin` file of u32 rows, eg. each query's nearest neighbors, nearest first.
pub struct Ibin(Table);

impl Ibin {
    pub fn open(path: &Path) -> Result<Ibin, Error> {
        Table::open(path).map(Ibin)
    }

    pub fn len(&self) -> usize {
        self.0.count
    }

    pub fn is_empty(&self) -> bool {
        self.0.count == 0
    }

    pub fn width(&self) -> usize {
        self.0.width
    }

    /// Every value, one row after another.
    pub fn values(&self) -> &[u32] {
        self.0.values()
    }

    pub fn row(&self, i: usize) -> &[u32] {
        &self.values()[i * self.width()..(i + 1) * self.width()]
    }
}

/// A guess file, with its label and the layer the label names, if any.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Guess {
    pub label: String,
    pub layer: Option<usize>,
    pub path: PathBuf,
}

/// The files of the dataset `name` in the data folder `root`.
pub struct Files {
    root: PathBuf,
    name: String,
}

impl Files {
    pub fn new(root: &Path, name: &str) -> Files {
        Files {
            root: root.to_path_buf(),
            name: name.to_string(),
        }
    }

    pub fn corpus(&self) -> PathBuf {
        self.root.join(format!("{}_corpus.fbin", self.name))
    }

    pub fn queries(&self) -> PathBuf {
        self.root.join(format!("{}_test.fbin", self.name))
    }

    pub fn truth(&self) -> PathBuf {
        self.root.join(format!("{}_test.gt.ibin", self.name))
    }

    /// The final embeddings that the guesses were fitted to.
    pub fn targets(&self) -> PathBuf {
        self.root.join("spec").join(&self.name).join("e.fbin")
    }

    pub fn centroids(&self, kind: &str) -> PathBuf {
        self.clusters().join(format!("{}.centroids.fbin", clusters_of(kind).0))
    }

    pub fn assignment(&self, kind: &str) -> PathBuf {
        self.clusters().join(format!("{}.assign.ibin", clusters_of(kind).0))
    }

    /// Where the index over one clustering goes.
    pub fn index(&self, kind: &str) -> PathBuf {
        self.root.join("coverage").join(format!("{}_{kind}.coldivf", self.name))
    }

    fn clusters(&self) -> PathBuf {
        self.root.join("spec").join(&self.name).join("clusters")
    }

    /// Every guess file, ordered by layer and then by label, with the guesses that name no layer last. A dataset whose
    /// guess folder doesn't exist yet has none.
    pub fn guesses(&self) -> Result<Vec<Guess>, Error> {
        let dir = self.root.join("spec").join(&self.name).join("ehat");
        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };
        let mut guesses = Vec::new();

        for entry in entries {
            let path = entry?.path();
            let label = path
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(|n| n.strip_suffix(".fbin"));
            if let Some(label) = label.filter(|l| !l.is_empty()) {
                guesses.push(Guess {
                    label: label.to_string(),
                    layer: layer_of(label),
                    path: path.clone(),
                });
            }
        }

        guesses.sort_by_key(|g| (g.layer.unwrap_or(usize::MAX), g.label.clone()));
        Ok(guesses)
    }
}

/// The clustering from spherical k-means, whose centers have length 1.
pub const SPHERICAL: &str = "sph";

/// The clustering that reads the clusters of `l2` with their centers scaled to length 1.
pub const COSINE: &str = "cos";

/// The kind whose cluster files the clustering `kind` reads, and whether its centers are scaled to length 1. An index
/// routes a query to the clusters whose centers are nearest by L2 (Euclidean) distance. For a query and centers of
/// length 1, that is the order by cosine. `cos` reads the files of `l2` and scales its centers, `sph` scales its own,
/// and any other kind reads its own files as they are.
pub fn clusters_of(kind: &str) -> (&str, bool) {
    match kind {
        COSINE => ("l2", true),
        SPHERICAL => (SPHERICAL, true),
        other => (other, false),
    }
}

/// The clustering `kind` of the documents `corpus`, from the files that [`clusters_of`] names, with its centers
/// scaled to length 1 when it says so. The index keeps the clusters in id order, so one group holds them all. Files
/// that don't fit the documents, or centers with a value that isn't finite, give `InvalidData`.
pub fn clusters(files: &Files, kind: &str, corpus: &Fbin) -> Result<Clusters, Error> {
    let (centers_path, assignment_path) = (files.centroids(kind), files.assignment(kind));
    let (centers, assignment) = (Fbin::open(&centers_path)?, Ibin::open(&assignment_path)?);
    let dim = corpus.dim();

    if centers.dim() != dim || centers.is_empty() {
        return Err(damaged(
            &centers_path,
            &format!("doesn't hold cluster centers of {dim} values"),
        ));
    }

    let nlist = centers.len();

    if assignment.width() != 1
        || assignment.len() != corpus.len()
        || assignment.values().iter().any(|&c| c as usize >= nlist)
    {
        let what = format!("doesn't hold one cluster of the {nlist} per document");
        return Err(damaged(&assignment_path, &what));
    }

    if let Some(i) = centers
        .values()
        .chunks_exact(dim)
        .position(|row| row.iter().any(|x| !x.is_finite()))
    {
        return Err(damaged(
            &centers_path,
            &format!("has a value in cluster center {i} that isn't finite"),
        ));
    }

    let mut centroids = centers.values().to_vec();

    if clusters_of(kind).1 {
        for (i, row) in centroids.chunks_exact_mut(dim).enumerate() {
            let norm = row.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
            if !(norm.is_finite() && norm > 0.0) {
                return Err(damaged(
                    &centers_path,
                    &format!("has a cluster center {i} with no direction"),
                ));
            }
            row.iter_mut().for_each(|x| *x = (f64::from(*x) / norm) as f32);
        }
    }

    let mut sum = vec![0f64; dim];

    for row in centroids.chunks_exact(dim) {
        sum.iter_mut().zip(row).for_each(|(s, &x)| *s += f64::from(x));
    }

    Ok(Clusters {
        dim,
        parents: sum.iter().map(|&s| (s / nlist as f64) as f32).collect(),
        parent_of: vec![0; nlist],
        centroids,
        assignment: assignment.values().to_vec(),
    })
}

/// The layer that a guess's label names, from its first part of the form `L<digits>`, eg. 9 for `openings.L9`. Parts
/// are split at dots and underscores, and a lowercase `l` works too.
pub fn layer_of(label: &str) -> Option<usize> {
    label.split(['.', '_']).find_map(|part| {
        let digits = part.strip_prefix(['L', 'l'])?;
        match !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) {
            true => digits.parse().ok(),
            false => None,
        }
    })
}

#[cfg(test)]
#[path = "../tests/unit/data.rs"]
mod tests;
