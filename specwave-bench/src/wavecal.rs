//! Times cold read waves on one disk volume and fits SpecWave's wave model, t = a + b·bytes, where t is a wave's time
//! in milliseconds (ms), a its fixed cost and b its cost per byte.
//!
//! A wave reads some pages of 16 KiB at random from a scratch file, the way ColdIVF's macOS backend reads. That is,
//! through a handle that skips the operating system's cache, with a read hint (`F_RDADVISE`) for every range first and
//! then each range in order, where pages next to each other make one range. Before each wave it drops the file from
//! the cache through a mapping it holds for the whole run, since a new mapping of a file can make another process read
//! its first pages back, then checks with `mincore` that none of the wave's pages is in memory. A wave whose pages stay
//! cached is left out. Each wave runs on a core kept busy for a moment first, as SpecWave's worker runs while the model
//! keeps the cores busy. The fit runs through the median time of each wave size.

use std::io::{Error, ErrorKind};
use std::path::{Path, PathBuf};

use specwave::scheduler::WaveModel;

#[cfg(target_os = "macos")]
use crate::coldcache::Mapped;

/// The bytes of a read page, the unit of ColdIVF's macOS reads.
pub const PAGE: usize = 16384;

/// One timed wave.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wave {
    pub pages: usize,
    /// The reads it made, one per run of neighboring pages.
    pub ranges: usize,
    pub bytes: u64,
    pub ms: f64,
}

/// The waves of one size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Size {
    pub pages: usize,
    pub bytes: u64,
    pub waves: usize,
    /// The median time and the 90th percentile, by nearest rank.
    pub median_ms: f64,
    pub p90_ms: f64,
}

/// The pages of each wave, in the order the waves run. That is `trials` rounds of one wave of each size of `sizes`, in
/// pages. Each wave's pages are distinct, drawn at random from the file's `pages` pages, and sorted. The same seed
/// gives the same plan. No sizes, a size of 0 or above `pages`, or no trials give `InvalidInput`.
pub fn plan(sizes: &[usize], trials: usize, pages: usize, seed: u64) -> Result<Vec<Vec<usize>>, Error> {
    if sizes.is_empty() || trials == 0 || sizes.iter().any(|&s| s == 0 || s > pages) {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "a plan needs trials and sizes of at least 1 page and at most the file's pages",
        ));
    }

    let mut rng = SplitMix64(seed);
    let mut waves = Vec::with_capacity(trials * sizes.len());

    for _ in 0..trials {
        for &size in sizes {
            let mut picked = std::collections::BTreeSet::new();
            while picked.len() < size {
                picked.insert(rng.below(pages));
            }
            waves.push(picked.into_iter().collect());
        }
    }

    Ok(waves)
}

/// The reads of a wave's sorted pages, as (first page, pages), where pages next to each other make one read.
pub fn ranges(pages: &[usize]) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();

    for &page in pages {
        match out.last_mut() {
            Some((first, count)) if *first + *count == page => *count += 1,
            _ => out.push((page, 1)),
        }
    }

    out
}

/// The waves of each size, with the median and the 90th percentile of their times, in increasing order of size.
pub fn summarize(waves: &[Wave]) -> Vec<Size> {
    let mut sizes: Vec<usize> = waves.iter().map(|w| w.pages).collect();
    sizes.sort_unstable();
    sizes.dedup();
    sizes
        .into_iter()
        .map(|pages| {
            let mut times: Vec<f64> = waves.iter().filter(|w| w.pages == pages).map(|w| w.ms).collect();
            times.sort_by(f64::total_cmp);
            Size {
                pages,
                bytes: (pages * PAGE) as u64,
                waves: times.len(),
                median_ms: rank(&times, 50.0),
                p90_ms: rank(&times, 90.0),
            }
        })
        .collect()
}

/// The value at percentile `p` of the values `sorted`, in increasing order, by nearest rank, i.e. the smallest that at
/// least `p` percent of them are at or below.
fn rank(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    sorted[((p / 100.0 * n as f64).ceil() as usize).clamp(1, n) - 1]
}

/// The least-squares fit through each size's median time.
pub fn fit(sizes: &[Size]) -> Result<WaveModel, Error> {
    WaveModel::fit(&points(sizes))
}

/// Each size's bytes and median time, the points of the fit.
pub fn points(sizes: &[Size]) -> Vec<(f64, f64)> {
    sizes.iter().map(|s| (s.bytes as f64, s.median_ms)).collect()
}

/// The command line's settings.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// The scratch file. It is made, and deleted at the end, unless it is there already.
    pub file: PathBuf,
    /// The size of a scratch file that has to be made, in MiB.
    pub size_mib: usize,
    /// The wave sizes, in pages.
    pub sizes: Vec<usize>,
    /// The waves of each size.
    pub trials: usize,
    /// How long each wave's core is kept busy before the wave, in ms.
    pub busy_ms: u64,
    pub seed: u64,
    /// The folder that the fit is cached in, or `None` to leave the cache alone.
    pub cache: Option<PathBuf>,
    /// The results file that a line for the run is appended to.
    pub out: PathBuf,
    /// Whether a scratch file made by the run stays.
    pub keep: bool,
}

/// Every argument name that takes a value.
const NAMES: [&str; 8] = ["file", "size-mib", "sizes", "trials", "busy-ms", "seed", "cache", "out"];

impl Settings {
    /// Reads `args`, the arguments after the program's name. `repo` is this repository's folder, which holds the
    /// default scratch file and results file, and `support` the default cache folder, if the system has one. A name
    /// that isn't known, a name given twice, or a bad value gives `InvalidInput`.
    pub fn parse(args: &[String], repo: &Path, support: Option<PathBuf>) -> Result<Settings, Error> {
        let mut values: Vec<(&str, &str)> = Vec::new();
        let mut keep = false;
        let mut it = args.iter();

        while let Some(arg) = it.next() {
            if arg == "--keep" {
                if keep {
                    return Err(invalid("--keep is given twice".to_string()));
                }
                keep = true;
                continue;
            }
            let name = arg
                .strip_prefix("--")
                .filter(|n| NAMES.contains(n))
                .ok_or_else(|| invalid(format!("unknown argument {arg}")))?;
            let value = it.next().ok_or_else(|| invalid(format!("--{name} needs a value")))?;
            if values.iter().any(|(n, _)| *n == name) {
                return Err(invalid(format!("--{name} is given twice")));
            }
            values.push((name, value));
        }

        let get = |name: &str| values.iter().find(|(n, _)| *n == name).map(|&(_, v)| v);
        let number = |name: &str, default: u64| -> Result<u64, Error> {
            get(name).map_or(Ok(default), |v| {
                v.trim()
                    .parse()
                    .map_err(|_| invalid(format!("--{name} has a bad value: {v}")))
            })
        };
        let sizes = match get("sizes") {
            Some(v) => v
                .split(',')
                .map(|s| s.trim().parse::<usize>().ok().filter(|&p| p > 0))
                .collect::<Option<Vec<usize>>>()
                .ok_or_else(|| invalid(format!("--sizes has a bad value: {v}")))?,
            None => vec![16, 32, 64, 128, 256, 512],
        };
        let cache = match get("cache") {
            Some("none") => None,
            Some(dir) => Some(PathBuf::from(dir)),
            None => {
                Some(support.ok_or_else(|| invalid("this system has no support folder, so give --cache".to_string()))?)
            }
        };
        let settings = Settings {
            file: get("file").map_or_else(|| repo.join("data/wavecal.bin"), PathBuf::from),
            size_mib: number("size-mib", 256)? as usize,
            sizes,
            trials: number("trials", 30)? as usize,
            busy_ms: number("busy-ms", 150)?,
            seed: number("seed", 1)?,
            cache,
            out: get("out").map_or_else(|| repo.join("results/wavecal.jsonl"), PathBuf::from),
            keep,
        };

        if settings.trials == 0 || settings.size_mib == 0 {
            return Err(invalid("--trials and --size-mib are at least 1".to_string()));
        }

        if settings.sizes.iter().any(|&s| s * PAGE > settings.size_mib << 20) {
            return Err(invalid("every wave must fit in the scratch file".to_string()));
        }

        Ok(settings)
    }
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

/// Writes `bytes` bytes of random data to a new file at `path`, through a handle that skips the cache, then waits until
/// they are on the disk. An existing file gives `AlreadyExists`.
pub fn make(path: &Path, bytes: usize, seed: u64) -> Result<(), Error> {
    let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(path)?;
    skip_cache(&file)?;
    let mut rng = SplitMix64(seed);
    let mut chunk = vec![0u8; 8 << 20];
    let mut left = bytes;

    while left > 0 {
        let n = left.min(chunk.len());
        chunk[..n]
            .as_chunks_mut::<8>()
            .0
            .iter_mut()
            .for_each(|c| *c = rng.next().to_le_bytes());
        std::io::Write::write_all(&mut file, &chunk[..n])?;
        left -= n;
    }

    file.sync_all()
}

/// Turns off the operating system's cache for `file`'s reads and writes.
#[cfg(target_os = "macos")]
fn skip_cache(file: &std::fs::File) -> Result<(), Error> {
    use std::os::fd::AsRawFd;

    // SAFETY: F_NOCACHE takes an integer and touches no memory.
    if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) } == -1 {
        return Err(Error::last_os_error());
    }

    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn skip_cache(_file: &std::fs::File) -> Result<(), Error> {
    Ok(())
}

/// Times the waves of `plan`, each a list of sorted pages of `file`, with each wave's core kept busy for `busy` first.
/// Returns each wave in the order of the plan, or `None` for one left out for starting with cached pages. It runs at
/// the user-interactive quality of service (QoS), as SpecWave's worker does. It is written for macOS only, so elsewhere
/// it gives `Unsupported`, and a page past the end of the file gives `InvalidInput`.
#[cfg(target_os = "macos")]
pub fn run(file: &Path, plan: &[Vec<usize>], busy: std::time::Duration) -> Result<Vec<Option<Wave>>, Error> {
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::FileExt;
    use std::time::Instant;

    let handle = std::fs::File::open(file)?;
    skip_cache(&handle)?;
    let map = Mapped::of(&handle)?;
    let largest = plan.iter().map(Vec::len).max().unwrap_or(0);
    let mut buffer = vec![0u8; largest * PAGE];
    let mut waves = Vec::with_capacity(plan.len());

    if plan.iter().flatten().any(|&p| (p + 1) * PAGE > map.len()) {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "a wave reads past the end of the file",
        ));
    }

    // SAFETY: the call changes only the calling thread's scheduling class, and it takes no pointers.
    unsafe { libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0) };

    for pages in plan {
        map.evict()?;
        if cached(&map, pages)? {
            map.evict()?;
            if cached(&map, pages)? {
                waves.push(None);
                continue;
            }
        }
        let reads = ranges(pages);
        let start = Instant::now();
        while start.elapsed() < busy {
            std::hint::spin_loop();
        }
        let start = Instant::now();
        for &(first, count) in &reads {
            let hint = libc::radvisory {
                ra_offset: (first * PAGE) as libc::off_t,
                ra_count: (count * PAGE) as libc::c_int,
            };
            // SAFETY: `hint` outlives the call, and F_RDADVISE only reads it.
            if unsafe { libc::fcntl(handle.as_raw_fd(), libc::F_RDADVISE, &hint as *const libc::radvisory) } == -1 {
                return Err(Error::last_os_error());
            }
        }
        let mut at = 0;
        for &(first, count) in &reads {
            handle.read_exact_at(&mut buffer[at..at + count * PAGE], (first * PAGE) as u64)?;
            at += count * PAGE;
        }
        waves.push(Some(Wave {
            pages: pages.len(),
            ranges: reads.len(),
            bytes: (pages.len() * PAGE) as u64,
            ms: start.elapsed().as_secs_f64() * 1e3,
        }));
    }

    Ok(waves)
}

#[cfg(not(target_os = "macos"))]
pub fn run(_file: &Path, _plan: &[Vec<usize>], _busy: std::time::Duration) -> Result<Vec<Option<Wave>>, Error> {
    Err(Error::new(
        ErrorKind::Unsupported,
        "the wave calibration is written for macOS only",
    ))
}

/// Whether any memory page of the read pages `pages` of the file that `map` maps is in the cache.
#[cfg(target_os = "macos")]
fn cached(map: &Mapped, pages: &[usize]) -> Result<bool, Error> {
    for &p in pages {
        if map.resident(p * PAGE..(p + 1) * PAGE)? > 0 {
            return Ok(true);
        }
    }

    Ok(false)
}

fn invalid(message: String) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

#[cfg(test)]
#[path = "../tests/unit/wavecal.rs"]
mod tests;
