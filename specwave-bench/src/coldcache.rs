//! Cold-cache tools for macOS. A `Mapped` file can be dropped from the operating system's cache and checked for how
//! much of it stays cached, `make_cold` drops files until they stay out, and `Usage` reads the process's counters, eg.
//! its memory footprint and its disk reads. Elsewhere dropping and checking give `Unsupported`, and the counters read
//! zeros.

use std::fs::File;
use std::io::{Error, ErrorKind};
use std::ops::Range;
use std::path::Path;
use std::time::{Duration, Instant};

use memmap2::Mmap;

/// How much of the start of each file `make_cold` watches while it waits.
const HEAD: usize = 1 << 20;

/// How long `make_cold` keeps dropping the files before it gives up, as a multiple of its wait. Pages can keep coming
/// back for a few seconds while the disk is busy, eg. reads that an earlier read hint asked for, or another process's
/// scan of a file.
const PATIENCE: u32 = 20;

/// A file mapped once, so it can be dropped from the cache and checked many times through the same mapping. On macOS a
/// new mapping of a file can make another process, eg. an endpoint-security agent, read the file's first pages back
/// into the cache up to a few hundred milliseconds later. Dropping and checking pages through a held mapping sets off
/// nothing. The map's bytes are never read.
pub struct Mapped {
    /// None for an empty file, which can't be mapped.
    map: Option<Mmap>,
}

impl Mapped {
    pub fn open(path: &Path) -> Result<Mapped, Error> {
        let file = File::open(path).map_err(|e| Error::new(e.kind(), format!("{}: {e}", path.display())))?;
        Mapped::of(&file)
    }

    /// The mapping of the open `file`.
    pub fn of(file: &File) -> Result<Mapped, Error> {
        if file.metadata()?.len() == 0 {
            return Ok(Mapped { map: None });
        }

        // SAFETY: `Mapped` never reads the map's bytes. It only passes the map's address range to msync and mincore.
        let map = unsafe { Mmap::map(file)? };
        Ok(Mapped { map: Some(map) })
    }

    pub fn len(&self) -> usize {
        self.map.as_ref().map_or(0, |m| m.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drops every cached page of the file, by asking the kernel to invalidate the mapping's pages with `msync`, which
    /// drops them from the unified buffer cache too. A page that another process maps stays cached.
    pub fn evict(&self) -> Result<(), Error> {
        let Some(map) = &self.map else {
            return Ok(());
        };
        imp::invalidate(map)
    }

    /// How many memory pages of the bytes `range` of the file are in the cache. The range is cut to the file and
    /// rounded out to whole pages.
    pub fn resident(&self, range: Range<usize>) -> Result<usize, Error> {
        let Some(map) = &self.map else {
            return Ok(0);
        };
        let end = range.end.min(map.len());

        if range.start >= end {
            return Ok(0);
        }

        let start = range.start / page_size() * page_size();
        imp::resident(&map[start..end])
    }

    /// The share of the file's memory pages that are in the cache, from 0 to 1, or 0 for an empty file.
    pub fn residency(&self) -> Result<f64, Error> {
        match self.len().div_ceil(page_size()) {
            0 => Ok(0.0),
            pages => Ok(self.resident(0..self.len())? as f64 / pages as f64),
        }
    }
}

/// Drops every file of `files` from the cache, then watches the first MiB of each for up to `wait`, since something
/// may read a file's first pages back soon after it was mapped or written. Whenever pages come back it drops the files
/// again, and once none came back for `wait` it checks every page of them. It gives `TimedOut` if they are still
/// cached after 20 times `wait`.
pub fn make_cold(files: &[&Mapped], wait: Duration) -> Result<(), Error> {
    let start = Instant::now();

    loop {
        for file in files {
            file.evict()?;
        }

        let round = Instant::now();

        while round.elapsed() < wait && !cached(files, 0..HEAD)? {
            std::thread::sleep(Duration::from_millis(10));
        }

        if !cached(files, 0..usize::MAX)? {
            return Ok(());
        }

        if start.elapsed() >= wait * PATIENCE {
            return Err(Error::new(
                ErrorKind::TimedOut,
                "the files' pages kept coming back into the cache",
            ));
        }
    }
}

/// Whether any page of the bytes `range` of any file of `files` is cached.
fn cached(files: &[&Mapped], range: Range<usize>) -> Result<bool, Error> {
    for file in files {
        if file.resident(range.clone())? > 0 {
            return Ok(true);
        }
    }

    Ok(false)
}

/// The size of a memory page, which is what mincore reports on, eg. 16 KiB on Apple silicon.
pub fn page_size() -> usize {
    imp::page_size().max(1)
}

#[cfg(target_os = "macos")]
mod imp {
    use std::io::Error;

    pub(super) fn invalidate(map: &[u8]) -> Result<(), Error> {
        // SAFETY: the range is the whole live mapping, which starts on a page.
        let rc = unsafe { libc::msync(map.as_ptr() as *mut libc::c_void, map.len(), libc::MS_INVALIDATE) };

        match rc {
            0 => Ok(()),
            _ => Err(Error::last_os_error()),
        }
    }

    /// The resident pages of `part`, a part of a live mapping that starts on a page.
    pub(super) fn resident(part: &[u8]) -> Result<usize, Error> {
        let mut pages = vec![0u8; part.len().div_ceil(super::page_size())];
        // SAFETY: the range lies in a live mapping and starts on a page, and `pages` has one byte per page of it, as
        // mincore writes.
        let rc = unsafe {
            libc::mincore(
                part.as_ptr() as *mut libc::c_void,
                part.len(),
                pages.as_mut_ptr().cast(),
            )
        };

        match rc {
            0 => Ok(pages.iter().filter(|&&p| p & 1 == 1).count()),
            _ => Err(Error::last_os_error()),
        }
    }

    pub(super) fn page_size() -> usize {
        // SAFETY: sysconf takes no pointers.
        unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.max(0) as usize
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    use std::io::{Error, ErrorKind};

    fn unsupported() -> Error {
        Error::new(
            ErrorKind::Unsupported,
            "the cold-cache tools are written for macOS only",
        )
    }

    pub(super) fn invalidate(_map: &[u8]) -> Result<(), Error> {
        Err(unsupported())
    }

    pub(super) fn resident(_part: &[u8]) -> Result<usize, Error> {
        Err(unsupported())
    }

    pub(super) fn page_size() -> usize {
        4096
    }
}

/// The process's counters at one moment, from macOS's `proc_pid_rusage`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    /// The bytes that the process read from disks, and the pages it faulted in from files.
    pub disk_bytes: u64,
    pub pageins: u64,
    /// The process's physical footprint, the memory macOS charges it, in bytes, now and at its highest so far.
    pub footprint: u64,
    pub peak_footprint: u64,
}

impl Usage {
    /// The counters now. On other systems they are all zero.
    pub fn now() -> Usage {
        #[cfg(target_os = "macos")]
        {
            // rusage_info_v6: 16 bytes of UUID (universally unique identifier), then 64-bit counters. The indexes
            // below follow <sys/resource.h>.
            let mut buf = [0u64; 2 + 56];
            // SAFETY: flavor 6 writes one rusage_info_v6, which is 2 + 56 u64s long, into `buf`.
            let rc = unsafe { proc_pid_rusage(std::process::id() as i32, 6, buf.as_mut_ptr()) };

            if rc != 0 {
                return Usage::default();
            }

            let at = |i: usize| buf[2 + i];
            Usage {
                disk_bytes: at(16),
                pageins: at(4),
                footprint: at(7),
                peak_footprint: at(28),
            }
        }

        #[cfg(not(target_os = "macos"))]
        Usage::default()
    }
}

// SAFETY: this matches <libproc.h>, with rusage_info_t taken as a pointer to the u64 counters it fills.
#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut u64) -> i32;
}

#[cfg(test)]
#[path = "../tests/unit/coldcache.rs"]
mod tests;
