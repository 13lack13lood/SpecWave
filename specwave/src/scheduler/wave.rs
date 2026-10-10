//! The time of a read wave, fitted once per disk volume and cached as JavaScript Object Notation (JSON).

use std::fs::{self, File};
use std::io::{Error, ErrorKind, Write};
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use super::volume::Volume;

/// The version of the cache files' format.
const FORMAT: u64 = 1;

/// A read wave takes t = a + b·bytes milliseconds (ms), where a is the fixed cost of a wave and b its cost per byte.
/// It is fitted once per disk volume, over pages known not to be in memory, and cached. Until then, a = 0.25 ms and
/// b = 1/3.4 gigabytes per second (GB/s), which suit a fast internal drive.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaveModel {
    /// a, in ms.
    pub a_ms: f64,
    /// b, in ms per megabyte (MB), i.e. per 10^6 bytes.
    pub b_ms_per_mb: f64,
}

impl Default for WaveModel {
    fn default() -> WaveModel {
        WaveModel::DEFAULT
    }
}

impl WaveModel {
    /// The model until a fit is cached.
    pub const DEFAULT: WaveModel = WaveModel {
        a_ms: 0.25,
        b_ms_per_mb: 1000.0 / 3400.0,
    };

    /// The time of a wave of `bytes` bytes, in ms, or 0 for no bytes, since then no wave runs.
    pub fn read_ms(&self, bytes: f64) -> f64 {
        match bytes > 0.0 {
            true => self.a_ms + self.b_ms_per_mb * bytes / 1e6,
            false => 0.0,
        }
    }

    /// The least-squares fit of t = a + b·bytes to `samples`, pairs of a wave's bytes and its time in ms, with a and b
    /// kept at 0 or above. Samples of fewer than two sizes, or with values that aren't finite or are below 0, give
    /// `InvalidInput`.
    pub fn fit(samples: &[(f64, f64)]) -> Result<WaveModel, Error> {
        if !samples
            .iter()
            .all(|&(bytes, t)| bytes.is_finite() && t.is_finite() && bytes >= 0.0 && t >= 0.0)
        {
            return Err(invalid("a wave's bytes and time must be finite and at least 0"));
        }

        let points: Vec<(f64, f64)> = samples.iter().map(|&(bytes, t)| (bytes / 1e6, t)).collect();

        if points.iter().all(|&(x, _)| Some(x) == points.first().map(|p| p.0)) {
            return Err(invalid("a fit needs waves of at least two sizes"));
        }

        let n = points.len() as f64;
        let (mean_x, mean_t) = (
            points.iter().map(|p| p.0).sum::<f64>() / n,
            points.iter().map(|p| p.1).sum::<f64>() / n,
        );
        let sxx: f64 = points.iter().map(|&(x, _)| (x - mean_x) * (x - mean_x)).sum();
        let sxt: f64 = points.iter().map(|&(x, t)| (x - mean_x) * (t - mean_t)).sum();
        let b = sxt / sxx;
        let a = mean_t - b * mean_x;

        if a >= 0.0 && b >= 0.0 {
            return Ok(WaveModel {
                a_ms: a,
                b_ms_per_mb: b,
            });
        }

        // The best line with a or b at 0 then runs through the origin or stays flat at the mean time.
        let through = WaveModel {
            a_ms: 0.0,
            b_ms_per_mb: points.iter().map(|&(x, t)| x * t).sum::<f64>()
                / points.iter().map(|&(x, _)| x * x).sum::<f64>(),
        };
        let flat = WaveModel {
            a_ms: mean_t,
            b_ms_per_mb: 0.0,
        };
        let squares = |m: &WaveModel| -> f64 {
            points
                .iter()
                .map(|&(x, t)| (m.a_ms + m.b_ms_per_mb * x - t).powi(2))
                .sum()
        };

        match squares(&through) <= squares(&flat) {
            true => Ok(through),
            false => Ok(flat),
        }
    }

    /// The fit cached for the volume that holds `path`, else the default. The cache is in [`support_dir`].
    pub fn for_volume(path: &Path) -> WaveModel {
        support_dir().map_or(WaveModel::DEFAULT, |dir| WaveModel::for_volume_in(&dir, path))
    }

    /// The fit cached in the folder `dir` for the volume that holds `path`, else the default, eg. when the volume can't
    /// be told apart from others or its cache file is damaged.
    pub fn for_volume_in(dir: &Path, path: &Path) -> WaveModel {
        Volume::of(path)
            .and_then(|volume| WaveModel::cached(dir, &volume))
            .ok()
            .flatten()
            .unwrap_or(WaveModel::DEFAULT)
    }

    /// The fit cached in the folder `dir` for `volume`. It is `None` when there is none, and when the cached one
    /// belongs to another drive, chip or operating system build, eg. from before an update. A volume whose UUID
    /// (universally unique identifier) isn't a plain name gives `InvalidInput`, a file that isn't a whole cache file
    /// `InvalidData`, and one of a newer format `Unsupported`.
    pub fn cached(dir: &Path, volume: &Volume) -> Result<Option<WaveModel>, Error> {
        let path = cache_path(dir, volume)?;
        let at = |e: Error| Error::new(e.kind(), format!("{}: {e}", path.display()));
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(at(e)),
        };
        decode(&bytes, volume).map_err(at)
    }

    /// Caches the fit for `volume` in the folder `dir`, making the folder first, with the samples it came from, pairs
    /// of a wave's bytes and its time in ms. It writes a temporary file next to the cache file and renames it over
    /// that, so a crash never leaves a partial one, and returns the cache file's path. A model or samples with values
    /// that aren't finite or are below 0, or a volume whose UUID isn't a plain name, give `InvalidInput`.
    pub fn save(&self, dir: &Path, volume: &Volume, samples: &[(f64, f64)]) -> Result<PathBuf, Error> {
        let path = cache_path(dir, volume)?;
        let good = |x: f64| x.is_finite() && x >= 0.0;

        if !(good(self.a_ms) && good(self.b_ms_per_mb)) || !samples.iter().all(|&(b, t)| good(b) && good(t)) {
            return Err(invalid("a cached fit and its samples must be finite and at least 0"));
        }

        let samples: Vec<Value> = samples.iter().map(|&(bytes, t)| json!([bytes, t])).collect();
        let text = json!({
            "format": FORMAT,
            "volume": volume.uuid,
            "drive": volume.drive,
            "chip": volume.chip,
            "os_build": volume.os_build,
            "a_ms": self.a_ms,
            "b_ms_per_mb": self.b_ms_per_mb,
            "samples": samples,
        });
        let mut bytes = serde_json::to_vec_pretty(&text).map_err(Error::other)?;
        bytes.push(b'\n');
        fs::create_dir_all(dir)?;
        let mut temporary = path.file_name().unwrap_or_default().to_os_string();
        temporary.push(format!(".tmp-{}", std::process::id()));
        let temporary = path.with_file_name(temporary);
        let written = File::create(&temporary).and_then(|mut file| {
            file.write_all(&bytes)?;
            file.sync_all()
        });
        let result = written.and_then(|()| fs::rename(&temporary, &path));

        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }

        result.map(|()| path)
    }
}

/// The cached fit in `bytes`, the contents of a cache file, if it belongs to `volume`.
fn decode(bytes: &[u8], volume: &Volume) -> Result<Option<WaveModel>, Error> {
    let v: Value = serde_json::from_slice(bytes).map_err(|e| damaged(&format!("isn't a cache file: {e}")))?;
    let format = v["format"].as_u64().ok_or_else(|| damaged("has no format"))?;

    if format != FORMAT {
        return Err(Error::new(
            ErrorKind::Unsupported,
            format!("is in cache format {format}, and this build reads only format {FORMAT}"),
        ));
    }

    let text = |key: &str| {
        v[key]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| damaged(&format!("has no {key}")))
    };
    let number = |key: &str| {
        v[key]
            .as_f64()
            .filter(|x| x.is_finite() && *x >= 0.0)
            .ok_or_else(|| damaged(&format!("has no {key} of 0 or more")))
    };
    let found = Volume {
        uuid: text("volume")?,
        drive: text("drive")?,
        chip: text("chip")?,
        os_build: text("os_build")?,
    };
    let model = WaveModel {
        a_ms: number("a_ms")?,
        b_ms_per_mb: number("b_ms_per_mb")?,
    };
    Ok((found == *volume).then_some(model))
}

/// Where the fit for `volume` is cached in the folder `dir`, i.e. `wave-<UUID>.json`. A UUID that is empty or holds
/// characters other than letters, digits and dashes gives `InvalidInput`.
pub fn cache_path(dir: &Path, volume: &Volume) -> Result<PathBuf, Error> {
    let plain = !volume.uuid.is_empty() && volume.uuid.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-');

    match plain {
        true => Ok(dir.join(format!("wave-{}.json", volume.uuid))),
        false => Err(invalid(&format!(
            "the volume UUID {:?} isn't a plain name",
            volume.uuid
        ))),
    }
}

/// The folder where SpecWave keeps what it measures on this machine. That is `~/Library/Application Support/SpecWave`
/// on macOS, and `$XDG_DATA_HOME/specwave` or else `~/.local/share/specwave` on Linux. It is `None` on other systems,
/// and without a home folder.
pub fn support_dir() -> Option<PathBuf> {
    let home = || std::env::var_os("HOME").map(PathBuf::from).filter(|h| h.is_absolute());

    if cfg!(target_os = "macos") {
        return home().map(|h| h.join("Library/Application Support/SpecWave"));
    }

    if cfg!(target_os = "linux") {
        let data = std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .filter(|d| d.is_absolute());
        return data
            .or_else(|| home().map(|h| h.join(".local/share")))
            .map(|d| d.join("specwave"));
    }

    None
}

fn invalid(message: &str) -> Error {
    Error::new(ErrorKind::InvalidInput, message.to_string())
}

fn damaged(message: &str) -> Error {
    Error::new(ErrorKind::InvalidData, message.to_string())
}

#[cfg(test)]
#[path = "../../tests/unit/scheduler/wave.rs"]
mod tests;
