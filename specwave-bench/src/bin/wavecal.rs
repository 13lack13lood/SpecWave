//! Fits SpecWave's wave model for one disk volume and caches it, from cold read waves of 16 to 512 pages on a scratch
//! file there, as the wavecal module describes.
//!
//! ```text
//! wavecal [--file data/wavecal.bin] [--size-mib 256] [--sizes 16,32,64,128,256,512] [--trials 30] [--busy-ms 150]
//!         [--seed 1] [--cache DIR|none] [--out results/wavecal.jsonl] [--keep]
//! ```
//!
//! The scratch file must be on the volume to fit, eg. next to the index. When it isn't there, the run writes one of
//! `--size-mib` MiB of random data and deletes it at the end, unless `--keep` is given. The waves run `--trials` rounds
//! of one wave of each size of `--sizes`, in pages of 16 KiB. The fit is cached in `--cache`, by default the system's
//! support folder, eg. `~/Library/Application Support/SpecWave` on macOS, as `wave-<volume UUID>.json`, where UUID
//! stands for universally unique identifier. `--cache none` leaves the cache alone. A line for the run, with each
//! size's times, is appended to `--out`. The times belong to this machine and its load, so run it on an idle machine.

use std::io::{Error, ErrorKind};
use std::path::Path;
use std::time::Duration;

use specwave::scheduler::{Volume, support_dir};
use specwave_bench::results::{Json, append, provenance};
use specwave_bench::wavecal::{PAGE, Settings, fit, make, plan, points, run, summarize};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");

    if let Err(e) = Settings::parse(&args, &repo, support_dir()).and_then(|s| calibrate(&s)) {
        eprintln!("wavecal: {e}");
        std::process::exit(if e.kind() == ErrorKind::InvalidInput { 2 } else { 1 });
    }
}

/// Makes the scratch file if it isn't there, times the waves and fits them, then deletes a file it made.
fn calibrate(s: &Settings) -> Result<(), Error> {
    let made = !s.file.exists();

    if made {
        if let Some(dir) = s.file.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        eprintln!("wavecal: writing {} MiB to {}", s.size_mib, s.file.display());
        make(&s.file, s.size_mib << 20, s.seed)?;
    }

    let result = measure(s);

    if made && !s.keep {
        let _ = std::fs::remove_file(&s.file);
    }

    result
}

fn measure(s: &Settings) -> Result<(), Error> {
    let at = |e: Error| Error::new(e.kind(), format!("{}: {e}", s.file.display()));
    let volume = Volume::of(&s.file).map_err(at)?;
    let bytes = std::fs::metadata(&s.file).map_err(at)?.len();
    let plan = plan(&s.sizes, s.trials, bytes as usize / PAGE, s.seed)?;
    let timed = run(&s.file, &plan, Duration::from_millis(s.busy_ms)).map_err(at)?;
    let dropped = timed.iter().filter(|w| w.is_none()).count();
    let waves: Vec<_> = timed.into_iter().flatten().collect();
    let sizes = summarize(&waves);
    let model = fit(&sizes)?;
    println!(
        "{} on {} ({}), {} {}: {} waves, {dropped} left out for starting with cached pages",
        volume.uuid,
        if volume.drive.is_empty() {
            "an unnamed drive"
        } else {
            &volume.drive
        },
        s.file.display(),
        volume.chip,
        volume.os_build,
        waves.len()
    );
    println!(
        "{:>6} {:>10} {:>6} {:>10} {:>10} {:>10}",
        "pages", "bytes", "waves", "median ms", "p90 ms", "fit ms"
    );

    for size in &sizes {
        println!(
            "{:>6} {:>10} {:>6} {:>10.3} {:>10.3} {:>10.3}",
            size.pages,
            size.bytes,
            size.waves,
            size.median_ms,
            size.p90_ms,
            model.read_ms(size.bytes as f64)
        );
    }

    println!(
        "fit: a = {:.4} ms, b = {:.4} ms per MB ({:.2} GB/s)",
        model.a_ms,
        model.b_ms_per_mb,
        1.0 / model.b_ms_per_mb
    );
    let cached = match &s.cache {
        Some(dir) => {
            let path = model.save(dir, &volume, &points(&sizes))?;
            println!("cached at {}", path.display());
            path.display().to_string()
        }
        None => String::new(),
    };
    let ints = |f: fn(&specwave_bench::wavecal::Size) -> u64| sizes.iter().map(f).collect::<Vec<u64>>();
    let nums = |f: fn(&specwave_bench::wavecal::Size) -> f64| sizes.iter().map(f).collect::<Vec<f64>>();
    let line = Json::new()
        .str("bench", "wavecal")
        .str("volume", &volume.uuid)
        .str("drive", &volume.drive)
        .str("chip", &volume.chip)
        .str("os_build", &volume.os_build)
        .str("file", &s.file.display().to_string())
        .int("file_bytes", bytes)
        .int("page", PAGE as u64)
        .int("trials", s.trials as u64)
        .int("busy_ms", s.busy_ms)
        .int("seed", s.seed)
        .int("dropped", dropped as u64)
        .ints("pages", &ints(|z| z.pages as u64))
        .ints("waves", &ints(|z| z.waves as u64))
        .nums("median_ms", &nums(|z| z.median_ms))
        .nums("p90_ms", &nums(|z| z.p90_ms))
        .num("a_ms", model.a_ms)
        .num("b_ms_per_mb", model.b_ms_per_mb)
        .str("cache", &cached);
    append(&s.out, &[provenance(line).line()])
}
