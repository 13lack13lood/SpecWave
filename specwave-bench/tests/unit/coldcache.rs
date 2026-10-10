//! Unit tests for the cold-cache tools, on a scratch file. They need macOS.

use super::*;
#[cfg(target_os = "macos")]
use crate::testutil::{cold, scratch};

#[cfg(target_os = "macos")]
#[test]
fn a_read_leaves_pages_cached_and_eviction_drops_them_through_the_held_mapping() {
    let _cold = cold();
    let dir = scratch("coldcache-evict");
    let path = dir.join("file.bin");
    let page = page_size();
    std::fs::write(&path, vec![7u8; 8 * page + 100]).unwrap();
    let map = Mapped::open(&path).unwrap();
    assert_eq!(map.len(), 8 * page + 100);

    let _ = std::fs::read(&path).unwrap();
    assert_eq!(map.resident(0..map.len()).unwrap(), 9);
    assert_eq!(map.resident(page + 1..3 * page).unwrap(), 2);
    assert_eq!(map.resident(9 * page..20 * page).unwrap(), 0);
    assert!((map.residency().unwrap() - 1.0).abs() < 1e-12);

    map.evict().unwrap();
    assert_eq!(map.resident(0..usize::MAX).unwrap(), 0);
    let _ = std::fs::read(&path).unwrap();
    make_cold(&[&map], Duration::from_millis(20)).unwrap();
    assert_eq!(map.residency().unwrap(), 0.0);

    // An empty file has nothing to map, drop or count.
    let empty = dir.join("empty.bin");
    std::fs::write(&empty, b"").unwrap();
    let none = Mapped::open(&empty).unwrap();
    assert!(none.is_empty());
    none.evict().unwrap();
    assert_eq!((none.resident(0..10).unwrap(), none.residency().unwrap()), (0, 0.0));
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(target_os = "macos")]
#[test]
fn the_counters_see_the_footprint_grow_and_the_disk_reads_add_up() {
    let _cold = cold();
    // Other tests run beside this one, so each check leaves room for what they allocate, free and read.
    let before = Usage::now();
    assert!(before.footprint > 0 && before.peak_footprint > 0, "{before:?}");
    let held = vec![1u8; 64 << 20];
    let after = Usage::now();
    assert!(after.footprint >= before.footprint + (32 << 20), "{before:?} {after:?}");
    drop(held);

    let dir = scratch("coldcache-usage");
    let path = dir.join("file.bin");
    std::fs::write(&path, vec![3u8; 4 << 20]).unwrap();
    let map = Mapped::open(&path).unwrap();
    map.evict().unwrap();
    let before = Usage::now();
    let _ = std::fs::read(&path).unwrap();
    let after = Usage::now();
    assert!(
        after.disk_bytes >= before.disk_bytes + (2 << 20),
        "{before:?} {after:?}"
    );
    assert!(after.pageins >= before.pageins);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Reads `path` about every millisecond on a thread of its own until `until` has passed, or until it is told to stop.
#[cfg(target_os = "macos")]
fn keep_reading(
    path: std::path::PathBuf,
    until: Duration,
) -> (std::sync::mpsc::Sender<()>, std::thread::JoinHandle<()>) {
    let (stop, stopped) = std::sync::mpsc::channel();
    let reader = std::thread::spawn(move || {
        let start = Instant::now();
        while start.elapsed() < until && stopped.try_recv().is_err() {
            let _ = std::fs::read(&path).unwrap();
            std::thread::sleep(Duration::from_millis(1));
        }
    });
    (stop, reader)
}

#[cfg(target_os = "macos")]
#[test]
fn making_files_cold_waits_out_a_reader_that_stops_and_gives_up_on_one_that_doesnt() {
    let _cold = cold();
    let dir = scratch("coldcache-patience");
    let path = dir.join("file.bin");
    std::fs::write(&path, vec![5u8; 4 * page_size()]).unwrap();
    let map = Mapped::open(&path).unwrap();

    // Pages that keep coming back for 0.5 s, five waits of 100 ms, are waited out.
    let (_stop, reader) = keep_reading(path.clone(), Duration::from_millis(500));
    make_cold(&[&map], Duration::from_millis(100)).unwrap();
    reader.join().unwrap();
    assert_eq!(map.residency().unwrap(), 0.0);

    // Pages that never stop coming back time out after 20 waits.
    let (stop, reader) = keep_reading(path.clone(), Duration::from_secs(60));
    let start = Instant::now();
    let error = make_cold(&[&map], Duration::from_millis(50)).unwrap_err();
    assert!(start.elapsed() < Duration::from_secs(30));
    stop.send(()).unwrap();
    reader.join().unwrap();
    assert_eq!(error.kind(), ErrorKind::TimedOut);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_missing_file_gives_its_error_with_the_path() {
    let error = Mapped::open(std::path::Path::new("/nonexistent/specwave/file.bin"))
        .err()
        .unwrap();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(error.to_string().contains("/nonexistent/specwave/file.bin"));
}
