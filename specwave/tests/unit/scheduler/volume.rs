//! Unit tests for telling disk volumes apart. The test that compares with `diskutil` is ignored, and run by hand.

use std::io::ErrorKind;

use super::*;
use crate::testutil::Scratch;

#[cfg(target_os = "macos")]
#[test]
fn paths_on_one_volume_share_its_uuid_drive_chip_and_build() {
    let dir = Scratch::new("volume");
    std::fs::write(dir.path("file"), b"x").unwrap();
    let folder = Volume::of(&dir.path("")).unwrap();
    let file = Volume::of(&dir.path("file")).unwrap();
    let groups: Vec<usize> = folder.uuid.split('-').map(str::len).collect();

    assert_eq!(folder, file);
    assert_eq!(groups, [8, 4, 4, 4, 12], "{}", folder.uuid);
    assert!(
        folder
            .uuid
            .bytes()
            .all(|b| b == b'-' || b.is_ascii_digit() || b.is_ascii_uppercase())
    );
    assert!(!folder.chip.is_empty() && !folder.os_build.is_empty(), "{folder:?}");
    assert!(!folder.drive.is_empty(), "{folder:?}");
}

#[cfg(target_os = "macos")]
#[test]
fn a_missing_path_has_no_volume() {
    let dir = Scratch::new("volume-missing");
    let found = Volume::of(&dir.path("missing")).unwrap_err();
    assert_eq!(found.kind(), ErrorKind::NotFound);

    let zero = std::path::Path::new("a\0b");
    assert_eq!(Volume::of(zero).unwrap_err().kind(), ErrorKind::InvalidInput);
}

#[cfg(not(target_os = "macos"))]
#[test]
fn volumes_are_only_told_apart_on_macos() {
    let dir = Scratch::new("volume");
    assert_eq!(Volume::of(&dir.path("")).unwrap_err().kind(), ErrorKind::Unsupported);
}

/// What `program` prints for `args`, trimmed.
#[cfg(target_os = "macos")]
fn output(program: &str, args: &[&str], input: &[u8]) -> String {
    let mut child = std::process::Command::new(program)
        .args(args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    std::io::Write::write_all(&mut child.stdin.take().unwrap(), input).unwrap();
    let out = child.wait_with_output().unwrap();
    String::from_utf8(out.stdout).unwrap().trim().to_string()
}

/// The value of `key` in what `diskutil info` says about the disk `disk`, eg. "disk3s1".
#[cfg(target_os = "macos")]
fn diskutil(disk: &str, key: &str) -> String {
    let info = output("diskutil", &["info", "-plist", disk], b"");
    output("plutil", &["-extract", key, "raw", "-o", "-", "-"], info.as_bytes())
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "runs diskutil, which can take a second"]
fn the_volume_matches_what_diskutil_says() {
    let dir = Scratch::new("volume-diskutil");
    let here = Volume::of(&dir.path("")).unwrap();
    let device = output("stat", &["-f", "%Sd", dir.path("").to_str().unwrap()], b"");
    let store = diskutil(&device, "APFSPhysicalStores.0.APFSPhysicalStore");
    let disk = diskutil(&store, "ParentWholeDisk");

    assert_eq!(here.uuid, diskutil(&device, "VolumeUUID"));
    assert_eq!(
        here.drive,
        diskutil(&disk, "MediaName"),
        "the drive of {device}, on {disk}"
    );
}
