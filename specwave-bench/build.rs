//! Records the code that this build holds, for this repository and for the ColdIVF checkout next to it, so every
//! result line can name it. `SPECWAVE_BENCH_BUILD` holds both, eg. "specwave 1d2a8be0c3f1+0f3a5c7e9b1d2f40 coldivf
//! d90edb1c2d3e", and the `_COMMIT` and `_CHANGES` variables hold the parts.

#[path = "src/fingerprint.rs"]
mod fingerprint;

use std::path::{Path, PathBuf};

/// The files of this repository that a build of this crate reads.
const HERE: [&str; 8] = [
    ".cargo",
    "Cargo.lock",
    "Cargo.toml",
    "specwave",
    "specwave-bench/Cargo.toml",
    "specwave-bench/build.rs",
    "specwave-bench/src",
    "specwave-bench/tests",
];

/// The files of the ColdIVF checkout that it reads, i.e. the workspace manifest and the library.
const COLDIVF: [&str; 2] = ["Cargo.toml", "coldivf"];

fn main() {
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
    let here = manifest.join("..");
    let repos = [
        ("specwave", "SPECWAVE", here.clone(), &HERE[..]),
        ("coldivf", "COLDIVF", here.join("../coldivf"), &COLDIVF[..]),
    ];
    let mut build = Vec::new();

    for (name, var, repo, paths) in repos {
        let repo = repo.canonicalize().unwrap_or(repo);
        let (commit, changes, label) = describe(&repo, paths);
        println!("cargo:rustc-env=SPECWAVE_BENCH_{var}_COMMIT={commit}");
        println!("cargo:rustc-env=SPECWAVE_BENCH_{var}_CHANGES={changes}");
        build.push(format!("{name} {label}"));
        for path in fingerprint::watched(&repo, paths) {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }

    println!("cargo:rustc-env=SPECWAVE_BENCH_BUILD={}", build.join(" "));
}

/// The commit, the hash of the changes or "", and the label of `paths` in `repo`, or "unknown" when git can't read it.
fn describe(repo: &Path, paths: &[&str]) -> (String, String, String) {
    match fingerprint::state(repo, paths) {
        Some(state) => (
            state.commit.clone(),
            state.changes.clone().unwrap_or_default(),
            state.label(),
        ),
        None => ("unknown".to_string(), String::new(), "unknown".to_string()),
    }
}
