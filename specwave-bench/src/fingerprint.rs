//! A fingerprint of the code in one git repository, which `build.rs` records so that every result line can name the
//! code that made it. It is the HEAD commit, plus a hash of the changes since that commit in the files a build reads,
//! i.e. of `git diff HEAD` and of the untracked files that git doesn't ignore.

use std::path::{Path, PathBuf};
use std::process::Command;

/// What git says about the files of one repository.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct State {
    /// The HEAD commit, as 12 hex digits.
    pub commit: String,
    /// A hash of the changes since that commit, as 16 hex digits, or none when there are no changes.
    pub changes: Option<String>,
}

impl State {
    /// The commit, then "+" and the hash of the changes when there are some, eg. "1d2a8be0c3f1+0f3a5c7e9b1d2f40".
    pub fn label(&self) -> String {
        match &self.changes {
            Some(changes) => format!("{}+{changes}", self.commit),
            None => self.commit.clone(),
        }
    }
}

/// The state of the files at `paths`, relative to the root of the git repository `repo`, or none when git can't read
/// the repository. Python's bytecode caches don't count as changes.
pub fn state(repo: &Path, paths: &[&str]) -> Option<State> {
    let commit = git(repo, &["rev-parse", "--short=12", "HEAD"], &[])?;
    let diff = git(
        repo,
        &["diff", "HEAD", "--binary", "--no-color", "--no-ext-diff"],
        paths,
    )?;
    let untracked = git(repo, &["ls-files", "--others", "--exclude-standard", "-z"], paths)?;
    let mut hash = Fnv::new();
    hash.write(&diff);
    let mut changed = !diff.is_empty();

    for name in untracked.split(|&b| b == 0).filter(|name| !name.is_empty()) {
        let name = String::from_utf8_lossy(name);
        if name.split('/').any(|part| part == "__pycache__") || name.ends_with(".pyc") {
            continue;
        }
        changed = true;
        hash.write(name.as_bytes());
        hash.write(&[0]);
        match std::fs::read(repo.join(&*name)) {
            Ok(bytes) => {
                hash.write(&(bytes.len() as u64).to_le_bytes());
                hash.write(&bytes);
            }
            Err(_) => hash.write(b"unreadable"),
        }
    }

    Some(State {
        commit: String::from_utf8_lossy(&commit).trim().to_string(),
        changes: changed.then(|| format!("{:016x}", hash.0)),
    })
}

/// The files and folders whose changes can change the state, for cargo to watch. They are git's HEAD, index and refs,
/// and the ones of `paths` that exist. Cargo would run the build script again on every build for a path that doesn't
/// exist.
pub fn watched(repo: &Path, paths: &[&str]) -> Vec<PathBuf> {
    let dir = git(repo, &["rev-parse", "--absolute-git-dir"], &[])
        .map(|out| PathBuf::from(String::from_utf8_lossy(&out).trim()));
    let git_files = dir
        .iter()
        .flat_map(|dir| ["HEAD", "index", "refs", "packed-refs"].map(|name| dir.join(name)));
    git_files
        .chain(paths.iter().map(|path| repo.join(path)))
        .filter(|path| path.exists())
        .collect()
}

/// The output of git with `args`, then `--` and `paths` when there are some, run in `repo`. Optional locks are off,
/// so git never rewrites its index here, which would make cargo run the build script again.
fn git(repo: &Path, args: &[&str], paths: &[&str]) -> Option<Vec<u8>> {
    let mut command = Command::new("git");
    command.arg("-C").arg(repo).args(args).env("GIT_OPTIONAL_LOCKS", "0");

    if !paths.is_empty() {
        command.arg("--").args(paths);
    }

    let out = command.output().ok()?;
    out.status.success().then_some(out.stdout)
}

/// The 64-bit FNV-1a (Fowler-Noll-Vo) hash, which is the same on every platform and in every Rust version.
struct Fnv(u64);

impl Fnv {
    fn new() -> Fnv {
        Fnv(0xcbf2_9ce4_8422_2325)
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/fingerprint.rs"]
mod tests;
