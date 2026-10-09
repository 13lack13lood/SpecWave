//! Unit tests for the build fingerprint, on a git repository made in the system's temporary folder.

use std::fs;

use super::*;

/// A fresh git repository with one commit of `src/a.rs`, `docs/a.md` and a `.gitignore` that ignores `*.log`, deleted
/// on drop. Its state covers `src` and `.gitignore`.
struct Repo(PathBuf);

impl Repo {
    fn new(name: &str) -> Repo {
        let dir = std::env::temp_dir().join(format!("specwave-bench-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("src")).unwrap();
        let repo = Repo(dir);
        repo.write("src/a.rs", "fn a() {}\n");
        repo.write("docs/a.md", "outside the paths\n");
        repo.write(".gitignore", "*.log\n");
        repo.git(&["init", "-q"]);
        repo.commit("first");
        repo
    }

    /// Commits every file, with a user and no signing of its own.
    fn commit(&self, message: &str) {
        let user = [
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "commit.gpgsign=false",
        ];
        self.git(&["add", "."]);
        self.git(&[&user[..], &["commit", "-q", "-m", message]].concat());
    }

    fn write(&self, path: &str, text: &str) {
        let path = self.0.join(path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }

    fn git(&self, args: &[&str]) -> String {
        let out = Command::new("git").arg("-C").arg(&self.0).args(args).output().unwrap();
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    fn state(&self) -> State {
        state(&self.0, &["src", ".gitignore"]).unwrap()
    }
}

impl Drop for Repo {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn a_clean_repository_is_its_commit() {
    let repo = Repo::new("clean");
    let head = repo.git(&["rev-parse", "HEAD"]);
    let state = repo.state();

    assert_eq!(state.commit, head[..12]);
    assert_eq!(state.changes, None);
    assert_eq!(state.label(), head[..12]);
    assert!(state.commit.bytes().all(|b| b.is_ascii_hexdigit()));
}

#[test]
fn changes_to_tracked_files_change_the_hash_whether_staged_or_not() {
    let repo = Repo::new("tracked");
    repo.write("src/a.rs", "fn a() { 1; }\n");
    let first = repo.state().changes.expect("a change");
    assert_eq!(first.len(), 16);
    assert_eq!(repo.state().label(), format!("{}+{first}", repo.state().commit));

    repo.git(&["add", "src/a.rs"]);
    assert_eq!(repo.state().changes.as_ref(), Some(&first), "staging changes nothing");

    repo.write("src/a.rs", "fn a() { 2; }\n");
    let second = repo.state().changes.expect("a change");
    assert_ne!(second, first);

    repo.write("src/a.rs", "fn a() {}\n");
    repo.git(&["add", "src/a.rs"]);
    assert_eq!(repo.state().changes, None, "the committed text again");

    repo.write("docs/a.md", "changed\n");
    assert_eq!(repo.state().changes, None, "a change outside the paths");
}

#[test]
fn every_byte_of_a_changed_binary_file_counts() {
    let repo = Repo::new("binary");
    let table = repo.0.join("src/table.bin");
    fs::write(&table, [0u8, 1, 2, 3, 0, 5]).unwrap();
    repo.commit("a binary file");
    fs::write(&table, [0u8, 1, 2, 3, 0, 6]).unwrap();
    let first = repo.state().changes.expect("a change");

    // Git shows no text lines for a binary file, but any changed byte still counts.
    fs::write(&table, [0u8, 1, 2, 3, 0, 7]).unwrap();
    assert_ne!(repo.state().changes.expect("a change"), first);
}

#[test]
fn untracked_files_count_unless_ignored_cached_or_elsewhere() {
    let repo = Repo::new("untracked");
    repo.write("src/b.rs", "fn b() {}\n");
    let first = repo.state().changes.expect("an untracked file");

    repo.write("src/b.rs", "fn c() {}\n");
    let second = repo.state().changes.expect("an untracked file");
    assert_ne!(second, first, "the untracked file's text counts, not only its length");

    fs::rename(repo.0.join("src/b.rs"), repo.0.join("src/e.rs")).unwrap();
    let renamed = repo.state().changes.expect("an untracked file");
    assert_ne!(renamed, second, "the untracked file's name counts too");

    fs::remove_file(repo.0.join("src/e.rs")).unwrap();
    repo.write("src/run.log", "ignored\n");
    repo.write("src/__pycache__/c.cpython-312.pyc", "cache\n");
    repo.write("src/d.pyc", "cache\n");
    repo.write("docs/notes.md", "outside the paths\n");
    assert_eq!(repo.state().changes, None);

    repo.write("src/c/d.rs", "fn d() {}\n");
    assert!(repo.state().changes.is_some(), "an untracked file in a new folder");
}

#[test]
fn a_folder_that_isnt_a_repository_has_no_state() {
    let dir = std::env::temp_dir().join(format!("specwave-bench-{}-not-a-repo", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    assert_eq!(state(&dir.join("missing"), &["src"]), None);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn cargo_watches_gits_files_and_the_paths_that_exist() {
    let repo = Repo::new("watched");
    let dir = PathBuf::from(repo.git(&["rev-parse", "--absolute-git-dir"]));
    let found = watched(&repo.0, &["src", "missing", ".gitignore"]);

    assert_eq!(
        found,
        [
            dir.join("HEAD"),
            dir.join("index"),
            dir.join("refs"),
            repo.0.join("src"),
            repo.0.join(".gitignore")
        ]
    );
}
