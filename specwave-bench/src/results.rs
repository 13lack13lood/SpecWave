//! The result files, in JavaScript Object Notation (JSON) lines form, i.e. one JSON object per line. Lines are appended
//! and flushed in groups, so a sleep or a kill loses at most the group in flight.

use std::fs::OpenOptions;
use std::io::{Error, Write};
use std::path::Path;
use std::process::Command;

/// One JSON object, built a field at a time and written as one line.
#[derive(Default)]
pub struct Json(Vec<String>);

impl Json {
    pub fn new() -> Json {
        Json::default()
    }

    pub fn str(mut self, key: &str, value: &str) -> Json {
        self.0.push(format!("{}:{}", quote(key), quote(value)));
        self
    }

    /// A number, or null when it isn't finite, which JSON can't hold.
    pub fn num(mut self, key: &str, value: f64) -> Json {
        self.0.push(format!("{}:{}", quote(key), number(value)));
        self
    }

    pub fn int(mut self, key: &str, value: u64) -> Json {
        self.0.push(format!("{}:{value}", quote(key)));
        self
    }

    /// A whole number, or null for `None`.
    pub fn opt(mut self, key: &str, value: Option<u64>) -> Json {
        let value = value.map_or_else(|| "null".to_string(), |v| v.to_string());
        self.0.push(format!("{}:{value}", quote(key)));
        self
    }

    pub fn bool(mut self, key: &str, value: bool) -> Json {
        self.0.push(format!("{}:{value}", quote(key)));
        self
    }

    /// An array of numbers, each null when it isn't finite.
    pub fn nums(mut self, key: &str, values: &[f64]) -> Json {
        let values: Vec<String> = values.iter().map(|&v| number(v)).collect();
        self.0.push(format!("{}:[{}]", quote(key), values.join(",")));
        self
    }

    pub fn ints(mut self, key: &str, values: &[u64]) -> Json {
        let values: Vec<String> = values.iter().map(u64::to_string).collect();
        self.0.push(format!("{}:[{}]", quote(key), values.join(",")));
        self
    }

    pub fn line(&self) -> String {
        format!("{{{}}}", self.0.join(","))
    }
}

fn number(value: f64) -> String {
    match value.is_finite() {
        true => format!("{value}"),
        false => "null".to_string(),
    }
}

fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');

    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }

    out.push('"');
    out
}

/// Appends `lines` to the results file `path` in one write, then flushes it.
pub fn append(path: &Path, lines: &[String]) -> Result<(), Error> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }

    let mut text = String::new();
    lines.iter().for_each(|line| {
        text.push_str(line);
        text.push('\n');
    });
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    file.write_all(text.as_bytes())?;
    file.flush()
}

/// The code this binary was built from, eg. "specwave 1d2a8be0c3f1+0f3a5c7e9b1d2f40 coldivf d90edb1c2d3e". Each
/// repository's HEAD commit comes first, then "+" and a hash of the changes since that commit in the files the build
/// reads, when there are some. A repository that git couldn't read when the binary was built is "unknown".
pub const BUILD: &str = env!("SPECWAVE_BENCH_BUILD");

/// Whether `build` names the code of both repositories, i.e. neither is "unknown". Only then does it change whenever
/// the code does.
pub fn names_its_code(build: &str) -> bool {
    !build.split(' ').any(|word| word == "unknown")
}

/// What the build recorded about one repository, its HEAD commit and the hash of its changes, which is "" for none.
struct Repo {
    commit: &'static str,
    changes: &'static str,
}

const SPECWAVE: Repo = Repo {
    commit: env!("SPECWAVE_BENCH_SPECWAVE_COMMIT"),
    changes: env!("SPECWAVE_BENCH_SPECWAVE_CHANGES"),
};

const COLDIVF: Repo = Repo {
    commit: env!("SPECWAVE_BENCH_COLDIVF_COMMIT"),
    changes: env!("SPECWAVE_BENCH_COLDIVF_CHANGES"),
};

/// Adds what every run records, i.e. the code it was built from and the machine. The code is `BUILD`, and also each
/// repository's commit and the hash of its changes.
pub fn provenance(json: Json) -> Json {
    code(json, BUILD, &SPECWAVE, &COLDIVF).str("machine", &machine())
}

/// Adds `build`, then for this repository and for ColdIVF's the commit, whether there are changes, and their hash.
fn code(json: Json, build: &str, specwave: &Repo, coldivf: &Repo) -> Json {
    json.str("build", build)
        .str("commit", specwave.commit)
        .bool("dirty", !specwave.changes.is_empty())
        .str("changes", specwave.changes)
        .str("coldivf_commit", coldivf.commit)
        .bool("coldivf_dirty", !coldivf.changes.is_empty())
        .str("coldivf_changes", coldivf.changes)
}

/// The machine's model and chip, eg. "Mac15,7 Apple M3 Pro".
pub fn machine() -> String {
    let sysctl = |name: &str| {
        let out = Command::new("sysctl").args(["-n", name]).output();
        out.map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default()
    };
    format!("{} {}", sysctl("hw.model"), sysctl("machdep.cpu.brand_string"))
        .trim()
        .to_string()
}

#[cfg(test)]
#[path = "../tests/unit/results.rs"]
mod tests;
