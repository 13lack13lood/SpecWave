//! Picks SpecWave's fire layer from measured inputs, as the schedule module describes, and prints a Markdown table.
//!
//! ```text
//! schedule --coverage FILE[,FILE...] --layers FILE --cold FILE [--dataset fiqa] [--fit openings] [--clustering l2]
//!          [--model arctic] [--definition rerank_coverage_normal] [--out FILE]
//! ```
//!
//! The coverage files hold `spec_coverage` results, judged with the verdict module's default rule. The layers file
//! holds `layers` results, and the cold file ColdIVF's cold trials. Every setup of `--model` in the layers file, every
//! source of the cold trials, and every nprobe and R of the coverage grid gets a row. `--model` takes a short name of
//! specwave's models list or a Hugging Face repository id. `--definition` names the rerank coverage that the rule
//! judges, `rerank_coverage` or `rerank_coverage_normal`. `--out` appends a result line for each row.

use std::io::{Error, ErrorKind};
use std::path::Path;

use specwave_bench::results::{append, provenance};
use specwave_bench::schedule::{Settings, json, markdown, rows};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if let Err(e) = Settings::parse(&args).and_then(|s| run(&s)) {
        eprintln!("schedule: {e}");
        std::process::exit(if e.kind() == ErrorKind::InvalidInput { 2 } else { 1 });
    }
}

fn run(s: &Settings) -> Result<(), Error> {
    let rows = rows(s)?;
    print!("{}", markdown(&rows));

    match &s.out {
        Some(out) => {
            let lines: Vec<String> = rows.iter().map(|row| provenance(json(row)).line()).collect();
            append(Path::new(out), &lines)
        }
        None => Ok(()),
    }
}
