//! Judges `spec_coverage` results with the accuracy rule, once with each rerank coverage, and writes a Markdown report.
//!
//! ```text
//! verdict FILE... [--min-gap -0.005] [--coverage 0.9] [--max-mult 4] [--max-beta 0.5|DATASET=BETA,...] [--out FILE]
//! ```
//!
//! A cell measured with several rotation seeds counts once, with the mean over the seeds and an interval from the
//! lowest low end to the highest high end. When the files measure a cell twice with the same seed, the later line
//! counts, and later files come after earlier ones. Every line of a cell must have the same R and the same number of
//! queries, so a run on part of the queries needs a report of its own. `--max-beta` takes one beta for every dataset,
//! or pairs of a dataset and its beta, eg. `quora=1,fiqa=0.5`, with a plain number for the rest. The verdict module
//! says how cells are judged. The report goes to `FILE`, or else to the standard output.

use std::io::{Error, ErrorKind};

use specwave_bench::verdict::{Settings, read, report};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if let Err(e) = Settings::parse(&args).and_then(|s| run(&s)) {
        eprintln!("verdict: {e}");
        std::process::exit(1);
    }
}

fn run(s: &Settings) -> Result<(), Error> {
    let mut lines = Vec::new();

    for file in &s.files {
        let text = std::fs::read_to_string(file).map_err(|e| Error::new(e.kind(), format!("{file}: {e}")))?;
        lines.extend(text.lines().map(str::to_string));
    }

    let cells = read(&lines)?;

    if cells.is_empty() {
        return Err(Error::new(ErrorKind::InvalidData, "the files hold no grid lines"));
    }

    let text = report(&cells, &s.rule);

    match &s.out {
        Some(path) => std::fs::write(path, text),
        None => {
            print!("{text}");
            Ok(())
        }
    }
}
