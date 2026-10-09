//! The accuracy rule over the coverage results. A cell of the grid passes when the guess's recall without a second
//! read is within `min_gap` of the normal search's, i.e. the gap is at least `min_gap`, and its rerank coverage is at
//! least `min_coverage`. Only cells with an effective beta of at most `max_beta` and a prefetch multiple of at most
//! `max_mult` count. Each cell is judged once with each rerank coverage, as [`DEFINITIONS`] lists them.
//! - `rerank_coverage` counts the rows that the true embedding rescores over the guess's clusters.
//! - `rerank_coverage_normal` counts the rows that a normal search rescores over its own clusters.
//!
//! A test passes when its whole 95% interval reaches the threshold, fails when none of it does, and is gray otherwise.
//! With several rotation seeds, a cell's value is the mean over the seeds and its interval runs from the lowest low
//! end to the highest high end. A cell passes when both its tests pass, fails when either fails, and is gray otherwise.
//! A guess, clustering, nprobe and R pass at the cell with the smallest beta, then the smallest multiple, that passes.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write;
use std::io::{Error, ErrorKind};

/// The two rerank coverages, by their field names in the result lines.
pub const DEFINITIONS: [&str; 2] = ["rerank_coverage", "rerank_coverage_normal"];

/// The slack of every comparison with a threshold, for the rounding of values that sit on it, eg. a gap of -0.005.
const SLACK: f64 = 1e-9;

/// A value with its 95% interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    pub mean: f64,
    pub lo: f64,
    pub hi: f64,
}

/// One cell of the grid, over every seed that measured it.
#[derive(Clone, Debug, PartialEq)]
pub struct Cell {
    pub dataset: String,
    pub clustering: String,
    pub guess: String,
    pub layer: Option<usize>,
    pub nprobe: usize,
    pub beta: f64,
    pub beta_effective: f64,
    pub r_label: String,
    /// The rescoring depth R itself, eg. 30 for the label "r384".
    pub rerank: usize,
    pub mult: usize,
    /// The number of queries it was measured on.
    pub queries: usize,
    pub seeds: Vec<u64>,
    /// The builds of the program that measured it, eg. "specwave 1d2a8be0c3f1 coldivf d90edb1c2d3e".
    pub builds: Vec<String>,
    /// The recall gap, the guess's recall minus the normal search's.
    pub gap: Interval,
    /// The rerank coverages, in the order of `DEFINITIONS`.
    pub coverage: [Interval; 2],
}

/// The fields that tell the cells apart, but for the seed.
type Key = (String, String, String, usize, u64, String, usize);

/// One seed's measure of one cell.
struct Measured {
    cell: Cell,
    seed: u64,
}

/// The cells of the result lines `lines`, the grid lines of `spec_coverage`, over every seed. Other lines are skipped.
/// When two lines measure the same cell with the same seed, the later one counts. A grid line without a field the rule
/// needs, eg. one written before the intervals, gives `InvalidData`. Every line of a dataset must have the same number
/// of queries, and every line of a dataset, clustering, nprobe and R label the same R, or that gives `InvalidData` too.
/// So a run on part of the queries, or one that found another R384, needs a report of its own.
pub fn read(lines: &[String]) -> Result<Vec<Cell>, Error> {
    let mut by_key: BTreeMap<Key, BTreeMap<u64, Cell>> = BTreeMap::new();
    // The query count of each dataset and the R of each R label, with the line that gave it first.
    let mut queries: HashMap<String, (usize, usize)> = HashMap::new();
    let mut reranks: HashMap<(String, String, usize, String), (usize, usize)> = HashMap::new();

    for (i, line) in lines.iter().enumerate() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if v["record"] != "grid" {
            continue;
        }
        let m = measured(&v, i + 1)?;
        let c = &m.cell;
        let (count, first) = *queries.entry(c.dataset.clone()).or_insert((c.queries, i + 1));
        if count != c.queries {
            return Err(damaged(&format!(
                "grid line {} measures {} over {} queries, but line {first} over {count}",
                i + 1,
                c.dataset,
                c.queries
            )));
        }
        let depth = (c.dataset.clone(), c.clustering.clone(), c.nprobe, c.r_label.clone());
        let (rerank, first) = *reranks.entry(depth).or_insert((c.rerank, i + 1));
        if rerank != c.rerank {
            return Err(damaged(&format!(
                "grid line {} measures {} {} nprobe {} at R {} for the label {}, but line {first} at R {rerank}",
                i + 1,
                c.dataset,
                c.clustering,
                c.nprobe,
                c.rerank,
                c.r_label
            )));
        }
        let key = (
            c.dataset.clone(),
            c.clustering.clone(),
            c.guess.clone(),
            c.nprobe,
            c.beta.to_bits(),
            c.r_label.clone(),
            c.mult,
        );
        by_key.entry(key).or_default().insert(m.seed, m.cell);
    }

    Ok(by_key.into_values().map(|seeds| combine(&seeds)).collect())
}

/// The cell of the grid line `v`, which is line `line` of the input. A missing field gives `InvalidData`.
fn measured(v: &serde_json::Value, line: usize) -> Result<Measured, Error> {
    let missing = |key: &str| damaged(&format!("grid line {line} has no {key}"));
    let text = |key: &str| v[key].as_str().map(str::to_string).ok_or_else(|| missing(key));
    let number = |key: &str| v[key].as_f64().ok_or_else(|| missing(key));
    let whole = |key: &str| v[key].as_u64().ok_or_else(|| missing(key));
    let interval = |key: &str| -> Result<Interval, Error> {
        Ok(Interval {
            mean: number(key)?,
            lo: number(&format!("{key}_lo"))?,
            hi: number(&format!("{key}_hi"))?,
        })
    };
    let cell = Cell {
        dataset: text("dataset")?,
        clustering: text("clustering")?,
        guess: text("guess")?,
        layer: v["layer"].as_u64().map(|l| l as usize),
        nprobe: whole("nprobe")? as usize,
        beta: number("beta")?,
        beta_effective: number("beta_effective")?,
        r_label: text("r_label")?,
        rerank: whole("rerank")? as usize,
        mult: whole("mult")? as usize,
        queries: whole("queries")? as usize,
        seeds: Vec::new(),
        builds: vec![v["build"].as_str().unwrap_or("unknown").to_string()],
        gap: interval("recall_gap")?,
        coverage: [interval(DEFINITIONS[0])?, interval(DEFINITIONS[1])?],
    };
    Ok(Measured {
        cell,
        seed: whole("seed")?,
    })
}

/// One cell from its measures by seed, with the mean over the seeds and an interval from the lowest low end to the
/// highest high end.
fn combine(seeds: &BTreeMap<u64, Cell>) -> Cell {
    let all: Vec<&Cell> = seeds.values().collect();
    let over = |pick: &dyn Fn(&Cell) -> Interval| {
        let picked: Vec<Interval> = all.iter().map(|c| pick(c)).collect();
        Interval {
            mean: picked.iter().map(|i| i.mean).sum::<f64>() / picked.len() as f64,
            lo: picked.iter().map(|i| i.lo).fold(f64::INFINITY, f64::min),
            hi: picked.iter().map(|i| i.hi).fold(f64::NEG_INFINITY, f64::max),
        }
    };
    let mut builds: Vec<String> = all.iter().flat_map(|c| c.builds.iter().cloned()).collect();
    builds.sort_unstable();
    builds.dedup();
    Cell {
        seeds: seeds.keys().copied().collect(),
        builds,
        gap: over(&|c| c.gap),
        coverage: [over(&|c| c.coverage[0]), over(&|c| c.coverage[1])],
        ..all[0].clone()
    }
}

/// A test's or a cell's outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Verdict {
    Pass,
    Gray,
    Fail,
}

impl Verdict {
    fn word(self) -> &'static str {
        match self {
            Verdict::Pass => "PASS",
            Verdict::Gray => "GRAY",
            Verdict::Fail => "FAIL",
        }
    }
}

/// Whether all of the interval `i` reaches `threshold`, none of it does, or only part of it does.
pub fn test(i: &Interval, threshold: f64) -> Verdict {
    if i.lo >= threshold - SLACK {
        Verdict::Pass
    } else if i.hi < threshold - SLACK {
        Verdict::Fail
    } else {
        Verdict::Gray
    }
}

/// The rule's thresholds and limits.
#[derive(Clone, Debug, PartialEq)]
pub struct Rule {
    pub min_gap: f64,
    pub min_coverage: f64,
    pub max_mult: usize,
    /// The most effective beta of each dataset, and of any other.
    pub max_beta: HashMap<String, f64>,
    pub max_beta_otherwise: f64,
}

impl Default for Rule {
    fn default() -> Rule {
        Rule {
            min_gap: -0.005,
            min_coverage: 0.9,
            max_mult: 4,
            max_beta: HashMap::new(),
            max_beta_otherwise: 0.5,
        }
    }
}

impl Rule {
    fn max_beta(&self, dataset: &str) -> f64 {
        self.max_beta.get(dataset).copied().unwrap_or(self.max_beta_otherwise)
    }

    /// Whether the cell reads no more extra clusters than the rule allows.
    fn allows_beta(&self, c: &Cell) -> bool {
        c.beta_effective <= self.max_beta(&c.dataset) + SLACK
    }

    /// Whether the cell is within the rule's limits.
    fn allows(&self, c: &Cell) -> bool {
        self.allows_beta(c) && c.mult <= self.max_mult
    }

    /// The recall test of `c`.
    pub fn recall(&self, c: &Cell) -> Verdict {
        test(&c.gap, self.min_gap)
    }

    /// The outcome of `c` with the rerank coverage `definition`, a position in `DEFINITIONS`.
    pub fn judge(&self, c: &Cell, definition: usize) -> Verdict {
        self.recall(c).max(test(&c.coverage[definition], self.min_coverage))
    }

    /// The outcome of the cells `group`, which share a dataset, guess, clustering, nprobe and R, with the cell that
    /// decides it. That is the first cell, by beta and then multiple, that passes, or else the first that is gray.
    /// When every cell fails, it is the cell with the best gap when none passes the recall test, and else the cell
    /// with the largest multiple at the first beta whose recall doesn't fail.
    pub fn decide<'a>(&self, group: &[&'a Cell], definition: usize) -> (Verdict, Option<&'a Cell>) {
        let mut allowed: Vec<&Cell> = group.iter().copied().filter(|c| self.allows(c)).collect();
        allowed.sort_by(|a, b| a.beta.total_cmp(&b.beta).then(a.mult.cmp(&b.mult)));

        for verdict in [Verdict::Pass, Verdict::Gray] {
            if let Some(c) = allowed.iter().find(|c| self.judge(c, definition) == verdict) {
                return (verdict, Some(c));
            }
        }

        let recalled = allowed.iter().find(|c| self.recall(c) != Verdict::Fail).map(|c| c.beta);
        let deciding = match recalled {
            Some(beta) => allowed.iter().filter(|c| c.beta == beta).max_by_key(|c| c.mult),
            None => allowed.iter().max_by(|a, b| a.gap.mean.total_cmp(&b.gap.mean)),
        };
        (Verdict::Fail, deciding.copied())
    }

    /// The smallest multiple at which the rerank coverage `definition` of `group` reaches the threshold at `beta`, by
    /// its mean and by its whole interval, over every multiple measured.
    pub fn smallest_mult(&self, group: &[&Cell], beta: f64, definition: usize) -> (Option<usize>, Option<usize>) {
        let mut at: Vec<&Cell> = group.iter().copied().filter(|c| c.beta == beta).collect();
        at.sort_by_key(|c| c.mult);
        let by_mean = at
            .iter()
            .find(|c| c.coverage[definition].mean >= self.min_coverage - SLACK);
        let by_interval = at
            .iter()
            .find(|c| test(&c.coverage[definition], self.min_coverage) == Verdict::Pass);
        (by_mean.map(|c| c.mult), by_interval.map(|c| c.mult))
    }

    /// The smallest allowed beta of `group` whose recall test passes, or else the smallest whose recall is gray. Recall
    /// doesn't depend on the multiple, so every multiple counts.
    pub fn recall_beta(&self, group: &[&Cell]) -> Option<f64> {
        let mut allowed: Vec<&Cell> = group.iter().copied().filter(|c| self.allows_beta(c)).collect();
        allowed.sort_by(|a, b| a.beta.total_cmp(&b.beta));
        [Verdict::Pass, Verdict::Gray]
            .iter()
            .find_map(|&v| allowed.iter().find(|c| self.recall(c) == v).map(|c| c.beta))
    }
}

/// The part of a guess's label without its layer, eg. "openings" for "openings.L9".
pub fn fit_of(guess: &str) -> String {
    let parts: Vec<&str> = guess
        .split('.')
        .filter(|part| {
            !(part.len() > 1 && part.starts_with(['L', 'l']) && part[1..].bytes().all(|b| b.is_ascii_digit()))
        })
        .collect();
    parts.join(".")
}

/// The groups of `cells`, i.e. the cells of each dataset, R, guess, clustering and nprobe.
type Groups<'a> = BTreeMap<(String, String, (String, Option<usize>, String), String, usize), Vec<&'a Cell>>;

fn groups(cells: &[Cell]) -> Groups<'_> {
    let mut groups: Groups = BTreeMap::new();

    for c in cells {
        let guess = (fit_of(&c.guess), c.layer, c.guess.clone());
        let key = (
            c.dataset.clone(),
            c.r_label.clone(),
            guess,
            c.clustering.clone(),
            c.nprobe,
        );
        groups.entry(key).or_default().push(c);
    }

    groups
}

/// A Markdown report of `cells` under `rule`, i.e. for each dataset and R a table of every guess's outcome by
/// clustering and nprobe under each rerank coverage, then the first passing layer of each fit, then the smallest
/// multiple that reaches the coverage threshold at each guess's first beta that passes the recall test.
pub fn report(cells: &[Cell], rule: &Rule) -> String {
    let groups = groups(cells);
    let mut out = String::new();
    let mut seeds: Vec<u64> = cells.iter().flat_map(|c| c.seeds.iter().copied()).collect();
    let mut builds: Vec<&str> = cells.iter().flat_map(|c| c.builds.iter().map(String::as_str)).collect();
    seeds.sort_unstable();
    seeds.dedup();
    builds.sort_unstable();
    builds.dedup();
    let betas: Vec<String> = rule.max_beta.iter().map(|(d, b)| format!("{b} for {d}")).collect();
    let _ = writeln!(
        out,
        "Rule: gap at least {} and rerank coverage at least {}, at an effective beta of at most {}{} and a multiple of \
         at most {}. A test is GRAY when its 95% interval holds the threshold. Seeds: {}. Builds: {}.\n",
        rule.min_gap,
        rule.min_coverage,
        rule.max_beta_otherwise,
        match betas.is_empty() {
            true => String::new(),
            false => format!(" ({})", betas.join(", ")),
        },
        rule.max_mult,
        seeds.iter().map(u64::to_string).collect::<Vec<_>>().join(", "),
        builds.join("; ")
    );
    let mut tables: BTreeMap<(String, String), Vec<(String, usize)>> = BTreeMap::new();

    for (dataset, r, _, clustering, nprobe) in groups.keys() {
        let columns = tables.entry((dataset.clone(), r.clone())).or_default();
        if !columns.contains(&(clustering.clone(), *nprobe)) {
            columns.push((clustering.clone(), *nprobe));
        }
    }

    for ((dataset, r), columns) in &tables {
        for (d, definition) in DEFINITIONS.iter().enumerate() {
            let head: Vec<String> = columns.iter().map(|(k, n)| format!("{k} {n}")).collect();
            let _ = writeln!(out, "## {dataset}, R {r}, {definition}\n");
            let _ = writeln!(out, "| Guess | {} |", head.join(" | "));
            let _ = writeln!(out, "|---|{}", "---|".repeat(columns.len()));
            let guesses: Vec<&(String, Option<usize>, String)> = groups
                .keys()
                .filter(|k| (&k.0, &k.1) == (dataset, r))
                .map(|k| &k.2)
                .fold(Vec::new(), |mut v, g| {
                    if !v.contains(&g) {
                        v.push(g);
                    }
                    v
                });
            for guess in guesses {
                let entries: Vec<String> = columns
                    .iter()
                    .map(|(k, n)| {
                        let key = (dataset.clone(), r.clone(), guess.clone(), k.clone(), *n);
                        groups.get(&key).map_or(String::new(), |g| entry(rule, g, d))
                    })
                    .collect();
                let _ = writeln!(out, "| {} | {} |", guess.2, entries.join(" | "));
            }
            out.push('\n');
        }
    }

    first_layers(&mut out, &groups, rule);
    smallest_mults(&mut out, &groups, rule);
    out
}

/// One table entry, eg. "PASS β0.25 m3" or "FAIL gap -0.0124 [-0.0158, -0.0091] β0.5 m2". A pass also names the
/// first gray cell before it, if any, and an outcome other than a pass shows every test that doesn't pass.
fn entry(rule: &Rule, group: &[&Cell], definition: usize) -> String {
    let (verdict, cell) = rule.decide(group, definition);
    let Some(c) = cell else {
        return format!("{} no allowed cell", verdict.word());
    };
    let beta = match (c.beta - c.beta_effective).abs() > SLACK {
        true => format!("β{} ({:.3} read)", c.beta, c.beta_effective),
        false => format!("β{}", c.beta),
    };
    let shown = |name: &str, i: &Interval, digits: usize| {
        format!("{name} {:.digits$} [{:.digits$}, {:.digits$}]", i.mean, i.lo, i.hi)
    };
    // A gray cell before the passing one might pass too, with fewer clusters or rows.
    let earlier = |p: &Cell| {
        let mut before: Vec<&Cell> = group
            .iter()
            .copied()
            .filter(|g| rule.allows(g) && (g.beta, g.mult) < (p.beta, p.mult))
            .collect();
        before.sort_by(|a, b| a.beta.total_cmp(&b.beta).then(a.mult.cmp(&b.mult)));
        before
            .into_iter()
            .find(|g| rule.judge(g, definition) == Verdict::Gray)
            .map_or(String::new(), |g| format!(" (GRAY at β{} m{})", g.beta, g.mult))
    };

    if verdict == Verdict::Pass {
        return format!("PASS {beta} m{}{}", c.mult, earlier(c));
    }

    // Every test that doesn't pass.
    let mut tests = Vec::new();

    if rule.recall(c) != Verdict::Pass {
        tests.push(shown("gap", &c.gap, 4));
    }

    if test(&c.coverage[definition], rule.min_coverage) != Verdict::Pass {
        tests.push(shown("rr", &c.coverage[definition], 3));
    }

    format!("{} {} {beta} m{}", verdict.word(), tests.join(", "), c.mult)
}

/// The layers of each dataset, fit, clustering, R and nprobe, with the outcome of each under both rerank coverages.
type Fits = BTreeMap<(String, String, String, String, usize), Vec<(usize, [Verdict; 2])>>;

/// The table of the first layer of each fit that passes, and the first that passes or is gray, under each rerank
/// coverage.
fn first_layers(out: &mut String, groups: &Groups, rule: &Rule) {
    let mut fits = Fits::new();

    for ((dataset, r, (fit, layer, _), clustering, nprobe), group) in groups {
        let Some(layer) = layer else {
            continue;
        };
        let verdicts = [rule.decide(group, 0).0, rule.decide(group, 1).0];
        let key = (dataset.clone(), fit.clone(), clustering.clone(), r.clone(), *nprobe);
        fits.entry(key).or_default().push((*layer, verdicts));
    }

    let _ = writeln!(
        out,
        "## First Passing Layer\n\nThe first layer that passes, then in brackets the first that passes or is gray.\n"
    );
    let _ = writeln!(
        out,
        "| Dataset | Fit | Clustering | R | nprobe | {} | {} |",
        DEFINITIONS[0], DEFINITIONS[1]
    );
    let _ = writeln!(out, "|---|---|---|---|---|---|---|");

    for ((dataset, fit, clustering, r, nprobe), mut layers) in fits {
        layers.sort_by_key(|&(layer, _)| layer);
        let first = |d: usize, ok: &dyn Fn(Verdict) -> bool| {
            layers
                .iter()
                .find(|(_, v)| ok(v[d]))
                .map_or("none".to_string(), |(l, _)| format!("L{l}"))
        };
        let shown: Vec<String> = (0..2)
            .map(|d| {
                format!(
                    "{} ({})",
                    first(d, &|v| v == Verdict::Pass),
                    first(d, &|v| v != Verdict::Fail)
                )
            })
            .collect();
        let _ = writeln!(
            out,
            "| {dataset} | {fit} | {clustering} | {r} | {nprobe} | {} | {} |",
            shown[0], shown[1]
        );
    }

    out.push('\n');
}

/// The table of the smallest multiple that reaches the coverage threshold, by mean and by interval, at the first
/// allowed beta of each guess whose recall test passes, or else is gray.
fn smallest_mults(out: &mut String, groups: &Groups, rule: &Rule) {
    let _ = writeln!(
        out,
        "## Smallest Multiple Reaching {}\n\nAt the first allowed beta whose recall passes, or else is gray, over every \
         multiple measured. Each entry is the multiple by the mean, then the multiple by the whole interval.\n",
        rule.min_coverage
    );
    let _ = writeln!(
        out,
        "| Dataset | R | Guess | Clustering | nprobe | β | {} | {} |",
        DEFINITIONS[0], DEFINITIONS[1]
    );
    let _ = writeln!(out, "|---|---|---|---|---|---|---|---|");

    for ((dataset, r, (_, _, guess), clustering, nprobe), group) in groups {
        let Some(beta) = rule.recall_beta(group) else {
            let _ = writeln!(
                out,
                "| {dataset} | {r} | {guess} | {clustering} | {nprobe} | none | | |"
            );
            continue;
        };
        let shown: Vec<String> = (0..2)
            .map(|d| {
                let (mean, interval) = rule.smallest_mult(group, beta, d);
                let m = |x: Option<usize>| x.map_or("none".to_string(), |m| format!("m{m}"));
                format!("{} / {}", m(mean), m(interval))
            })
            .collect();
        let _ = writeln!(
            out,
            "| {dataset} | {r} | {guess} | {clustering} | {nprobe} | {beta} | {} | {} |",
            shown[0], shown[1]
        );
    }
}

/// The command line of the `verdict` binary, i.e. the result files, the rule and where the report goes.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub files: Vec<String>,
    pub rule: Rule,
    pub out: Option<String>,
}

impl Settings {
    /// Reads `args`, the arguments after the program's name. A bad argument gives `InvalidInput`.
    pub fn parse(args: &[String]) -> Result<Settings, Error> {
        let mut settings = Settings {
            files: Vec::new(),
            rule: Rule::default(),
            out: None,
        };
        let mut it = args.iter();
        let mut seen = Vec::new();

        while let Some(arg) = it.next() {
            let Some(name) = arg.strip_prefix("--") else {
                settings.files.push(arg.clone());
                continue;
            };
            let value = it.next().ok_or_else(|| invalid(&format!("--{name} needs a value")))?;
            if seen.contains(&name) {
                return Err(invalid(&format!("--{name} is given twice")));
            }
            seen.push(name);
            let number = |v: &str| v.trim().parse::<f64>().ok().filter(|x| x.is_finite());
            let bad = || invalid(&format!("--{name} has a bad value: {value}"));
            match name {
                "min-gap" => settings.rule.min_gap = number(value).ok_or_else(bad)?,
                "coverage" => settings.rule.min_coverage = number(value).ok_or_else(bad)?,
                "max-mult" => settings.rule.max_mult = value.trim().parse().map_err(|_| bad())?,
                "max-beta" => {
                    for part in value.split(',') {
                        match part.split_once('=') {
                            Some((dataset, beta)) => {
                                let beta = number(beta).filter(|b| *b >= 0.0).ok_or_else(bad)?;
                                settings.rule.max_beta.insert(dataset.trim().to_string(), beta);
                            }
                            None => {
                                settings.rule.max_beta_otherwise = number(part).filter(|b| *b >= 0.0).ok_or_else(bad)?
                            }
                        }
                    }
                }
                "out" => settings.out = Some(value.clone()),
                _ => return Err(invalid(&format!("unknown argument {arg}"))),
            }
        }

        match settings.files.is_empty() {
            true => Err(invalid("name at least one results file")),
            false => Ok(settings),
        }
    }
}

fn invalid(message: &str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

fn damaged(message: &str) -> Error {
    Error::new(ErrorKind::InvalidData, message)
}

#[cfg(test)]
#[path = "../tests/unit/verdict.rs"]
mod tests;
