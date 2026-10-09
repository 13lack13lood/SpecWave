//! SpecWave's measurements. The library holds what the binaries share, and the work of `spec_coverage` and `verdict`,
//! so the unit tests can reach it.
//!
//! | Module | Role |
//! |---|---|
//! | [`data`] | The vector files and their names |
//! | [`coverage`] | How much of a search a guessed embedding already covers, with ColdIVF's own codes |
//! | [`results`] | The result files, one JavaScript Object Notation (JSON) object per line, and the build behind them |
//! | [`spec_coverage`] | The run of the `spec_coverage` binary, from its command line to its result lines |
//! | [`verdict`] | The accuracy rule over the coverage results, with both rerank coverages and their intervals |

pub mod coverage;
pub mod data;
pub mod results;
pub mod spec_coverage;
pub mod verdict;

#[cfg(test)]
mod fingerprint;

#[cfg(test)]
#[path = "../tests/unit/testutil.rs"]
mod testutil;
