//! SpecWave's measurements. The library holds what the binaries share, and the work of `spec_coverage`, `verdict`,
//! `schedule`, `wavecal`, `spec` and `synthetic`, so the unit tests can reach it.
//!
//! | Module | Role |
//! |---|---|
//! | [`coldcache`] | Dropping files from the system's cache, checking what stays, and the process's counters |
//! | [`data`] | The vector files and their names |
//! | [`coverage`] | How much of a search a guessed embedding already covers, with ColdIVF's own codes |
//! | [`results`] | The result files, one JavaScript Object Notation (JSON) object per line, and the build behind them |
//! | [`spec_coverage`] | The run of the `spec_coverage` binary, from its command line to its result lines |
//! | [`verdict`] | The accuracy rule over the coverage results, with both rerank coverages and their intervals |
//! | [`schedule`] | The fire layer that SpecWave's scheduler picks from measured layer times, reads and coverage |
//! | [`synthetic`] | Guesses made from final embeddings and noise, for an index whose queries have no model states |
//! | [`trials`] | The end-to-end cold trials of SpecWave, from a query's text to its hits |
//! | [`wavecal`] | The cold read waves on one disk volume that SpecWave's wave model is fitted to |

pub mod coldcache;
pub mod coverage;
pub mod data;
pub mod results;
pub mod schedule;
pub mod spec_coverage;
pub mod synthetic;
pub mod trials;
pub mod verdict;
pub mod wavecal;

#[cfg(test)]
mod fingerprint;

#[cfg(test)]
#[path = "../tests/unit/testutil.rs"]
mod testutil;
