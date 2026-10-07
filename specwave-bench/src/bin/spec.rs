//! Runs SpecWave end to end with the real model (feature `candle`). `--profile` times each layer. Otherwise it runs
//! cold trials with speculation on and off. Each trial is a fresh process with the model loaded and the index out
//! of memory.

fn main() {
    todo!("load encoder + predictors → schedule → SpecWave::search per trial → print results")
}
