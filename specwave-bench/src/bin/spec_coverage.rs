//! Measures how good SpecWave's guesses are, with ColdIVF's real code. It builds the in-memory index over the
//! financial question answering (FiQA) and Quora documents. It reads the true and guessed embeddings for each
//! layer. For each combination of layer, beta, nprobe, rerank and rerank_mult, it reports:
//! - how much of what the true embedding needs was already read with the guess
//! - how many true nearest neighbors it finds without a second read, compared with a normal search
//! - how many of the rows to rescore were prefetched

fn main() {
    todo!("InMemoryIndex → grid over (layer, β, nprobe, R, m) → JSON lines")
}
