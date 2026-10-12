# SpecWave

A search layer in Rust that hides ColdIVF's disk reads under the embedding model: partway through encoding a query, it predicts the final embedding from an intermediate layer and starts the index reads with that guess. When the query text arrives, its worker thread calls ColdIVF's `prepare` and routes once, so the cluster tables load while the model runs. At the chosen layer, it uses `route` and `fetch` to read a few more clusters than a normal search would, to cover the guess's error. Once the model finishes, it ranks those clusters with the real embedding (`Fetched::search`) and reads only the rows the guess missed.

## Results

These times come from one Apple M3 Pro and run from the query text's arrival to its hits, i.e. the embedding plus the search. Each trial is a fresh process. It loads the arctic model (snowflake-arctic-embed-m-v1.5) on 5 threads, drops the index from the cache and times one query. A pair is one query in one start state, run once with speculation on and once off. The start states are idle cores and a core kept busy for 150 ms first (prewarmed), and the Both rows pool them. FiQA (financial opinion mining and question answering) uses the model's own guesses, probing 9 clusters and rescoring 29 candidates. Cohere 1M probes 96 clusters and rescores 35. Its guesses are synthetic, with noise fitted to match the model's own guesses on FiQA, and the model runs on a FiQA text. Each dataset has 100 queries.

| Dataset | Start | Pairs | p50 On (ms) | p50 Off (ms) | p99 On (ms) | p99 Off (ms) | Faster With Speculation |
|---|---|---|---|---|---|---|---|
| FiQA | Idle | 99 | 35.58 | 35.52 | 44.16 | 62.16 | 56 of 99 |
| FiQA | Prewarmed | 99 | 18.35 | 19.10 | 26.47 | 28.07 | 73 of 99 |
| FiQA | Both | 198 | 21.81 | 22.35 | 43.45 | 47.23 | 129 of 198 |
| Cohere 1M | Idle | 100 | 23.29 | 24.66 | 28.79 | 44.22 | 65 of 100 |
| Cohere 1M | Prewarmed | 94 | 19.32 | 21.19 | 70.02 | 46.64 | 77 of 94 |
| Cohere 1M | Both | 194 | 21.28 | 22.40 | 46.76 | 46.64 | 142 of 194 |

With speculation on, the full query is 0.55 ms faster at the median on FiQA (2.4%) and 1.12 ms faster on Cohere 1M (5.0%), and the median pair saves 0.79 and 1.76 ms. At the median, speculation cuts the time after the embedding from 1.246 to 0.135 ms on FiQA and from 3.265 to 0.443 ms on Cohere 1M, but the embedding itself takes 0.56 and 1.43 ms longer. From idle cores on FiQA the two modes tie at the median. Recall@10 is 0.763 with speculation and 0.751 without on FiQA, and 0.940 in both modes on Cohere 1M.

No job was flagged. Each one started with the CPU at least 85% idle, and every trial finished. The 8 trials that started with an index page in the cache, 2 on FiQA and 6 on Cohere 1M, are left out with their runs in the other mode. With fewer than 100 pairs the p99 is the slowest pair. FiQA's p99s come from slow embeddings, except its idle p99 with speculation off, one query whose reads took 25 ms. Cohere 1M's come from reads that stalled for 19 to 52 ms, except its idle p99 with speculation on, a slow embedding. Its prewarmed p99 with speculation on is one query whose speculative reads took 52 ms there and 27 ms from idle cores.
