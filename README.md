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

### More Embedding Models

We timed FiQA on 7 more models, each with its own FiQA index and guesses. The runs match arctic's above: 9 clusters, 29 candidates, the CPU with 5 threads and 100 queries in each start state. The table repeats arctic's FiQA results for comparison.

Before timing a model, we test it on each device, i.e. the CPU with 5 and with 6 threads, and Metal. The test looks at the fire layer, the latest layer whose remaining model time covers the guess's reads with 10% to spare. A model passes on a device when these hold at its fire layer:
- The guess's reads cover at least 90% of the rows that a normal search rescores.
- A search over the guess's reads alone loses at most 0.005 of recall@10 against a normal search.
- The guess reads at most 1.5 times the clusters of a normal search and 4 times its rows.
- At least 2.5 ms of model time remains after the fire layer.

| Model | Layers | Passes On | Fire Layer | After the Embedding p50, Off to On (ms) | Full Query p50, Off to On (ms) | Median Pair Saves (ms) | Recall@10, Off to On |
|---|---|---|---|---|---|---|---|
| [arctic](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5) | 12 | CPU | 10 | 1.246 to 0.135 | 22.35 to 21.81 | 0.79 | 0.7510 to 0.7626 |
| [bge-small](https://huggingface.co/BAAI/bge-small-en-v1.5) | 12 | None | 10 | 1.272 to 0.224 | 9.76 to 10.33 | 0.67 | 0.7758 to 0.7939 |
| [bge-base](https://huggingface.co/BAAI/bge-base-en-v1.5) | 12 | CPU | 10 | 1.378 to 0.213 | 23.19 to 23.09 | 0.41 | 0.7969 to 0.8265 |
| [e5-base](https://huggingface.co/intfloat/e5-base-v2) | 12 | CPU | 10 | 1.530 to 0.232 | 22.62 to 21.60 | 0.18 | 0.7811 to 0.8071 |
| [nomic](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) | 12 | CPU, 5 threads | 9 | 1.430 to 0.217 | 23.69 to 21.97 | 1.22 | 0.8015 to 0.8026 |
| [bge-large](https://huggingface.co/BAAI/bge-large-en-v1.5) | 24 | CPU and Metal | 22 | 2.027 to 0.211 | 67.20 to 65.48 | 1.05 | 0.8410 to 0.8513 |
| [mxbai-large](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) | 24 | CPU | 22 | 2.261 to 0.202 | 68.69 to 66.81 | 1.09 | 0.8719 to 0.8844 |
| [qwen3-0.6b](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 28 | CPU and Metal | 27 | 2.015 to 0.165 | 98.88 to 97.96 | 1.52 | 0.8717 to 0.8687 |

Every new model but bge-small passes on the CPU. For those 6 models, speculation cuts the time after the embedding by 84.5% to 91.8% at the median, against 89.2% for arctic. The median pair saves 0.18 to 1.52 ms of the full query, under 2% for all but nomic, since the embedding takes 21 to 97 ms. bge-small fails since layer 10, its first passing layer, leaves 0.84 to 1.02 ms for reads that need 1.55 to 2.04 ms. Of its 198 runs with speculation, 6 also waited 5.0 to 23.6 ms after the embedding, while none without speculation waited over 2.5 ms. On Metal, only bge-large and qwen3-0.6b pass. Recall@10 rises with speculation for every model but qwen3-0.6b, which loses 0.0030.

<details>
<summary>Test at the fire layer, by device</summary>

Each cell gives the fire layer, then the time its reads need with 10% to spare, of the model time left after that layer, in ms. None means that no passing layer leaves enough time, and the cell gives the first passing layer's numbers instead. The read times add up medians from 100 cold searches per start state, run with ColdIVF's cold harness. The CPU's come from a prewarmed core, since the model keeps the CPU's cores busy, and Metal's from idle cores. Passing layers count only the layers with a guess, eg. qwen3-0.6b has none at layer 22.

| Model | Passing Layers | CPU, 5 Threads | CPU, 6 Threads | Metal | Passes On |
|---|---|---|---|---|---|
| arctic | 8 to 10 | 10: 1.26 of 2.78 | 10: 1.26 of 2.90 | 9: 1.69 of 2.15 | CPU |
| bge-small | 10 and 11 | None, 10: 1.55 of 0.95 | None, 10: 1.55 of 1.02 | None, 10: 2.04 of 0.84 | None |
| bge-base | 10 and 11 | 10: 1.95 of 2.63 | 10: 1.95 of 2.84 | None, 10: 2.21 of 1.43 | CPU |
| e5-base | 10 and 11 | 10: 2.30 of 2.95 | 10: 2.30 of 2.70 | None, 10: 2.82 of 1.32 | CPU |
| nomic | 9 to 11 | 10: 1.97 of 2.96 | 11: 1.73 of 1.81 | None, 9: 2.50 of 2.00 | CPU, 5 threads |
| bge-large | 21 to 23 | 23: 2.04 of 2.50 | 23: 2.04 of 2.63 | 21: 2.77 of 2.81 | CPU and Metal |
| mxbai-large | 21 to 23 | 22: 2.35 of 4.38 | 23: 2.35 of 2.65 | None, 21: 3.01 of 2.83 | CPU |
| qwen3-0.6b | 21 and 23 to 27 | 27: 1.77 of 3.93 | 27: 1.77 of 3.27 | 25: 2.59 of 3.05 | CPU and Metal |

Arctic on Metal and nomic on 6 threads fail only the 2.5 ms rule. The timed runs pick their fire layer from their own read model, which expects slower reads, thus nomic and bge-large fire 1 layer earlier there. Of the 6,400 cold searches, 3 started with a cached page and are left out.

</details>
<br>

<details>
<summary>Full query by start state</summary>

| Model | Start | Pairs | p50 On (ms) | p50 Off (ms) | p99 On (ms) | p99 Off (ms) | Faster With Speculation | Median Pair Saves (ms) |
|---|---|---|---|---|---|---|---|---|
| bge-small | Idle | 100 | 20.16 | 22.56 | 30.91 | 43.45 | 63 of 100 | 2.10 |
| bge-small | Prewarmed | 98 | 7.52 | 7.74 | 33.46 | 17.17 | 57 of 98 | 0.41 |
| bge-small | Both | 198 | 10.33 | 9.76 | 33.46 | 43.45 | 120 of 198 | 0.67 |
| bge-base | Idle | 99 | 36.45 | 36.51 | 45.17 | 51.86 | 52 of 99 | 0.36 |
| bge-base | Prewarmed | 97 | 18.55 | 19.34 | 28.43 | 26.84 | 61 of 97 | 0.52 |
| bge-base | Both | 196 | 23.09 | 23.19 | 43.99 | 50.54 | 113 of 196 | 0.41 |
| e5-base | Idle | 98 | 37.02 | 34.34 | 51.29 | 50.30 | 37 of 98 | −2.26 |
| e5-base | Prewarmed | 98 | 18.50 | 19.05 | 24.53 | 26.93 | 64 of 98 | 0.72 |
| e5-base | Both | 196 | 21.60 | 22.62 | 45.67 | 44.00 | 101 of 196 | 0.18 |
| nomic | Idle | 97 | 34.81 | 39.65 | 48.89 | 47.95 | 58 of 97 | 1.55 |
| nomic | Prewarmed | 97 | 20.01 | 20.97 | 26.91 | 47.03 | 65 of 97 | 1.19 |
| nomic | Both | 194 | 21.97 | 23.69 | 48.48 | 47.03 | 123 of 194 | 1.22 |
| bge-large | Idle | 98 | 81.17 | 79.98 | 128.32 | 104.68 | 47 of 98 | −0.78 |
| bge-large | Prewarmed | 97 | 60.40 | 62.07 | 79.28 | 86.54 | 66 of 97 | 1.69 |
| bge-large | Both | 195 | 65.48 | 67.20 | 108.57 | 101.08 | 113 of 195 | 1.05 |
| mxbai-large | Idle | 100 | 80.19 | 82.35 | 99.72 | 96.71 | 54 of 100 | 0.57 |
| mxbai-large | Prewarmed | 99 | 60.82 | 62.61 | 82.53 | 79.20 | 65 of 99 | 1.38 |
| mxbai-large | Both | 199 | 66.81 | 68.69 | 99.72 | 96.71 | 119 of 199 | 1.09 |
| qwen3-0.6b | Idle | 98 | 105.64 | 109.40 | 182.44 | 148.38 | 56 of 98 | 2.02 |
| qwen3-0.6b | Prewarmed | 100 | 82.91 | 93.10 | 126.79 | 128.97 | 59 of 100 | 1.37 |
| qwen3-0.6b | Both | 198 | 97.96 | 98.88 | 160.43 | 138.23 | 115 of 198 | 1.52 |

From idle cores at least 1 pair in 5 differs by over 11 ms, against the ~1 to 2 ms that speculation saves. Thus the prewarmed rows show the gain most clearly.

</details>
<br>

No job was flagged. Each one started with the CPU at least 85% idle, and every trial finished. The 24 trials that started with an index page in the cache are left out with their runs in the other mode.
