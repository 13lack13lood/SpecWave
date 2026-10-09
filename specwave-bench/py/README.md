# specwave-bench/py

Python scripts that make the data of the SpecWave benchmarks. They write only to `data/` and `results/`, which git
ignores. Each one skips the outputs that already exist, so a killed run picks up where it stopped, and deleting an
output makes the next run remake it. An output made from the evaluation queries also records their id, and a run
remakes it when the evaluation queries change (see below). They need numpy, pyarrow, torch, sentence-transformers and
faiss, with the models and the BeIR (Benchmarking Information Retrieval) datasets in the Hugging Face cache. They never
go online.

| Script | What It Does |
|---|---|
| `export_beir.py` | Writes the FiQA (financial opinion mining and question answering) and Quora texts as JSON (JavaScript Object Notation) lines in parquet order, the cached embeddings as `.fbin`, the evaluation queries and their exact 100 nearest documents |
| `spec_states.py` | Writes each layer's pooled states, the ridge predictions of the final embedding from them, and the k-means clusters |
| `common.py` | Holds the file formats, the exact nearest-neighbor search, the ridge regression and the model runs that both share |
| `test_common.py` | Tests `common.py`, and how the two scripts keep or remake their outputs, on small synthetic data |

## Running

```sh
python3 export_beir.py --cache DIR   # DIR holds the cached .npy embeddings and the BeIR relevance judgments
python3 spec_states.py fiqa          # or name steps, eg. spec_states.py fiqa timing
python3 spec_states.py quora
python3 test_common.py
```

Each step of `spec_states.py` runs in its own process, named or not, since faiss and PyTorch load clashing copies of
the OpenMP (Open Multi-Processing) runtime.

## Files

Vectors are `.fbin` files and integers `.ibin` files, both an int32 row count and an int32 width, then the rows as
little-endian float32 or int32 values. Row i of a `.jsonl` file is row i of the `.fbin` file with the same stem. FiQA
uses snowflake-arctic-embed-m-v1.5 (768 dimensions, the CLS (classification) token, a query prompt) and Quora all-MiniLM-L6-v2 (384
dimensions, the token mean, no prompt). Every embedding is normalized.

In `data/`, for `<ds>` fiqa or quora:

| File | Contents |
|---|---|
| `<ds>_corpus.jsonl`, `<ds>_corpus.fbin` | Every document as {row, id, text}, and its embedding |
| `<ds>_queries.jsonl`, `<ds>_queries.fbin` | Every query, and its embedding with the query prompt |
| `<ds>_queries.splits.json` | The query rows of each BeIR split with relevance judgments, eg. test |
| `<ds>_test.jsonl`, `<ds>_test.fbin` | The evaluation queries, 1,548 for FiQA and 1,000 for Quora |
| `<ds>_test.meta.json` | Their rows in `<ds>_queries`, the seed and the subsets, eg. FiQA's 648 BeIR test queries |
| `<ds>_test.gt.ibin` | The 100 corpus rows nearest each evaluation query by cosine, nearest first, ties to the lower row |
| `<ds>_test.gt.json` | How the ground truth was made and checked |
| `<ds>_export.json` | Counts, timings and the checks that each `.fbin` row embeds its `.jsonl` row |

FiQA's evaluation queries are rows `default_rng(11).permutation(6648)[:1000]` of `fiqa_queries`, then the BeIR test
queries not among them, in parquet order. Quora's are `default_rng(11).choice` of 1,000 of its 10,000 BeIR test
queries.

In `data/spec/<ds>/`, where layer l is the hidden state after l transformer layers and l = 0 is the embedding output:

| File | Contents |
|---|---|
| `e.fbin` | A byte copy of `<ds>_test.fbin`, the evaluation queries' final embeddings, written last by the ridge step |
| `ehat/<fit>.L<l>.fbin` | The normalized predicted final embeddings from layer l, in the rows of `e.fbin`, with a `.json` of the penalty, the split seed and the cosines |
| `ehat/minilm.<fit>.fbin` | FiQA only, arctic's final embedding predicted from MiniLM's last layer |
| `openings.jsonl` | The 20,000 document openings, i.e. the first 6 to 24 words, drawn with seed 12, as {row, doc, id, words, text} |
| `queries.jsonl` | The real training queries, all other queries for FiQA and the BeIR dev split for Quora, in parquet order |
| `openings.meta.json`, `queries.meta.json` | How each training set was drawn |
| `states/<set>.L<l>.fbin` | The pooled states (n × 2d, the masked mean then the CLS vector) of set openings, queries or test |
| `states/<set>.final.fbin` | The same texts' final embeddings, the ridge targets |
| `clusters/<kind>.centroids.fbin` | N/160 cluster centers for N documents, from L2 (Euclidean, `l2`) or spherical (`sph`) k-means with seed 1 |
| `clusters/<kind>.assign.ibin` | Each document's cluster, one per row |
| `clusters/<kind>.parents.fbin`, `<kind>.parent_assign.ibin` | Groups of clusters (the square root of their count) and each cluster's group |
| `clusters/<kind>.ceiling.json` | The recall of an exact scan of each query's nearest clusters, picked by L2 distance (`l2`) or by the raw inner product with the centers as they are (`ip-raw`) |
| `timing.json` | The time to each layer at batch 1 on the processor with 4 threads |
| `summary.json` | All of the numbers above in one file |

Fit is `openings`, which needs no query log, or `queries`. The layers are 0, 3, 6, 8, 9 and 10 of arctic's 12, and 0,
1, 3, 4 and 5 of MiniLM's 6. The penalty is the one that, fitted on the first 85% of the training pairs, best predicts
the last 15%. The openings are in random draw order already. The real queries are in parquet order, where the BeIR
splits sit in blocks, so their pairs are shuffled with seed 13 before this split. The cluster files map to coldivf's
`Clusters`: `centroids`, `assignment`, `parents` and `parent_of`. `results/spec_states.jsonl` gets one line per run
that made something new.

## Evaluation Ids

The id of the evaluation queries is `test_rows_sha256`, the SHA-256 (Secure Hash Algorithm) hash of their rows in
`<ds>_queries` as little-endian int64 values, from `<ds>_test.meta.json`. The `.json` file of every output made from
them records it, i.e. `<ds>_test.gt.json`, the training sets' `.meta.json` files, `states/*.json`, `e.json`,
`ehat/*.json` and `clusters/*.ceiling.json`. A step makes such an output again when the id differs, and stops when an
input from an earlier step has another id. The ridge step also checks the id of the ground truth. It makes every
prediction first, then copies `e.fbin` again whenever its bytes differ from `<ds>_test.fbin`. So the two match only
once every prediction and the ground truth belong to the current evaluation queries, which `spec_coverage` checks before
it measures. `export_beir.py` stops when kept test files hold other rows than its rule gives. `timing.json` isn't tied
to the evaluation queries, since it only times the model.
