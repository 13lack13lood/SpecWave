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
| `encoder_refs.py` | Writes the outputs of sentence-transformers and transformers that the Rust encoder's checks compare with, for each model of the models list |
| `common.py` | Holds the file formats, the exact nearest-neighbor search, the ridge regression, the models list and the model runs that the scripts share |
| `test_common.py` | Tests `common.py`, and how the first two scripts keep or remake their outputs, on small synthetic data |
| `test_models.py` | Tests how the scripts serve any model of the models list: its folders, settings, prompts, pooling and layer summaries, and the embeddings the export makes and resumes |

## Running

```sh
python3 export_beir.py --cache DIR   # DIR holds the cached .npy embeddings and the BeIR relevance judgments
python3 spec_states.py fiqa          # or name steps, eg. spec_states.py fiqa timing
python3 spec_states.py quora
python3 export_beir.py --cache DIR --model bge-small fiqa   # another model of the models list, in its own folders
python3 spec_states.py --model bge-small fiqa
python3 encoder_refs.py              # or name models, eg. encoder_refs.py bge-small e5-base
python3 test_common.py
python3 test_models.py
```

Each step of `spec_states.py` runs in its own process, named or not, since faiss and PyTorch load clashing copies of
the OpenMP (Open Multi-Processing) runtime.

## Models

`--model` takes a short name of the models list, eg. `bge-small`, or a repository id. Each dataset's own model is the
default, arctic for FiQA and MiniLM for Quora, and its files are the ones in `data/` and `results/`. Any other model
gets folders of its own, `data/models/<name>/` and `results/models/<name>/`, laid out like `data/` and `results/`.
`$SPECWAVE_DATA` and `$SPECWAVE_RESULTS` name other folders, but never the shared ones for another model. The texts
and the evaluation queries don't depend on the model, so the export links them from `data/`.

The layers that get predictors come from the list, and the sizes, the pooling and the prompts from the model's
snapshot or the list, as `common.model` and `common.prompts` find them. Every model runs in float32. The prompts are
passed as their text, since sentence-transformers' prompt names give no prompt for a model whose settings hold none,
eg. bge. A layer's summary ends with the first (CLS) token, or with the last token for a causal model, eg. Qwen3,
whose last token is the end-of-text token its tokenizer appends. nomic-embed-text-v1.5 runs on transformers' own
nomic-bert code, which gives the same embeddings and states as the code its snapshot points at. transformers 5
can't run that code.

The export embeds a model's documents with its passage prompt and its queries with its query prompt, each text cut at
512 tokens, or at the model's own limit if it is lower, so every model reads as much of each document. It saves them
4,096 texts at a time in `<ds>_<part>.fbin.parts/`, so a stopped run makes only the parts that are missing, and
`<ds>_<part>.embed.json` says how they were made. The MiniLM fit to arctic's embedding is made for arctic on FiQA only.

## Files

Vectors are `.fbin` files and integers `.ibin` files, both an int32 row count and an int32 width, then the rows as
little-endian float32 or int32 values. Row i of a `.jsonl` file is row i of the `.fbin` file with the same stem. FiQA
uses snowflake-arctic-embed-m-v1.5 (768 dimensions, the CLS (classification) token, a query prompt) and Quora all-MiniLM-L6-v2 (384
dimensions, the token mean, no prompt), unless `--model` names another model. Every embedding is normalized.

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
| `states/<set>.L<l>.fbin` | The layer summaries (n × 2d, the masked mean then the CLS vector, or the last token's for a causal model) of set openings, queries or test |
| `states/<set>.final.fbin` | The same texts' final embeddings, the ridge targets |
| `clusters/<kind>.centroids.fbin` | N/160 cluster centers for N documents, from L2 (Euclidean, `l2`) or spherical (`sph`) k-means with seed 1 |
| `clusters/<kind>.assign.ibin` | Each document's cluster, one per row |
| `clusters/<kind>.parents.fbin`, `<kind>.parent_assign.ibin` | Groups of clusters (the square root of their count) and each cluster's group |
| `clusters/<kind>.ceiling.json` | The recall of an exact scan of each query's nearest clusters, picked by L2 distance (`l2`) or by the raw inner product with the centers as they are (`ip-raw`) |
| `timing.json` | The time to each layer at batch 1 on the processor with 4 threads |
| `summary.json` | All of the numbers above in one file |

Fit is `openings`, which needs no query log, or `queries`. The layers are the models list's, eg. 0, 3, 6, 8, 9 and 10 of
arctic's 12, and 0, 1, 3, 4 and 5 of MiniLM's 6. The penalty is the one that, fitted on the first 85% of the training
pairs, best predicts the last 15%. The openings are in random draw order already. The real queries are in parquet
order, where the BeIR splits sit in blocks, so their pairs are shuffled with seed 13 before this split. The cluster
files map to coldivf's `Clusters`: `centroids`, `assignment`, `parents` and `parent_of`. `spec_states.jsonl` in the
model's results folder gets one line per run that made something new.

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

## Encoder References

The models list is `specwave/src/encoder/models.json`, which the Rust encoder reads too. It names each model by a short
name, eg. `bge-small`, and holds its layers that may fire and the prompts that its sentence-transformers settings leave
out, with the file of the snapshot that gives them. `common.prompts` finds a model's query and passage prompts the way
the encoder does, and checks that each listed prompt is in that file.

`encoder_refs.py` writes `data/encoders/<name>/` for each model of the list whose snapshot is cached. It embeds the
first 50 evaluation queries of `fiqa_test.jsonl` and 50 documents of `fiqa_corpus.jsonl`, drawn with seed 5 from those
with text, on the processor in float32, one text at a time. The ignored tests of `specwave/tests/unit/encoder/candle.rs`
read these files.

| File | Contents |
|---|---|
| `<set>.jsonl` | The texts of set `queries` or `passages` as {row, text}, with each text's row in its source file |
| `<set>.ids.ibin`, `<set>.lengths.ibin` | Every text's token ids, one text after another, and each text's token count |
| `<set>.st.fbin` | sentence-transformers' embeddings, as the model gives them, so not always normalized |
| `<set>.L<l>.fbin` | concat(mean over the tokens, first token) of hidden state l, for l = 0 to the last layer |
| `queries.tokens.L<l>.fbin` | Every query token's hidden state l, one query after another |
| `meta.json` | The model's settings and prompts, how far the files agree among themselves, and the versions |
