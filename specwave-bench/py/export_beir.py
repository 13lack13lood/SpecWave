"""Exports two BeIR (Benchmarking Information Retrieval) datasets for the Rust benchmarks, FiQA (financial opinion
mining and question answering) with its snowflake-arctic-embed-m-v1.5 embeddings and Quora with its all-MiniLM-L6-v2
embeddings.

For each dataset it writes these files under data/.
- <ds>_corpus.jsonl and <ds>_queries.jsonl hold every document and every query as {row, id, text}, in the order of the
  BeIR parquet files. JSON (JavaScript Object Notation) lines hold one record per line.
- <ds>_corpus.fbin and <ds>_queries.fbin hold the cached normalized embeddings of the same rows. The queries were
  embedded with the model's query prompt, if it has one.
- <ds>_queries.splits.json holds the query rows of each BeIR split whose relevance judgments are in the cache.
- <ds>_test.jsonl, <ds>_test.fbin and <ds>_test.meta.json hold the evaluation queries, their embeddings, the rows they
  come from and the seed.
- <ds>_test.gt.ibin holds the 100 documents nearest each evaluation query by cosine, nearest first, with ties going to
  the lower row. <ds>_test.gt.json says how it was checked and records test_rows_sha256, the SHA-256 (Secure Hash
  Algorithm) hash of the evaluation queries' rows in <ds>_queries.
- <ds>_export.json holds the counts and the checks that row i of each .fbin file embeds row i of its .jsonl file.

FiQA's evaluation queries are the first 1,000 of a permutation of all its queries with seed 11, then the queries of
its BeIR test split that are not among them, in parquet order. Quora's are 1,000 queries of its BeIR test split, drawn
with seed 11.

Usage: export_beir.py --cache DIR [fiqa] [quora]

DIR holds the cached embeddings, eg. fiqa_corpus_arctic.npy and fiqa_queries_arctic.npy, the corpus ids they were made
with, eg. fiqa_corpus_ids.json, and the BeIR relevance judgments, eg. quora_dev.tsv. It can also come from
$SPECWAVE_CACHE. A query file that isn't cached is embedded again. Outputs that exist are skipped, so delete one to
make it again. The ground truth is also made again whenever the hash it records differs from the one of the rows in
<ds>_test.meta.json."""
import argparse
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Keeps the repo free of compiled bytecode folders.
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import common as c  # noqa: E402

QUERY_SEED = 11
TEST_SIZE = 1000
GT_K = 100
CHUNK = 100_000
CHECK_ROWS = 48
CHECK_SEED = 5
MIN_COS = 0.999


def write_texts(file, ids, texts):
    if os.path.exists(file):
        return

    c.write_jsonl(file, ({"row": i, "id": d, "text": t} for i, (d, t) in enumerate(zip(ids, texts))))
    print("wrote", file, len(ids), flush=True)


def check_ids(ids, file):
    """Checks the parquet ids against the ids the cached embeddings were made with, when the cache has them."""
    if os.path.exists(file) and c.read_json(file) != ids:
        raise ValueError(f"the parquet ids don't match {file}, so the cached embeddings are in another order")


def convert(src, dst, n, dim):
    """Copies the cached .npy embeddings `src` into the .fbin file `dst`, in chunks."""
    if os.path.exists(dst):
        return

    x = np.load(src, mmap_mode="r")

    if x.shape != (n, dim) or x.dtype != np.float32:
        raise ValueError(f"{src} is {x.dtype} {x.shape}, expected float32 ({n}, {dim})")

    out = c.new_table(dst, n, dim)

    for s in range(0, n, CHUNK):
        out[s:s + CHUNK] = x[s:s + CHUNK]

    c.finish(dst, out)
    print("wrote", dst, x.shape, flush=True)


def embed(st, texts, prompt_name):
    """Normalized embeddings of `texts`, made the way the cached ones were."""
    kw = dict(prompt_name=prompt_name) if prompt_name else {}
    e = st.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False, **kw)
    return e.astype(np.float32)


def embed_queries(dst, name, texts):
    """Embeds every query when the cache has no embeddings of them."""
    if os.path.exists(dst):
        return

    st = c.load_model(name, c.device())
    out = c.new_table(dst, len(texts), c.MODELS[name]["dim"])

    for s in range(0, len(texts), 4096):
        out[s:s + 4096] = embed(st, texts[s:s + 4096], c.MODELS[name]["prompt"])

    c.finish(dst, out)
    print("embedded", dst, len(texts), flush=True)


def stats(file):
    """The shape of an .fbin file and the range of its row norms."""
    x = c.open_table(file)
    lo, hi, finite = np.inf, 0.0, True

    for s in range(0, len(x), CHUNK):
        part = np.asarray(x[s:s + CHUNK])
        finite = finite and bool(np.isfinite(part).all())
        norms = np.linalg.norm(part.astype(np.float64), axis=1)
        lo, hi = min(lo, float(norms.min())), max(hi, float(norms.max()))

    if not finite or abs(lo - 1) > 1e-4 or abs(hi - 1) > 1e-4:
        raise ValueError(f"{file} has rows that aren't finite and normalized (norms {lo} to {hi})")

    return dict(rows=len(x), dim=x.shape[1], norm_min=lo, norm_max=hi)


def splits(ds, cache, qids):
    """The query rows of each BeIR split whose relevance judgments are cached, in parquet order."""
    row = {q: i for i, q in enumerate(qids)}
    out = {}

    for split in ("train", "dev", "test"):
        tsv = os.path.join(cache, f"{ds}_{split}.tsv")
        if not os.path.exists(tsv):
            continue
        with open(tsv, encoding="utf-8") as f:
            ids = {line.split("\t")[0] for line in f.read().splitlines()[1:] if line.strip()}
        if not ids <= row.keys():
            raise ValueError(f"{tsv} names queries that the parquet file doesn't have")
        out[split] = sorted(row[q] for q in ids)

    if "test" not in out:
        raise FileNotFoundError(f"no {ds}_test.tsv in {cache}")

    file = c.path(f"{ds}_queries.splits.json")

    if not os.path.exists(file):
        c.write_json(file, out)

    return out


def test_rows(ds, nq, split):
    """The query rows of the evaluation set, and the positions in it of each named subset."""
    if ds == "fiqa":
        first = np.random.default_rng(QUERY_SEED).permutation(nq)[:TEST_SIZE].tolist()
        taken, beir_test = set(first), set(split["test"])
        rows = first + [r for r in split["test"] if r not in taken]
        subsets = dict(permutation=list(range(TEST_SIZE)), beir_test=[j for j, r in enumerate(rows) if r in beir_test])
        rule = (f"rows default_rng({QUERY_SEED}).permutation(all queries)[:{TEST_SIZE}], then the BeIR test queries"
                " not among them, in parquet order")
        return rows, subsets, rule

    test = split["test"]
    rows = [test[i] for i in np.random.default_rng(QUERY_SEED).choice(len(test), TEST_SIZE, replace=False)]
    rule = (f"default_rng({QUERY_SEED}).choice(BeIR test queries in parquet order, {TEST_SIZE}, replace=False),"
            " in draw order")
    return rows, {"beir_test": list(range(TEST_SIZE))}, rule


def test_set(ds, qids, qtexts, split):
    """Writes the evaluation queries unless they are there already, and checks that their files hold the rule's rows."""
    rows, subsets, rule = test_rows(ds, len(qids), split)
    meta_file = c.path(f"{ds}_test.meta.json")
    remake = f"delete the {ds}_test files to remake them"

    if os.path.exists(meta_file) and c.read_json(meta_file)["query_rows"] != rows:
        raise ValueError(f"{meta_file} has other rows than the rule gives, {remake}")

    write_texts(c.path(f"{ds}_test.jsonl"), [qids[r] for r in rows], [qtexts[r] for r in rows])
    dst = c.path(f"{ds}_test.fbin")

    if not os.path.exists(dst):
        c.write_table(dst, c.open_table(c.path(f"{ds}_queries.fbin"))[rows])

    # The ground truth is made from these files but tied to the rows of the .meta.json file, so they must agree.
    same_ids = [t["id"] for t in c.read_jsonl(c.path(f"{ds}_test.jsonl"))] == [qids[r] for r in rows]
    same_vecs = np.array_equal(c.open_table(c.path(f"{ds}_queries.fbin"))[rows], c.open_table(dst))

    if not (same_ids and same_vecs):
        raise ValueError(f"{ds}_test.jsonl or {ds}_test.fbin doesn't hold the rule's rows of {ds}_queries, {remake}")

    if not os.path.exists(meta_file):
        c.write_json(meta_file, dict(seed=QUERY_SEED, rule=rule, rows=len(rows), query_rows=rows, subsets=subsets))

    return dict(rows=len(rows), seed=QUERY_SEED, rule=rule, subsets={k: len(v) for k, v in subsets.items()},
                copies_queries=True)


def ground_truth(ds):
    """The 100 documents nearest each evaluation query, made again unless the files belong to these queries."""
    dst, info_file = c.path(f"{ds}_test.gt.ibin"), c.path(f"{ds}_test.gt.json")
    made = c.evaluation_id(ds)

    if c.up_to_date([dst], info_file, **made):
        return c.read_json(info_file)

    base = c.open_table(c.path(f"{ds}_corpus.fbin"))
    q = np.asarray(c.open_table(c.path(f"{ds}_test.fbin")))
    t = time.time()
    found = c.twins(base)
    rows, scores = c.exact_topk(q, base, GT_K, found=found)
    took = time.time() - t
    sample = np.random.default_rng(CHECK_SEED).choice(len(q), 8, replace=False)

    for i in sample:
        if not np.array_equal(c.naive_topk(q[i], base, GT_K, found=found), rows[i]):
            raise ValueError(f"the chunked top {GT_K} of query {i} differs from a full sort")

    c.write_table(dst, rows, dtype="<i4")
    info = dict(made, k=GT_K, rows=len(q), metric="cosine in float64, ties to the lower row", seconds=round(took, 1),
                rows_with_identical_twins=len(found[0]), twin_sets=len(np.unique(found[1])),
                checked_by_full_sort=sample.tolist(), top1_mean=float(scores[:, 0].mean()),
                kth_mean=float(scores[:, -1].mean()))
    c.write_json(info_file, info)
    print("wrote", dst, rows.shape, f"{took:.0f}s", flush=True)
    return info


def check(ds, name):
    """Embeds a sample of rows of each .jsonl file again and compares them with the same rows of its .fbin file."""
    st = c.load_model(name, c.device())
    prompt = c.MODELS[name]["prompt"]
    rng = np.random.default_rng(CHECK_SEED)
    out = {}

    for part, prompt_name in (("corpus", None), ("queries", prompt), ("test", prompt)):
        recs = c.read_jsonl(c.path(f"{ds}_{part}.jsonl"))
        vecs = c.open_table(c.path(f"{ds}_{part}.fbin"))
        n = len(recs)
        if n != len(vecs) or any(r["row"] != i for i, r in enumerate(recs)):
            raise ValueError(f"{ds}_{part}.jsonl and .fbin don't have the same rows")
        rows = np.unique(np.concatenate([[0, n - 1], rng.choice(n, CHECK_ROWS - 2, replace=False)]))
        e = embed(st, [recs[i]["text"] for i in rows], prompt_name)
        cos = np.sum(e * vecs[rows], 1)
        shifted = np.sum(e * vecs[(rows + 1) % n], 1)
        out[part] = dict(rows_checked=len(rows), min_cos=float(cos.min()), mean_cos=float(cos.mean()),
                         next_row_mean_cos=float(shifted.mean()))
        print(ds, part, "check", out[part], flush=True)
        if cos.min() < MIN_COS:
            raise ValueError(f"{ds}_{part}: a row's text doesn't embed to its vector (cosine {cos.min():.4f})")

    return out


def export(ds, cache):
    name = c.DATASETS[ds]
    dim = c.MODELS[name]["dim"]
    t = time.time()
    ids, texts = c.beir(ds, "corpus")
    check_ids(ids, os.path.join(cache, f"{ds}_corpus_ids.json"))
    write_texts(c.path(f"{ds}_corpus.jsonl"), ids, texts)
    convert(os.path.join(cache, f"{ds}_corpus_{name}.npy"), c.path(f"{ds}_corpus.fbin"), len(ids), dim)
    qids, qtexts = c.beir(ds, "queries")
    check_ids(qids, os.path.join(cache, f"{ds}_query_ids.json"))
    write_texts(c.path(f"{ds}_queries.jsonl"), qids, qtexts)
    cached = os.path.join(cache, f"{ds}_queries_{name}.npy")

    if os.path.exists(cached):
        convert(cached, c.path(f"{ds}_queries.fbin"), len(qids), dim)
    else:
        embed_queries(c.path(f"{ds}_queries.fbin"), name, qtexts)

    split = splits(ds, cache, qids)
    report = dict(dataset=ds, model=c.MODELS[name]["repo"], query_prompt=c.MODELS[name]["prompt"],
                  corpus=stats(c.path(f"{ds}_corpus.fbin")), queries=stats(c.path(f"{ds}_queries.fbin")),
                  splits={k: len(v) for k, v in split.items()}, test=test_set(ds, qids, qtexts, split))
    report["gt"] = ground_truth(ds)
    report["checks"] = check(ds, name)
    report["seconds"] = round(time.time() - t, 1)
    c.write_json(c.path(f"{ds}_export.json"), report)
    print("done", ds, f"{report['seconds']:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Export FiQA and Quora texts, embeddings, query sets and ground truth.")
    ap.add_argument("--cache", default=os.environ.get("SPECWAVE_CACHE"), help="the folder of cached embeddings")
    ap.add_argument("datasets", nargs="*", help="fiqa, quora or both (the default)")
    a = ap.parse_args()
    datasets = a.datasets or ["fiqa", "quora"]

    if not a.cache or not os.path.isdir(a.cache):
        ap.error("give the folder of cached embeddings with --cache or $SPECWAVE_CACHE")

    if any(ds not in c.DATASETS for ds in datasets):
        ap.error(f"datasets are {', '.join(c.DATASETS)}")

    for ds in datasets:
        export(ds, a.cache)


if __name__ == "__main__":
    main()
