"""Exports two BeIR (Benchmarking Information Retrieval) datasets for the Rust benchmarks, FiQA (financial opinion
mining and question answering) and Quora, with the embeddings of one model of the models list. By default that is
each dataset's own model, snowflake-arctic-embed-m-v1.5 for FiQA and all-MiniLM-L6-v2 for Quora.

For each dataset it writes these files under data/, or under data/models/<name>/ for another model.
- <ds>_corpus.jsonl and <ds>_queries.jsonl hold every document and every query as {row, id, text}, in the order of the
  BeIR parquet files. JSON (JavaScript Object Notation) lines hold one record per line.
- <ds>_corpus.fbin and <ds>_queries.fbin hold the normalized embeddings of the same rows, cached or made here. The
  documents were embedded with the model's passage prompt and the queries with its query prompt, if it has them.
- <ds>_queries.splits.json holds the query rows of each BeIR split whose relevance judgments are in the cache.
- <ds>_test.jsonl, <ds>_test.fbin and <ds>_test.meta.json hold the evaluation queries, their embeddings, the rows they
  come from and the seed.
- <ds>_test.gt.ibin holds the 100 documents nearest each evaluation query by cosine, nearest first, with ties going to
  the lower row. <ds>_test.gt.json says how it was checked and records test_rows_sha256, the SHA-256 (Secure Hash
  Algorithm) hash of the evaluation queries' rows in <ds>_queries.
- <ds>_<part>.embed.json says how the embeddings of part corpus or queries were made, for those made here.
- <ds>_export.json holds the counts and the checks that row i of each .fbin file embeds row i of its .jsonl file.

FiQA's evaluation queries are the first 1,000 of a permutation of all its queries with seed 11, then the queries of
its BeIR test split that are not among them, in parquet order. Quora's are 1,000 queries of its BeIR test split, drawn
with seed 11. They don't depend on the model, and neither do the texts, so a model's own folder links those files from
data/ when data/ has them.

Embeddings that aren't cached are made here in float32, each text cut at 512 tokens, or at the model's own limit if it
is lower. That is the limit of the BERT (Bidirectional Encoder Representations from Transformers) models, so every
model reads as much of each document. They are made 4,096 texts at a time, and each finished part is saved at once, so
a stopped run makes only the parts that are missing.

Usage: export_beir.py --cache DIR [--model NAME] [fiqa] [quora]

DIR holds the cached embeddings, eg. fiqa_corpus_arctic.npy and fiqa_queries_arctic.npy, the corpus ids they were made
with, eg. fiqa_corpus_ids.json, and the BeIR relevance judgments, eg. quora_dev.tsv. It can also come from
$SPECWAVE_CACHE. NAME is a short name of the models list, eg. bge-small. Outputs that exist are skipped, so delete one
to make it again. The ground truth is also made again whenever the hash it records differs from the one of the rows in
<ds>_test.meta.json."""
import argparse
import os
import shutil
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
# The most tokens of a text that a model reads here.
MAX_TOKENS = 512
# The texts in each saved part of an embedding made here, and in each batch of the model.
PART = 4096
BATCH = 32
# The files of a dataset that don't depend on the model.
TEXTS = ("corpus.jsonl", "queries.jsonl", "queries.splits.json", "test.jsonl", "test.meta.json")


def write_texts(file, ids, texts):
    if os.path.exists(file):
        return

    c.write_jsonl(file, ({"row": i, "id": d, "text": t} for i, (d, t) in enumerate(zip(ids, texts))))
    print("wrote", file, len(ids), flush=True)


def link_texts(ds):
    """Links the texts and the evaluation queries of `ds` from data/ into another model's folder, where it lacks them.
    A file that data/ doesn't have is made here instead."""
    if os.path.realpath(c.DATA) == os.path.realpath(c.SHARED_DATA):
        return

    os.makedirs(c.DATA, exist_ok=True)

    for part in TEXTS:
        src, dst = os.path.join(c.SHARED_DATA, f"{ds}_{part}"), c.path(f"{ds}_{part}")
        if os.path.exists(src) and not os.path.lexists(dst):
            os.symlink(src, dst)


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


def load(name):
    """The model `name` on the fastest device here, reading at most MAX_TOKENS tokens of each text."""
    st = c.load_model(name, c.device())
    st.max_seq_length = min(st.max_seq_length or MAX_TOKENS, MAX_TOKENS)
    return st


def embed(st, texts, prompt):
    """Normalized embeddings of `texts`, each read after the literal text `prompt`."""
    e = st.encode(texts, prompt=prompt, batch_size=BATCH, normalize_embeddings=True, convert_to_numpy=True,
                  show_progress_bar=False)
    return e.astype(np.float32)


def save_part(file, rows):
    with open(file + ".part", "wb") as f:
        np.save(f, rows)

    os.replace(file + ".part", file)


def embed_table(dst, st, name, texts, prompt):
    """Embeds `texts` after `prompt` with model `name` into the .fbin file `dst`, PART texts at a time. Each part goes
    to <dst>.parts/ as soon as it is done, with the settings it was made with, and the parts are joined once they are
    all there. Parts made with other settings are made again. Writes the settings and the model's seconds to
    <stem>.embed.json, next to `dst`."""
    if os.path.exists(dst):
        return

    import torch

    folder, info_file = dst + ".parts", dst[:-len(".fbin")] + ".embed.json"
    dev = next(st.parameters()).device
    settings = dict(model=c.model(name)["repo"], prompt=prompt, rows=len(texts), part=PART, batch=BATCH,
                    max_tokens=st.max_seq_length, dtype=str(next(st.parameters()).dtype), normalized=True)

    if os.path.isdir(folder) and not c.up_to_date([], os.path.join(folder, "settings.json"), **settings):
        shutil.rmtree(folder)

    os.makedirs(folder, exist_ok=True)
    c.write_json(os.path.join(folder, "settings.json"), settings)
    starts = range(0, len(texts), PART)

    for s in starts:
        file = os.path.join(folder, f"{s:08d}.npy")
        if os.path.exists(file):
            continue
        t = time.time()
        rows = embed(st, texts[s:s + PART], prompt)
        save_part(file, rows)
        c.write_json(file[:-len(".npy")] + ".json", dict(seconds=round(time.time() - t, 2), device=dev.type))
        print(os.path.basename(dst), f"rows {s} to {s + len(rows)} of {len(texts)} in {time.time() - t:.0f}s",
              flush=True)
        if dev.type == "mps":
            torch.mps.empty_cache()

    dim = st.get_embedding_dimension()
    out = c.new_table(dst, len(texts), dim)
    seconds = 0.0

    for s in starts:
        rows = np.load(os.path.join(folder, f"{s:08d}.npy"))
        if rows.shape != (min(PART, len(texts) - s), dim) or rows.dtype != np.float32:
            raise ValueError(f"{folder} has a part of {rows.dtype} {rows.shape} at row {s}, so delete it")
        out[s:s + len(rows)] = rows
        seconds += c.read_json(os.path.join(folder, f"{s:08d}.json"))["seconds"]

    c.finish(dst, out)
    c.write_json(info_file, dict(settings, device=dev.type, seconds=round(seconds, 1), parts=len(starts)))
    shutil.rmtree(folder)
    print("embedded", dst, len(texts), f"{seconds:.0f}s", flush=True)


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


def check(ds, name, st=None):
    """Embeds a sample of rows of each .jsonl file again, with the prompt of its part, and compares them with the same
    rows of its .fbin file."""
    st = st or load(name)
    m = c.model(name)
    rng = np.random.default_rng(CHECK_SEED)
    out = {}

    for part, prompt in (("corpus", m["passage_prompt"]), ("queries", m["query_prompt"]), ("test", m["query_prompt"])):
        recs = c.read_jsonl(c.path(f"{ds}_{part}.jsonl"))
        vecs = c.open_table(c.path(f"{ds}_{part}.fbin"))
        n = len(recs)
        if n != len(vecs) or any(r["row"] != i for i, r in enumerate(recs)):
            raise ValueError(f"{ds}_{part}.jsonl and .fbin don't have the same rows")
        rows = np.unique(np.concatenate([[0, n - 1], rng.choice(n, CHECK_ROWS - 2, replace=False)]))
        e = embed(st, [recs[i]["text"] for i in rows], prompt)
        cos = np.sum(e * vecs[rows], 1)
        shifted = np.sum(e * vecs[(rows + 1) % n], 1)
        out[part] = dict(rows_checked=len(rows), min_cos=float(cos.min()), mean_cos=float(cos.mean()),
                         next_row_mean_cos=float(shifted.mean()))
        print(ds, part, "check", out[part], flush=True)
        if cos.min() < MIN_COS:
            raise ValueError(f"{ds}_{part}: a row's text doesn't embed to its vector (cosine {cos.min():.4f})")

    return out


def export(ds, cache, name):
    """Exports `ds` with the listed model `name` into the data folder, which `c.use_model` points at that model's."""
    m = c.model(name)
    t = time.time()
    link_texts(ds)
    ids, texts = c.beir(ds, "corpus")
    check_ids(ids, os.path.join(cache, f"{ds}_corpus_ids.json"))
    write_texts(c.path(f"{ds}_corpus.jsonl"), ids, texts)
    qids, qtexts = c.beir(ds, "queries")
    check_ids(qids, os.path.join(cache, f"{ds}_query_ids.json"))
    write_texts(c.path(f"{ds}_queries.jsonl"), qids, qtexts)
    st = None

    for part, items, prompt in (("corpus", texts, m["passage_prompt"]), ("queries", qtexts, m["query_prompt"])):
        cached, dst = os.path.join(cache, f"{ds}_{part}_{name}.npy"), c.path(f"{ds}_{part}.fbin")
        if os.path.exists(cached):
            convert(cached, dst, len(items), m["dim"])
        elif not os.path.exists(dst):
            st = st or load(name)
            embed_table(dst, st, name, items, prompt)

    split = splits(ds, cache, qids)
    infos = {p: c.path(f"{ds}_{p}.embed.json") for p in ("corpus", "queries")}
    made_here = {p: c.read_json(f) for p, f in infos.items() if os.path.exists(f)}
    report = dict(dataset=ds, model=m["repo"], name=name, query_prompt=m["query_prompt"],
                  passage_prompt=m["passage_prompt"], embedded_here=made_here,
                  corpus=stats(c.path(f"{ds}_corpus.fbin")), queries=stats(c.path(f"{ds}_queries.fbin")),
                  splits={k: len(v) for k, v in split.items()}, test=test_set(ds, qids, qtexts, split))
    report["gt"] = ground_truth(ds)
    report["checks"] = check(ds, name, st)
    report["seconds"] = round(time.time() - t, 1)
    c.write_json(c.path(f"{ds}_export.json"), report)
    print("done", ds, name, f"{report['seconds']:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Export FiQA and Quora texts, embeddings, query sets and ground truth.")
    ap.add_argument("--cache", default=os.environ.get("SPECWAVE_CACHE"), help="the folder of cached embeddings")
    ap.add_argument("--model", help="a short name of the models list, each dataset's own model by default")
    ap.add_argument("datasets", nargs="*", help="fiqa, quora or both (the default)")
    a = ap.parse_args()
    datasets = a.datasets or ["fiqa", "quora"]

    if not a.cache or not os.path.isdir(a.cache):
        ap.error("give the folder of cached embeddings with --cache or $SPECWAVE_CACHE")

    if any(ds not in c.DATASETS for ds in datasets):
        ap.error(f"datasets are {', '.join(c.DATASETS)}")

    for ds in datasets:
        try:
            name = c.use_model(a.model or c.DATASETS[ds], ds)
        except KeyError as e:
            ap.error(str(e))
        export(ds, a.cache, name)


if __name__ == "__main__":
    main()
