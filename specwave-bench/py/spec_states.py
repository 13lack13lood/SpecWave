"""Computes the layer states and the ridge predictors that SpecWave starts its early reads with, plus the cluster files
of the Rust benchmarks, for FiQA (financial opinion mining and question answering) with snowflake-arctic-embed-m-v1.5
and Quora with all-MiniLM-L6-v2. It reads the exports of export_beir.py.

Layer l is the hidden state after l transformer layers, and l = 0 is the embedding output. Its pooled state is the
mean over the attention mask followed by the CLS (classification token) vector, so 2d values for a model of d
dimensions. Every input is encoded with at most 128 tokens and the model's query prompt, if it has one. The inputs
are these.
- The evaluation queries of export_beir.py.
- Real queries outside them, all the other queries for FiQA and the BeIR (Benchmarking Information Retrieval) dev
  split for Quora.
- The openings of 20,000 random documents, i.e. each one's first 6 to 24 words, a uniform draw per document.

For each layer, a ridge regression from the pooled state to the final embedding predicts the evaluation queries' final
embeddings. It is fitted once on the openings, which need no query log, and once on the real queries, which bound what
a query log could give. X is standardized and Y centered. Of the penalties 1e-3, 1e-2 and 1e-1, the one that fitted on
the first 85% of the pairs best predicts the last 15% is fitted again on all of them. The openings are in random draw
order already. The real queries are in parquet order, where the BeIR splits sit in blocks, so their pairs are shuffled
with seed 13 before this split. For FiQA it also fits MiniLM's last layer state to arctic's final embedding, so a
second, smaller model can make the prediction.

The clusters come from k-means in faiss, the similarity search library, over the corpus embeddings, with N/160
clusters for N documents. They are made twice, with plain L2 (Euclidean distance) k-means and with spherical k-means,
each with groups of nearby clusters from a k-means over the cluster centers.

It writes these files under data/spec/<ds>/.
- openings.jsonl and queries.jsonl hold the training texts, eg. {row, doc, id, words, text} per opening.
  openings.meta.json and queries.meta.json say how they were drawn.
- states/<set>.L<l>.fbin (n × 2d) and states/<set>.final.fbin (n × d, normalized) hold the states of each set, i.e.
  openings, queries or test. For FiQA, states/<set>.minilm.L6.fbin holds MiniLM's last layer state.
- e.fbin is a byte copy of data/<ds>_test.fbin, the evaluation queries' final embeddings, which the ridge step writes
  last.
- ehat/<fit>.L<l>.fbin holds the normalized predictions for the evaluation queries, in the same row order, for fit
  openings or queries. For FiQA, ehat/minilm.<fit>.fbin holds MiniLM's predictions. Each has a .json file.
- clusters/<kind>.centroids.fbin, <kind>.assign.ibin (each document's cluster), <kind>.parents.fbin (the group
  centers) and <kind>.parent_assign.ibin (each cluster's group) hold the clusters, for kind l2 or sph. <kind>.json
  holds the settings and the cluster sizes, and <kind>.ceiling.json the recall of an exact scan of the clusters each
  evaluation query probes. The query picks them by L2 distance (l2) or by the raw inner product with the centers as
  they are (ip-raw).
- timing.json holds the time to each layer on the processor at batch 1.
- summary.json holds the numbers of all of these. results/spec_states.jsonl gets a line per run that made something.

Every output made from the evaluation queries records test_rows_sha256 in its .json file, the SHA-256 (Secure Hash
Algorithm) hash of their rows in data/<ds>_queries. A step makes such an output again when that hash differs from the
one of data/<ds>_test.meta.json, and stops when an input from an earlier step has another hash. The ridge step also
checks the ground truth's hash. It makes every prediction first, then copies e.fbin again whenever its bytes differ
from data/<ds>_test.fbin. So the two files match only once every prediction and the ground truth belong to the current
evaluation queries, which the Rust benchmarks check. Other outputs that exist are skipped, so delete one to make it
again.

Usage: spec_states.py fiqa|quora [states] [ridge] [clusters] [timing]

With no step it runs them all. Each step runs in its own process, named or not, since faiss and PyTorch load clashing
copies of the OpenMP (Open Multi-Processing) runtime."""
import argparse
import os
import subprocess
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Keeps the repo free of compiled bytecode folders.
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import common as c  # noqa: E402

LAYERS = {"fiqa": [0, 3, 6, 8, 9, 10], "quora": [0, 1, 3, 4, 5]}
OPENINGS = 20_000
OPENING_SEED = 12
# A fit on fewer openings shows how much the cosine gains from more training pairs.
SMALL_FIT = {"fiqa": 8000, "quora": 10_000}
LAMBDAS = [1e-3, 1e-2, 1e-1]
# The seed that shuffles a training set's pairs before the penalty's validation split, or None for no shuffle.
SPLIT_SEED = {"openings": None, "queries": 13}
MAX_LEN = 128
VECTORS_PER_LIST = 160
# Probe shares of the clusters, eg. 0.0256 of 360 clusters is 9.
PROBE_SHARES = [0.017, 0.0256, 0.0307]
TIMING_WARMUP, TIMING_RUNS, TIMING_THREADS = 20, 100, 4
STEPS = ["states", "ridge", "clusters", "timing"]


def spec(ds, *parts):
    return c.path("spec", ds, *parts)


def training_sets(ds, made):
    """The texts of each set, after writing the openings and the real training queries unless they were made for the
    current evaluation queries, whose id is `made`. The openings skip documents with an evaluation query's text, and
    the real queries skip the evaluation queries."""
    test = c.read_jsonl(c.path(f"{ds}_test.jsonl"))
    taken = set(c.read_json(c.path(f"{ds}_test.meta.json"))["query_rows"])
    opening_file, query_file = spec(ds, "openings.jsonl"), spec(ds, "queries.jsonl")

    if not c.up_to_date([opening_file], spec(ds, "openings.meta.json"), **made):
        corpus = c.read_jsonl(c.path(f"{ds}_corpus.jsonl"))
        texts = [r["text"] for r in corpus]
        docs, words, opened = c.openings(texts, OPENINGS, OPENING_SEED, skip={t["text"] for t in test})
        records = ({"row": i, "doc": int(d), "id": corpus[d]["id"], "words": int(w), "text": t}
                   for i, (d, w, t) in enumerate(zip(docs, words, opened)))
        c.write_jsonl(opening_file, records)
        rule = "the first 6 to 24 words of random documents whose text isn't an evaluation query, in draw order"
        c.write_json(spec(ds, "openings.meta.json"), dict(made, rows=OPENINGS, seed=OPENING_SEED, rule=rule))

    if not c.up_to_date([query_file], spec(ds, "queries.meta.json"), **made):
        queries = c.read_jsonl(c.path(f"{ds}_queries.jsonl"))
        split = c.read_json(c.path(f"{ds}_queries.splits.json"))
        pool = range(len(queries)) if ds == "fiqa" else split["dev"]
        rows = [r for r in pool if r not in taken]
        records = ({"row": i, "query": r, "id": queries[r]["id"], "text": queries[r]["text"]}
                   for i, r in enumerate(rows))
        c.write_jsonl(query_file, records)
        which = "the queries" if ds == "fiqa" else "the BeIR dev queries"
        rule = f"{which} outside the evaluation queries, in parquet order"
        c.write_json(spec(ds, "queries.meta.json"), dict(made, rows=len(rows), rule=rule))

    return {
        "openings": [r["text"] for r in c.read_jsonl(opening_file)],
        "queries": [r["text"] for r in c.read_jsonl(query_file)],
        "test": [r["text"] for r in test],
    }


def state_infos(ds):
    """The .json file of each set's states, for each model that encodes the sets."""
    tags = ["", "minilm."] if ds == "fiqa" else [""]
    return [spec(ds, "states", f"{s}.{tag}json") for tag in tags for s in ("openings", "queries", "test")]


def states(ds):
    """Encodes every set with the dataset's model, and for FiQA also with MiniLM, writing the pooled states. A set is
    encoded again when its states were made for other evaluation queries."""
    name = c.DATASETS[ds]
    made = c.evaluation_id(ds)
    sets = training_sets(ds, made)
    runs = [(name, "", LAYERS[ds], True)]

    if ds == "fiqa":
        runs.append(("minilm", "minilm.", [c.MODELS["minilm"]["layers"]], False))

    for model, tag, layers, with_final in runs:
        st = None
        cfg = c.MODELS[model]
        for set_name, texts in sets.items():
            files = {layer: spec(ds, "states", f"{set_name}.{tag}L{layer}.fbin") for layer in layers}
            final = spec(ds, "states", f"{set_name}.final.fbin") if with_final else None
            info_file = spec(ds, "states", f"{set_name}.{tag}json")
            if c.up_to_date([f for f in [*files.values(), final] if f], info_file, **made):
                continue
            st = st or c.load_model(model, c.device())
            prompt = c.prompt_of(st, model)
            sinks = {layer: c.new_table(f, len(texts), 2 * cfg["dim"]) for layer, f in files.items()}
            final_sink = c.new_table(final, len(texts), cfg["dim"]) if final else None
            t = time.time()
            lengths = c.encode_states(st, texts, prompt, layers, sinks, final_sink, cls=cfg["cls"], max_len=MAX_LEN)
            took = time.time() - t
            for layer, f in files.items():
                c.finish(f, sinks[layer])
            if final:
                c.finish(final, final_sink)
            info = dict(made, model=cfg["repo"], rows=len(texts), layers=layers, prompt=prompt, max_len=MAX_LEN,
                        tokens_mean=float(np.mean(lengths)), at_max_len=int(np.sum(lengths >= MAX_LEN)),
                        seconds=round(took, 1), texts_per_second=round(len(texts) / took, 1))
            c.write_json(info_file, info)
            print(ds, model, set_name, f"{len(texts)} texts in {took:.0f}s", flush=True)


def table(file):
    return np.asarray(c.open_table(file))


def fit(X, Y, Z, e, subsets, seed):
    """Chooses the penalty, with the pairs shuffled by `seed` first unless it is None, fits on all pairs and predicts
    the evaluation queries. Returns the normalized predictions and their numbers."""
    t = time.time()
    lam, val = c.choose_lambda(X, Y, LAMBDAS, seed=seed)
    ridge = c.Ridge(X, Y)
    preds = {p: c.normalize(ridge.predict(Z, ridge.weights(p))) for p in LAMBDAS}
    cos = np.sum(preds[lam] * e, 1)
    info = dict(rows_train=len(X), lam=lam, val_cos={str(p): v for p, v in val.items()}, cos=float(cos.mean()),
                cos_p10=float(np.percentile(cos, 10)), cos_min=float(cos.min()),
                cos_subsets={k: float(cos[v].mean()) for k, v in subsets.items()},
                cos_by_lam={str(p): float(np.mean(np.sum(q * e, 1))) for p, q in preds.items()},
                seconds=round(time.time() - t, 1))
    return preds[lam].astype(np.float32), info


def predict(stem, made, X, Y, Z, e, subsets):
    """Fits one predictor and writes its predictions to <stem>.fbin and its numbers to <stem>.json, unless they were
    made for the same evaluation queries and split seed, which `made` holds."""
    if c.up_to_date([stem + ".fbin"], stem + ".json", **made):
        return

    ehat, info = fit(X, Y, Z, e, subsets, made["split_seed"])
    c.write_table(stem + ".fbin", ehat)
    c.write_json(stem + ".json", dict(made, **info))
    print(os.path.basename(stem), f"lam={info['lam']} cos={info['cos']:.4f}", flush=True)


def ridge(ds):
    """Fits every predictor and writes its predictions for the evaluation queries. It stops when the states or the
    ground truth were made for other evaluation queries, and it writes e.fbin last. So e.fbin holds the bytes of
    data/<ds>_test.fbin only once a ridge step has finished with the current evaluation queries."""
    made = c.evaluation_id(ds)
    stale = [os.path.basename(f)[:-len(".json")] for f in state_infos(ds) if not c.up_to_date([], f, **made)]

    if stale:
        raise ValueError(f"the {', '.join(stale)} states weren't made for {ds}'s evaluation queries, so run the states"
                         " step first")

    if not c.up_to_date([c.path(f"{ds}_test.gt.ibin")], c.path(f"{ds}_test.gt.json"), **made):
        raise ValueError(f"{ds}_test.gt.ibin wasn't made for {ds}'s evaluation queries, so run export_beir.py first")

    test_file, e_file = c.path(f"{ds}_test.fbin"), spec(ds, "e.fbin")
    e = table(test_file).astype(np.float64)
    subsets = c.read_json(c.path(f"{ds}_test.meta.json"))["subsets"]
    # The model run here must reproduce the exported embeddings, or the states belong to other texts.
    agree = np.sum(c.normalize(table(spec(ds, "states", "test.final.fbin")).astype(np.float64)) * e, 1)

    if agree.min() < 0.999:
        raise ValueError(f"the final embeddings of the test states differ from {ds}_test.fbin (cosine {agree.min()})")

    targets = {s: table(spec(ds, "states", f"{s}.final.fbin")) for s in ("openings", "queries")}

    for fit_set, Y in targets.items():
        split = dict(made, split_seed=SPLIT_SEED[fit_set])
        for layer in LAYERS[ds]:
            X = table(spec(ds, "states", f"{fit_set}.L{layer}.fbin"))
            Z = table(spec(ds, "states", f"test.L{layer}.fbin"))
            predict(spec(ds, "ehat", f"{fit_set}.L{layer}"), split, X, Y, Z, e, subsets)

    if ds == "fiqa":
        last = c.MODELS["minilm"]["layers"]
        Z = table(spec(ds, "states", f"test.minilm.L{last}.fbin"))
        for fit_set, Y in targets.items():
            X = table(spec(ds, "states", f"{fit_set}.minilm.L{last}.fbin"))
            split = dict(made, split_seed=SPLIT_SEED[fit_set])
            predict(spec(ds, "ehat", f"minilm.{fit_set}"), split, X, Y, Z, e, subsets)

    n = SMALL_FIT[ds]
    small_file = spec(ds, "ehat", f"openings{n}.json")
    small_made = dict(made, split_seed=SPLIT_SEED["openings"])

    if not c.up_to_date([], small_file, **small_made):
        small = {}
        for layer in LAYERS[ds]:
            X = table(spec(ds, "states", f"openings.L{layer}.fbin"))[:n]
            Z = table(spec(ds, "states", f"test.L{layer}.fbin"))
            _, info = fit(X, targets["openings"][:n], Z, e, subsets, SPLIT_SEED["openings"])
            small[str(layer)] = dict(lam=info["lam"], cos=info["cos"], cos_subsets=info["cos_subsets"])
        c.write_json(small_file, dict(small_made, rows_train=n, layers=small))

    if not c.same_bytes(test_file, e_file):
        c.copy_file(test_file, e_file)

    if not c.up_to_date([e_file], spec(ds, "e.json"), **made):
        c.write_json(spec(ds, "e.json"), dict(made, rows=len(e), states_final_min_cos=float(agree.min()),
                                              states_final_mean_cos=float(agree.mean())))


def probe_ceilings(ds, stem):
    """The mean share of each evaluation query's 10 true nearest documents inside the clusters it probes, i.e. the
    recall of an exact scan of those clusters, for each probe share. Queries probe their nearest clusters by L2
    distance (l2) or by the raw inner product with the centers as they are (ip-raw). The two only agree when every
    center has the same length."""
    e = table(c.path(f"{ds}_test.fbin")).astype(np.float64)
    gt = np.asarray(c.open_table(c.path(f"{ds}_test.gt.ibin"), dtype="<i4"))[:, :10]
    cent = table(stem + ".centroids.fbin").astype(np.float64)
    home = np.asarray(c.open_table(stem + ".assign.ibin", dtype="<i4"))[:, 0][gt]
    ip = e @ cent.T
    out = {}

    for routing, score in (("l2", 2 * ip - np.sum(cent * cent, 1)), ("ip-raw", ip)):
        order = np.argsort(-score, axis=1, kind="stable")
        for share in PROBE_SHARES:
            nprobe = int(round(share * len(cent)))
            hit = [np.isin(home[i], order[i, :nprobe]).mean() for i in range(len(e))]
            out.setdefault(routing, {})[str(nprobe)] = float(np.mean(hit))

    return out


def clusters(ds):
    """Clusters the corpus with plain and spherical k-means, then groups the clusters, and writes both. Then it writes
    the probe ceilings of both, again whenever they were made for other evaluation queries."""
    base = c.open_table(c.path(f"{ds}_corpus.fbin"))
    n, d = base.shape
    nlist = n // VECTORS_PER_LIST
    ngroups = int(round(np.sqrt(nlist)))
    names = [".centroids.fbin", ".assign.ibin", ".parents.fbin", ".parent_assign.ibin", ".json"]
    kinds = [(kind, spherical) for kind, spherical in (("l2", False), ("sph", True))
             if not all(os.path.exists(spec(ds, "clusters", kind) + s) for s in names)]

    if kinds:
        import faiss
        faiss.omp_set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

    for kind, spherical in kinds:
        stem = spec(ds, "clusters", kind)
        t = time.time()
        km = faiss.Kmeans(d, nlist, niter=20, seed=1, max_points_per_centroid=256, spherical=spherical, verbose=False)
        # faiss reads the mapped corpus in place, since its rows are already contiguous float32 values.
        km.train(base)
        cent = np.ascontiguousarray(km.centroids, dtype=np.float32)
        index = faiss.IndexFlatIP(d) if spherical else faiss.IndexFlatL2(d)
        index.add(cent)
        _, assign = index.search(base, 1)
        groups = faiss.Kmeans(d, ngroups, niter=50, seed=2, max_points_per_centroid=256, verbose=False)
        groups.train(cent)
        gindex = faiss.IndexFlatL2(d)
        gindex.add(groups.centroids)
        _, parent = gindex.search(cent, 1)
        took = time.time() - t
        sizes = np.bincount(assign[:, 0], minlength=nlist)
        info = dict(kind=kind, rows=n, nlist=nlist, ngroups=ngroups, spherical=spherical, niter=20, seed=1,
                    group_niter=50, group_seed=2, objective=float(km.obj[-1]), seconds=round(took, 1),
                    size_min=int(sizes.min()), size_max=int(sizes.max()), size_p99=float(np.percentile(sizes, 99)),
                    empty=int(np.sum(sizes == 0)), groups_empty=int(ngroups - len(np.unique(parent))))
        c.write_table(stem + ".centroids.fbin", cent)
        c.write_table(stem + ".assign.ibin", assign.astype(np.int32), dtype="<i4")
        c.write_table(stem + ".parents.fbin", groups.centroids)
        c.write_table(stem + ".parent_assign.ibin", parent.astype(np.int32), dtype="<i4")
        c.write_json(stem + ".json", info)
        print(ds, kind, f"nlist={nlist} groups={ngroups} {took:.0f}s", flush=True)

    made = c.evaluation_id(ds)

    for kind in ("l2", "sph"):
        stem = spec(ds, "clusters", kind)
        if c.up_to_date([], stem + ".ceiling.json", **made):
            continue
        if not c.up_to_date([c.path(f"{ds}_test.gt.ibin")], c.path(f"{ds}_test.gt.json"), **made):
            raise ValueError(f"{ds}_test.gt.ibin wasn't made for {ds}'s evaluation queries, so run export_beir.py"
                             " first")
        recall = probe_ceilings(ds, stem)
        c.write_json(stem + ".ceiling.json", dict(made, recall=recall))
        print(ds, kind, "probe ceilings", recall, flush=True)


def timing(ds):
    """Times the model to each layer at batch 1 on the processor, the way a query arrives, after a warm-up."""
    out = spec(ds, "timing.json")

    if os.path.exists(out):
        return

    import torch

    torch.set_num_threads(TIMING_THREADS)
    name = c.DATASETS[ds]
    st = c.load_model(name, "cpu")
    net, tok, prompt = st[0].auto_model, st.tokenizer, c.prompt_of(st, name)
    texts = [r["text"] for r in c.read_jsonl(c.path(f"{ds}_test.jsonl"))][:TIMING_WARMUP + TIMING_RUNS]
    marks = []

    def stamp(module, args, output):
        marks.append(time.perf_counter())

    hooks = [net.embeddings.register_forward_hook(stamp)]
    hooks += [layer.register_forward_hook(stamp) for layer in net.encoder.layer]
    per, total = [], []
    load_before = os.getloadavg()

    with torch.inference_mode():
        for text in texts:
            marks.clear()
            t0 = time.perf_counter()
            net(**tok([prompt + text], truncation=True, max_length=MAX_LEN, return_tensors="pt"))
            total.append(time.perf_counter() - t0)
            per.append([m - t0 for m in marks])

    for h in hooks:
        h.remove()

    per = np.array(per[TIMING_WARMUP:]) * 1e3
    total = np.array(total[TIMING_WARMUP:]) * 1e3
    info = dict(device="cpu", threads=torch.get_num_threads(), batch=1, warmup=TIMING_WARMUP, runs=len(total),
                total_ms_p50=float(np.median(total)), total_ms_p90=float(np.percentile(total, 90)),
                layer_ms_p50={str(layer): float(np.median(per[:, layer])) for layer in range(per.shape[1])},
                loadavg_before=list(load_before), loadavg_after=list(os.getloadavg()),
                note="from the start of tokenization to the end of each layer, l = 0 being the embedding output")
    c.write_json(out, info)
    print(ds, "timing", f"total {info['total_ms_p50']:.2f} ms", info["layer_ms_p50"], flush=True)


def read(file):
    return c.read_json(file) if os.path.exists(file) else None


def summary(ds):
    """Collects the numbers of every step that has run into summary.json, and returns it."""
    folder = spec(ds, "states")
    names = sorted(f for f in os.listdir(folder) if f.endswith(".json")) if os.path.isdir(folder) else []
    sets = {f[:-len(".json")]: read(os.path.join(folder, f)) for f in names}
    preds, cos, lams = {}, {}, {}

    for fit_set in ("openings", "queries"):
        for layer in LAYERS[ds]:
            info = read(spec(ds, "ehat", f"{fit_set}.L{layer}.json"))
            if info:
                preds[f"{fit_set}.L{layer}"] = info
                cos.setdefault(fit_set, {})[str(layer)] = round(info["cos"], 4)
                lams.setdefault(fit_set, {})[str(layer)] = info["lam"]

    for fit_set in ("openings", "queries"):
        info = read(spec(ds, "ehat", f"minilm.{fit_set}.json"))
        if info:
            preds[f"minilm.{fit_set}"] = info
            cos.setdefault("minilm", {})[fit_set] = round(info["cos"], 4)

    out = dict(dataset=ds, model=c.MODELS[c.DATASETS[ds]]["repo"], layers=LAYERS[ds],
               final_layer=c.MODELS[c.DATASETS[ds]]["layers"], openings_seed=OPENING_SEED, max_len=MAX_LEN,
               lambdas=LAMBDAS, split_seed=SPLIT_SEED, cos=cos, lam=lams,
               training_sets={s: read(spec(ds, f"{s}.meta.json")) for s in ("openings", "queries")},
               sets=sets, e=read(spec(ds, "e.json")), predictors=preds,
               small_fit=read(spec(ds, "ehat", f"openings{SMALL_FIT[ds]}.json")),
               clusters={k: read(spec(ds, "clusters", f"{k}.json")) for k in ("l2", "sph")},
               probe_ceiling={k: read(spec(ds, "clusters", f"{k}.ceiling.json")) for k in ("l2", "sph")},
               timing_cpu=read(spec(ds, "timing.json")))
    c.write_json(spec(ds, "summary.json"), out)
    return out


def outputs(ds):
    """The modification time of every output of `ds`, to tell whether a run made something."""
    found = {}

    for root, _, files in os.walk(spec(ds)):
        for f in files:
            if not f.endswith(".part") and f != "summary.json":
                found[os.path.join(root, f)] = os.path.getmtime(os.path.join(root, f))

    return found


STEP = {"states": states, "ridge": ridge, "clusters": clusters, "timing": timing}
# Marks the process that runs one step for a run of the script. That run writes the summary itself.
CHILD = "SPEC_STATES_STEP"


def main():
    ap = argparse.ArgumentParser(description="Layer states, ridge predictors and clusters for SpecWave.")
    ap.add_argument("dataset", choices=sorted(c.DATASETS))
    ap.add_argument("steps", nargs="*", help=f"any of {', '.join(STEPS)}, all of them by default, run in this order")
    a = ap.parse_args()

    if any(s not in STEPS for s in a.steps):
        ap.error(f"steps are {', '.join(STEPS)}")

    if os.environ.get(CHILD):
        for step in a.steps:
            STEP[step](a.dataset)
        return

    before = outputs(a.dataset)
    env = dict(os.environ, **{CHILD: "1"})

    for step in [s for s in STEPS if s in a.steps] or STEPS:
        subprocess.run([sys.executable, os.path.abspath(__file__), a.dataset, step], check=True, env=env)

    s = summary(a.dataset)

    if outputs(a.dataset) != before:
        compact = dict(dataset=s["dataset"], model=s["model"], cos=s["cos"], lam=s["lam"],
                       clusters={k: v and v["nlist"] for k, v in s["clusters"].items()},
                       probe_ceiling={k: v and v.get("recall") for k, v in s["probe_ceiling"].items()},
                       timing_cpu=s["timing_cpu"] and dict(total=s["timing_cpu"]["total_ms_p50"],
                                                           layers=s["timing_cpu"]["layer_ms_p50"]))
        c.append_result("spec_states", compact)

    print("summary", spec(a.dataset, "summary.json"), flush=True)


if __name__ == "__main__":
    main()
