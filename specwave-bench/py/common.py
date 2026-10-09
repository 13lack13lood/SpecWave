"""Helpers shared by the SpecWave data scripts.

Vectors are stored as .fbin files and integers as .ibin files. Both hold an int32 row count and an int32 width, then
the rows as little-endian float32 or int32 values. Every output is first written under a temporary name and renamed
once it is complete, so a killed job never leaves a partial file under the final name and a rerun can skip the
outputs that exist. An output made from the evaluation queries records their id, so a rerun can also tell when it
belongs to other evaluation queries. Texts are JSON (JavaScript Object Notation) lines, one record per line."""
import filecmp
import glob
import hashlib
import json
import os
import shutil
import subprocess
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "data")
RESULTS = os.path.join(REPO, "results")
# The Hugging Face (HF) cache, where the models and the BeIR (Benchmarking Information Retrieval) parquet files are.
HUB = os.environ.get("HF_HUB_CACHE", os.path.join(os.environ.get("HF_HOME", "~/.cache/huggingface"), "hub"))
HUB = os.path.expanduser(HUB)

# The embedding model of each dataset. FiQA (financial opinion mining and question answering) uses arctic, which
# pools the CLS (classification) token and prefixes queries with a prompt. Quora uses MiniLM, which pools the mean of
# the tokens and has no prompt.
MODELS = {
    "arctic": dict(repo="Snowflake/snowflake-arctic-embed-m-v1.5", layers=12, dim=768, cls=True, prompt="query"),
    "minilm": dict(repo="sentence-transformers/all-MiniLM-L6-v2", layers=6, dim=384, cls=False, prompt=None),
}
DATASETS = {"fiqa": "arctic", "quora": "minilm"}


def path(*parts):
    return os.path.join(DATA, *parts)


def table_shape(file):
    """The (rows, width) of an .fbin or .ibin file, after checking that its size matches its header."""
    head = np.fromfile(file, dtype="<i4", count=2)

    if len(head) != 2 or head[0] < 0 or head[1] <= 0:
        raise ValueError(f"{file} has no valid header")

    n, d = int(head[0]), int(head[1])

    if os.path.getsize(file) != 8 + 4 * n * d:
        raise ValueError(f"{file} doesn't hold the {n} × {d} values its header says")

    return n, d


def open_table(file, dtype="<f4"):
    """Maps an .fbin (float32) or .ibin (int32) file read-only as an n × d array."""
    n, d = table_shape(file)
    return np.memmap(file, dtype=dtype, mode="r", offset=8, shape=(n, d))


def new_table(file, n, d, dtype="<f4"):
    """Creates the temporary file of an n × d table and maps it for writing. Call `finish(file)` once it is full."""
    os.makedirs(os.path.dirname(file), exist_ok=True)
    part = file + ".part"

    with open(part, "wb") as f:
        np.array([n, d], dtype="<i4").tofile(f)
        f.truncate(8 + 4 * n * d)

    return np.memmap(part, dtype=dtype, mode="r+", offset=8, shape=(n, d))


def finish(file, table=None):
    """Renames the temporary file of `file` to its final name, after flushing its map."""
    if table is not None:
        table.flush()

    os.replace(file + ".part", file)


def write_table(file, rows, dtype="<f4"):
    """Writes a small array as an .fbin or .ibin file in one go."""
    rows = np.ascontiguousarray(rows, dtype=dtype)
    out = new_table(file, rows.shape[0], rows.shape[1], dtype)
    out[:] = rows
    finish(file, out)


def write_jsonl(file, records):
    os.makedirs(os.path.dirname(file), exist_ok=True)

    with open(file + ".part", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    os.replace(file + ".part", file)


def read_jsonl(file):
    with open(file, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_json(file, obj):
    os.makedirs(os.path.dirname(file), exist_ok=True)

    with open(file + ".part", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)

    os.replace(file + ".part", file)


def read_json(file):
    with open(file, encoding="utf-8") as f:
        return json.load(f)


def same_bytes(a, b):
    """Whether files `a` and `b` both exist and hold the same bytes."""
    if not (os.path.exists(a) and os.path.exists(b)):
        return False

    filecmp.clear_cache()
    return filecmp.cmp(a, b, shallow=False)


def copy_file(src, dst):
    """Copies file `src` to `dst` byte for byte, through a temporary file."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst + ".part")
    os.replace(dst + ".part", dst)


def evaluation_id(dataset):
    """The id of a dataset's evaluation queries, {"test_rows_sha256": hash}, where the hash is the SHA-256 (Secure Hash
    Algorithm) hash of their rows in <ds>_queries as little-endian int64 values, read from <ds>_test.meta.json. Every
    output made from the evaluation queries records it in its .json file."""
    rows = read_json(path(f"{dataset}_test.meta.json"))["query_rows"]
    return {"test_rows_sha256": hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()}


def up_to_date(files, info_file, **inputs):
    """Whether all of `files` and the .json file `info_file` exist and `info_file` records the same value for each of
    `inputs`, eg. the id of the evaluation queries. A step keeps such outputs and makes the others again."""
    if not all(os.path.exists(f) for f in [*files, info_file]):
        return False

    info = read_json(info_file)
    return all(info.get(k) == v for k, v in inputs.items())


def beir(dataset, part):
    """The ids and texts of a BeIR dataset's "corpus" or "queries" parquet file, in file order."""
    import pyarrow.parquet as pq

    files = glob.glob(f"{HUB}/datasets--BeIR--{dataset}/snapshots/*/{part}/*.parquet")

    if len(files) != 1:
        raise FileNotFoundError(f"expected one {dataset} {part} parquet file in {HUB}, found {len(files)}")

    t = pq.read_table(files[0], columns=["_id", "text"])
    return t.column("_id").to_pylist(), t.column("text").to_pylist()


def twins(base, chunk=100_000):
    """The rows of `base` that have a byte-identical twin, in row order, and the first row of each one's set of twins.

    A matrix product can round the scores of identical rows differently, depending on where they sit in the matrix,
    so the nearest-neighbor search gives each set of twins one shared score."""
    weights = np.random.default_rng(0).integers(1, 2**63, size=base.shape[1], dtype=np.uint64)
    bits = (np.ascontiguousarray(base[s:s + chunk]).view(np.uint32) for s in range(0, len(base), chunk))
    hashes = np.concatenate([(b.astype(np.uint64) * weights).sum(1) for b in bits])
    order = np.argsort(hashes, kind="stable")
    starts = np.flatnonzero(np.r_[True, hashes[order][1:] != hashes[order][:-1]])
    rows, firsts = [], []

    for a, b in zip(starts, np.r_[starts[1:], len(order)]):
        if b - a < 2:
            continue
        members = np.sort(order[a:b])
        first = np.asarray(base[members[0]]).view(np.uint32)
        same = [int(r) for r in members if np.array_equal(np.asarray(base[r]).view(np.uint32), first)]
        if len(same) > 1:
            rows += same
            firsts += [same[0]] * len(same)

    by_row = np.argsort(rows, kind="stable")
    return np.array(rows, dtype=np.int64)[by_row], np.array(firsts, dtype=np.int64)[by_row]


def twin_scores(q, base, found):
    """The rows with twins, the column of each one's first twin, and every query's score of each first twin."""
    rows, firsts = found
    unique, col = np.unique(firsts, return_inverse=True)
    shared = q @ normalize(np.asarray(base[unique], dtype=np.float64)).T if len(unique) else np.zeros((len(q), 0))
    return rows, col, shared


def exact_topk(queries, base, k, query_chunk=256, base_chunk=100_000, found=None):
    """The k rows of `base` nearest each query by cosine, nearest first, and their scores.

    Scores are computed in float64 on normalized copies of both sides, and ties go to the lower row. `base` is read
    `base_chunk` rows at a time, so it can be a memory map larger than the memory a job may use. `found` is the result
    of `twins(base)`, computed here when it isn't given."""
    q = normalize(np.asarray(queries, dtype=np.float64))
    n = len(base)
    k = min(k, n)
    rows, col, shared = twin_scores(q, base, twins(base) if found is None else found)
    best_s = np.full((len(q), k), -np.inf)
    best_i = np.full((len(q), k), n, dtype=np.int64)

    for b0 in range(0, n, base_chunk):
        x = normalize(np.asarray(base[b0:b0 + base_chunk], dtype=np.float64))
        m = len(x)
        lo, hi = np.searchsorted(rows, [b0, b0 + m])
        for q0 in range(0, len(q), query_chunk):
            s = q[q0:q0 + query_chunk] @ x.T
            s[:, rows[lo:hi] - b0] = shared[q0:q0 + len(s)][:, col[lo:hi]]
            kth = np.partition(s, m - k, axis=1)[:, m - k] if m > k else np.full(len(s), -np.inf)
            for r in range(len(s)):
                pick = np.flatnonzero(s[r] >= kth[r])
                cand_s = np.concatenate([best_s[q0 + r], s[r, pick]])
                cand_i = np.concatenate([best_i[q0 + r], b0 + pick])
                keep = np.lexsort((cand_i, -cand_s))[:k]
                best_s[q0 + r], best_i[q0 + r] = cand_s[keep], cand_i[keep]

    return best_i, best_s


def naive_topk(query, base, k, found=None, chunk=100_000):
    """The same as `exact_topk` for one query, by sorting every score. It checks `exact_topk` on a few queries."""
    q = normalize(np.asarray(query, dtype=np.float64)[None, :])
    s = np.concatenate([normalize(np.asarray(base[i:i + chunk], dtype=np.float64)) @ q[0]
                        for i in range(0, len(base), chunk)])
    rows, col, shared = twin_scores(q, base, twins(base) if found is None else found)
    s[rows] = shared[0, col]
    return np.lexsort((np.arange(len(s)), -s))[:k]


def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def mean_cos(a, b):
    """The mean cosine between matching rows of `a` and `b`."""
    return float(np.mean(np.sum(normalize(np.asarray(a, np.float64)) * normalize(np.asarray(b, np.float64)), 1)))


class Ridge:
    """Ridge regression from X to Y. X is standardized, Y is centered, and the weights W solve
    (Xsᵀ·Xs + λ·n·I)·W = Xsᵀ·(Y − Ȳ) in float64, where Xs is the standardized X, n its row count, λ the penalty, I the
    identity matrix and Ȳ the mean row of Y. The products are kept, so each penalty costs one solve."""

    def __init__(self, X, Y):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-6
        self.ybar = Y.mean(0)
        xs = (X - self.mu) / self.sd
        self.n = len(xs)
        self.gram = xs.T @ xs
        self.cross = xs.T @ (Y - self.ybar)

    def weights(self, lam):
        a = self.gram + lam * self.n * np.eye(len(self.gram))
        return np.linalg.solve(a, self.cross)

    def predict(self, Z, w):
        return ((np.asarray(Z, dtype=np.float64) - self.mu) / self.sd) @ w + self.ybar


def choose_lambda(X, Y, lams, split=0.85, seed=None):
    """Fits on the first 85% of the pairs for each penalty, and returns the one whose normalized predictions have the
    highest mean cosine with the last 15%, with every penalty's cosine. A tie keeps the earlier penalty. With a seed,
    the pairs are first put in the order of default_rng(seed).permutation, for pairs whose stored order isn't random."""
    if seed is not None:
        order = np.random.default_rng(seed).permutation(len(X))
        X, Y = X[order], Y[order]

    cut = int(split * len(X))
    fit = Ridge(X[:cut], Y[:cut])
    val = {lam: mean_cos(fit.predict(X[cut:], fit.weights(lam)), Y[cut:]) for lam in lams}
    best = lams[0]

    for lam in lams[1:]:
        if val[lam] > val[best]:
            best = lam

    return best, val


def openings(texts, n, seed, skip=()):
    """Draws `n` documents with text, none of them in `skip`, and keeps the first 6 to 24 of each one's words, a
    uniform draw per document. Returns the document rows, the words kept and the opening texts, in draw order."""
    rng = np.random.default_rng(seed)
    skip = set(skip)
    pool = np.array([i for i, t in enumerate(texts) if t.split() and t not in skip])

    if len(pool) < n:
        raise ValueError(f"only {len(pool)} documents can give openings, {n} wanted")

    docs = rng.choice(pool, n, replace=False)
    words = rng.integers(6, 25, size=n)
    out = [" ".join(texts[d].split()[:w]) for d, w in zip(docs, words)]
    kept = np.array([len(t.split()) for t in out])
    return docs, kept, out


def device():
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(name, dev):
    """The sentence-transformers model `name` ("arctic" or "minilm") from the local cache, on device `dev`."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(MODELS[name]["repo"], device=dev)
    st.eval()
    return st


def prompt_of(st, name):
    """The text a query starts with for model `name`, eg. arctic's "Represent this sentence for searching ..."."""
    which = MODELS[name]["prompt"]
    return st.prompts.get(which, "") if which else ""


def encode_states(st, texts, prompt, layers, states, final=None, cls=True, batch=64, max_len=128):
    """Runs the model on `prompt + text` for every text and writes the pooled state of each layer in `layers` into
    `states[layer]` (n × 2d) and the normalized final embedding into `final` (n × d), in input order.

    Layer l is the hidden state after l transformer layers, and l = 0 is the embedding output. A pooled state is the
    mean over the attention mask, then the CLS vector. The final embedding pools the last layer like the model does,
    with the CLS vector (`cls`) or the masked mean. Batches hold texts of similar token counts, so little of each is
    padding. Returns each text's token count."""
    import torch

    tok, net = st.tokenizer, st[0].auto_model
    dev = next(net.parameters()).device
    full = [prompt + t for t in texts]
    lengths = [len(ids) for ids in tok(full, truncation=True, max_length=max_len)["input_ids"]]
    order = np.argsort(lengths, kind="stable")[::-1]

    with torch.inference_mode():
        for b, s in enumerate(range(0, len(order), batch)):
            idx = order[s:s + batch]
            enc = tok([full[i] for i in idx], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            enc = enc.to(dev)
            hs = net(**enc, output_hidden_states=True).hidden_states
            mask = enc["attention_mask"].unsqueeze(-1).to(hs[0].dtype)
            for layer in layers:
                h = hs[layer]
                mean = (h * mask).sum(1) / mask.sum(1)
                states[layer][idx] = torch.cat([mean, h[:, 0]], 1).float().cpu().numpy()
            if final is not None:
                h = hs[-1]
                e = h[:, 0] if cls else (h * mask).sum(1) / mask.sum(1)
                final[idx] = torch.nn.functional.normalize(e, dim=-1).float().cpu().numpy()
            if dev.type == "mps" and b % 20 == 0:
                torch.mps.empty_cache()

    if dev.type == "mps":
        torch.mps.empty_cache()

    return np.array(lengths)


def git_rev():
    try:
        rev = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain"], capture_output=True, text=True)
        return rev.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except OSError:
        return "unknown"


def machine():
    """The processor, core counts and system version, from sysctl on macOS."""
    out = {}

    for key in ("machdep.cpu.brand_string", "hw.perflevel0.physicalcpu", "hw.perflevel1.physicalcpu",
                "hw.memsize", "kern.osproductversion"):
        r = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True)
        out[key] = r.stdout.strip()

    return out


def append_result(name, record):
    """Appends one JSON line to results/<name>.jsonl, with the date, the git revision and the machine."""
    os.makedirs(RESULTS, exist_ok=True)
    line = dict(date=time.strftime("%Y-%m-%dT%H:%M:%S"), git=git_rev(), machine=machine(), **record)

    with open(os.path.join(RESULTS, name + ".jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
