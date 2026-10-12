"""Writes reference outputs of the models in the models list, for the Rust encoder's checks against
sentence-transformers and transformers, on FiQA (financial opinion mining and question answering) texts.

For each model it takes the first 50 evaluation queries of data/fiqa_test.jsonl and 50 documents of
data/fiqa_corpus.jsonl, drawn with seed 5 from those with text. sentence-transformers embeds them with the model's
prompts, and transformers gives every layer's hidden states for the same tokens. The model runs on the processor in
float32, one text at a time, so no text is padded. sentence-transformers would otherwise run a model in the precision
that its snapshot stores, eg. float16 for mxbai-embed-large-v1.

It writes these files under data/encoders/<name>/, for set queries or passages.
- <set>.jsonl holds the texts as {row, text}, with each text's row in its source file.
- <set>.ids.ibin holds every text's token ids, one text after another, and <set>.lengths.ibin each text's count.
- <set>.st.fbin holds sentence-transformers' embeddings as the model gives them, so not always normalized.
- <set>.L<l>.fbin holds concat(mean over the tokens, first token) of hidden state l, for l = 0 to L, where l = 0 is
  the embedding output and L the last layer.
- queries.tokens.L<l>.fbin holds every query token's hidden state l, one query after another.
- meta.json holds the model's settings, its prompts, how far the stored outputs agree among themselves, and the
  versions. For a model stored in another precision it also holds how close sentence-transformers' default run comes.

Usage: encoder_refs.py [NAME ...]

With no name it writes every model of the list whose snapshot is in the cache. A model whose meta.json exists is
skipped, so delete its folder to make it again. It never goes online."""
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Keeps the repo free of compiled bytecode folders.
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import common as c  # noqa: E402

QUERIES = 50
PASSAGES = 50
PASSAGE_SEED = 5
THREADS = 4


def texts():
    """The query and passage texts, each as (row, text), in their order in the files."""
    queries = c.read_jsonl(c.path("fiqa_test.jsonl"))[:QUERIES]
    corpus = c.read_jsonl(c.path("fiqa_corpus.jsonl"))
    pool = [i for i, r in enumerate(corpus) if r["text"].strip()]
    rows = sorted(np.random.default_rng(PASSAGE_SEED).choice(pool, PASSAGES, replace=False).tolist())
    return {
        "queries": [(r["row"], r["text"]) for r in queries],
        "passages": [(row, corpus[row]["text"]) for row in rows],
    }


def load(repo, dtype=None):
    """The sentence-transformers model `repo` on the processor, in `dtype`, else in the precision it would pick."""
    import torch
    from sentence_transformers import SentenceTransformer

    kw = dict(model_kwargs={"dtype": dtype}) if dtype is not None else {}
    st = SentenceTransformer(repo, device="cpu", **kw)
    st.eval()
    torch.set_num_threads(THREADS)
    return st


def run(st, prompt, items, every_token):
    """Each text's token ids, sentence-transformers' embedding, the summary of each hidden state, and with
    `every_token` each hidden state of every token, plus how far the embedding is from the pooled last state."""
    import torch

    pooling = st[1].pooling_mode
    out = dict(ids=[], st=[], layers=None, tokens=None, pooled_gap=0.0)

    with torch.inference_mode():
        for _, text in items:
            enc = st.preprocess([prompt + text])
            feed = {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
            hs = st[0].auto_model(**feed, output_hidden_states=True).hidden_states
            e = st.encode([text], prompt=prompt, batch_size=1, convert_to_numpy=True, normalize_embeddings=False)
            states = [h[0].float().numpy() for h in hs]
            if out["layers"] is None:
                out["layers"] = [[] for _ in states]
                out["tokens"] = [[] for _ in states]
            for layer, h in enumerate(states):
                out["layers"][layer].append(np.concatenate([h.mean(0), h[0]]))
                if every_token:
                    out["tokens"][layer].append(h)
            last = states[-1][0] if pooling == "cls" else states[-1].mean(0)
            out["pooled_gap"] = max(out["pooled_gap"], float(1 - cosine(last, e[0])))
            out["ids"].append(enc["input_ids"][0].numpy())
            out["st"].append(e[0])

    return out


def cosine(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def write(folder, name, rows, dtype="<f4"):
    c.write_table(os.path.join(folder, name), np.asarray(rows), dtype=dtype)


def refs(entry, sets):
    """Writes the reference files of the listed model `entry`."""
    import sentence_transformers
    import torch
    import transformers

    folder = c.path("encoders", entry["name"])
    meta_file = os.path.join(folder, "meta.json")

    if os.path.exists(meta_file):
        print(entry["name"], "done before, skipped", flush=True)
        return

    t = time.time()
    query, passage = c.prompts(entry)
    st = load(entry["repo"], torch.float32)
    dtype = str(next(st.parameters()).dtype)

    if dtype != "torch.float32":
        raise ValueError(f"{entry['name']} loaded in {dtype}, not float32")

    config = st[0].auto_model.config
    meta = dict(name=entry["name"], repo=entry["repo"], snapshot=os.path.basename(c.snapshot(entry["repo"])),
                query_prompt=query, passage_prompt=passage, prompts_from=entry.get("prompts_from"),
                model_type=config.model_type, layers=config.num_hidden_layers, dim=config.hidden_size,
                pooling=st[1].pooling_mode, modules=[type(m).__name__ for m in st],
                max_seq_length=st.max_seq_length, dtype=dtype, threads=THREADS,
                normalizer=str(st.tokenizer.backend_tokenizer.normalizer), versions=dict(
                    sentence_transformers=sentence_transformers.__version__, transformers=transformers.__version__,
                    torch=torch.__version__))
    results = {}

    for set_name, prompt in (("queries", query), ("passages", passage)):
        items = sets[set_name]
        out = run(st, prompt, items, every_token=set_name == "queries")
        results[set_name] = out
        c.write_jsonl(os.path.join(folder, f"{set_name}.jsonl"), ({"row": row, "text": text} for row, text in items))
        write(folder, f"{set_name}.ids.ibin", np.concatenate(out["ids"])[:, None], "<i4")
        write(folder, f"{set_name}.lengths.ibin", np.array([len(i) for i in out["ids"]])[:, None], "<i4")
        write(folder, f"{set_name}.st.fbin", np.stack(out["st"]))
        for layer, rows in enumerate(out["layers"]):
            write(folder, f"{set_name}.L{layer}.fbin", np.stack(rows))
        if set_name == "queries":
            for layer, states in enumerate(out["tokens"]):
                write(folder, f"queries.tokens.L{layer}.fbin", np.concatenate(states))
        lengths = [len(i) for i in out["ids"]]
        meta[set_name] = dict(texts=len(items), tokens_mean=float(np.mean(lengths)), tokens_max=int(max(lengths)),
                              at_max_seq_length=int(sum(n >= st.max_seq_length for n in lengths)),
                              pooled_last_vs_st_worst=out["pooled_gap"])

    del st
    default = load(entry["repo"])
    default_dtype = str(next(default.parameters()).dtype)

    if default_dtype != dtype:
        worst = {}
        for set_name, prompt in (("queries", query), ("passages", passage)):
            found = default.encode([t for _, t in sets[set_name]], prompt=prompt, batch_size=1,
                                   convert_to_numpy=True, normalize_embeddings=False)
            worst[set_name] = min(cosine(a, b) for a, b in zip(found.astype(np.float64), results[set_name]["st"]))
        meta["default_run"] = dict(dtype=default_dtype, min_cos_with_float32=worst)

    meta["seconds"] = round(time.time() - t, 1)
    c.write_json(meta_file, meta)
    print(entry["name"], f"{meta['seconds']:.0f}s", {k: meta[k] for k in ("layers", "dim", "pooling", "dtype")},
          meta.get("default_run", ""), flush=True)


def main():
    wanted = sys.argv[1:]
    entries = [e for e in c.listed_models() if not wanted or e["name"] in wanted or e["repo"] in wanted]

    if wanted and len(entries) != len(wanted):
        sys.exit(f"the models list has no model {', '.join(set(wanted) - {e['name'] for e in entries})}")

    sets = texts()

    for entry in entries:
        if not os.path.isdir(os.path.join(c.HUB, "models--" + entry["repo"].replace("/", "--"))):
            print(entry["name"], "isn't in the cache, skipped", flush=True)
            continue
        refs(entry, sets)


if __name__ == "__main__":
    main()
