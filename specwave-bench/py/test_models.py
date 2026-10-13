"""Unit tests of how the data scripts serve any model of the models list: each model's folders, settings, prompts,
pooling and layer summaries, and the embeddings that export_beir.py makes and resumes. They run on small synthetic data
with stand-in models. Run with `python3 test_models.py`."""
import contextlib
import io
import os
import pathlib
import sys
import tempfile
import types
import unittest
import zlib
from unittest import mock

sys.dont_write_bytecode = True
# Keeps every test away from the real data and results folders, even one that doesn't patch them.
SCRATCH = tempfile.TemporaryDirectory()
os.environ["SPECWAVE_DATA"] = os.environ["SPECWAVE_RESULTS"] = SCRATCH.name

import numpy as np  # noqa: E402
import torch  # noqa: E402

import common as c  # noqa: E402
import export_beir  # noqa: E402

# The width of the synthetic embeddings and states.
DIM = 8


def vec(text):
    """A unit vector made from the text alone."""
    v = np.random.default_rng(zlib.crc32(text.encode())).standard_normal(DIM)
    return v / np.linalg.norm(v)


def unit(ids):
    """A fixed vector of each token id, DIM values per id."""
    return torch.sin(ids.unsqueeze(-1).to(torch.float32) * torch.arange(1, DIM + 1, dtype=torch.float32))


class Batch(dict):
    """A stand-in for a tokenizer's batch, which can move to a device."""

    def to(self, dev):
        return Batch({k: v.to(dev) for k, v in self.items()})


class Tokenizer:
    """A stand-in for a Hugging Face tokenizer. Each word gets an id from its hash. A bidirectional model's tokenizer
    puts the CLS id 1 first, and a causal model's appends the end id 2. Pads are id 0, on the side `side`."""

    def __init__(self, causal, side="right"):
        self.causal, self.side = causal, side

    def ids(self, text, max_length=128):
        head, tail = ([], [2]) if self.causal else ([1], [])
        words = [3 + zlib.crc32(w.encode()) % 997 for w in text.split()]
        return head + words[:max_length - len(head) - len(tail)] + tail

    def __call__(self, texts, padding=False, truncation=True, max_length=128, return_tensors=None):
        rows = [self.ids(t, max_length) for t in texts]
        if not padding:
            return {"input_ids": rows}
        width = max(len(r) for r in rows)
        pads = [[0] * (width - len(r)) for r in rows]
        ids = [r + p if self.side == "right" else p + r for r, p in zip(rows, pads)]
        mask = [[1] * len(r) + p if self.side == "right" else p + [1] * len(r) for r, p in zip(rows, pads)]
        return Batch(input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask))


class Net(torch.nn.Module):
    """A stand-in for a transformers model of `layers` layers. Hidden state l of a token with id i is (l + 1)·u(i) + l,
    where u(i) is the fixed vector of the id. Every pad holds 1,000 instead, so a pooling that reads a pad shows."""

    def __init__(self, model_type, layers=3):
        super().__init__()
        self.config = types.SimpleNamespace(model_type=model_type)
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.count = layers

    def forward(self, input_ids, attention_mask, output_hidden_states=False):
        u = unit(input_ids)
        pad = (attention_mask == 0).unsqueeze(-1)
        states = tuple(torch.where(pad, torch.full_like(u, 1000.0), (k + 1) * u + k) for k in range(self.count + 1))
        return types.SimpleNamespace(hidden_states=states if output_hidden_states else None)


class SentenceModel:
    """A stand-in for a sentence-transformers model, a transformers model `net` with its tokenizer."""

    def __init__(self, net, tokenizer):
        self.tokenizer = tokenizer
        self.modules = [types.SimpleNamespace(auto_model=net)]

    def __getitem__(self, i):
        return self.modules[i]


def expected(tok, text, layer, causal):
    """The summary of hidden state `layer` of the stand-in model on `text`, the mean then the first or last token."""
    h = (layer + 1) * unit(torch.tensor(tok.ids(text))).numpy() + layer
    return np.concatenate([h.mean(0), h[-1] if causal else h[0]]), h


class Summaries(unittest.TestCase):
    TEXTS = ["a b c", "d", "e f g h i j", "k l", "m n o p", "q r s t u v w x"]

    def encode(self, model_type, side, pooling):
        """The summaries of layers 0 and 2 and the final embeddings that c.encode_states writes, in batches of 2."""
        tok = Tokenizer(causal=model_type in c.CAUSAL, side=side)
        st = SentenceModel(Net(model_type), tok)
        states = {layer: np.zeros((len(self.TEXTS), 2 * DIM), np.float32) for layer in (0, 2)}
        final = np.zeros((len(self.TEXTS), DIM), np.float32)
        lengths = c.encode_states(st, self.TEXTS, "q: ", [0, 2], states, final, pooling=pooling, batch=2)
        self.assertEqual(lengths.tolist(), [len(tok.ids("q: " + t)) for t in self.TEXTS])
        return tok, states, final

    def check(self, model_type, side, pooling):
        tok, states, final = self.encode(model_type, side, pooling)
        causal = model_type in c.CAUSAL
        for i, text in enumerate(self.TEXTS):
            for layer in (0, 2):
                want, _ = expected(tok, "q: " + text, layer, causal)
                self.assertTrue(np.allclose(states[layer][i], want, atol=1e-5), (model_type, side, layer, i))
            _, h = expected(tok, "q: " + text, 3, causal)
            e = {"cls": h[0], "lasttoken": h[-1], "mean": h.mean(0)}[pooling]
            self.assertTrue(np.allclose(final[i], e / np.linalg.norm(e), atol=1e-5), (model_type, pooling, i))

    def test_a_bidirectional_model_summarizes_with_its_first_token(self):
        for pooling in ("cls", "mean"):
            self.check("bert", "right", pooling)

    def test_a_causal_model_summarizes_with_its_last_token_on_either_padding_side(self):
        for side in ("right", "left"):
            self.check("qwen3", side, "lasttoken")

    def test_the_summary_token_differs_from_the_first_one_for_a_causal_model(self):
        tok, states, _ = self.encode("qwen3", "right", "lasttoken")
        h = expected(tok, "q: " + self.TEXTS[2], 2, True)[1]
        self.assertFalse(np.allclose(states[2][2][DIM:], h[0], atol=1e-3))

    def test_an_unknown_pooling_is_refused(self):
        with self.assertRaises(ValueError):
            self.encode("bert", "right", "max")


class Layers(unittest.TestCase):
    def test_the_layers_of_each_kind_of_model(self):
        bert = types.SimpleNamespace(embeddings="e", encoder=types.SimpleNamespace(layer=["l1", "l2"]))
        nomic = types.SimpleNamespace(embeddings="e", layers=["l1"])
        qwen = types.SimpleNamespace(embed_tokens="t", layers=["l1", "l2", "l3"])
        self.assertEqual(c.layer_modules(bert), ("e", ["l1", "l2"]))
        self.assertEqual(c.layer_modules(nomic), ("e", ["l1"]))
        self.assertEqual(c.layer_modules(qwen), ("t", ["l1", "l2", "l3"]))


class Folders(unittest.TestCase):
    def roots(self, name, dataset, **env):
        """The data and results folders that c.use_model picks with only `env` of the two variables set."""
        keep = {k: v for k, v in os.environ.items() if k not in ("SPECWAVE_DATA", "SPECWAVE_RESULTS")}
        with mock.patch.dict(os.environ, dict(keep, **env), clear=True), mock.patch.object(c, "DATA", c.DATA), \
                mock.patch.object(c, "RESULTS", c.RESULTS):
            short = c.use_model(name, dataset)
            return short, c.DATA, c.RESULTS

    def test_the_own_model_uses_the_shared_folders_and_any_other_its_own(self):
        self.assertEqual(self.roots("arctic", "fiqa"), ("arctic", c.SHARED_DATA, c.SHARED_RESULTS))
        self.assertEqual(self.roots("minilm", "quora"), ("minilm", c.SHARED_DATA, c.SHARED_RESULTS))
        for name, dataset in (("BAAI/bge-small-en-v1.5", "fiqa"), ("arctic", "quora")):
            short = c.listed(name)["name"]
            self.assertEqual(self.roots(name, dataset), (short, os.path.join(c.SHARED_DATA, "models", short),
                                                         os.path.join(c.SHARED_RESULTS, "models", short)))

    def test_the_variables_name_other_folders(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.roots("e5-base", "fiqa", SPECWAVE_DATA=d), (
                "e5-base", d, os.path.join(c.SHARED_RESULTS, "models", "e5-base")))
            self.assertEqual(self.roots("e5-base", "fiqa", SPECWAVE_RESULTS=d)[2], d)

    def test_another_model_can_t_use_the_shared_folders(self):
        with self.assertRaises(ValueError):
            self.roots("bge-small", "fiqa", SPECWAVE_DATA=c.SHARED_DATA)
        with self.assertRaises(ValueError):
            self.roots("bge-small", "fiqa", SPECWAVE_RESULTS=os.path.join(c.SHARED_RESULTS, "."))
        with self.assertRaises(KeyError):
            self.roots("org/other", "fiqa")


class Settings(unittest.TestCase):
    def hub(self, hub, repo, config, readme=""):
        model = os.path.join(hub, "models--" + repo.replace("/", "--"))
        folder = os.path.join(model, "snapshots", "c0")
        os.makedirs(os.path.join(model, "refs"), exist_ok=True)
        os.makedirs(folder, exist_ok=True)
        pathlib.Path(model, "refs", "main").write_text("c0")
        pathlib.Path(folder, "README.md").write_text(readme)
        c.write_json(os.path.join(folder, "config.json"), config)

    def test_settings_come_from_the_list_and_the_snapshot(self):
        with tempfile.TemporaryDirectory() as hub:
            listed = [dict(name="x", repo="org/x", layers=[0, 2, 3], query_prompt="search_query: ",
                           passage_prompt="search_document: ", prompts_from="README.md"),
                      dict(name="y", repo="org/y", layers=[0, 5])]
            c.write_json(os.path.join(hub, "models.json"), listed)
            self.hub(hub, "org/x", dict(model_type="nomic_bert", num_hidden_layers=4, hidden_size=16),
                     "Use search_query: and search_document: in front.")
            self.hub(hub, "org/y", dict(model_type="qwen3", num_hidden_layers=6, hidden_size=32))
            with mock.patch.object(c, "HUB", hub), mock.patch.object(c, "MODEL_LIST", os.path.join(hub, "models.json")):
                self.assertEqual(c.model("org/x"), dict(
                    name="x", repo="org/x", layers=4, dim=16, fire=[0, 2, 3], query_prompt="search_query: ",
                    passage_prompt="search_document: ", model_type="nomic_bert"))
                y = c.model("y")
                self.assertEqual((y["layers"], y["dim"], y["fire"], y["model_type"]), (6, 32, [0, 5], "qwen3"))
                self.assertEqual((y["query_prompt"], y["passage_prompt"]), ("", ""))

    def test_every_cached_model_of_the_list(self):
        """Reads the real list and every snapshot of it in the cache. Each model's layers that may fire stop before its
        last layer, whose state is the embedding itself."""
        sizes = {"arctic": (12, 768), "minilm": (6, 384), "bge-small": (12, 384), "bge-base": (12, 768),
                 "e5-base": (12, 768), "bge-large": (24, 1024), "mxbai-large": (24, 1024), "nomic": (12, 768),
                 "qwen3-0.6b": (28, 1024)}
        prompts = {"arctic": ("Represent this sentence for searching relevant passages: ", ""), "minilm": ("", ""),
                   "e5-base": ("query: ", "passage: "), "nomic": ("search_query: ", "search_document: "),
                   "qwen3-0.6b": ("Instruct: Given a web search query, retrieve relevant passages that answer the "
                                  "query\nQuery:", "")}
        seen = 0
        for entry in c.listed_models():
            if not os.path.isdir(os.path.join(c.HUB, "models--" + entry["repo"].replace("/", "--"))):
                continue
            m = c.model(entry["name"])
            seen += 1
            self.assertLess(max(m["fire"]), m["layers"], entry["name"])
            if entry["name"] in sizes:
                self.assertEqual((m["layers"], m["dim"]), sizes[entry["name"]], entry["name"])
            if entry["name"] in prompts:
                self.assertEqual((m["query_prompt"], m["passage_prompt"]), prompts[entry["name"]], entry["name"])
        if not seen:
            self.skipTest("no model of the list is in the cache")


class Embedder:
    """A stand-in for a sentence-transformers model that embeds `prompt + text` as vec(prompt + text). It stops with
    RuntimeError at its call number `stop`, if one is given."""

    max_seq_length = 512

    def __init__(self, stop=None):
        self.calls, self.stop = [], stop

    def encode(self, texts, prompt=None, **kwargs):
        if self.stop is not None and len(self.calls) + 1 == self.stop:
            raise RuntimeError("stopped")
        self.calls.append((prompt, len(texts)))
        return np.array([vec(prompt + t) for t in texts], dtype=np.float32)

    def parameters(self):
        return iter([torch.zeros(1)])

    def get_embedding_dimension(self):
        return DIM


def listed_model(name):
    """Stands in for c.model, a model with a query and a passage prompt."""
    return dict(name=name, repo=f"org/{name}", layers=12, dim=DIM, fire=[0, 3], query_prompt="query: ",
                passage_prompt="passage: ", model_type="bert")


class Embeddings(unittest.TestCase):
    TEXTS = [f"text {i} " + "w " * (i % 5) for i in range(10)]

    def setUp(self):
        self.dir = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(mock.patch.object(c, "model", listed_model))
        self.enterContext(mock.patch.object(export_beir, "PART", 3))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.dst = os.path.join(self.dir, "x_corpus.fbin")

    def table(self):
        return np.asarray(c.open_table(self.dst))

    def test_a_stopped_run_makes_only_the_missing_parts(self):
        first = Embedder(stop=3)
        with self.assertRaises(RuntimeError):
            export_beir.embed_table(self.dst, first, "m", self.TEXTS, "passage: ")
        self.assertEqual(first.calls, [("passage: ", 3), ("passage: ", 3)])
        self.assertFalse(os.path.exists(self.dst))
        again = Embedder()
        export_beir.embed_table(self.dst, again, "m", self.TEXTS, "passage: ")
        self.assertEqual(again.calls, [("passage: ", 3), ("passage: ", 1)])
        want = np.array([vec("passage: " + t) for t in self.TEXTS], dtype=np.float32)
        self.assertTrue(np.array_equal(self.table(), want))
        self.assertFalse(os.path.exists(self.dst + ".parts"))
        info = c.read_json(os.path.join(self.dir, "x_corpus.embed.json"))
        self.assertEqual((info["model"], info["prompt"], info["rows"], info["parts"]), ("org/m", "passage: ", 10, 4))
        self.assertEqual(info["max_tokens"], 512)

    def test_parts_made_with_other_settings_are_made_again(self):
        with self.assertRaises(RuntimeError):
            export_beir.embed_table(self.dst, Embedder(stop=3), "m", self.TEXTS, "query: ")
        again = Embedder()
        export_beir.embed_table(self.dst, again, "m", self.TEXTS, "passage: ")
        self.assertEqual(len(again.calls), 4)
        want = np.array([vec("passage: " + t) for t in self.TEXTS], dtype=np.float32)
        self.assertTrue(np.array_equal(self.table(), want))

    def test_a_finished_table_is_kept(self):
        export_beir.embed_table(self.dst, Embedder(), "m", self.TEXTS, "passage: ")
        again = Embedder()
        export_beir.embed_table(self.dst, again, "m", self.TEXTS, "other: ")
        self.assertEqual(again.calls, [])


class Export(unittest.TestCase):
    """The export of a tiny FiQA (financial opinion mining and question answering) dataset for a model of its own, with
    a stand-in embedder, so no model runs."""

    def setUp(self):
        self.dir = self.enterContext(tempfile.TemporaryDirectory())
        self.shared, self.own = os.path.join(self.dir, "data"), os.path.join(self.dir, "data", "models", "m")
        self.cache = os.path.join(self.dir, "cache")
        os.makedirs(self.cache)
        self.docs = [" ".join(f"d{i}w{j}" for j in range(4 + i % 30)) for i in range(200)]
        self.queries = [f"query {i} " + " ".join(f"q{i}w{j}" for j in range(3 + i % 7)) for i in range(60)]
        corpus = ([str(i) for i in range(200)], self.docs)
        queries = ([f"q{i}" for i in range(60)], self.queries)
        tsv = "query-id\tcorpus-id\tscore\n" + "".join(f"q{i}\t{i}\t1\n" for i in range(50, 60))
        pathlib.Path(self.cache, "fiqa_test.tsv").write_text(tsv)
        self.embedder = Embedder()
        self.enterContext(mock.patch.object(c, "SHARED_DATA", self.shared))
        self.enterContext(mock.patch.object(c, "DATA", self.own))
        self.enterContext(mock.patch.object(c, "model", listed_model))
        self.enterContext(mock.patch.object(c, "beir", lambda ds, part: corpus if part == "corpus" else queries))
        self.enterContext(mock.patch.object(export_beir, "load", lambda name: self.embedder))
        self.enterContext(mock.patch.object(export_beir, "TEST_SIZE", 10))
        self.enterContext(mock.patch.object(export_beir, "GT_K", 20))
        self.enterContext(mock.patch.object(export_beir, "CHECK_ROWS", 6))
        self.enterContext(mock.patch.object(export_beir, "PART", 64))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def test_the_corpus_gets_the_passage_prompt_and_the_queries_the_query_prompt(self):
        export_beir.export("fiqa", self.cache, "m")
        corpus = np.asarray(c.open_table(c.path("fiqa_corpus.fbin")))
        queries = np.asarray(c.open_table(c.path("fiqa_queries.fbin")))
        self.assertTrue(np.array_equal(corpus, np.array([vec("passage: " + t) for t in self.docs], np.float32)))
        self.assertTrue(np.array_equal(queries, np.array([vec("query: " + t) for t in self.queries], np.float32)))
        report = c.read_json(c.path("fiqa_export.json"))
        self.assertEqual((report["query_prompt"], report["passage_prompt"]), ("query: ", "passage: "))
        self.assertEqual(sorted(report["embedded_here"]), ["corpus", "queries"])
        self.assertEqual(report["embedded_here"]["corpus"]["prompt"], "passage: ")
        for part in ("corpus", "queries", "test"):
            self.assertGreater(report["checks"][part]["min_cos"], 0.9999, part)
        # The check embeds its samples with each part's prompt too.
        self.assertEqual({p for p, _ in self.embedder.calls}, {"passage: ", "query: "})

    def test_another_model_links_the_shared_texts_and_never_writes_over_them(self):
        os.makedirs(self.shared)
        with mock.patch.object(c, "DATA", self.shared):
            export_beir.export("fiqa", self.cache, "m")
        before = {f: pathlib.Path(self.shared, f).read_bytes() for f in os.listdir(self.shared) if "." in f}
        self.embedder.calls.clear()
        export_beir.export("fiqa", self.cache, "m")
        for part in export_beir.TEXTS:
            self.assertEqual(os.readlink(c.path(f"fiqa_{part}")), os.path.join(self.shared, f"fiqa_{part}"), part)
        for part in ("corpus.fbin", "queries.fbin", "test.fbin", "test.gt.ibin", "export.json"):
            self.assertFalse(os.path.islink(c.path(f"fiqa_{part}")), part)
        after = {f: pathlib.Path(self.shared, f).read_bytes() for f in os.listdir(self.shared) if "." in f}
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
