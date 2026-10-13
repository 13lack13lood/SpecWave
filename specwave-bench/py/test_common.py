"""Unit tests of common.py, and of how export_beir.py and spec_states.py keep or remake their outputs, on small
synthetic data. Run with `python3 test_common.py`."""
import contextlib
import glob
import hashlib
import io
import os
import pathlib
import sys
import tempfile
import time
import unittest
import zlib
from unittest import mock

sys.dont_write_bytecode = True
# Keeps every test away from the real data and results folders, even one that doesn't patch them.
SCRATCH = tempfile.TemporaryDirectory()
os.environ["SPECWAVE_DATA"] = os.environ["SPECWAVE_RESULTS"] = SCRATCH.name

import numpy as np  # noqa: E402

import common as c  # noqa: E402
import export_beir  # noqa: E402
import spec_states  # noqa: E402

# The width of the synthetic embeddings.
DIM = 8


def fake_model(name):
    """Stands in for c.model, with DIM dimensions, no prompts, and arctic's layers that may fire cut to 0 and 3."""
    layers = {"arctic": 12, "minilm": 6}.get(name, 12)
    return dict(name=name, repo=f"org/{name}", layers=layers, dim=DIM, fire=[0, 3] if name != "minilm" else [0, 1],
                query_prompt="", passage_prompt="", model_type="bert")


class Tables(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            x = np.random.default_rng(1).standard_normal((7, 5)).astype(np.float32)
            ids = np.arange(21, dtype=np.int32).reshape(7, 3)
            c.write_table(os.path.join(d, "x.fbin"), x)
            c.write_table(os.path.join(d, "i.ibin"), ids, dtype="<i4")
            self.assertTrue(np.array_equal(c.open_table(os.path.join(d, "x.fbin")), x))
            self.assertTrue(np.array_equal(c.open_table(os.path.join(d, "i.ibin"), dtype="<i4"), ids))
            self.assertEqual(sorted(os.listdir(d)), ["i.ibin", "x.fbin"])

    def test_header(self):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "x.fbin")
            c.write_table(f, np.ones((4, 3), np.float32))
            with open(f, "r+b") as h:
                h.truncate(8 + 4 * 11)
            with self.assertRaises(ValueError):
                c.open_table(f)
            with open(f, "wb") as h:
                h.write(b"\x01\x00")
            with self.assertRaises(ValueError):
                c.table_shape(f)

    def test_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            recs = [{"row": 0, "id": "7", "text": "two\nlines, é and \u2028"}, {"row": 1, "id": "9", "text": ""}]
            c.write_jsonl(os.path.join(d, "t.jsonl"), recs)
            self.assertEqual(c.read_jsonl(os.path.join(d, "t.jsonl")), recs)


class TopK(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(3)
        base = rng.standard_normal((500, 8)).astype(np.float32)
        # Exact duplicates tie, so their order must follow the row.
        base[[40, 41, 300, 499]] = base[7]
        base[[100, 200]] = base[9]
        q = np.concatenate([rng.standard_normal((20, 8)), base[[7, 9]]]).astype(np.float32)
        return q, base

    def test_matches_full_sort(self):
        q, base = self.data()
        for k, base_chunk, query_chunk in ((10, 64, 8), (100, 64, 5), (7, 1000, 256), (500, 128, 3)):
            rows, scores = c.exact_topk(q, base, k, query_chunk=query_chunk, base_chunk=base_chunk)
            for i in range(len(q)):
                self.assertTrue(np.array_equal(rows[i], c.naive_topk(q[i], base, k)), (k, base_chunk, i))
            self.assertTrue(np.all(np.diff(scores, axis=1) <= 0))

    def test_ties_go_to_the_lower_row(self):
        q, base = self.data()
        for base_chunk in (7, 13, 50, 64, 499, 500, 1000):
            rows, scores = c.exact_topk(q[-2:], base, 5, base_chunk=base_chunk)
            self.assertEqual(rows[0].tolist(), [7, 40, 41, 300, 499], base_chunk)
            self.assertEqual(rows[1][:3].tolist(), [9, 100, 200], base_chunk)
            self.assertEqual(len(set(scores[0].tolist())), 1)

    def test_twins(self):
        _, base = self.data()
        rows, firsts = c.twins(base, chunk=64)
        self.assertEqual(rows.tolist(), [7, 9, 40, 41, 100, 200, 300, 499])
        self.assertEqual(firsts.tolist(), [7, 9, 7, 7, 9, 9, 7, 7])

    def test_k_above_rows(self):
        q, base = self.data()
        rows, _ = c.exact_topk(q[:2], base[:30], 100, base_chunk=7)
        self.assertEqual(rows.shape, (2, 30))
        self.assertEqual(sorted(rows[0].tolist()), list(range(30)))


class RidgeFit(unittest.TestCase):
    def test_matches_least_squares(self):
        rng = np.random.default_rng(5)
        X = rng.standard_normal((300, 12)) * rng.uniform(0.1, 3, 12) + rng.uniform(-1, 1, 12)
        Y = X @ rng.standard_normal((12, 4)) + 0.1 * rng.standard_normal((300, 4)) + 2
        lam = 0.05
        r = c.Ridge(X, Y)
        xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
        a = np.vstack([xs, np.sqrt(lam * len(X)) * np.eye(12)])
        b = np.vstack([Y - Y.mean(0), np.zeros((12, 4))])
        w = np.linalg.lstsq(a, b, rcond=None)[0]
        self.assertTrue(np.allclose(r.weights(lam), w, atol=1e-8))
        self.assertTrue(np.allclose(r.predict(X, w), xs @ w + Y.mean(0)))

    def test_choose_lambda_matches_an_independent_fit(self):
        rng = np.random.default_rng(6)
        X = rng.standard_normal((400, 30)) * rng.uniform(0.5, 2, 30) + rng.uniform(-1, 1, 30)
        Y = X @ rng.standard_normal((30, 3)) + 3 * rng.standard_normal((400, 3))
        # The last rows follow another map, like a block of another split, so the order of the pairs matters.
        Y[300:] = X[300:] @ rng.standard_normal((30, 3)) + 3 * rng.standard_normal((100, 3))
        lams = [1e-3, 1e-2, 1e-1, 1.0]
        for seed in (None, 13):
            order = np.arange(400) if seed is None else np.random.default_rng(seed).permutation(400)
            first, last = order[:340], order[340:]
            mu, sd, ybar = X[first].mean(0), X[first].std(0) + 1e-6, Y[first].mean(0)
            want = {}
            for lam in lams:
                a = np.vstack([(X[first] - mu) / sd, np.sqrt(lam * len(first)) * np.eye(30)])
                b = np.vstack([Y[first] - ybar, np.zeros((30, 3))])
                p = ((X[last] - mu) / sd) @ np.linalg.lstsq(a, b, rcond=None)[0] + ybar
                cos = np.sum(p * Y[last], 1) / np.linalg.norm(p, axis=1) / np.linalg.norm(Y[last], axis=1)
                want[lam] = float(cos.mean())
            best, val = c.choose_lambda(X, Y, lams, seed=seed)
            self.assertEqual(sorted(val), lams)
            for lam in lams:
                self.assertAlmostEqual(val[lam], want[lam], places=9, msg=(seed, lam))
            self.assertEqual(best, max(lams, key=lambda p: (want[p], -lams.index(p))), seed)


class Ids(unittest.TestCase):
    def test_the_evaluation_id_hashes_the_rows_in_their_order(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(c, "DATA", d):
            def id_of(rows):
                c.write_json(c.path("x_test.meta.json"), {"query_rows": rows})
                return c.evaluation_id("x")["test_rows_sha256"]

            first = id_of([3, 1, 2])
            self.assertEqual(first, hashlib.sha256(np.array([3, 1, 2], dtype="<i8").tobytes()).hexdigest())
            self.assertNotEqual(id_of([3, 1, 4]), first, "other rows of the same count")
            self.assertNotEqual(id_of([1, 2, 3]), first, "the same rows in another order")
            self.assertEqual(id_of([3, 1, 2]), first)

    def test_an_output_is_up_to_date_only_with_every_file_and_every_value(self):
        with tempfile.TemporaryDirectory() as d:
            out, info = os.path.join(d, "x.fbin"), os.path.join(d, "x.json")
            c.write_json(info, {"a": 1, "b": 2})
            self.assertFalse(c.up_to_date([out], info, a=1, b=2), "no output file")
            pathlib.Path(out).touch()
            self.assertTrue(c.up_to_date([out], info, a=1, b=2))
            self.assertFalse(c.up_to_date([out], info, a=1, b=3))
            self.assertFalse(c.up_to_date([out], info, a=0, b=2))
            self.assertFalse(c.up_to_date([out], os.path.join(d, "y.json"), a=1), "no .json file")


class ModelList(unittest.TestCase):
    def test_entries_by_short_name_or_repository(self):
        names = [e["name"] for e in c.listed_models()]
        self.assertEqual(names[:2], ["arctic", "minilm"])
        self.assertEqual(c.listed("BAAI/bge-small-en-v1.5"), c.listed("bge-small"))
        self.assertEqual(c.listed("arctic")["layers"], [0, 3, 6, 8, 9, 10])
        with self.assertRaises(KeyError):
            c.listed("org/other")

    def snapshot(self, hub, prompts=None, readme=""):
        """A snapshot of org/x in the cache folder `hub`, with these sentence-transformers prompts and model card."""
        model = os.path.join(hub, "models--org--x")
        folder = os.path.join(model, "snapshots", "c0")
        os.makedirs(folder, exist_ok=True)
        os.makedirs(os.path.join(model, "refs"), exist_ok=True)
        pathlib.Path(model, "refs", "main").write_text("c0\n")
        pathlib.Path(folder, "README.md").write_text(readme)
        settings = pathlib.Path(folder, "config_sentence_transformers.json")
        if prompts is None:
            settings.unlink(missing_ok=True)
        else:
            c.write_json(str(settings), {"prompts": prompts})
        return folder

    def test_prompts_come_from_the_snapshot_else_from_the_list(self):
        listed = dict(name="x", repo="org/x", layers=[0], query_prompt="query: ", passage_prompt="passage: ",
                      prompts_from="README.md")
        card = 'Start each text with "query: " or "passage: ".'
        with tempfile.TemporaryDirectory() as hub, mock.patch.object(c, "HUB", hub):
            folder = self.snapshot(hub, readme=card)
            self.assertEqual(c.snapshot("org/x"), folder)
            self.assertEqual(c.prompts(listed), ("query: ", "passage: "))
            self.snapshot(hub, {"query": "query: ", "document": "passage: "}, card)
            self.assertEqual(c.prompts(listed), ("query: ", "passage: "))

            own = dict(name="x", repo="org/x", layers=[0])
            for prompts, expected in (({"query": "q: ", "corpus": "c: ", "passage": "p: ", "document": "d: "},
                                       ("q: ", "d: ")), ({"corpus": "c: ", "passage": "p: "}, ("", "p: ")),
                                      ({}, ("", ""))):
                self.snapshot(hub, prompts)
                self.assertEqual(c.prompts(own), expected, prompts)

            self.snapshot(hub, {"query": "search: "}, card)
            with self.assertRaises(ValueError):
                c.prompts(listed)
            self.snapshot(hub, None, "no prompts here")
            with self.assertRaises(ValueError):
                c.prompts(listed)


class Openings(unittest.TestCase):
    def test_draw(self):
        texts = [" ".join(f"w{i}x{j}" for j in range(i % 40)) for i in range(300)]
        docs, words, out = c.openings(texts, 100, seed=9, skip={texts[5]})
        again = c.openings(texts, 100, seed=9, skip={texts[5]})
        self.assertTrue(np.array_equal(docs, again[0]) and out == again[2])
        self.assertEqual(len(set(docs.tolist())), 100)
        for d, w, t in zip(docs, words, out):
            n = len(texts[d].split())
            self.assertTrue(n > 0 and d != 5)
            self.assertTrue(texts[d].startswith(t) and len(t.split()) == w)
            self.assertTrue(6 <= w <= 24 or w == n < 6)

    def test_too_few(self):
        with self.assertRaises(ValueError):
            c.openings(["a b", "", "c"], 3, seed=1)


class Steps(unittest.TestCase):
    def main(self, *steps, child=False):
        """Runs spec_states.py's main with these steps. Returns the steps it started a process for, each with the
        marker it gave that process, and the steps it ran itself."""
        ran = []
        with tempfile.TemporaryDirectory() as d:
            env = {k: v for k, v in os.environ.items() if k != spec_states.CHILD}
            env.update(SPECWAVE_DATA=d, SPECWAVE_RESULTS=d)
            if child:
                env[spec_states.CHILD] = "1"
            steps_of = {k: lambda ds, name, k=k: ran.append((k, name)) for k in spec_states.STEPS}
            with mock.patch.object(c, "DATA", d), mock.patch.object(c, "RESULTS", d), \
                    mock.patch.object(c, "model", fake_model), mock.patch.object(spec_states, "subprocess") as sp, \
                    mock.patch.dict(spec_states.STEP, steps_of), mock.patch.dict(os.environ, env, clear=True), \
                    mock.patch.object(sys, "argv", ["spec_states.py", *steps]), \
                    contextlib.redirect_stdout(io.StringIO()):
                spec_states.main()
        started = [(run.args[0][2:], run.kwargs["env"].get(spec_states.CHILD)) for run in sp.run.call_args_list]
        return started, ran

    def test_each_step_runs_in_its_own_process(self):
        started, ran = self.main("fiqa", "clusters", "states")
        self.assertEqual(started, [(["--model", "arctic", "fiqa", "states"], "1"),
                                   (["--model", "arctic", "fiqa", "clusters"], "1")])
        self.assertEqual(ran, [])
        started, ran = self.main("--model", "BAAI/bge-small-en-v1.5", "fiqa")
        self.assertEqual(started, [(["--model", "bge-small", "fiqa", step], "1") for step in spec_states.STEPS])
        self.assertEqual(ran, [])

    def test_a_child_runs_its_step(self):
        self.assertEqual(self.main("fiqa", "ridge", child=True), ([], [("ridge", "arctic")]))
        self.assertEqual(self.main("--model", "e5-base", "quora", "states", child=True), ([], [("states", "e5-base")]))


def vec(text):
    """A unit vector made from the text alone, so the exported embeddings and the stand-in encoder agree."""
    v = np.random.default_rng(zlib.crc32(text.encode())).standard_normal(DIM)
    return v / np.linalg.norm(v)


class Model:
    """A stand-in for a sentence-transformers model, with no prompts."""

    def __init__(self, name):
        self.name = name
        self.prompts = {}


def spec(*parts):
    return spec_states.spec("fiqa", *parts)


def jsonl_texts(file):
    return [r["text"] for r in c.read_jsonl(file)]


class Remake(unittest.TestCase):
    """The export and the steps on a tiny FiQA (financial opinion mining and question answering) dataset, with a
    stand-in encoder and cluster files made here, so no model or faiss runs."""

    def setUp(self):
        self.dir = self.enterContext(tempfile.TemporaryDirectory())
        self.encoded = []
        self.enterContext(mock.patch.object(c, "DATA", self.dir))
        self.enterContext(mock.patch.object(c, "RESULTS", self.dir))
        self.enterContext(mock.patch.object(c, "device", lambda: "cpu"))
        self.enterContext(mock.patch.object(c, "load_model", lambda name, dev: Model(name)))
        self.enterContext(mock.patch.object(c, "encode_states", self.encode))
        self.enterContext(mock.patch.object(c, "model", fake_model))
        self.enterContext(mock.patch.object(c, "pooling_of", lambda st: "cls"))
        self.enterContext(mock.patch.object(export_beir, "TEST_SIZE", 10))
        self.enterContext(mock.patch.object(export_beir, "GT_K", 20))
        self.enterContext(mock.patch.object(spec_states, "OPENINGS", 40))
        self.enterContext(mock.patch.dict(spec_states.SMALL_FIT, fiqa=20))
        self.enterContext(mock.patch.object(spec_states, "PROBE_SHARES", [0.34, 0.5]))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        docs = [" ".join(f"d{i}w{j}" for j in range(4 + i % 30)) for i in range(200)]
        queries = [f"query {i} " + " ".join(f"q{i}w{j}" for j in range(3 + i % 7)) for i in range(60)]
        for part, texts in (("corpus", docs), ("queries", queries)):
            records = ({"row": i, "id": str(i), "text": t} for i, t in enumerate(texts))
            c.write_jsonl(c.path(f"fiqa_{part}.jsonl"), records)
            c.write_table(c.path(f"fiqa_{part}.fbin"), np.array([vec(t) for t in texts]))
        c.write_json(c.path("fiqa_queries.splits.json"), {"test": list(range(50, 60))})
        rng = np.random.default_rng(4)
        base = np.array([vec(t) for t in docs])
        # The l2 centers have lengths from 0.2 to 3, so the raw inner product ranks them unlike L2 distance does.
        for kind, lengths in (("l2", rng.uniform(0.2, 3, 6)), ("sph", np.ones(6))):
            cent = c.normalize(rng.standard_normal((6, DIM))) * lengths[:, None]
            assign = np.argmin(((base[:, None] - cent[None]) ** 2).sum(-1), 1)
            stem = spec("clusters", kind)
            c.write_table(stem + ".centroids.fbin", cent)
            c.write_table(stem + ".assign.ibin", assign[:, None], dtype="<i4")
            c.write_table(stem + ".parents.fbin", cent[:2])
            c.write_table(stem + ".parent_assign.ibin", (np.arange(6) % 2)[:, None], dtype="<i4")
            c.write_json(stem + ".json", dict(kind=kind, nlist=6))

    def encode(self, st, texts, prompt, layers, states, final=None, pooling="cls", batch=64, max_len=128):
        """Stands in for c.encode_states. Each layer's state is the text's vector twice plus noise that shrinks with
        the layer, and the final embedding is the vector itself."""
        self.encoded.append((st.name, list(texts)))
        for i, t in enumerate(texts):
            v = vec(t)
            for layer in layers:
                noise = np.random.default_rng(zlib.crc32(f"{t} {layer}".encode())).standard_normal(2 * DIM)
                states[layer][i] = np.concatenate([v, v]) + noise / (2 + layer)
            if final is not None:
                final[i] = v
        # Keeps the encode time above zero, since the step divides by it.
        time.sleep(0.002)
        return np.array([len(t.split()) for t in texts])

    def evaluation_set(self):
        recs = c.read_jsonl(c.path("fiqa_queries.jsonl"))
        split = c.read_json(c.path("fiqa_queries.splits.json"))
        export_beir.test_set("fiqa", [r["id"] for r in recs], [r["text"] for r in recs], split)

    def export(self):
        self.evaluation_set()
        export_beir.ground_truth("fiqa")

    def run_all(self):
        self.export()
        spec_states.states("fiqa")
        spec_states.ridge("fiqa")
        spec_states.clusters("fiqa")

    def records(self):
        """Every .json file of an output made from the evaluation queries."""
        found = [c.path("fiqa_test.gt.json"), spec("openings.meta.json"), spec("queries.meta.json"), spec("e.json")]
        found += sorted(glob.glob(spec("states", "*.json")) + glob.glob(spec("ehat", "*.json")))
        found += sorted(glob.glob(spec("clusters", "*.ceiling.json")))
        return found

    def files(self):
        return glob.glob(os.path.join(self.dir, "**", "*.*"), recursive=True)

    def new_rows(self):
        """Moves the evaluation queries to other rows of the same count, as a new rule would, and writes their test
        files. The ground truth and every later output still belong to the old rows, whose list it returns."""
        old_rows = c.read_json(c.path("fiqa_test.meta.json"))["query_rows"]
        self.enterContext(mock.patch.object(export_beir, "QUERY_SEED", 16))
        with self.assertRaises(ValueError):
            self.export()
        for part in ("jsonl", "fbin", "meta.json"):
            os.remove(c.path(f"fiqa_test.{part}"))
        self.evaluation_set()
        rows = c.read_json(c.path("fiqa_test.meta.json"))["query_rows"]
        self.assertEqual(len(rows), len(old_rows))
        self.assertNotEqual(sorted(rows), sorted(old_rows))
        return old_rows

    def test_new_test_rows_remake_every_output(self):
        self.run_all()
        old = {f: pathlib.Path(f).read_bytes() for f in self.files()}
        self.encoded.clear()
        self.new_rows()
        with self.assertRaises(ValueError):
            spec_states.ridge("fiqa")
        with self.assertRaises(ValueError):
            spec_states.clusters("fiqa")
        self.run_all()
        rows = c.read_json(c.path("fiqa_test.meta.json"))["query_rows"]
        made = c.evaluation_id("fiqa")
        self.assertEqual(len(glob.glob(spec("ehat", "*.json"))), 7)
        self.assertEqual(len(glob.glob(spec("states", "*.json"))), 6)
        for f in self.records():
            self.assertEqual(c.read_json(f)["test_rows_sha256"], made["test_rows_sha256"], f)
        self.assertEqual([r["query"] for r in c.read_jsonl(spec("queries.jsonl"))],
                         [r for r in range(60) if r not in set(rows)])
        sets = [jsonl_texts(f) for f in (spec("openings.jsonl"), spec("queries.jsonl"), c.path("fiqa_test.jsonl"))]
        self.assertEqual(sorted(self.encoded), sorted((m, t) for m in ("arctic", "minilm") for t in sets))
        self.assertTrue(c.same_bytes(c.path("fiqa_test.fbin"), spec("e.fbin")))
        for f in [c.path("fiqa_test.gt.ibin"), *glob.glob(spec("ehat", "*.fbin"))]:
            self.assertEqual(c.table_shape(f)[0], len(rows), f)
            self.assertNotEqual(pathlib.Path(f).read_bytes(), old[f], f)

    def test_e_matches_the_test_queries_only_once_a_ridge_step_finishes(self):
        self.run_all()
        self.new_rows()
        export_beir.ground_truth("fiqa")
        spec_states.states("fiqa")
        fit, calls = spec_states.fit, []

        def stops(*args):
            calls.append(args)
            if len(calls) > 1:
                raise RuntimeError("stopped after one predictor")
            return fit(*args)

        with mock.patch.object(spec_states, "fit", stops), self.assertRaises(RuntimeError):
            spec_states.ridge("fiqa")
        self.assertFalse(c.same_bytes(c.path("fiqa_test.fbin"), spec("e.fbin")))
        spec_states.ridge("fiqa")
        self.assertTrue(c.same_bytes(c.path("fiqa_test.fbin"), spec("e.fbin")))
        made = c.evaluation_id("fiqa")
        for f in [spec("e.json"), *glob.glob(spec("ehat", "*.json"))]:
            self.assertEqual(c.read_json(f)["test_rows_sha256"], made["test_rows_sha256"], f)

    def test_ridge_stops_on_a_ground_truth_of_other_queries(self):
        self.run_all()
        self.new_rows()
        # The export stops inside the ground truth, so the old one stays, and then the states step runs.
        with mock.patch.object(c, "exact_topk", side_effect=RuntimeError("stopped")), self.assertRaises(RuntimeError):
            export_beir.ground_truth("fiqa")
        spec_states.states("fiqa")
        with self.assertRaises(ValueError):
            spec_states.ridge("fiqa")
        self.assertFalse(c.same_bytes(c.path("fiqa_test.fbin"), spec("e.fbin")))
        export_beir.ground_truth("fiqa")
        spec_states.ridge("fiqa")
        self.assertTrue(c.same_bytes(c.path("fiqa_test.fbin"), spec("e.fbin")))

    def test_a_states_step_stopped_while_renaming_encodes_that_set_again(self):
        self.run_all()
        self.new_rows()
        finish = c.finish

        def stops(file, table=None):
            if os.path.basename(file).startswith("queries."):
                raise RuntimeError("stopped")
            finish(file, table)

        with mock.patch.object(c, "finish", stops), self.assertRaises(RuntimeError):
            spec_states.states("fiqa")
        spec_states.states("fiqa")
        final = np.asarray(c.open_table(spec("states", "queries.final.fbin")))
        want = np.array([vec(t) for t in jsonl_texts(spec("queries.jsonl"))], dtype=np.float32)
        self.assertTrue(np.array_equal(final, want))

    def test_a_rerun_keeps_current_outputs(self):
        self.run_all()
        before = {f: os.stat(f).st_mtime_ns for f in self.files()}
        self.encoded.clear()
        self.run_all()
        self.assertEqual(self.encoded, [])
        self.assertEqual({f: os.stat(f).st_mtime_ns for f in self.files()}, before)

    def test_another_model_fits_its_own_layers_with_its_prompt_and_no_minilm(self):
        other = dict(fake_model("bge-small"), fire=[0, 1, 3], query_prompt="Represent: ")
        prompts, encode = [], self.encode

        def recording(st, texts, prompt, *args, **kwargs):
            prompts.append(prompt)
            return encode(st, texts, prompt, *args, **kwargs)

        with mock.patch.object(c, "model", lambda name: other if name == "bge-small" else fake_model(name)), \
                mock.patch.object(c, "encode_states", recording):
            self.export()
            spec_states.states("fiqa", "bge-small")
            spec_states.ridge("fiqa", "bge-small")
            summary = spec_states.summary("fiqa", "bge-small")
        stems = sorted(os.path.basename(f) for f in glob.glob(spec("ehat", "*.fbin")))
        self.assertEqual(stems, sorted(f"{s}.L{layer}.fbin" for s in ("openings", "queries") for layer in (0, 1, 3)))
        self.assertEqual({m for m, _ in self.encoded}, {"bge-small"})
        self.assertEqual(set(prompts), {"Represent: "})
        self.assertEqual(glob.glob(spec("states", "*minilm*")), [])
        self.assertEqual((summary["model"], summary["layers"]), ("org/bge-small", [0, 1, 3]))
        self.assertTrue(c.same_bytes(c.path("fiqa_test.fbin"), spec("e.fbin")))

    def test_e_stays_a_byte_copy_of_the_test_queries(self):
        self.run_all()
        with open(spec("e.fbin"), "r+b") as f:
            f.seek(20)
            byte = f.read(1)[0]
            f.seek(20)
            f.write(bytes([byte ^ 1]))
        self.assertFalse(c.same_bytes(c.path("fiqa_test.fbin"), spec("e.fbin")))
        spec_states.ridge("fiqa")
        self.assertTrue(c.same_bytes(c.path("fiqa_test.fbin"), spec("e.fbin")))

    def test_steps_stop_on_inputs_made_for_other_queries(self):
        self.run_all()
        # MiniLM's states of an older evaluation set, as when a states step stops before its MiniLM half.
        info = c.read_json(spec("states", "queries.minilm.json"))
        c.write_json(spec("states", "queries.minilm.json"), dict(info, test_rows_sha256="0" * 64))
        with self.assertRaises(ValueError):
            spec_states.ridge("fiqa")
        os.remove(spec("clusters", "l2.ceiling.json"))
        info = c.read_json(c.path("fiqa_test.gt.json"))
        c.write_json(c.path("fiqa_test.gt.json"), dict(info, test_rows_sha256="0" * 64))
        with self.assertRaises(ValueError):
            spec_states.clusters("fiqa")

    def test_only_the_real_queries_are_shuffled_before_the_split(self):
        self.run_all()
        for fit_set, seed in (("openings", None), ("queries", 13)):
            Y = np.asarray(c.open_table(spec("states", f"{fit_set}.final.fbin")))
            for stem, states in ((f"{fit_set}.L0", f"{fit_set}.L0"), (f"{fit_set}.L3", f"{fit_set}.L3"),
                                 (f"minilm.{fit_set}", f"{fit_set}.minilm.L6")):
                X = np.asarray(c.open_table(spec("states", f"{states}.fbin")))
                lam, val = c.choose_lambda(X, Y, spec_states.LAMBDAS, seed=seed)
                info = c.read_json(spec("ehat", f"{stem}.json"))
                self.assertEqual(info["split_seed"], seed, stem)
                self.assertEqual(info["val_cos"], {str(p): v for p, v in val.items()}, stem)
                self.assertEqual(info["lam"], lam, stem)

    def test_a_new_split_seed_refits_only_the_fits_it_shuffles(self):
        self.run_all()
        before = {f: (os.stat(f).st_mtime_ns, c.read_json(f)) for f in glob.glob(spec("ehat", "*.json"))}
        with mock.patch.dict(spec_states.SPLIT_SEED, queries=14):
            spec_states.ridge("fiqa")
        shuffled = {"queries.L0.json", "queries.L3.json", "minilm.queries.json"}
        for f, (mtime, info) in before.items():
            if os.path.basename(f) in shuffled:
                self.assertEqual(c.read_json(f)["split_seed"], 14, f)
            else:
                self.assertEqual((os.stat(f).st_mtime_ns, c.read_json(f)), (mtime, info), f)
        self.assertEqual(len(shuffled & {os.path.basename(f) for f in before}), 3)

    def test_openings_skip_documents_with_the_text_of_an_evaluation_query(self):
        self.export()
        test_texts = jsonl_texts(c.path("fiqa_test.jsonl"))
        corpus = c.read_jsonl(c.path("fiqa_corpus.jsonl"))
        # Every second document holds an evaluation query's text.
        for r in corpus[::2]:
            r["text"] = test_texts[r["row"] % len(test_texts)]
        c.write_jsonl(c.path("fiqa_corpus.jsonl"), corpus)
        spec_states.states("fiqa")
        drawn = [r["doc"] for r in c.read_jsonl(spec("openings.jsonl"))]
        self.assertEqual(len(drawn), spec_states.OPENINGS)
        self.assertTrue(all(d % 2 == 1 for d in drawn), drawn)

    def test_kept_test_files_must_hold_the_rows_of_the_rule(self):
        self.run_all()
        rows = c.read_json(c.path("fiqa_test.meta.json"))["query_rows"]
        queries = np.asarray(c.open_table(c.path("fiqa_queries.fbin")))
        c.write_table(c.path("fiqa_test.fbin"), queries[rows[::-1]])
        with self.assertRaises(ValueError):
            self.export()
        # The vectors are right again, but the texts are in another order.
        c.write_table(c.path("fiqa_test.fbin"), queries[rows])
        self.export()
        c.write_jsonl(c.path("fiqa_test.jsonl"), c.read_jsonl(c.path("fiqa_test.jsonl"))[::-1])
        with self.assertRaises(ValueError):
            self.export()

    def test_probe_ceilings_rank_by_l2_and_by_the_raw_inner_product(self):
        self.run_all()
        e = np.asarray(c.open_table(c.path("fiqa_test.fbin")), dtype=np.float64)
        gt = np.asarray(c.open_table(c.path("fiqa_test.gt.ibin"), dtype="<i4"))[:, :10]
        for kind in ("l2", "sph"):
            cent = np.asarray(c.open_table(spec("clusters", f"{kind}.centroids.fbin")), dtype=np.float64)
            home = np.asarray(c.open_table(spec("clusters", f"{kind}.assign.ibin"), dtype="<i4"))[:, 0][gt]
            got = c.read_json(spec("clusters", f"{kind}.ceiling.json"))["recall"]
            self.assertEqual(sorted(got), ["ip-raw", "l2"])
            for routing, far in (("l2", ((e[:, None] - cent[None]) ** 2).sum(-1)), ("ip-raw", -e @ cent.T)):
                for nprobe in (2, 3):
                    near = np.argsort(far, axis=1)[:, :nprobe]
                    want = np.mean([np.isin(home[i], near[i]).mean() for i in range(len(e))])
                    self.assertAlmostEqual(got[routing][str(nprobe)], want, places=12, msg=(kind, routing))
            # Centers of one length rank the same both ways.
            self.assertEqual(got["l2"] == got["ip-raw"], kind == "sph", kind)


if __name__ == "__main__":
    unittest.main()
