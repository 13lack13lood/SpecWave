# specwave-bench/py

Python helpers. They write only to `data/`.

| script | what it does |
|---|---|
| `export_beir.py` | FiQA / Quora texts → JSONL in parquet order; cached `.npy` → `.fbin`; query sets |
| `spec_states.py` | per-layer pooled states and ridge predictors → e and ê as `.fbin` |
