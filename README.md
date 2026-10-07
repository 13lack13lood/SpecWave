# SpecWave

Fires ColdIVF's read waves from inside the embedding model. At an intermediate layer, a ridge predictor guesses the final embedding, so the index I/O hides under the remaining layers.

```text
specwave/
├── specwave/        encoder (layer hook) · predictor (ridge W) · scheduler (fire layer)
│                    · speculate (worker on coldivf's route/fetch) · train
├── specwave-bench/  bin/{spec, spec_coverage, wavecal} · py/ (BeIR export, layer states)
└── scripts/check-all.sh
```

- **Dependencies:** this repo uses the sibling repo [~/vectordb/coldivf](../coldivf) by path, through ColdIVF's generic API only (`prepare`, `route`, `fetch`, `Fetched`); ColdIVF has no SpecWave code. Check both out side by side; once both are on GitHub, the path becomes a git dependency with a local `[patch]`.
- **Features:** `specwave/candle` (the real encoder, which has C/C++ build scripts) is off by default. `specwave-bench/candle` (on by default) turns it on.
- **Status:** skeleton only (`todo!()` stubs).

```bash
./scripts/check-all.sh
```
