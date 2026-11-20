# Reproducibility

Two different questions are answered here, and they should not be confused:

* **Can the reported numbers be checked?** Yes, from the artefacts in this package, without a GPU and
  without re-training. §2–§5 give the exact commands.
* **Can the models be re-built from scratch?** Yes in principle, with the caveats in §6. Re-training is
  not necessary to verify any claim in this package, and was deliberately not performed.

---

## 1. What you need

| Requirement | Detail |
|---|---|
| Read-only verification | Python 3.10+, `pandas`, `numpy`, `matplotlib` (figures only). No torch. |
| Re-running loss/generation evaluation | the above plus `torch` 2.14.x, `transformers` 5.23.x, `tokenizers` |
| Re-training | the above plus ~8 GB RAM; CPU only is sufficient (v1 279 s, v2 1,864 s) |
| Re-deploying to Ollama | Ollama 0.33.x; the `.gguf` files are already in `models/` |
| Disk | the two model directories total ≈ 1.2 GB (weights, GGUF, tokenizer) |

All commands assume the project root as the working directory, on Windows with Git Bash:

```bash
python -m venv venv
./venv/Scripts/python.exe -m pip install -U pip
./venv/Scripts/python.exe -m pip install torch transformers tokenizers numpy pandas gguf tqdm ollama requests matplotlib
```

## 2. Verify the dataset numbers without any model

This recomputes every statistic in `dataset/dataset_documentation.md` §3 directly from the raw CSV.

```bash
./venv/Scripts/python.exe - <<'PY'
import pandas as pd
raw = pd.read_csv("data/raw/air+quality/AirQualityUCI.csv", sep=";", decimal=",")
raw = raw.dropna(how="all", axis=1).dropna(how="all")
print("rows", len(raw), "cols", list(raw.columns))
num = raw.drop(columns=["Date", "Time"])
print("cells == -200:", int((num == -200).sum().sum()))
print("NaNs per column:\n", num.replace(-200, pd.NA).isna().sum())
PY
```

Expected: 9,357 rows, 15 columns, 16,701 sentinel cells, NaNs CO 1,683 / NOx 1,639 / NO2 1,642 /
NMHC 8,443 / other 366 each.

Verify the splits:

```bash
./venv/Scripts/python.exe - <<'PY'
import pandas as pd
for s in ("train", "validation", "test"):
    d = pd.read_csv(f"data/processed/{s}.csv")
    print(s, len(d), d["Date"].iloc[0], d["Time"].iloc[0], "->", d["Date"].iloc[-1], d["Time"].iloc[-1])
PY
```

Expected: 6,549 / 936 / 1,872 rows with the contiguous boundaries quoted in `METHODOLOGY.md` §2.

## 3. Verify the training curves

The two curves are stored verbatim from the training runs:

| Artefact | Content |
|---|---|
| `models/air_quality_slm/training_summary.json` | v1: four eval points (steps 40/80/120/149) plus initial/final losses, runtime |
| `training/loss_curve.csv` | v2: step, train EMA loss, val loss, val PPL, tokens, epochs, elapsed |
| `training/training_log_finetune.jsonl` | v2: one JSON object per eval |

Regenerate figures 5–7 from them:

```bash
./venv/Scripts/python.exe research_package/Air_Quality_SLM_Research_Package/figures/scripts/make_figures.py
```

## 4. Verify the loss and perplexity table

`evaluation/results/loss_v1_128.json`, `loss_v2_128.json`, `loss_v1_64.json` hold the exact values in
`EXPERIMENTAL_RESULTS.md` §3. To recompute them from the checkpoints (slower, needs torch):

```bash
./venv/Scripts/python.exe -u src/evaluate_v2_full.py --phase suite --suite loss --threads 6
```

The harness scores every token once over non-overlapping 128-token windows; `tokens_scored` must equal
431,488 (val) and 870,784 (test).

## 5. Verify every MAE and RMSE from the per-row dumps

`evaluation/results/metrics_v1_val.json`, `metrics_v1_test.json`, `metrics_v2_val.json`,
`metrics_v2_test.json` contain one record per generated output: prompt, generated text, extracted
number, gold label. Five lines of pandas reproduce any MAE in §5 of `EXPERIMENTAL_RESULTS.md`:

Row schema: `{"row": int, "kind": "REPORT"|"AH"|"TEMP_RH"|"POLLUTANT", "gold": {...},
"pred": {...}|null, "output": str, "n_new": int, "tf_loss": float, "tf_top1_hit": float,
"struct": {...}}`, plus `ok_<quantity>` booleans for extractability.

```bash
./venv/Scripts/python.exe - <<'PY'
import json, statistics as st
rows = json.load(open("research_package/Air_Quality_SLM_Research_Package/evaluation/results/metrics_v2_val.json"))
pairs = [(r["pred"]["AH"], r["gold"]["AH"])
         for r in rows if r["kind"] == "AH"
         and r.get("pred") and "AH" in r["gold"]]
mae  = sum(abs(p - g) for p, g in pairs) / len(pairs)
rmse = (sum((p - g) ** 2 for p, g in pairs) / len(pairs)) ** 0.5
print(f"n={len(pairs)} MAE={mae:.4f} RMSE={rmse:.4f} pred_std={st.pstdev([p for p,_ in pairs]):.4f}")
PY
```

Expected for v2/AH/val: `n=804 MAE=0.4335 RMSE=0.4871 pred_std=0.003`.

## 6. Re-building the models (not required, and not performed here)

```bash
# 1. clean + split the raw CSV
./venv/Scripts/python.exe src/preprocess_air_quality.py
# 2. build instruction texts and token streams
./venv/Scripts/python.exe src/prepare_training_data.py
# 3. v1: 150 steps from random init, block 64  (~5 min CPU)
./venv/Scripts/python.exe src/train_slm.py
# 4. v2: 1,200 steps continued from v1, block 128  (~31 min CPU)
./venv/Scripts/python.exe src/continue_training.py
# 5. export GGUF
./venv/Scripts/python.exe src/convert_to_gguf.py
./venv/Scripts/python.exe src/export_gguf_from_checkpoint.py
# 6. register with Ollama
ollama create air-quality-slm-v3 -f Modelfile.v3
```

**Caveats.** v1 samples batches with replacement, so its loss trace is not bit-reproducible. v2 resumes
*weights only* — no optimiser or scheduler state was carried over, because the exported v1 weights had
none attached. A fresh run with a different torch build will differ in the last digits. The stored JSON
is the record of truth; the figures are drawn from it, not from a re-run.

## 7. Verify the deployment

```bash
# proxy topology: 11434 = proxy, 11435 = real Ollama
OLLAMA_HOST=http://127.0.0.1:11435 ollama list
OLLAMA_HOST=http://127.0.0.1:11435 ollama show air-quality-slm-v3:latest --modelfile
OLLAMA_HOST=http://127.0.0.1:11435 ollama run air-quality-slm-v3:latest "Calculate absolute humidity for T = 16.7 C and RH = 54.7%"
```

The GGUF context length must report 128, and `prompt_eval_count` for the 86-token humidity prompt must
equal 86 (not 34 — 34 was the v1 truncation defect).

## 8. Checksums

`FILE_INVENTORY.md` in the package root lists SHA-256 for every substantial artefact, including:

| File | SHA-256 (prefix) |
|---|---|
| `models/air_quality_slm/air_quality_slm.gguf` | `10bb0ead…` |
| `models/air_quality_slm/air_quality_slm_llama_adapted.gguf` | `5ad37e73…` |
| `models/air_quality_slm_finetuned/air_quality_slm_finetuned.gguf` | `ebf9037d…` |
| `models/air_quality_slm/pytorch_model.bin` | `77be6061…` |
| `training/checkpoints/best_model.pt` | `32ae7374…` |
| `data/raw/air+quality/AirQualityUCI.csv` | `13277ae5…` |

## 9. What cannot be reproduced from this package

* The original dataset download event (no URL, no timestamp recorded).
* The exact v1/v2 loss values to the last digit (sampling with replacement, thread-order effects).
* Any result for v3 (never evaluated; see `MISSING_EVIDENCE.md` §8).
* The two Ollama Desktop screenshots (never captured; see `MISSING_EVIDENCE.md` §1).
