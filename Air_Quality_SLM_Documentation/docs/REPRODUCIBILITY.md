# Reproducibility

Three levels, from cheapest to most expensive. You do **not** need to retrain anything to check every claim
in this documentation.

---

## Level 1 — Verify the numbers from stored artefacts (minutes, no torch)

### 1.1 Environment you need

```bash
python -m venv venv
./venv/Scripts/python.exe -m pip install -U pip
./venv/Scripts/python.exe -m pip install pandas numpy matplotlib
```

### 1.2 Dataset statistics

```bash
./venv/Scripts/python.exe - <<'PY'
import pandas as pd
raw = pd.read_csv("data/raw/air+quality/AirQualityUCI.csv", sep=";", decimal=",")
raw = raw.dropna(how="all", axis=1).dropna(how="all")
num = raw.drop(columns=["Date", "Time"])
print("rows", len(raw), "cols", len(raw.columns))
print("sentinel cells", int((num == -200).sum().sum()))
print(num.replace(-200, pd.NA).isna().sum())
PY
```

Expected: `rows 9357 cols 15`, `sentinel cells 16701`, NaNs CO 1,683 / NOx 1,639 / NO2 1,642 /
NMHC 8,443 / the remaining ten columns 366 each.

### 1.3 Split boundaries

```bash
./venv/Scripts/python.exe - <<'PY'
import pandas as pd
for s in ("train", "validation", "test"):
    d = pd.read_csv(f"data/processed/{s}.csv")
    print(s, len(d), d["Date"].iloc[0], d["Time"].iloc[0], "->", d["Date"].iloc[-1], d["Time"].iloc[-1])
PY
```

Expected: 6,549 rows from `10/03/2004 18.00.00` to `08/12/2004 14.00.00`; 936 rows from
`08/12/2004 15.00.00` to `16/01/2005 14.00.00`; 1,872 rows from `16/01/2005 15.00.00` to
`04/04/2005 14.00.00`.

### 1.4 Any MAE in the evaluation report

```bash
./venv/Scripts/python.exe - <<'PY'
import json, statistics as st
rows = json.load(open("docs/eval_v2/metrics_v2_val.json"))
pairs = [(r["pred"]["AH"], r["gold"]["AH"])
         for r in rows if r["kind"] == "AH"
         and r.get("pred") and "AH" in r["gold"]]
mae  = sum(abs(p - g) for p, g in pairs) / len(pairs)
rmse = (sum((p - g) ** 2 for p, g in pairs) / len(pairs)) ** 0.5
print(f"n={len(pairs)} MAE={mae:.4f} RMSE={rmse:.4f} pred_std={st.pstdev([p for p,_ in pairs]):.4f}")
PY
```

Expected: `n=804 MAE=0.4335 RMSE=0.4871 pred_std=0.0030`. The same pattern works for `T`, `RH`, `CO`,
`NOx`, `NO2`, `C6H6` and for `metrics_v2_test.json`, `metrics_v1_val.json`, `metrics_v1_test.json`.

### 1.5 Loss and perplexity

Open `docs/eval_v2/loss_v2_128.json` and compare with `docs/EVALUATION_REPORT.md` §2. The tokens-scored
values (431,488 val / 870,784 test) are the self-check: if the harness had skipped or double-counted
tokens, those totals would not match.

### 1.6 The humidity formula's current behaviour

```bash
./venv/Scripts/python.exe -c "import humidity_calculator as h; print(h.get_scale_factor(), h.calculate_absolute_humidity(16.7,54.7))"
```

Expected: `0.12874776284191383 {'result': 1.0284, ...}`. If you get 0.976, the file has been corrected since
this documentation was written — see `docs/KNOWN_LIMITATIONS.md` §3.

---

## Level 2 — Re-run the evaluation (needs torch, ~1–2 hours CPU)

```bash
./venv/Scripts/python.exe -m pip install torch transformers tokenizers gguf tqdm
./venv/Scripts/python.exe -u src/evaluate_v2_full.py --phase baseline
./venv/Scripts/python.exe -u src/evaluate_v2_full.py --phase suite --suite loss    --threads 6
./venv/Scripts/python.exe -u src/evaluate_v2_full.py --phase suite --suite metrics --max-new 50 --threads 6
./venv/Scripts/python.exe -u src/summarize_eval_v2.py
./venv/Scripts/python.exe -u src/diagnose_v2_numbers.py
```

This overwrites `docs/eval_v2/*`. Copy the directory first if you want to compare.

Expected agreement: loss and perplexity to ~4 decimal places; generation-based metrics exactly, because
decoding is greedy and deterministic for a fixed torch build.

---

## Level 3 — Rebuild the models (needs ~8 GB RAM, ~40 minutes CPU)

```bash
./venv/Scripts/python.exe src/preprocess_air_quality.py
./venv/Scripts/python.exe src/prepare_training_data.py
./venv/Scripts/python.exe src/train_slm.py           # v1, ~5 min
./venv/Scripts/python.exe src/continue_training.py   # v2, ~31 min
./venv/Scripts/python.exe src/convert_to_gguf.py
./venv/Scripts/python.exe src/export_gguf_from_checkpoint.py
ollama create air-quality-slm-v3 -f Modelfile.v3
```

Caveats that must be stated if you report a re-run:

* v1 samples batches **with replacement**, so its loss trace is not bit-reproducible.
* v2 resumes **weights only**; the v1 export contained no optimiser or scheduler state, so a fresh AdamW is
  initialised.
* Floating-point reduction order depends on thread count (6 was used here).
* A different torch build moves the last digits.

The stored JSON is the record of truth; a re-run should reproduce the *conclusions* (66 % loss reduction,
constant outputs, 18–28× worse than the closed form), not the last digit.

---

## Level 4 — Verify the deployment

```bash
# 1. the proxy topology
OLLAMA_HOST=http://127.0.0.1:11435 ollama list
OLLAMA_HOST=http://127.0.0.1:11435 ollama show air-quality-slm-v3:latest --modelfile

# 2. the context window is 128, not 64
OLLAMA_HOST=http://127.0.0.1:11435 ollama run air-quality-slm-v3:latest "Calculate absolute humidity for T = 16.7 C and RH = 54.7%"

# 3. the system path (proxy on 11434)
./venv/Scripts/python.exe ollama_proxy.py
# then, in Ollama Desktop: "Hey, what can you do?"
```

Check `proxy_access.log` afterwards: each intercepted request appends one line with its intent. That file is
the proof that the desktop session is routed through the proxy.

---

## What cannot be reproduced

| Item | Why |
|---|---|
| The dataset download event | no URL, mirror or timestamp was recorded |
| v1/v2 losses to the last digit | sampling with replacement; thread-order effects |
| Any v3 result | never evaluated |
| The Ollama Desktop screenshots | never captured; `assets/screenshots/README.md` says how |
| A quantised-model comparison | no quantised build exists |
| Precision / recall / F1 | never computed; the task is generation, not classification |

---

## File-level integrity

SHA-256 for every artefact is in `../Air_Quality_SLM_Research_Package/FILE_INVENTORY.md` (that package ships
the source files and the results inside its ZIP) and in the research package's `conversion/`
documentation. Key values:

| File | SHA-256 (prefix) |
|---|---|
| `models/air_quality_slm/air_quality_slm.gguf` | `10bb0ead…` |
| `models/air_quality_slm_finetuned/air_quality_slm_finetuned.gguf` | `ebf9037d…` |
| `models/air_quality_slm/pytorch_model.bin` | `77be6061…` |
| `training/checkpoints/best_model.pt` | `32ae7374…` |
| `data/raw/air+quality/AirQualityUCI.csv` | `13277ae5…` |
