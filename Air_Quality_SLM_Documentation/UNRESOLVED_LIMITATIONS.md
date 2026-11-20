# Unresolved Limitations

Open issues at the time this package was produced. Each entry is a defect or gap that **still exists in the
code or the evidence** — not a limitation of the domain. Fixing these does not require retraining.

---

## 1. `humidity_calculator.py` documents coefficients it does not use

`magnus_saturation_vp` documents `T + 243.5` and executes `T + 237.3`. The docstring also describes a fixed
scale of 0.976, while `get_scale_factor()` computes a single-row ratio at import time (0.12874776 here).

**Impact:** the documented worked example returns 1.0284 g/m³ instead of the ≈ 7.80 g/m³ the docstring
implies. **Status:** open. **Owner action:** choose one definition, implement it, recompute the example.

## 2. `app.py` prints CO and NOx in the wrong unit

`handle_retrieval` and `_format_sensor_values` print every pollutant as `ug/m3`; the dataset stores CO in
mg/m³ and NOx in ppb. `ollama_proxy.py` already has the correct map (`_POLLUTANT_UNITS`).

**Impact:** terminal transcripts are unit-mislabelled by up to 10³. **Status:** open.

## 3. The application still targets v2

`ollama_client.MODEL_NAME = "air-quality-slm-improved:latest"`, while `Modelfile.v3` expects `{{ .System }}`.

**Impact:** the terminal assistant never exercises v3, and v3's system-prompt path is unmeasured.
**Status:** open.

## 4. v3 has no evaluation

Same weights as v2, different template, zero measurements. **Status:** open. **Action:** run the existing
harness against the tag.

## 5. The model outputs constants

Prediction standard deviation 0.000–0.003; 1–5 distinct outputs per 936 prompts; RH and NOx worse than a
constant predictor. **Impact:** no numeric result from the network is usable as an estimate.
**Status:** open, and probably not fixable at 20M parameters on 0.2 epoch.

## 6. No screenshots exist

The brief asks for two Ollama Desktop captures; the repository contains none, and none was fabricated.
**Status:** open. **Action:** `docs/assets/screenshots/README.md` specifies exactly what to capture.

## 7. The Desktop path bypasses the model

`ollama_proxy.py` answers `/api/chat` and `/api/generate` in Python. **Impact:** a demonstration proves the
system, not the network. **Status:** by design, but unresolved as a demonstration strategy — the two paths
should be shown and labelled separately.

## 8. `NMHC(GT)` is unevaluable on held-out data

100 % missing in validation and test. **Status:** unfixable with this dataset; must simply not be reported.

## 9. The full-report template does not fit the context

193 tokens vs 128. It is excluded from teacher-forced scoring. **Status:** open.

## 10. Training stopped at 0.202 epoch

2,400 steps were planned; early stopping fired at 1,200. **Impact:** some constant-output behaviour may be
budget, not capacity. **Status:** open.

## 11. Evaluation and documentation are uncommitted

The newest commit is `e967ef9`. Everything in `docs/eval_v2/`, the figures, both ZIP packages and this
documentation are working-tree files. **Impact:** a clone reproduces neither the numbers nor the docs.
**Status:** open.

## 12. No dependency manifest

No `requirements.txt` or lockfile. Versions are recorded in prose only. **Status:** open.

## 13. Dataset download is undocumented

No URL, mirror or checksum captured at download time. Only the file's last-write date (2026-08-05) survives.
**Status:** open.

## 14. Date-overlap field is misleading

`baseline.json` reports `date_overlap_with_train = 1` for validation because the boundary date appears in
both splits. No `(Date, Time)` pair is shared. **Status:** cosmetic, but it will mislead a reviewer who reads
the raw JSON.

## 15. Greedy decoding is the only characterised mode

Sampling variance was never measured — sampling was in fact the original bug. **Status:** open.

## 16. No quantised model, latency benchmark or human evaluation

**Status:** open. Any such claim would currently be unsupported.
