# Known Limitations

Written to be quoted in a limitations section without further editing. Each item states the limitation, the
evidence, and what would fix it. None of these is a guess: each was measured or read from the source.

---

## 1. The model produces constant numbers

**What.** v2 emits one value per template regardless of the input row.

**Evidence.** Prediction standard deviation is `0.000` for T, RH, CO, NOx, NO₂, C₆H₆ and `0.003` for AH
(`docs/eval_v2/summary_tables.txt`). Distinct outputs per 936 validation prompts: **3** for AH, **5** for
TEMP_RH, **1** for POLLUTANT (`summary_tables.txt`, "OUTPUT DIVERSITY").

**Effect.** No numeric metric from this model should be described as a prediction in the inferential sense.
MAE measures the distance between one constant and the split distribution.

**Fix.** A model with capacity for arithmetic, or (better) keep numbers in Python — which is what the
application already does.

## 2. The humidity estimate is 19–28× worse than the formula it was given

**What.** The AH prompt contains `T` and `RH`; the dataset's `AH` is a deterministic function of them.

**Evidence.** Closed-form MAE 0.02296 (val) / 0.01880 (test) versus v2 0.4335 / 0.5345 —
`numeric_diagnostics.json`.

**Fix.** Do not ask the network for arithmetic. `humidity_calculator.py` exists for this reason.

## 3. `humidity_calculator.py` contradicts its own documentation

**What.** Three different numbers are in play and they do not agree:

1. the docstring says `e_s = 6.112·exp(17.67·T/(T+243.5))`; the code uses **237.3**;
2. the docstring says the scale factor is a fixed fit of **0.976**; the code computes a **single-row ratio**
   at import time and caches it (`get_scale_factor()`), which evaluates to **0.12874776** on this machine;
3. `numeric_diagnostics.json` reports a train-split fit of ≈ 0.1351678.

The worked example `T = 16.7 °C, RH = 54.7 %` therefore returns **1.0284 g/m³**, while pure Tetens gives
≈ 7.99 and `0.976 ×` Tetens gives ≈ 7.80.

**Effect.** The application's answer happens to sit near the correct dataset value (1.0311) — but for an
undocumented reason. It must not be presented as a validated physical calculation.

**Fix.** Choose one definition (dataset-calibrated ratio fitted over the whole training split), implement it,
and recompute the documented example.

## 4. `app.py` mislabels CO and NOx units

**What.** `handle_retrieval` and `_format_sensor_values` print every pollutant as `ug/m3`.

**Evidence.** `app.py` lines 140–148 and 194–203; the dataset stores CO in mg/m³ and NOx in ppb.

**Effect.** Any terminal transcript showing "CO: 1.2 ug/m3" is wrong by three orders of magnitude in unit.
The proxy (`ollama_proxy.py`) uses the correct map, so the two user interfaces disagree.

**Fix.** Reuse `_POLLUTANT_UNITS` from `ollama_proxy.py` in `app.py`.

## 5. Historical data only — the model cannot monitor anything

**What.** The dataset ends 04/04/2005. Every lookup outside 10/03/2004–04/04/2005 returns "No data found".

**Effect.** "Air quality monitoring" in a title must be qualified as *historical record retrieval*. No live
sensor, no API, no forecast exists in this project.

## 6. `NMHC(GT)` is unusable on held-out data

**What.** 90.23 % missing overall and **100 % missing in both validation and test**.

**Effect.** No NMHC metric can be reported. Any table that includes NMHC numbers for val/test is fabricated.

## 7. Context window is 128 tokens; one template does not fit

**What.** The full-report template is 193 tokens (100 in earlier measurements of a shorter variant). The
REPORT template is excluded from teacher-forced scoring because it exceeds the window.

**Effect.** The model has never been trained or evaluated on a complete report with 128-token windows; only
the three instruction templates are.

## 8. v3 has no evaluation

**What.** `air-quality-slm-v3:latest` uses the same GGUF as v2; only the Modelfile differs (template with
`{{ .System }}`). No metrics file, dump or report exists for it.

**Effect.** v3 cannot be called better, worse or improved. It is the newest tag, not the best-evidenced model.

## 9. Ollama Desktop shows the system, not the model

**What.** `ollama_proxy.py` intercepts `POST /api/chat` and `POST /api/generate` and answers in Python
(`handle_message`), forwarding only model-management endpoints.

**Evidence.** `ollama_proxy.py` lines 518–573; `proxy_access.log` records every interception.

**Effect.** A desktop demonstration proves routing, dataset lookup and the formula — **not** neural quality.
`app.py` and `ollama_client.py` are the only paths that actually call the network.

## 10. The neural network has no refusal, retrieval or tool use

**What.** Refusals come from `domain_guard.py`; retrieval from pandas; arithmetic from Python. The GGUF's
chat template is a raw passthrough with no tool markers, and no tool calling is configured.

**Effect.** Bypassing the proxy (`OLLAMA_HOST=...:11435`) yields raw continuation, including for
out-of-domain prompts.

## 11. The model was trained for 0.202 epoch

**What.** v2 saw 614,400 of 3,041,432 training tokens before early stopping.

**Effect.** Undertrained relative to the corpus. Some of the constant-output behaviour may be a training-budget
artefact rather than a capacity limit — but the closed-form gap (§2) is large enough that capacity is a
sufficient explanation.

## 12. Single deterministic decode

**What.** All reported generations use greedy decoding (`temperature 0`, `top_k 1`).

**Effect.** Sampling variance is not characterised. No claim can be made about output diversity under
sampling, because sampling was the original cause of the garbled-output bug.

## 13. No quantised model, no latency benchmark, no human evaluation

**What.** Only F32 GGUFs exist; run time is anecdotal; no human rating of usefulness was collected.

**Effect.** Performance claims must be limited to what was measured: loss, perplexity, MAE/RMSE, structural
metrics, and dataset statistics.

## 14. Dataset date-overlap counter is misleading

**What.** `baseline.json` reports `date_overlap_with_train = 1` for validation, because the boundary date
`08/12/2004` appears in both splits (train ends 14:00, validation starts 15:00).

**Effect.** Not leakage — 0 `(Date, Time)` pairs are shared — but the raw field says 1. State the nuance.

## 15. The application still points at v2

**What.** `ollama_client.MODEL_NAME = "air-quality-slm-improved:latest"`, while `Modelfile.v3` is designed to
receive a system prompt.

**Effect.** The terminal assistant does not use the v3 tag, and v3's system-prompt path is unexercised.

## 16. No screenshot exists in the repository

**What.** A recursive search for image files outside `venv/` finds only two fixtures inside the vendored
`markitdown` package.

**Effect.** No deployment screen can be shown without capturing one now. See
`assets/screenshots/README.md` for exactly what to capture and what it will demonstrate.

## 17. Evaluation and documentation work is uncommitted

**What.** The most recent git commit is `e967ef9` (proxy hardening). The evaluation dumps, reports, figures
and both packages are working-tree files.

**Effect.** A clone of the repository will not contain them until they are committed.
