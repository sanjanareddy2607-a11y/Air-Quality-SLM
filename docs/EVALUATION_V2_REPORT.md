# Air Quality SLM — v1 vs v2 Evaluation Report

**Date:** 19 November 2026
**Checkpoints evaluated:** `models/air_quality_slm_original` (v1, 64-token training window) and `models/air_quality_slm_finetuned` (v2, 128-token continued training)
**Splits:** complete validation (936 rows) and complete test (1,872 rows)
**Raw results:** `docs/eval_v2/*.json`, tables in `docs/eval_v2/summary_tables.txt`
**No checkpoint was modified and no training was run to produce this report.**

---

## 1. Verdict

| Question | Answer |
|---|---|
| Did continued training improve language modelling? | **Yes, decisively.** Full-split loss 1.4252 → 0.4877 (val) and 1.4474 → 0.5555 (test); perplexity 4.16 → 1.63 (val), 4.25 → 1.74 (test). |
| Did it improve formatting and instruction-following? | **Yes.** v2 emits EOS and correct units in 100 % of outputs with 0 % repetition and 0 % malformed output at 128-token budget. v1 never learns to stop (100 % hit the token cap, 35 % repetitive). |
| Did it improve numerical prediction accuracy? | **No.** v2 emits essentially one constant value per template (prediction std 0.000–0.003, 1–5 distinct outputs per 936–1,872 prompts), so its error is at or slightly better than a constant predictor and correlation with the labels is ≈0 (r = 0.006 val, 0.101 test for AH). |
| Was absolute humidity computed correctly? | **No.** v2's AH MAE is 0.4335 (val) / 0.5345 (test) g/m³, while the dataset's own `AH` column is reproducible from the prompt's `T` and `RH` to **0.023 / 0.019 g/m³** by a closed-form formula. v2 is ≈19–28× worse than the formula that its own inputs determine. |
| Is more training the fix? | **No.** The quantity to be predicted is not present in the prompt; it is a measurement at a timestamp. Val/test dates never occur in training (`date_overlap_with_train = 0`), so the task is a lookup, not a generalisation that more gradient steps can solve. |
| Is v2 ready for Hugging Face? | **Yes, but only as a format-following tiny-SLM artefact with an explicit non-predictor card** — not as an air-quality predictor. See §8. |

---

## 2. What was measured, and how

Both checkpoints were scored by one script (`src/evaluate_v2_full.py`) using one procedure, so every difference below is a model difference, not a harness difference.

* **Loss / perplexity** — next-token loss over *every* token of `data/training/{val,test}.bin`, in non-overlapping 128-token windows (val: 431,488 tokens / 3,371 windows; test: 870,784 tokens / 6,803 windows). No sampling, no truncation of the scored region.
* **Generations** — greedy (temperature 0, top-k 1) decoding from the four templates produced by `prepare_training_data.row_to_texts`, the same function used to build training data. Budget: 50 new tokens, capped by `BLOCK − prompt_len`.
* **Teacher-forced answer loss** — loss over answer tokens only, prompt given.
* **Numeric scoring** — numbers extracted from generated text by regex and compared to the dataset's own columns (`AH`, `T`, `RH`, `CO(GT)`, `NOx(GT)`, `NO2(GT)`, `C6H6(GT)`). Rows whose label is missing (`−200` sentinel / NaN) are dropped: 132 val and 86 test AH rows, similar counts for pollutants. Tolerance for "within 5 %" is `|pred − gold| ≤ max(0.05, 0.05|gold|)`.
* **Cached decoding** was validated token-for-token against the uncached path (`--phase verify-cache`, 6 prompts, PASSED) so the speed-up did not change outputs.
* **Baselines** — constant train-mean predictors (`docs/eval_v2/baseline.json`) give the no-information floor.

Template sizes (measured with the model's tokenizer):

| Template | Prompt tokens | Answer tokens | Total |
|---|---|---|---|
| REPORT (model card / summary) | 193 | — | 193 |
| AH (`Response:`) | 86 | 14 | 100 |
| TEMP_RH (`Answer:`) | 36 | 47 | 83 |
| POLLUTANT (`Result:`) | 43 | 45 | 88 |

This table matters: the *smallest* complete example is 83 tokens. v1 was trained with a 64-token window, so **no complete example ever fit inside one window** — which is the mechanism behind its degenerate outputs, independent of how it is evaluated.

---

## 3. Language modelling: loss and perplexity

Full splits, block 128, every token scored exactly once:

| Model | Split | Tokens | Loss | Perplexity |
|---|---|---:|---:|---:|
| v1 | val | 431,488 | 1.4252 | 4.159 |
| v1 | test | 870,784 | 1.4474 | 4.252 |
| **v2** | **val** | 431,488 | **0.4877** | **1.629** |
| **v2** | **test** | 870,784 | **0.5555** | **1.743** |

Supplementary, because v1 was trained at 64 (400-window subsample):

| Model | Split | Tokens | Loss | PPL |
|---|---|---:|---:|---:|
| v1 | val@64 | 25,600 | 1.3583 | 3.890 |
| v1 | test@64 | 25,600 | 1.4130 | 4.108 |

* Val loss falls **65.8 %** (ppl 2.55× lower); test loss falls **61.6 %** (ppl 2.44× lower).
* Giving v1 its native 64-token window only improves it from 1.4252 to 1.3583, so the gap is not an artefact of the context length — v2 is better by a wide margin under either setting.
* Generalisation gap (test − val): v2 **+0.068**, v1 **+0.022**. The gap widens, as expected for a later time period, but test loss is still 2.6× below v1's.

Teacher-forced answer loss (answer tokens only):

| Model | Split | AH | TEMP_RH | POLLUTANT | All | Mean top-1 token acc |
|---|---|---:|---:|---:|---:|---:|
| v1 | val | 1.4798 | 1.5306 | 1.3656 | 1.4587 | 0.68 |
| v1 | test | 1.4632 | 1.5762 | 1.3453 | 1.4615 | 0.67 |
| **v2** | **val** | **0.3607** | **0.5947** | **0.4720** | **0.4758** | **0.82** |
| **v2** | **test** | **0.4086** | **0.7086** | **0.4934** | **0.5369** | **0.81** |

v2's gains are largest exactly where the output is most predictable (AH template: fixed sentence, one number) and smallest on TEMP_RH, which carries two numbers plus a date.

---

## 4. Numerical accuracy

Gold = the dataset's own label for the row. `const` = train-mean predictor; `ratio` = model MAE / constant MAE (<1 means better than a constant).

### 4.1 Absolute humidity (val)

| Model | Extracted | MAE | RMSE | within 5 % | r | pred std | const MAE | ratio | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 | 94.4 % | 9.3380 | 9.8151 | 1.6 % | −0.038 | 3.009 | 0.459 | 20.36 | 752 |
| **v2** | **100 %** | **0.4335** | **0.4871** | **1.4 %** | **0.006** | **0.003** | 0.459 | **0.945** | 804 |

### 4.2 Absolute humidity (test)

| Model | Extracted | MAE | RMSE | within 5 % | r | pred std | const MAE | ratio | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 | 93.6 % | 9.0992 | 9.7186 | 0.7 % | −0.002 | 3.416 | 0.558 | 16.32 | 1,667 |
| **v2** | **100 %** | **0.5345** | **0.5849** | **3.4 %** | **0.101** | **0.003** | 0.558 | **0.959** | 1,786 |

v1's AH numbers do exist in its output but are corrupted into strings such as `11.0.0.0.0.0.0.0.0.0.0.0.0.04`; the extractor still credits 94 % of them, which is why its MAE is 9.3 rather than infinite. v2 always produces a well-formed number — the same one.

### 4.3 All quantities, v2

| Quantity | Split | MAE | RMSE | within 5 % | pred std | const MAE | ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| AH (g/m³) | val | 0.4335 | 0.4871 | 1.4 % | 0.003 | 0.459 | 0.945 |
| AH (g/m³) | test | 0.5345 | 0.5849 | 3.4 % | 0.003 | 0.558 | 0.959 |
| T (°C) | val | 11.0262 | 11.5868 | 0.0 % | 0.000 | 11.491 | 0.960 |
| T (°C) | test | 11.5356 | 12.7249 | 3.1 % | 0.000 | 11.951 | 0.965 |
| RH (%) | val | 36.3190 | 39.4636 | 0.9 % | 0.000 | 15.713 | **2.311** |
| RH (%) | test | 29.3173 | 33.4316 | 1.4 % | 0.000 | 13.608 | **2.154** |
| CO (mg/m³) | val | 1.2180 | 1.6551 | 8.3 % | 0.000 | 1.224 | 0.995 |
| CO (mg/m³) | test | 1.1197 | 1.3933 | 8.1 % | 0.000 | 1.100 | 1.018 |
| NOx (ppb) | val | 300.8435 | 382.9045 | 1.0 % | 0.000 | 233.888 | **1.286** |
| NOx (ppb) | test | 202.5402 | 285.2763 | 2.5 % | 0.000 | 157.689 | **1.284** |
| NO2 (µg/m³) | val | 35.9242 | 45.4307 | 11.4 % | 0.000 | 42.188 | 0.852 |
| NO2 (µg/m³) | test | 49.4129 | 62.9303 | 7.4 % | 0.000 | 55.509 | 0.890 |
| C6H6 (µg/m³) | val | 5.8106 | 6.9737 | 5.0 % | 0.000 | 5.678 | 1.023 |
| C6H6 (µg/m³) | test | 6.3469 | 7.3235 | 3.4 % | 0.000 | 6.128 | 1.036 |

v1 produces no extractable predictions at all for T, RH, CO, NOx, NO2 or C6H6 — its output never reaches the trained answer form. A lenient offline re-parse of v1's stored output recovers NOx only (`NOx(GT)=111`, 936/936, MAE 301.3) and nothing for the other five quantities, so this is a genuine v1 failure, not an extractor artefact.

**Reading of the table:** `pred std = 0.000` for every quantity except AH (0.003) means v2 answers with a constant. Its error is therefore the *offset between that constant and the split's distribution*. It is marginally better than a train-mean constant only because the constant it picked was nudged slightly downward. RH and NOx are **worse than the constant baseline**. "Within 5 %" scores of 3–11 % are the small fraction of rows that happen to sit near the constant, not skill.

### 4.4 Can absolute humidity be computed from the prompt? Yes.

The dataset's `AH` column is, to within measurement rounding, a scaled Magnus formula of the *same row's* `T` and `RH`:

```
e_s  = 6.112 · exp(17.67·T / (T + 243.5))          # hPa, Tetens
AH  ≈ 0.1351678 · 216.7 · (e_s · RH / 100) / (T + 273.15)
```

(`0.1351678` is a single scale factor fitted on **train only**, least squares; the rest is the physical form, so the fit uses no val/test information)

| Split | n | MAE | RMSE | r | constant MAE |
|---|---:|---:|---:|---:|---:|
| train | 6,401 | 0.01881 | 0.02327 | 0.99807 | 0.29173 |
| val | 804 | **0.02296** | 0.02388 | **0.99979** | 0.45874 |
| test | 1,786 | **0.01880** | 0.02024 | **0.99956** | 0.55756 |

So the AH prompt already contains everything needed to answer exactly. v2's 0.4335 / 0.5345 is an arithmetic failure, not an information limit: it is **18.9× (val) and 28.4× (test) worse than the closed form**. This is the strongest single piece of evidence that the remaining error is architectural, not statistical.

### 4.5 Paired comparison on identical prompts (AH only)

| Split | v2 better | v2 worse | n |
|---|---:|---:|---:|
| val | 679 | 73 | 752 |
| test | 1,458 | 209 | 1,667 |

Paired comparison is possible only for AH; v1 yields no comparable numbers elsewhere.

---

## 5. Output structure, repetition, units

Fraction of generated outputs (2,808 val / 5,616 test):

| Metric | v1 val | v1 test | v2 val | v2 test |
|---|---:|---:|---:|---:|
| Emitted EOS | 0.000 | 0.000 | **1.000** | **1.000** |
| Hit token cap | **1.000** | **1.000** | **0.000** | **0.000** |
| Repetitive text | 0.352 | 0.355 | **0.000** | **0.000** |
| No digits at all | 0.019 | 0.021 | 0.000 | 0.000 |
| Correct unit | 0.055 | 0.053 | **1.000** | **1.000** |
| Wrong unit | 0.333 | 0.333 | **0.000** | **0.000** |
| Replacement characters | 0 | 0 | 0 | 0 |
| Empty output | 0.000 | 0.000 | 0.000 | 0.000 |
| Malformed (any of the above) | **1.000** | **1.000** | **0.000** | **0.000** |
| Hallucinated date | 0.000 | 0.000 | **0.333** | **0.333** |

Distinct outputs and the modal answer:

| Split / template | Distinct | Modal share | Modal output |
|---|---:|---:|---|
| v2 val AH | 3 | 84.3 % | `The absolute humidity is 1.15 g/m3.` |
| v2 val TEMP_RH | 5 | 83.5 % | `On 04/04/2004 at 04.00.00, the ambient temperature was 21.2 degrees Celsius and the relative humidity was 21.2%.` |
| v2 val POLLUTANT | **1** | **100 %** | `Recorded CO is 2.2 mg/m3, NOx is 111 ppb, and NO2 is 111 ug/m3 with benzene level at 11.2 ug/m3.` |
| v2 test AH | 3 | 93.1 % | identical to val |
| v2 test TEMP_RH | 5 | 89.2 % | identical to val |
| v2 test POLLUTANT | **1** | **100 %** | identical to val |

v2 also opens with exactly the gold answer's leading text in **66.7 %** of generations (two of the three templates), i.e. it reproduces the surface form of the trained answer while inventing the payload. It is a flawless *formatter* and a constant *speaker*. Note that the TEMP_RH sentence echoes a fixed date and time (`04/04/2004 04.00.00`) in every one of 2,808 generations, i.e. 100 % of TEMP_RH prompts and 33.3 % of all prompts hallucinate the timestamp; it also reports temperature and humidity as the same number (21.2), which is a semantic error a unit check does not catch.

---

## 6. Memorisation

| Check | v1 val | v1 test | v2 val | v2 test |
|---|---:|---:|---:|---:|
| Output verbatim equals a training answer | 0.00 % | 0.00 % | **33.33 %** | **33.33 %** |
| Output timestamp matches an existing train row | 0 | 0 | 936 | 1,872 |
| Dates in output never seen in the dataset | 0 | 0 | 0 | 0 |
| Date overlap between split and train | 0 | 0 | 0 | 0 |

The 33.33 % "verbatim" figure is exactly one third of prompts: the POLLUTANT template, whose single constant output happens to be a string that appears in the training set. It is **not** record-level recall — the model does not reproduce the right numbers for any specific row. The 100 % "train row key" figure comes from the hallucinated `04/04/2004 04.00.00` timestamp; the answers attached to it are wrong by 23.9 (val) / 24.9 (test) on average, so nothing is being retrieved either.

Splits are temporal: no val or test date appears in training, so there is no leakage to explain the constant behaviour.

---

## 7. Two concrete defects found and fixed in the Ollama artefact

### 7.1 `num_predict 40` silently truncated the temperature/RH answer

Scoring the same val split with a 40-token budget versus 50:

| Budget | Ran to cap | RH values extracted |
|---|---:|---:|
| `num_new = 40` | **66.7 %** | **0 / 936 (0 %)** |
| `num_new = 50` | 0.0 % | 936 / 936 (100 %) |

The TEMP_RH answer is 47 tokens; at a 40-token budget it is cut before the RH value is emitted. Any user of the original model would have seen a temperature-only sentence and concluded the model had no idea about humidity. `Modelfile.improved` now sets **`num_predict 48`**, which covers the 47-token answer while keeping prompt + answer inside 128 positions (36 + 48 = 84 for TEMP_RH, 43 + 48 = 91 for POLLUTANT; the AH answer is only 14 tokens and stops at EOS well before the limit).

### 7.2 Rebuild and verification

`air-quality-slm-improved` was rebuilt from `Modelfile.improved` and exercised through the Ollama HTTP API:

| Check | Result |
|---|---|
| Context length reported by `api/ps` | **128** |
| GGUF metadata (`qwen3.context_length`) | **128** |
| AH prompt (86 tokens) fully evaluated | `prompt_eval_count = 86`, no truncation |
| TEMP_RH prompt evaluated | `prompt_eval_count = 36` |
| TEMP_RH answer completes | `eval_count = 47`, `done_reason = stop`, RH value present |
| Sampling | deterministic (temp 0, top_k 1) |

Example (val row, gold `16.7 °C`, `54.7 %`):

```
prompt (36 tok) → ' On 04/04/2004 at 04.00.00, the ambient temperature was 21.2 degrees Celsius and the relative humidity was 21.2%.'
```

The pipeline is correct and the context is configured; the *content* is the constant described in §5.

---

## 8. Decision: more training, or retrieval?

**Retrieval (or a deterministic formula) is required. More training on the present objective will not help.**

The reason is structural, not a matter of scale or epochs:

1. **The answer is not in the prompt.** The prompts give a date and a time. The temperature, RH and pollutant concentrations for that instant are measurements from a specific sensor, recorded in the dataset. There is no function of `(date, time)` that predicts them; the split is temporal and `date_overlap_with_train = 0`, so no val/test timestamp was ever seen.
2. **Every number v2 could learn is a constant or a memorised row.** It chose the constant (prediction std 0.000–0.003, 1–5 distinct outputs). Training longer, or with a larger model, can only move that constant toward the train mean, which is already where it sits — and on RH and NOx it is *already worse* than the constant.
3. **Where the prompt does determine the answer, the model still fails.** AH is a pure function of T and RH, both in the prompt, and the closed form reaches MAE 0.023 g/m³ against v2's 0.4335. The gap is 19–28×, so this is not a data-quantity problem either.
4. **The 20.4 M-parameter, 4-layer, d=128 architecture** is a formatting engine. It does that job perfectly (100 % EOS, units, no repetition) and has no capacity left for multi-step arithmetic, which is what AH from T/RH requires.

What to build instead, in order of value:

* **(a) Deterministic lookup / tool call.** `(date, time) → row` from the dataset or a small SQLite/D1 table; `(T, RH) → AH` from the Magnus formula in §4.4. Exact, auditable, ~0 cost. The SLM stays as the natural-language front end: it parses the question and formats the tool's answer.
* **(b) Reformulate the tasks so the inputs determine the outputs** — unit conversion, daily averages over readings supplied in the prompt, threshold classification ("is CO above 8.5 mg/m³?"), AH from given T and RH. A model this size can genuinely learn these; AH-from-T/RH with ~10⁴ synthetic examples would be a fair next training run and would be measurable against a formula rather than against a constant.
* **(c) Only then, scale.** If the goal is forecasting rather than lookup, the input must change to a *window of prior readings* (a time-series model), which is a different project, not a continuation of this one.

Do not run another epoch on the current `(date, time) → value` templates expecting accuracy to improve.

---

## 9. Publication recommendation

**Publish v2 — as an explicitly scoped tiny-SLM demonstration, not as an air-quality predictor. Keep v1 unpublished or as the baseline in the same repository.**

Recommended framing for the model card:

* **Name/tags:** educational, tiny (20.4 M params), instruction-following / structured-output demo, GGUF for Ollama, trained from scratch on a single UCI Air Quality table.
* **State the numbers measured here:** AH MAE 0.4335 val / 0.5345 test (r ≈ 0.01 / 0.10); every other quantity emitted as a per-template constant; 33 % of outputs echo a fixed `04/04/2004 04.00.00` timestamp; RH and NOx worse than a train-mean baseline.
* **State plainly:** "Outputs are not measurements. Do not use for regulatory, health or safety decisions. For the correct value at a timestamp, look up the dataset row; for AH, use `AH = 0.1352 · 216.7 · (6.112·exp(17.67T/(T+243.5)) · RH/100) / (T+273.15)`."
* **Ship `num_predict 48`** with the GGUF (a 40-token budget truncates the temperature/RH answer, §7.1), and include this report plus `docs/eval_v2/` so the claims are reproducible.

If the intent is to publish something people can *use* as an air-quality model, then **hold v2** and ship option (a) first: retrieval or the formula behind the same interface. v2 is publishable as a learning artefact today; it is not publishable as a predictor.

---

## 10. Limitations of this evaluation

* **v1 is judged partly outside its design envelope.** It was trained at 64 tokens; the generation pass used the shared 128-token budget. The loss comparison is therefore also reported at 64 (1.3583 / 1.4130), and the conclusion is unchanged: v2 is far better under both settings. v1's degenerate generations are not *only* a context-length effect — the smallest complete template is 83 tokens, so v1 never saw a complete example in one window at all.
* **v1's numeric scores understate its outputs' brokenness.** The extractor still credits v1 with 94 % of AH values despite digit corruption, producing a finite 9.34 MAE; a stricter parser would fail them outright. Conversely, v1's format (`NOx(GT)=111`) is invisible to the shared extractor for most pollutants, so its real numbers are not scored. Both directions are disclosed rather than resolved, since the instruction was to use one identical procedure.
* **Single deterministic decode.** Greedy decoding was used for both models, so sampling variance is not characterised.
* **Missing labels dropped, not imputed.** 132 val / 86 test rows lack an AH label and 58–131 lack pollutant labels; those rows are excluded per quantity, so `n` differs slightly by quantity.
* **The REPORT template (193 tokens) exceeds the 128-token window** and is never generated; it is scored only as language-modelling loss.
* **The test split was used for reporting only.** It was not used for training, checkpoint selection, or tuning of the extractor — the extractor and tolerance were fixed on val before test was scored.

---

## 11. Reproduction

```
./venv/Scripts/python.exe -u src/evaluate_v2_full.py --phase baseline
./venv/Scripts/python.exe -u src/evaluate_v2_full.py --phase suite --suite loss    --threads 6
./venv/Scripts/python.exe -u src/evaluate_v2_full.py --phase suite --suite metrics --max-new 50 --threads 6
./venv/Scripts/python.exe -u src/summarize_eval_v2.py      # → docs/eval_v2/summary.json + tables
./venv/Scripts/python.exe -u src/diagnose_v2_numbers.py    # → docs/eval_v2/numeric_diagnostics.json
```

`src/evaluate_v2_full.py` supports `--phase loss|metrics|verify-cache|baseline|suite`, `--block`, `--max-windows`, `--max-new` and `--threads`. Decode speed on this machine: ~0.17 s/prompt cached, ~1,144 tok/s at batch 8 × 128 with 12 threads.
