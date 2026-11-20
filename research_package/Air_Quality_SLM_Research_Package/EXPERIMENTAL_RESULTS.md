# Experimental Results

Every number in this file is transcribed from a stored artefact in this package or computed from the
dataset CSVs; the source file is named in each table. Nothing here was re-run or re-estimated for the
package.

**Source files:** `evaluation/results/loss_v1_128.json`, `loss_v2_128.json`, `loss_v1_64.json`,
`summary_tables.txt`, `baseline.json`, `numeric_diagnostics.json`, and the per-row generation dumps
`metrics_v1_{val,test}.json`, `metrics_v2_{val,test}.json`.

---

## 1. Models compared

| Label | Checkpoint | Context | Training |
|---|---|---|---|
| **v1** | `models/air_quality_slm/pytorch_model.bin` (SHA256 `77be6061…`) | 64 | 150 steps from random init, batch 8, block 64 |
| **v2** | `models/air_quality_slm_finetuned/pytorch_model.bin` | 128 | continued from v1, 1,200 steps, block 128, 0.202 epoch |
| **v3** | *same weights as v2* | 128 | **no training, no evaluation** — Modelfile variant only (`TEMPLATE "{{ .System }}\n{{ .Prompt }}"`) |

## 2. Training dynamics

### 2.1 v1 (`models/air_quality_slm/training_summary.json`)

| Step | Learning rate | Train loss | Val loss | Val perplexity |
|---:|---:|---:|---:|---:|
| 0 | 1.00e-4 (warmup start) | 11.8904 | 11.8830 | 144,785.12 |
| 40 | 9.48e-4 | 5.0992 | 5.1368 | 170.18 |
| 80 | 6.04e-4 | 2.2676 | 2.2905 | 9.88 |
| 120 | 2.13e-4 | 1.5242 | 1.5600 | 4.76 |
| 149 | 1.00e-4 | **1.3390** | **1.3651** | **3.92** |

Test loss 1.4295, test perplexity 4.18. Runtime 279.01 s on CPU. Figure `fig05`.

### 2.2 v2 (`training/logs/loss_curve.csv`, `training_log_finetune.jsonl`)

| Step | Train loss (EMA) | Val loss | Val PPL | Tokens seen | Epochs | Elapsed (s) |
|---:|---:|---:|---:|---:|---:|---:|
| baseline | — | 1.4433 | 4.235 | 0 | — | — |
| 200 | 0.6425 | 0.6957 | 2.005 | 102,400 | 0.034 | 274.8 |
| 400 | 0.4955 | 0.5559 | 1.744 | 204,800 | 0.067 | 586.2 |
| 600 | 0.4699 | 0.5279 | 1.695 | 307,200 | 0.101 | 900.6 |
| 800 | 0.4470 | 0.5187 | 1.680 | 409,600 | 0.135 | 1,222.6 |
| **1,000** | 0.4325 | **0.4896** | **1.632** | 512,000 | 0.168 | 1,544.2 |
| 1,200 | 0.4229 | 0.4905 | 1.633 | 614,400 | 0.202 | 1,864.1 |

Validation loss fell 66 % (1.4433 → 0.4896), perplexity 4.24 → 1.63. Training stopped on the
overfitting signature (val rising while train keeps falling). Figure `fig06`.

## 3. Language modelling: loss and perplexity (full splits)

Identical harness, block 128, every token scored once:

| Model | Split | Tokens | Loss | Perplexity |
|---|---|---:|---:|---:|
| v1 | val | 431,488 | 1.4252 | 4.159 |
| v1 | test | 870,784 | 1.4474 | 4.252 |
| **v2** | **val** | 431,488 | **0.4877** | **1.629** |
| **v2** | **test** | 870,784 | **0.5555** | **1.743** |

Supplementary (v1 at its native 64-token window, 400-window subsample):

| Model | Split | Tokens | Loss | PPL |
|---|---|---:|---:|---:|
| v1 | val@64 | 25,600 | 1.3583 | 3.890 |
| v1 | test@64 | 25,600 | 1.4130 | 4.108 |

* Val loss −65.8 %, test loss −61.6 %.
* v1 at 64 is still far worse than v2 at 128 (1.3583 vs 0.4877), so the gap is not an artefact of
  context length.
* Generalisation gap (test − val): v2 +0.068, v1 +0.022 — larger, as expected for a later period.

Figure `fig07`.

## 4. Teacher-forced answer loss (answer tokens only)

| Model | Split | AH | TEMP_RH | POLLUTANT | All | Mean top-1 token accuracy |
|---|---|---:|---:|---:|---:|---:|
| v1 | val | 1.4798 | 1.5306 | 1.3656 | 1.4587 | 0.68 |
| v1 | test | 1.4632 | 1.5762 | 1.3453 | 1.4615 | 0.67 |
| **v2** | **val** | **0.3607** | **0.5947** | **0.4720** | **0.4758** | **0.82** |
| **v2** | **test** | **0.4086** | **0.7086** | **0.4934** | **0.5369** | **0.81** |

v2's gain is largest on the most predictable template (AH: fixed sentence, one number) and smallest
on TEMP_RH (two numbers plus a date).

## 5. Numerical accuracy

Gold = the dataset's own label. `ratio` = model MAE / constant (train-mean) MAE; < 1 beats a constant.

### 5.1 Absolute humidity

| Model | Split | Extracted | MAE | RMSE | within 5 % | r | pred std | const MAE | ratio | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 | val | 94.4 % | 9.3380 | 9.8151 | 1.6 % | −0.038 | 3.009 | 0.459 | 20.36 | 752 |
| **v2** | val | 100 % | **0.4335** | 0.4871 | 1.4 % | 0.006 | 0.003 | 0.459 | 0.945 | 804 |
| v1 | test | 93.6 % | 9.0992 | 9.7186 | 0.7 % | −0.002 | 3.416 | 0.558 | 16.32 | 1,667 |
| **v2** | test | 100 % | **0.5345** | 0.5849 | 3.4 % | 0.101 | 0.003 | 0.558 | 0.959 | 1,786 |

v1's AH numbers are corrupted strings (`11.0.0.0.0.0.0.0.0.0.0.0.0.04`); the extractor still credits
94 % of them, which is why v1's MAE is finite (9.34) rather than infinite. v2 always emits a
well-formed number — the same one every time.

### 5.2 All quantities, v2

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

**Interpretation.** `pred std = 0.000` for every quantity except AH (0.003): v2 answers with a
constant. Its error is the offset between that constant and the split distribution. RH and NOx are
**worse than a constant predictor**. "Within 5 %" hits are rows that happen to sit near the constant,
not skill. Figures `fig08`, `fig09`.

v1 produces **no extractable predictions at all** for T, RH, CO, NOx, NO2, C6H6; a lenient offline
re-parse recovers NOx only (`NOx(GT)=111`, 936/936, MAE 301.3). Both directions are disclosed rather
than resolved.

### 5.3 The prompt already determines the answer (AH)

| Split | n | MAE | RMSE | r | constant MAE |
|---|---:|---:|---:|---:|---:|
| train | 6,401 | 0.01881 | 0.02327 | 0.99807 | 0.29173 |
| val | 804 | **0.02296** | 0.02388 | **0.99979** | 0.45874 |
| test | 1,786 | **0.01880** | 0.02024 | **0.99956** | 0.55756 |

v2 is **18.9× (val) / 28.4× (test) worse than the closed form** whose inputs are both in its prompt.
This is the single strongest piece of evidence that the remaining error is architectural, not
statistical. Figure `fig10`.

### 5.4 Paired comparison on identical prompts (AH)

| Split | v2 better | v2 worse | n |
|---|---:|---:|---:|
| val | 679 | 73 | 752 |
| test | 1,458 | 209 | 1,667 |

## 6. Output structure, repetition and units

Fraction of generated outputs (2,808 val / 5,616 test):

| Metric | v1 val | v1 test | v2 val | v2 test |
|---|---:|---:|---:|---:|
| Emitted EOS | 0.000 | 0.000 | **1.000** | **1.000** |
| Hit token cap | **1.000** | **1.000** | 0.000 | 0.000 |
| Repetitive text | 0.352 | 0.355 | **0.000** | **0.000** |
| No digits at all | 0.019 | 0.021 | 0.000 | 0.000 |
| Correct unit | 0.055 | 0.053 | **1.000** | **1.000** |
| Wrong unit | 0.333 | 0.333 | **0.000** | 0.000 |
| Replacement characters | 0 | 0 | 0 | 0 |
| Empty output | 0.000 | 0.000 | 0.000 | 0.000 |
| Malformed (any of the above) | **1.000** | **1.000** | **0.000** | **0.000** |
| Hallucinated date | 0.000 | 0.000 | **0.333** | **0.333** |

Distinct outputs:

| Split / template | Distinct | Modal share | Modal output |
|---|---:|---:|---|
| v2 val AH | 3 | 84.3 % | `The absolute humidity is 1.15 g/m3.` |
| v2 val TEMP_RH | 5 | 83.5 % | `On 04/04/2004 at 04.00.00, the ambient temperature was 21.2 degrees Celsius and the relative humidity was 21.2%.` |
| v2 val POLLUTANT | **1** | **100 %** | `Recorded CO is 2.2 mg/m3, NOx is 111 ppb, and NO2 is 111 ug/m3 with benzene level at 11.2 ug/m3.` |
| v2 test AH | 3 | 93.1 % | identical to val |
| v2 test TEMP_RH | 5 | 89.2 % | identical to val |
| v2 test POLLUTANT | **1** | **100 %** | identical to val |

v2 opens with the gold answer's leading text in 66.7 % of generations: it reproduces the *surface
form* of the trained answer while inventing the payload. It is a flawless formatter and a constant
speaker. The TEMP_RH sentence echoes a fixed date/time in every generation and reports temperature
and humidity as the same number (21.2) — a semantic error no unit check detects. Figure `fig11`.

## 7. Memorisation

| Check | v1 val | v1 test | v2 val | v2 test |
|---|---:|---:|---:|---:|
| Output verbatim equals a training answer | 0.00 % | 0.00 % | 33.33 % | 33.33 % |
| Output timestamp matches an existing train row | 0 | 0 | 936 | 1,872 |
| Dates in output never seen in the dataset | 0 | 0 | 0 | 0 |
| Date overlap between split and train | 0 | 0 | 0 | 0 |

The 33.33 % is exactly one third of prompts — the POLLUTANT template, whose single constant output
is a string that also appears in training. It is not record-level recall: the numbers attached to the
matched timestamp are wrong by 23.9 (val) / 24.9 (test) on average, so nothing is being retrieved.

## 8. Two deployed-artefact defects found and fixed

### 8.1 `num_predict 40` truncated the temperature/RH answer

| Budget | Ran to cap | RH values extracted |
|---|---:|---:|
| 40 tokens | 66.7 % | **0 / 936 (0 %)** |
| 50 tokens | 0.0 % | 936 / 936 (100 %) |

The TEMP_RH answer is 47 tokens; at 40 the sentence is cut before the RH value. `Modelfile.improved`
now sets `num_predict 48`.

### 8.2 Context truncation at 64 tokens (v1 deployment)

Ollama's GGUF declared `context_length 64`, so an 86-token AH prompt was reduced to its last 34
tokens (`prompt_eval_count` measured as 34 for any prompt ≥ 64 tokens). The model never received the
instruction. Fixed by exporting v2 with context 128; `api/ps` and GGUF metadata then report 128, and
`prompt_eval_count` equals the true token count (86).

## 9. v3: what the evidence supports

* `air-quality-slm-v3:latest` is registered (modified 2026-11-19 18:57) with `num_ctx 128`,
  `num_predict 48`, greedy decoding, and a template that prepends `{{ .System }}`.
* Its GGUF `FROM` line points at **`models/air_quality_slm_finetuned/air_quality_slm_finetuned.gguf`
  — the same file v2 uses**.
* **No evaluation of v3 exists**: no metrics file, no report, no generation dump. Its digest differs
  from v2 only because the Modelfile (parameters/template) differs.

Therefore: v3 **cannot** be described as better than v2. If the paper compares versions, it compares
v1 and v2, and states that v3 is an interface variant whose system-prompt path was never measured.

## 10. Documented limitations (all measured, none hidden)

| Limitation | Evidence |
|---|---|
| **Constant numerical output** | prediction std 0.000–0.003; 1–5 distinct outputs per 936–1,872 prompts (§6) |
| **Incorrect humidity prediction** | AH MAE 0.4335 / 0.5345 vs 0.023 / 0.019 for the closed form (§5.3) |
| **Hallucinated readings and timestamps** | 33.3 % of outputs echo `04/04/2004 04.00.00`; pollutant values 2.2/111/111 vs true 1.2/179/82 |
| **Incorrect or mislabelled units in the application** | `app.py::handle_retrieval` prints CO and NOx as `ug/m3` although the dataset is mg/m³ and ppb — see `MISSING_EVIDENCE.md` §5 |
| **Repetitive output (v1)** | 35 % repetitive, 100 % hit the token cap (§6) |
| **Truncated responses** | 40-token cap removed the RH value in 100 % of TEMP_RH answers (§8.1) |
| **Context-window limit** | 64-token window vs 83-token smallest template; report template is 193 tokens and never fits 128 |
| **Failure on unsupported questions** | the model has no refusal training; refusals are produced by `domain_guard.py`, not by the network |
| **v1 numerical scores understate breakage** | 94 % of v1 AH values are digit-corrupted but still extracted as finite numbers |
| **Single deterministic decode** | greedy only; sampling variance not characterised |
| **Only 0.202 epoch of v2 training** | 614,400 of 3,041,432 tokens |
| **NMHC unevaluable on val/test** | 100 % missing in both splits |

## 11. Separation of concerns for the paper

| Statement | Belongs to |
|---|---|
| "v2 reduces full-split test loss from 1.4474 to 0.5555" | language modelling — legitimate claim |
| "v2 emits correct units in 100 % of outputs" | formatting — legitimate claim |
| "v2 predicts absolute humidity with MAE 0.53 g/m³" | numeric failure — report as such |
| "The application returns 1.0284 g/m³ for T = 16.7 °C, RH = 54.7 %" | **Python formula**, not the model |
| "The application returns NOx 179.0 ppb for 08/12/2004 15:00" | **dataset lookup**, not the model |
| "The model answers questions about air quality" | **do not claim** — it continues text patterns |

## 12. Reproduce these tables

```bash
./venv/Scripts/python.exe -u src/evaluate_v2_full.py --phase baseline
./venv/Scripts/python.exe -u src/evaluate_v2_full.py --phase suite --suite loss    --threads 6
./venv/Scripts/python.exe -u src/evaluate_v2_full.py --phase suite --suite metrics --max-new 50 --threads 6
./venv/Scripts/python.exe -u src/summarize_eval_v2.py     # -> docs/eval_v2/summary.json, summary_tables.txt
./venv/Scripts/python.exe -u src/diagnose_v2_numbers.py   # -> docs/eval_v2/numeric_diagnostics.json
```

Internally consistent check: the per-row dumps let every MAE above be recomputed with a five-line
pandas script (see `REPRODUCIBILITY.md` §5).
