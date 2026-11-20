# Verified Performance Metrics

The complete list of numbers this project can legitimately report, with the artefact each comes from. If a
metric is not on this page, it was not measured.

All values were read from the stored JSON/CSV artefacts in this repository; none was re-estimated.

---

## 1. Dataset (measured from `data/raw/air+quality/AirQualityUCI.csv`)

| Metric | Value |
|---|---|
| Data rows | 9,357 |
| Columns | 15 |
| Unique calendar dates | 391 |
| Duplicate `(Date, Time)` pairs | 0 |
| Sentinel (`-200`) cells | 16,701 (12.0 % of measurement cells) |
| NaNs: `CO(GT)` | 1,683 (17.99 %) |
| NaNs: `NOx(GT)` | 1,639 (17.52 %) |
| NaNs: `NO2(GT)` | 1,642 (17.55 %) |
| NaNs: `NMHC(GT)` | 8,443 (90.23 %) |
| NaNs: each PT08 / T / RH / AH | 366 (3.91 %) |
| `NMHC(GT)` missing in val and test | **100 %** |
| `T` range | −1.9 … 44.6 °C |
| `AH` range / mean | 0.1847 … 2.231 / 1.0255 g/m³ |

| Split | Rows | Texts | Tokens |
|---|---:|---:|---:|
| train | 6,549 | 26,196 | 3,041,432 |
| validation | 936 | 3,744 | 431,740 |
| test | 1,872 | 7,488 | 870,987 |

## 2. Model size

| Metric | Value | Source |
|---|---|---|
| Trainable parameters | 20,397,568 | `models/air_quality_slm/training_summary.json` |
| Stored parameters | 39,810,688 (embeddings stored twice, tied) | weight file sizes |
| Vocabulary | 151,665 | tokenizer |
| Layers / width / heads | 4 / 128 / 4 (2 KV) | `config.json` |
| GGUF size | 165,774,560 B (v1) · 165,774,528 B (v2) | file sizes |

## 3. Training

| Metric | v1 | v2 | Source |
|---|---|---|---|
| Steps | 150 | 1,200 | training summary / log |
| Tokens seen | 76,800 (2.53 %) | 614,400 (0.202 epoch) | derived from config / log |
| Initial val loss | 11.8830 | 1.4433 | summary / `loss_curve.csv` |
| Final train loss | 1.3390 | 0.4229 (EMA) | summary / log |
| Best val loss | 1.3651 | **0.4896** (step 1,000) | summary / log |
| Best val perplexity | 3.92 | 1.632 | summary / log |
| Test loss / PPL (own harness) | 1.4295 / 4.18 | — | `training_summary.json` |
| Runtime | 279.01 s | 1,864 s | summary / log |

## 4. Full-split language modelling (identical harness)

| Model | Split | Tokens | Windows | Loss | Perplexity |
|---|---|---:|---:|---:|---:|
| v1 | val | 431,488 | 3,371 | 1.4252 | 4.159 |
| v1 | test | 870,784 | 6,803 | 1.4474 | 4.252 |
| v2 | val | 431,488 | 3,371 | **0.4877** | **1.629** |
| v2 | test | 870,784 | 6,803 | **0.5555** | **1.743** |
| v1 | val@64 | 25,600 | 400 | 1.3583 | 3.890 |
| v1 | test@64 | 25,600 | 400 | 1.4130 | 4.108 |

Derived: val loss −65.8 %, test loss −61.6 %, test perplexity 4.252 → 1.743.

## 5. Teacher-forced answer loss

| Model | Split | AH | TEMP_RH | POLLUTANT | All | Top-1 |
|---|---|---:|---:|---:|---:|---:|
| v1 | val | 1.4798 | 1.5306 | 1.3656 | 1.4587 | 0.68 |
| v1 | test | 1.4632 | 1.5762 | 1.3453 | 1.4615 | 0.67 |
| v2 | val | 0.3607 | 0.5947 | 0.4720 | **0.4758** | 0.82 |
| v2 | test | 0.4086 | 0.7086 | 0.4934 | **0.5369** | 0.81 |

## 6. Numerical accuracy — v2

| Quantity | Split | MAE | RMSE | within 5 % | r | pred std | const MAE | ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| AH (g/m³) | val | 0.4335 | 0.4871 | 1.4 % | 0.006 | 0.003 | 0.459 | 0.945 |
| AH (g/m³) | test | 0.5345 | 0.5849 | 3.4 % | 0.101 | 0.003 | 0.558 | 0.959 |
| T (°C) | val | 11.0262 | 11.5868 | 0.0 % | n/a | 0.000 | 11.491 | 0.960 |
| T (°C) | test | 11.5356 | 12.7249 | 3.1 % | n/a | 0.000 | 11.951 | 0.965 |
| RH (%) | val | 36.3190 | 39.4636 | 0.9 % | n/a | 0.000 | 15.713 | **2.311** |
| RH (%) | test | 29.3173 | 33.4316 | 1.4 % | n/a | 0.000 | 13.608 | **2.154** |
| CO (mg/m³) | val | 1.2180 | 1.6551 | 8.3 % | n/a | 0.000 | 1.224 | 0.995 |
| CO (mg/m³) | test | 1.1197 | 1.3933 | 8.1 % | n/a | 0.000 | 1.100 | 1.018 |
| NOx (ppb) | val | 300.8435 | 382.9045 | 1.0 % | n/a | 0.000 | 233.888 | **1.286** |
| NOx (ppb) | test | 202.5402 | 285.2763 | 2.5 % | n/a | 0.000 | 157.689 | **1.284** |
| NO2 (µg/m³) | val | 35.9242 | 45.4307 | 11.4 % | n/a | 0.000 | 42.188 | 0.852 |
| NO2 (µg/m³) | test | 49.4129 | 62.9303 | 7.4 % | n/a | 0.000 | 55.509 | 0.890 |
| C₆H₆ (µg/m³) | val | 5.8106 | 6.9737 | 5.0 % | n/a | 0.000 | 5.678 | 1.023 |
| C₆H₆ (µg/m³) | test | 6.3469 | 7.3235 | 3.4 % | n/a | 0.000 | 6.128 | 1.036 |

## 7. Numerical accuracy — v1

| Quantity | Split | Extracted | MAE | RMSE | pred std |
|---|---|---:|---:|---:|---:|
| AH (g/m³) | val | 94.4 % | 9.3380 | 9.8151 | 3.009 |
| AH (g/m³) | test | 93.6 % | 9.0992 | 9.7186 | 3.416 |
| T, RH, CO, NOx, NO₂, C₆H₆ | both | **0 %** | not extractable | — | — |

## 8. Closed-form reference (the strongest result)

| Split | n | MAE | RMSE | r | constant MAE |
|---|---:|---:|---:|---:|---:|
| train | 6,401 | 0.01881 | 0.02327 | 0.99807 | 0.29173 |
| val | 804 | **0.02296** | 0.02388 | 0.99979 | 0.45874 |
| test | 1,786 | **0.01880** | 0.02024 | 0.99956 | 0.55756 |

Model/closed-form MAE ratio: **18.9× (val)**, **28.4× (test)**.

## 9. Output structure (fraction of outputs)

| Metric | v1 val | v1 test | v2 val | v2 test |
|---|---:|---:|---:|---:|
| Emitted EOS | 0.000 | 0.000 | 1.000 | 1.000 |
| Ran to token cap | 1.000 | 1.000 | 0.000 | 0.000 |
| Repetitive | 0.352 | 0.355 | 0.000 | 0.000 |
| Correct unit | 0.055 | 0.053 | 1.000 | 1.000 |
| Wrong unit | 0.333 | 0.333 | 0.000 | 0.000 |
| Hallucinated date | 0.000 | 0.000 | 0.333 | 0.333 |
| Malformed | 1.000 | 1.000 | 0.000 | 0.000 |

| Diversity (v2, val) | Distinct | Modal share |
|---|---:|---:|
| AH | 3 | 84.3 % |
| TEMP_RH | 5 | 83.5 % |
| POLLUTANT | 1 | 100 % |

## 10. Paired comparison (identical prompts, AH)

| Split | v2 better | v2 worse | n |
|---|---:|---:|---:|
| val | 679 | 73 | 752 |
| test | 1,458 | 209 | 1,667 |

## 11. Deployment checks

| Check | Result |
|---|---|
| Tokenizer parity, HF vs GGUF | 13/13 identical |
| `n_ctx_train` | 64 (v1), 128 (v2, v3) |
| `prompt_eval_count`, 86-token prompt, ctx 128 | 86 |
| `prompt_eval_count`, ctx 64 | 34 (truncation defect) |
| Greedy Ollama output vs PyTorch | token-identical |
| RH values extracted at `num_predict 40` / 50 | 0/936 / 936/936 |
| Registered tags | 3 (`:latest`, `-improved:latest`, `-v3:latest`) |

## 12. Application values (Python, **not** model output)

| Query | Result | Source |
|---|---|---|
| `T = 16.7 °C, RH = 54.7 %` | **1.0284 g/m³** | `humidity_calculator.calculate_absolute_humidity` |
| Dataset row 08/12/2004 15:00 | CO 1.2 mg/m³, NOx 179 ppb, NO₂ 82 µg/m³, C₆H₆ 4.6 µg/m³, T 16.7 °C, RH 54.7 %, AH 1.0311 g/m³ | `data_retrieval.find_record` |

## 13. Metrics that do **not** exist

Precision, recall, F1, AUC · latency or throughput benchmarks · human preference ratings · memory peak
measurements · quantised-model comparisons · v3 metrics of any kind · training-time FLOPs.
