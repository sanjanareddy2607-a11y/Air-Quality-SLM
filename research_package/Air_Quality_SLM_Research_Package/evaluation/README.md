# Evaluation

One harness scores both checkpoints identically, so every difference attributed to a model is a model
difference. `results/` holds the machine-readable output; `../EXPERIMENTAL_RESULTS.md` holds the
transcribed tables.

| File | Content |
|---|---|
| `results/loss_v1_128.json`, `loss_v2_128.json` | Full-split loss/perplexity, block 128, every token scored once |
| `results/loss_v1_64.json` | v1 at its native 64-token window (supplementary control) |
| `results/baseline.json` | Constant train-mean predictor per quantity, plus `date_overlap_with_train` |
| `results/numeric_diagnostics.json` | Closed-form AH reconstruction and the train-split scale fit |
| `results/summary.json` | The aggregated summary written by `summarize_eval_v2.py` |
| `results/summary_tables.txt` | The same summary as fixed-width tables (human-readable) |
| `results/metrics_v{1,2}_{val,test}.json` | **Per-row generations** — prompt kind, generated text, extracted number, gold label, structural flags. These make every MAE independently recomputable; see `../REPRODUCIBILITY.md` §5 |
| `scripts/evaluate_v2_full.py` | The harness: `--phase baseline | suite --suite loss | metrics` |
| `scripts/summarize_eval_v2.py` | Aggregates the dumps into `summary.json` / `summary_tables.txt` |
| `scripts/diagnose_v2_numbers.py` | Closed-form comparison and the constant-output diagnosis |
| `scripts/evaluate_slm.py` | Earlier, single-model evaluation script (superseded by `evaluate_v2_full.py`) |

## Headline results

| Model | Split | Loss | PPL |
|---|---|---:|---:|
| v1 | val | 1.4252 | 4.159 |
| v1 | test | 1.4474 | 4.252 |
| v2 | val | **0.4877** | **1.629** |
| v2 | test | **0.5555** | **1.743** |

| Quantity | Split | v2 MAE | Constant MAE | Ratio |
|---|---|---:|---:|---:|
| AH (g/m³) | val | 0.4335 | 0.459 | 0.945 |
| AH (g/m³) | test | 0.5345 | 0.558 | 0.959 |
| RH (%) | val | 36.32 | 15.71 | **2.311** |
| NOx (ppb) | val | 300.84 | 233.89 | **1.286** |

`ratio > 1` means **worse than predicting the training mean**. v2 emits a constant value per template
(prediction standard deviation 0.000–0.003), so these numbers measure offset from the split
distribution, not skill. The AH figure is 18.9× (val) / 28.4× (test) worse than the closed form whose
inputs are both present in the prompt.

## Reproduce

```bash
./venv/Scripts/python.exe -u src/evaluate_v2_full.py --phase baseline
./venv/Scripts/python.exe -u src/evaluate_v2_full.py --phase suite --suite loss    --threads 6
./venv/Scripts/python.exe -u src/evaluate_v2_full.py --phase suite --suite metrics --max-new 50 --threads 6
./venv/Scripts/python.exe -u src/summarize_eval_v2.py
```
