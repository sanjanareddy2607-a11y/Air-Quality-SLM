# Figures Generated

21 image files (18 PNG, 3 PDF), all produced during this documentation work by three scripts. No figure
contains an invented value: data figures read a stored artefact, and schematics are drawn from the source
code and labelled as schematics in their captions.

```bash
./venv/Scripts/python.exe docs/scripts/make_figures.py        # data-driven figures
./venv/Scripts/python.exe docs/scripts/make_diagrams.py       # six schematics
./venv/Scripts/python.exe docs/scripts/make_docs_diagrams.py  # three further schematics
```

`make_figures.py` locates the project root by walking up for `data/processed/train.csv`, or honours
`AQSLM_ROOT`. All three scripts exit 0 on this machine.

---

## Diagrams — `docs/assets/diagrams/`

| # | File | Type | Shows | Source of truth |
|---|---|---|---|---|
| 01 | `01_preprocessing_workflow.png` | schematic | raw CSV → sentinel handling → chronological split → four templates → token stream | `src/preprocess_air_quality.py`, `src/prepare_training_data.py` |
| 02 | `02_development_workflow.png` · `.pdf` | schematic | the twelve stages, Day 0 → v3, with the script that implements each | `docs/PROJECT_HISTORY.md` |
| 03 | `03_slm_architecture.png` | schematic | Qwen3-style block: RMSNorm → GQA (4 q / 2 kv, QK-norm) → RoPE → SwiGLU, tied embeddings | `src/train_slm.py`, `config.json` |
| 04 | `04_training_workflow.png` · `.pdf` | schematic | v1 from scratch and v2 continued, side by side, with the recorded settings | `training_config.json`, `src/continue_training.py` |
| 05 | `05_pytorch_to_gguf.png` | schematic | checkpoint → tensor map → GGUF (F32, `qwen3`, tokenizer, template) | `src/export_gguf_from_checkpoint.py` |
| 06 | `06_ollama_deployment.png` | schematic | Modelfile → `ollama create` → tag → run → llama.cpp | `Modelfile*` |
| 07 | `07_system_architecture.png` | schematic | app, Python lookups, proxy (11434 → 11435), Ollama server, GGUF | `app.py`, `ollama_proxy.py` |
| 08 | `08_user_interaction.png` · `.pdf` | schematic | one user turn: routing, four modes, answer assembly | `app.py`, `domain_guard.py` |
| 09 | `09_domain_routing.png` | schematic | greeting / retrieval / AH / out-of-domain branches | `domain_guard.py` |

## Graphs — `docs/assets/graphs/`

| # | File | Shows | Data source |
|---|---|---|---|
| 01 | `01_training_loss_v1.png` | v1 train and validation loss against step, with the LR schedule | `models/air_quality_slm/training_summary.json` |
| 02 | `02_validation_loss_v2.png` | v2 train (EMA) and validation loss, early-stopping point marked | `docs/loss_curve.csv` |
| 04 | `04_version_comparison.png` | v1 vs v2 full-split loss and perplexity, val and test | `docs/eval_v2/loss_v*.json` |
| 05 | `05_mae_rmse_vs_baseline.png` | v2 MAE per quantity against the constant-mean baseline | `docs/eval_v2/{summary.json,baseline.json}` |
| 07 | `07_ah_actual_vs_predicted.png` | AH actual vs predicted, one point per validation row, identity line, closed-form reference | `docs/eval_v2/metrics_v2_val.json` |
| 08 | `08_ah_error_distribution.png` | dataset AH is a deterministic function of T and RH; model error distribution | `docs/eval_v2/numeric_diagnostics.json` |
| 09 | `09_output_structure.png` | EOS rate, token-cap rate, repetition, unit correctness per model/split | `docs/eval_v2/summary_tables.txt` |
| 10 | `10_feature_distributions.png` | distribution of every numeric column, units in the axis labels | `data/processed/train.csv` |
| 11 | `11_missing_values.png` | missing-cell counts and percentages per column | `data/raw/air+quality/AirQualityUCI.csv` |

## Requested figures that were **not** produced

| Requested | Status | Reason |
|---|---|---|
| Validation loss (separate from training) | **produced** as a two-panel graph, 02 | v2's train and val curves are in one figure; separating them would duplicate data |
| RMSE comparison | **produced** within 05 | MAE and RMSE share the figure because both compare against the same baseline |
| Actual vs predicted humidity | **produced** as 07 | |
| Prediction error distribution | **produced** as 08 (right panel) | |
| Evaluation results summary | **produced** as 09 | |
| Screenshots | **not produced** | no screenshot exists in the repository; fabricating one would misrepresent the project. See `docs/assets/screenshots/README.md` |

## Visual checks performed

Each new schematic (02, 04, 08) was rendered and inspected for overlapping or clipped labels; label overflow
in 02 and 04 and a colliding caption in 08 were fixed before the final render. Figures 01–09 in `graphs/` and
the six copied diagrams were produced by the research-package scripts and previously inspected there.

## Known figure limitations

* 02's graph has only four evaluation points (steps 40/80/120/149); it is not a per-step trace.
* 07's predictions collapse onto one horizontal band — that collapse *is* the finding.
* No figure shows v3: there is no v3 data to plot.
