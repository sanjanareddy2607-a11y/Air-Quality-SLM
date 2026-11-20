# Figure Index

15 figures, each rendered at 300 dpi as PNG and vector PDF. Two committed scripts generate them; no
plotted value is typed in by hand.

```bash
# rebuild every figure (both scripts exit 0; ~20 s)
./venv/Scripts/python.exe research_package/Air_Quality_SLM_Research_Package/figures/scripts/make_figures.py
./venv/Scripts/python.exe research_package/Air_Quality_SLM_Research_Package/figures/scripts/make_diagrams.py
```

`make_figures.py` locates the project root by walking up for `data/processed/train.csv`, or honours the
`AQSLM_ROOT` environment variable. Fig 1–2 read the CSVs; fig 5–7 read the training JSON/CSV; fig 8–11
read `docs/eval_v2/*`; fig 3, 4, 12–15 are schematics drawn from the source files and are labelled
"schematic" in-panel.

---

## Data-driven figures (every number comes from a stored artefact)

| # | File | Shows | Source |
|---|---|---|---|
| 01 | `figures/dataset/fig01_feature_distributions.{png,pdf}` | Distribution of every numeric column (histograms, units in the axis labels) | `data/processed/train.csv` |
| 02 | `figures/dataset/fig02_missing_values.{png,pdf}` | Missing-cell counts and percentages per column for the whole dataset | `data/raw/air+quality/AirQualityUCI.csv` (sentinel `-200`) |
| 05 | `figures/training/fig05_loss_curve_v1.{png,pdf}` | v1 train and validation loss against step, with the LR schedule | `models/air_quality_slm/training_summary.json` |
| 06 | `figures/training/fig06_loss_curve_v2.{png,pdf}` | v2 train (EMA) and validation loss, with the early-stopping point | `training/loss_curve.csv` |
| 07 | `figures/evaluation/fig07_v1_v2_loss_perplexity.{png,pdf}` | v1 vs v2 full-split loss and perplexity, val and test | `docs/eval_v2/loss_v*.json` |
| 08 | `figures/evaluation/fig08_mae_rmse_vs_baseline.{png,pdf}` | v2 MAE/RMSE per quantity against the constant-mean baseline (`ratio` < 1 beats a constant) | `docs/eval_v2/{summary.json,baseline.json}` |
| 09 | `figures/evaluation/fig09_ah_actual_vs_predicted.{png,pdf}` | Absolute humidity: actual vs predicted, one point per validation row, with the identity line and the closed-form reference | `docs/eval_v2/metrics_v2_val.json` |
| 10 | `figures/evaluation/fig10_ah_formula_reconstruction.{png,pdf}` | AH from the dataset's own formula vs the model's output, and the error distributions | `docs/eval_v2/numeric_diagnostics.json` |
| 11 | `figures/evaluation/fig11_output_structure.{png,pdf}` | Structural metrics per model/split: EOS rate, token-cap rate, repetition, unit correctness | `docs/eval_v2/summary_tables.txt` |

## Schematic figures (drawn from the code, not from data)

| # | File | Shows |
|---|---|---|
| 03 | `figures/architecture/fig03_preprocessing_workflow.{png,pdf}` | Raw CSV → sentinel handling → split → 4 instruction templates → token stream |
| 04 | `figures/architecture/fig04_model_architecture.{png,pdf}` | Qwen3-style block: RMSNorm → GQA (4 q / 2 kv, QK-norm) → RoPE → SwiGLU, tied embeddings |
| 12 | `figures/architecture/fig12_conversion_workflow.{png,pdf}` | `pytorch_model.bin` → tensor map → GGUF (F32, `qwen3`, tokenizer, template) |
| 13 | `figures/architecture/fig13_ollama_deployment.{png,pdf}` | Modelfile → `ollama create` → tag → `ollama run` → llama.cpp |
| 14 | `figures/architecture/fig14_query_routing.{png,pdf}` | `domain_guard` branches: greeting / retrieval / AH calculation / out-of-domain |
| 15 | `figures/architecture/fig15_system_architecture.{png,pdf}` | Application, Python lookups, proxy (11434→11435), Ollama server, GGUF |

## How to cite a figure in the paper

Each figure's caption should state the source file, which is given above. For fig 9 and fig 10 in
particular, the caption must say that the plotted predictions are the **model's** outputs, whereas the
closed-form curve is arithmetic — mixing the two is the single easiest way to misrepresent this project.

## Known figure limitations

* Fig 5 has only four evaluation points (steps 40/80/120/149); the curve is drawn between them and should
  not be read as a per-step trace.
* Fig 9's predictions collapse onto one horizontal band (standard deviation 0.003 g/m³). That is the
  finding, not a plotting artefact.
* No figure shows v3. There is no v3 data to plot (`MISSING_EVIDENCE.md` §8).
* No screenshot figure exists; none was captured (`MISSING_EVIDENCE.md` §1).
