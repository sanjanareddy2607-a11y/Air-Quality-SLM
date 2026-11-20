# References and Data Sources

Only sources that were actually consulted or that the code actually depends on are listed. Where a
claim in this package rests on a stored artefact rather than an external source, the artefact is named
instead of a citation.

---

## 1. Dataset

**Primary citation (required by the UCI repository):**

> S. De Vito, E. Massera, M. Piga, L. Martinotto, and G. Di Francia, "On field calibration of an
> electronic nose for benzene estimation in an urban pollution monitoring scenario," *Sensors and
> Actuators B: Chemical*, vol. 129, no. 2, pp. 750–757, 2008.
> doi:10.1016/j.snb.2007.09.060

**Dataset record:**

> S. De Vito, E. Massera, M. Piga, L. Martinotto, and G. Di Francia, "Air Quality," UCI Machine Learning
> Repository, dataset 360, 2008. doi:10.24432/C59K5F.
> URL: https://archive.ics.uci.edu/dataset/360/air+quality
>
> (Note: UCI also lists dataset 387 as a duplicate of 360 under doi:10.24432/C5060Z. Use C59K5F for
> dataset 360, which is the identifier the repository's own record displays.)

**What is present in this project:** `data/raw/air+quality/AirQualityUCI.csv` (also `.xlsx`), 9,357 data
rows after removing trailing blank rows, 15 columns, semicolon-separated, comma decimal separator.
SHA-256 of the CSV: `13277ae5…` (full value in `FILE_INVENTORY.md`).

**What is not recorded:** the download date, the mirror used, or a checksum captured at download time.
The model card in `models/air_quality_slm/README.md` names a Hugging Face mirror,
`sanju7/air-quality-uci`; no upload log for it exists in the repository.

## 2. Licensing

| Item | Statement | Evidence |
|---|---|---|
| Dataset | UCI datasets are distributed for research use; the Air Quality dataset is commonly cited under CC BY 4.0 | UCI record; the project's own model card declares `license: cc-by-4.0` |
| Project code | No `LICENSE` file exists in the repository | directory listing |
| Model card | declares `license: cc-by-4.0` and `language: en` | `models/air_quality_slm/README.md` front matter |

The absence of a repository `LICENSE` file is recorded in the "unavailable information" list rather than
assumed to be CC BY.

## 3. Architecture and framework references

**Qwen3** — the design the custom model follows (Grouped-Query Attention with Q/K RMSNorm, RoPE,
SwiGLU, RMSNorm, tied embeddings):

> A. Yang et al., "Qwen3 Technical Report," arXiv:2505.09388, 2025.

**Qwen2.5 tokenizer** — the BPE vocabulary actually loaded (`Qwen/Qwen2.5-0.5B`, vocab 151,665):

> Qwen Team, "Qwen2.5 Technical Report," arXiv:2412.15115, 2024.

**Grouped-Query Attention:**

> J. Ainslie, J. Lee-Thorp, M. de Jong, Y. Zemlyanskiy, F. Lebrón, and S. Sanghai, "GQA: Training
> Generalized Multi-Query Transformer Models from Multi-Head Checkpoints," arXiv:2305.13245, 2023.

**Rotary position embeddings:**

> J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, and Y. Liu, "RoFormer: Enhanced Transformer with Rotary
> Position Embedding," arXiv:2104.09864, 2021 (RoFormer).

**SwiGLU / GLU variants:**

> N. Shazeer, "GLU Variants Improve Transformer," arXiv:2002.05202, 2020.

**RMSNorm:**

> B. Zhang and R. Sennrich, "Root Mean Square Layer Normalization," NeurIPS 2019, arXiv:1910.07467.

**AdamW:**

> I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," ICLR 2019, arXiv:1711.05101.

**Cosine schedule with linear warmup:**

> I. Loshchilov and F. Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts," ICLR 2017,
> arXiv:1608.03983.

**Perplexity as a language-modelling metric** — standard definition, `exp(mean negative log-likelihood
per token)`; no external citation needed beyond any LM textbook, but the harness that computes it is
`src/evaluate_v2_full.py`.

## 4. Software documentation

| Tool | Version used | Reference |
|---|---|---|
| PyTorch | 2.14.1 (CPU) | https://pytorch.org/docs/ |
| Hugging Face `transformers` | 5.23.0 | https://huggingface.co/docs/transformers |
| Hugging Face `tokenizers` | 0.23.3 | https://huggingface.co/docs/tokenizers |
| `gguf` Python package | 0.19.0 | https://github.com/ggerganov/ggml/blob/master/docs/gguf.md |
| llama.cpp | build `b12503` (bundled with Ollama) | https://github.com/ggml-org/llama.cpp |
| Ollama | 0.33.2 | https://github.com/ollama/ollama/tree/main/docs |
| Ollama Modelfile reference | 0.33.x | https://github.com/ollama/ollama/blob/main/docs/modelfile.md |
| pandas | 3.1.1 | https://pandas.pydata.org/docs/ |
| matplotlib | 3.11.2 (figures only) | https://matplotlib.org/stable/ |

## 5. Metrology references for the domain

**Absolute humidity from temperature and relative humidity** (the Tetens/Magnus form used in
`humidity_calculator.py`, with the caveat in `MISSING_EVIDENCE.md` §4):

> O. A. Alduchov and R. E. Eskridge, "Improved Magnus Form Approximation of Saturation Vapor Pressure,"
> *Journal of Applied Meteorology*, vol. 35, no. 4, pp. 601–609, 1996.

**Air-quality measurement units** used in the dataset (mg/m³ for CO, ppb for NOx, µg/m³ for NO₂, C₆H₆,
NMHC) follow the UCI dataset's own variable descriptions; see `dataset/dataset_documentation.md` §4.

## 6. Internal sources of record

These files, not external literature, are the authority for every number in this package:

| File | Role |
|---|---|
| `docs/eval_v2/loss_v1_128.json`, `loss_v2_128.json`, `loss_v1_64.json` | full-split loss and perplexity |
| `docs/eval_v2/summary_tables.txt`, `summary.json` | all tables reproduced in `EXPERIMENTAL_RESULTS.md` |
| `docs/eval_v2/metrics_v{1,2}_{val,test}.json` | per-row generations, enabling independent MAE recomputation |
| `docs/eval_v2/baseline.json` | constant-predictor baselines per quantity |
| `docs/eval_v2/numeric_diagnostics.json` | closed-form AH reconstruction, scale fit |
| `models/air_quality_slm/training_summary.json` | v1 training trace, runtime, final losses |
| `training/logs/loss_curve.csv`, `training/training_log_finetune.jsonl` | v2 training trace |
| `docs/EVALUATION_V2_REPORT.md` | narrative interpretation of the above |
| `docs/CONTINUED_TRAINING_REPORT.md` | v2 training design and early-stopping rationale |
| `docs/OLLAMA_INTEGRATION.md` | deployment diagnosis (flat distribution, garbled output) |
| `docs/VALIDATION_REPORT.md` | dataset validation |

## 7. Citation block for the paper

```
S. De Vito, E. Massera, M. Piga, L. Martinotto, G. Di Francia, "On field calibration of an electronic
nose for benzene estimation in an urban pollution monitoring scenario," Sensors and Actuators B, 129(2),
750-757, 2008. Dataset: UCI Machine Learning Repository, doi:10.24432/C59C74.

A. Yang et al., "Qwen3 Technical Report," arXiv:2505.09388, 2025.
```
