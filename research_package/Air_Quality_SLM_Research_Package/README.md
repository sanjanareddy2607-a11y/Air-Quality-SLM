# Air Quality SLM — Research Evidence Package

**Project:** Domain-specific Small Language Model for air-quality measurements
**Author of the underlying work:** Sanjana Reddy (B.Tech CSE — AI & ML)
**Package compiled:** 20 November 2026
**Purpose:** collect, verify and package the evidence of work already completed, so an IEEE-format
paper can cite each claim and reproduce each number.

This package contains **no new model and no new training**. It is an audit of the existing
workspace: the dataset and its preprocessing, the training and continued-training runs, the full
evaluation of both checkpoints, the PyTorch-to-GGUF conversion, the Ollama deployment, the Python
application layer, and the figures/reports derived from those artefacts.

---

## 1. Read this first: what the evidence actually shows

The project has three registered Ollama models. They are **not** three independent models:

| Version | Ollama tag | Weights | Context | Status of evidence |
|---|---|---|---|---|
| v1 | `air-quality-slm:latest` | `models/air_quality_slm/pytorch_model.bin` (SHA256 `77be6061…`) | 64 | Trained 150 steps; fully evaluated (val/test loss, generations, numeric scoring) |
| v2 | `air-quality-slm-improved:latest` | `models/air_quality_slm_finetuned/pytorch_model.bin` | 128 | Continued from v1 for 1,200 steps; fully evaluated with the same harness as v1 |
| v3 | `air-quality-slm-v3:latest` | **the same GGUF as v2** | 128 | **No separate evaluation exists.** Differs only in the `TEMPLATE` (adds `{{ .System }}`) |

**Which version has the strongest experimentally supported results?**
On language modelling and output structure: **v2**. It reduces full-split test loss from 1.4474 to
0.5555 (perplexity 4.25 → 1.74), emits EOS in 100 % of outputs, produces correct units in 100 %,
and never repeats. On **numerical accuracy**: **no version is competent**. v2 emits essentially one
constant value per template (prediction std 0.000–0.003); absolute-humidity MAE is 0.53 g/m³ on
test versus 0.019 g/m³ for the closed-form formula its own prompt determines. v3 has no evidence of
being better than v2 because **no v3-specific measurement was ever recorded** — it must not be
described as the best model merely because it is the newest.

The strongest *usable* result of the whole project is therefore the combination documented in
`EXPERIMENTAL_RESULTS.md`: a tiny model that formats air-quality language perfectly, wrapped in a
deterministic Python layer (dataset lookup + closed-form humidity) that supplies the numbers.

---

## 2. Package contents

```
Air_Quality_SLM_Research_Package/
├── README.md                        this file
├── PROJECT_OVERVIEW.md              problem, motivation, objectives, contributions
├── METHODOLOGY.md                   dataset -> training -> evaluation -> deployment pipeline
├── EXPERIMENTAL_RESULTS.md          every measured number, comparison tables, failure cases
├── SYSTEM_ARCHITECTURE.md           components, data flow, what runs where
├── REPRODUCIBILITY.md               exact commands for every stage, with environment caveats
├── ENVIRONMENT.md                   verified versions, hardware, installation
├── REFERENCES_AND_DATA_SOURCES.md   dataset citation, model/library references
├── MISSING_EVIDENCE.md              everything missing, disputed or unverified
├── FILE_INVENTORY.md                path, size and SHA-256 for every file in this package
│
├── dataset/
│   ├── dataset_documentation.md     full dataset card with measured statistics
│   ├── preprocess_air_quality.py    parsing, sentinel handling, chronological split
│   ├── prepare_training_data.py     four instruction templates per row + tokenisation
│   ├── validate_dataset.py          validation pass
│   ├── raw/AirQualityUCI.csv        the original file, unmodified
│   └── processed/{train,validation,test}.csv
├── training/
│   ├── README.md                    what was run, with both settings side by side
│   ├── scripts/train_slm.py         v1: 150 steps, block 64, from random init
│   ├── scripts/continue_training.py v2: 1,200 steps, block 128, from v1 weights
│   ├── training_config.json         full CPU and GPU configuration
│   ├── training_summary.md          narrative summary of the v1 run
│   ├── loss_curve.csv               v2 evaluation trace
│   ├── training_log_finetune.jsonl  v2 per-evaluation log
│   ├── EVALUATION_V2_REPORT.md      authoritative v1-vs-v2 report
│   ├── CONTINUED_TRAINING_REPORT.md why v2 exists (the 64-token window defect)
│   └── checkpoints/dataset_metadata.json
├── evaluation/
│   ├── README.md                    harness description and headline tables
│   ├── scripts/                     evaluate_v2_full, summarize_eval_v2, diagnose_v2_numbers, evaluate_slm
│   └── results/                     loss JSON, metrics JSON (per row), baseline, diagnostics, tables
├── conversion/
│   ├── conversion_documentation.md  tensor mapping, tokenizer preservation, verification, checksums
│   ├── config_v1_context64.json     exported v1 configuration
│   ├── config_v2_context128.json    exported v2 configuration
│   └── scripts/                     both GGUF exporters + tokenizer verifier
├── ollama/
│   ├── deployment_documentation.md  registration, API behaviour, proxy architecture, troubleshooting
│   ├── Modelfile                    v1 (num_ctx 64)
│   ├── Modelfile.improved           v2 (num_ctx 128, num_predict 48)
│   ├── Modelfile.v3                 v3 (same GGUF, system-prompt template)
│   └── chat_template.jinja          the Qwen template shipped with the checkpoints
├── application/
│   ├── README.md                    routing model and every verified quirk
│   ├── app.py, domain_guard.py, data_retrieval.py, humidity_calculator.py
│   ├── ollama_client.py, ollama_proxy.py, proxy_watchdog.py
│   └── INTEGRATION_OPERATIONS.md    start/stop/check procedure
├── figures/
│   ├── dataset/       fig01, fig02
│   ├── training/      fig05, fig06
│   ├── evaluation/    fig07, fig08, fig09, fig10, fig11
│   ├── architecture/  fig03, fig04, fig12, fig13, fig14, fig15
│   ├── scripts/       make_figures.py (data) and make_diagrams.py (schematics)
│   └── FIGURE_INDEX.md   caption, source and limitation for every figure
└── supplementary_material/
    ├── README.md
    ├── notebooks/     four development notebooks
    ├── reports/       probe JSONs and secondary reports
    └── scripts/       supporting development scripts
```

All figures exist as 300 dpi PNG and vector PDF. **Large binary artefacts (PyTorch weights, GGUF
files, token `.bin` streams) are not included**; their paths, sizes and SHA-256 values are in
`FILE_INVENTORY.md` and `conversion/conversion_documentation.md` §6.

---

## 3. How to use this package for the paper

1. **Section "Dataset"** → `dataset/dataset_documentation.md` and figures `fig01`, `fig02`.
2. **Section "Methodology"** → `METHODOLOGY.md`, figures `fig03`, `fig04`, `fig12`.
3. **Section "Results"** → `EXPERIMENTAL_RESULTS.md`, figures `fig05`–`fig11`.
4. **Section "Deployment/System"** → `SYSTEM_ARCHITECTURE.md`, figures `fig13`–`fig15`.
5. **Every claim that could not be supported** → `MISSING_EVIDENCE.md`. Read it before writing the
   limitations section; it lists the exact evidence still to be produced.

## 4. Honesty rules applied in this package

* Every number in the documents is copied from a file in this package or computed from the dataset
  CSVs; each table names its source file.
* Figures are regenerated from those files by the two scripts in `figures/scripts/`. Schematic
  diagrams are labelled as schematics and contain no measurements.
* Where an existing document contradicts the stored data, both are reported and the discrepancy is
  flagged in `MISSING_EVIDENCE.md` (for example, the report's "date overlap = 0" versus
  `baseline.json`'s `date_overlap_with_train = 1` for the validation split).
* Python-computed values (dataset lookup, humidity formula) are never presented as model outputs.

## 5. Citation of the underlying dataset

De Vito, S., Massera, E., Piga, M., Martinotto, L., Di Francia, G. (2008). *On-field calibration of
an electronic nose for benzene estimation in an urban pollution monitoring scenario.* Sensors and
Actuators B: Chemical, 129(2), 750–757. Dataset: UCI Machine Learning Repository, dataset 360,
DOI [10.24432/C59K5F](https://doi.org/10.24432/C59K5F), CC BY 4.0. Full reference list in
`REFERENCES_AND_DATA_SOURCES.md`.

## 6. Rebuilding the figures

```bash
./venv/Scripts/python.exe research_package/Air_Quality_SLM_Research_Package/figures/scripts/make_figures.py
./venv/Scripts/python.exe research_package/Air_Quality_SLM_Research_Package/figures/scripts/make_diagrams.py
```
