[README_Air_Quality_SLM.md](https://github.com/user-attachments/files/35069758/README_Air_Quality_SLM.md)
# Air Quality Small Language Model (SLM)

A custom-trained, lightweight Transformer language model for generating structured text from air-quality records, with local inference through **Ollama**.

> **Project status:** Experimental research prototype. The model has been trained, improved, exported to GGUF and registered in Ollama. It can generate structured air-quality-style text, but its numerical predictions are not sufficiently reliable for scientific, public-health or operational decisions. It is not a real-time monitoring system or a general-purpose chatbot.

---

## Table of contents

- [1. Project overview](#1-project-overview)
- [2. Problem statement and objectives](#2-problem-statement-and-objectives)
- [3. Dataset](#3-dataset)
- [4. Data preparation](#4-data-preparation)
- [5. Model architecture](#5-model-architecture)
- [6. Training approach](#6-training-approach)
- [7. Evaluation and results](#7-evaluation-and-results)
- [8. PyTorch-to-GGUF conversion](#8-pytorch-to-gguf-conversion)
- [9. Running the model with Ollama](#9-running-the-model-with-ollama)
- [10. Training and evaluation scripts](#10-training-and-evaluation-scripts)
- [11. Repository layout](#11-repository-layout)
- [12. Current limitations and future work](#12-current-limitations-and-future-work)
- [13. Reproducibility and model files](#13-reproducibility-and-model-files)
- [14. Dataset reference and acknowledgement](#14-dataset-reference-and-acknowledgement)

---

## 1. Project overview

The **Air Quality SLM** explores how a small, domain-specific language model can learn the structure and terminology of air-quality measurements and generate text about those measurements.

The project covers the complete experimental workflow:

1. Obtain an environmental dataset.
2. Validate and preprocess the records.
3. Convert selected measurements into text-based instruction examples.
4. Build a compact Transformer language model in PyTorch.
5. Train and evaluate the model.
6. Improve the training pipeline after identifying limitations in the first run.
7. Export the improved checkpoint to GGUF.
8. Register and run the model locally through Ollama Desktop.

The model is intended as a learning and research prototype for **structured air-quality text generation**. The current version does not reliably retrieve historical measurements or predict accurate pollutant values.

### Project pipeline

```text
UCI Air Quality Dataset (CSV)
            |
            v
Data validation and preprocessing
            |
            v
Structured instruction / response examples
            |
            v
Qwen2.5-based tokenizer
            |
            v
Custom PyTorch Transformer SLM
            |
            v
Training, validation and evaluation
            |
            v
Improved PyTorch checkpoint (v2)
            |
            v
GGUF export and verification
            |
            v
Ollama registration
            |
            v
Local chat inference in Ollama Desktop
```

## 2. Problem statement and objectives

Air-quality monitoring produces structured measurements that include pollutant concentrations, chemical-sensor responses and meteorological conditions. Understanding these records requires familiarity with the parameters, their units and their relationships.

This project investigates whether a compact, custom-trained language model can learn common air-quality record formats and generate domain-specific text from them.

### Objectives

- Prepare air-quality records for language-model training.
- Implement a small decoder-only Transformer architecture.
- Train the model on structured air-quality text examples.
- Evaluate language modelling and task-specific generation behaviour.
- Convert the trained model into GGUF format.
- Run the model locally using Ollama, without depending on a hosted LLM API for inference.

## 3. Dataset

### UCI Air Quality Dataset

The project uses the **Air Quality** dataset from the UCI Machine Learning Repository. It contains hourly measurements from a field-deployed air-quality multisensor device in an Italian city, collected from March 2004 to February 2005. The source dataset contains pollutant reference-analyser measurements, five metal-oxide sensor responses and meteorological variables.

The UCI repository describes 9,358 instances and 15 features. The project's recorded train/validation/test splits total 9,357 records; consult the preprocessing scripts to verify the exact reason for the one-record difference in this project copy.

### Important fields

| Field | Meaning | Unit / interpretation |
|---|---|---|
| `Date`, `Time` | Measurement timestamp | Date and time fields |
| `CO(GT)` | Carbon monoxide reference measurement | mg/m³ |
| `NMHC(GT)` | Non-methane hydrocarbons reference measurement | µg/m³ |
| `C6H6(GT)` | Benzene reference measurement | µg/m³ |
| `NOx(GT)` | Nitrogen oxides reference measurement | ppb |
| `NO2(GT)` | Nitrogen dioxide reference measurement | µg/m³ |
| `PT08.S1`–`PT08.S5` | Chemical sensor responses | Sensor response values |
| `T` | Temperature | °C |
| `RH` | Relative humidity | % |
| `AH` | Absolute humidity field supplied in the dataset | Dataset-provided value |

The original UCI data uses `-200` as a missing-value marker. The exact handling of missing and invalid values in this project is implemented in the preprocessing/validation scripts; review those scripts when reproducing the dataset preparation rather than assuming a particular imputation or row-removal policy.

### Dataset split used in this project

| Split | Records |
|---|---:|
| Training | 6,549 |
| Validation | 936 |
| Testing | 1,872 |
| **Total** | **9,357** |

The test split is reserved for final evaluation and should not be used for training or hyperparameter selection.

## 4. Data preparation

The raw CSV records were converted into text examples suitable for next-token language-model training.

### Preprocessing workflow

```text
Raw CSV
  -> Data validation
  -> Missing / invalid-value handling
  -> Feature selection
  -> Text-template generation
  -> Tokenisation
  -> Train / validation / test split
```

The exact transformations and missing-value policy are defined in the repository's preprocessing scripts.

### Four instruction formats

The project prepares four main types of text examples:

1. **Full measurement-report completion** — a structured report containing pollutant, sensor and meteorological fields.
2. **Absolute-humidity task** — a prompt containing date/time, selected pollutant fields, temperature and relative humidity, followed by an absolute-humidity response.
3. **Temperature and relative-humidity task** — a prompt asking for the environmental measurements in a record.
4. **Pollutant-concentration task** — a prompt requesting values such as CO, NOx and NO2.

### Tokenisation

The project uses a tokenizer based on the **Qwen2.5 tokenizer**. The tokenizer was adopted for this project; it was not independently trained from scratch. The vocabulary contains 151,665 tokens. The model uses the tokenizer's special-token configuration, including `<|endoftext|>` as the end-of-sequence token.

## 5. Model architecture

The model is a custom, compact, decoder-only Transformer language model implemented in PyTorch. Its architecture uses Qwen3-style components, while the project uses a Qwen2.5-based tokenizer.

### Main configuration

| Component | Configuration |
|---|---:|
| Effective model parameters | Approximately 20.4 million |
| Transformer layers | 4 |
| Hidden dimension | 128 |
| Query attention heads | 4 |
| Key/value heads | 2 |
| Attention type | Grouped Query Attention (GQA) |
| Feed-forward network | SwiGLU |
| Normalisation | RMSNorm, including per-head Q/K RMSNorm |
| Positional encoding | Rotary Positional Embeddings (RoPE) |
| Vocabulary size | 151,665 |
| v1 context length | 64 tokens |
| v2 context length | 128 tokens |

The effective parameter count is approximately 20.4M. GGUF tools may report a larger stored tensor count because tied input/output embedding tensors can be represented separately in the exported file; stored tensor count should not be confused with the number of unique trainable parameters.

### Architecture in simple terms

```text
Input text
   -> Tokenizer
   -> Token embeddings
   -> Transformer block 1
   -> Transformer block 2
   -> Transformer block 3
   -> Transformer block 4
   -> Final RMSNorm
   -> Language-modelling head
   -> Next-token prediction
   -> Generated text
```

- **Transformer:** learns relationships between tokens in a sequence.
- **GQA:** uses shared key/value heads to reduce attention computation.
- **RoPE:** supplies positional information to the attention mechanism.
- **RMSNorm:** normalises activations to help stabilise training.
- **SwiGLU:** provides a gated non-linear transformation in the feed-forward network.
- **Q/K RMSNorm:** applies normalisation to query and key representations before rotary positioning.

The architecture and trained weights are custom to this project. The tokenizer is reused from the Qwen2.5 family; this is not a full pretrained Qwen model and the project does not use a downloaded Qwen language-model checkpoint as its trained model.

## 6. Training approach

The work progressed through two main versions.

### Version 1 — Initial training

The initial model was trained using the custom PyTorch architecture. The first training run used:

- 150 training iterations.
- Batch size 8.
- Sequence/context window of 64 tokens.
- Approximately 76,800 tokens processed.
- Random sampling of windows from a concatenated token stream.

A key issue was identified: many instruction examples were longer than 64 tokens. Random windows could split a prompt from its answer, so the model did not receive enough complete examples of the intended instruction-to-answer relationship.

### Version 2 — Continued training and pipeline improvement

Rather than restarting from random weights, training continued from the original learned checkpoint. The original checkpoint did not contain optimiser state, so a fresh AdamW optimiser was initialised at a lower learning rate. This is continued weight training, not an exact optimiser-state resume.

The improved run used:

- Context window: 128 tokens.
- Learning rate: `2e-4`.
- Batch size: 2.
- Gradient accumulation: 2 steps.
- Warmup followed by cosine learning-rate scheduling.
- Systematic training-data sampling.
- Validation checks and early stopping when validation loss plateaued and overfitting appeared.
- 1,200 training steps and approximately 614,400 tokens processed.
- Best checkpoint selected at step 1,000.
- Approximately 0.202 epoch completed.

Training was performed on CPU using an AMD Ryzen 5 5600H system. The extended training run was stopped based on validation behaviour rather than simply running for a predetermined large number of iterations.

### Training loss comparison

The following figures are from the later full-split evaluation using a matched block-128 procedure:

| Metric | Original v1 | Improved v2 |
|---|---:|---:|
| Validation loss | 1.4252 | **0.4877** |
| Validation perplexity | 4.159 | **1.629** |
| Test loss | 1.4474 | **0.5555** |
| Test perplexity | 4.252 | **1.743** |

The validation loss decreased by approximately 66% under this matched evaluation. Lower language-modelling loss indicates better next-token modelling on the evaluated text; it does **not** by itself prove accurate numerical prediction.

## 7. Evaluation and results

Evaluation considered both language-modelling metrics and generated responses.

### What improved in v2

- The model produces more consistently structured responses for the project templates.
- It is more likely to terminate generated responses with the expected end-of-sequence marker.
- Output units and formatting are more consistent in the full-split generation evaluation.
- The extended 128-token context allows the model to receive the complete 86-token absolute-humidity prompt used in the diagnostic test.

The full-split evaluation reported 100% EOS termination, 100% correct-unit formatting, 0% repetitive outputs and 0% malformed outputs for the evaluated v2 generation set. These are formatting/generation checks, not measures of whether the numerical values are correct.

### Numerical prediction limitations

The same evaluation found that v2 does not reliably predict numerical values:

| Absolute-humidity metric | Validation | Test |
|---|---:|---:|
| MAE | 0.4335 g/m³ | 0.5345 g/m³ |
| Correlation with target | ≈ 0.01 | ≈ 0.10 |
| Predictions within 5% | 1.4% | 3.4% |

The model produced only a very small number of distinct outputs per template, with near-constant predictions. Relative humidity and NOx results were worse than a simple constant baseline in the reported evaluation. Temperature/relative-humidity responses also showed systematic date and value hallucinations.

### Example: absolute-humidity response

For one validation example:

| Item | Value |
|---|---:|
| Dataset target | 1.0311 g/m³ |
| v2 generated output | 1.15 g/m³ |
| Absolute error | 0.1189 g/m³ |
| Relative error | ≈ 11.53% |

This example demonstrates improved response structure, but it should not be interpreted as proof of reliable prediction. The wider validation/test metrics show that numerical performance remains weak.

### Important interpretation of the AH field

The dataset's `AH` field is the target used in the project's training and evaluation. The project investigation found a strong relationship between this field and temperature/relative humidity under a fitted dataset-specific formula. The scale differs from a conventional physical absolute-humidity calculation. Therefore, comparisons in this project should state clearly whether they refer to the UCI dataset's `AH` target or a separately calculated physical quantity.

**Conclusion from evaluation:** v2 is a better structured-text generator than v1, but it is not currently a dependable air-quality numerical predictor. It should not be used for real-world environmental decisions.

## 8. PyTorch-to-GGUF conversion

After training, the improved PyTorch checkpoint was exported to GGUF so that it could be loaded by the llama.cpp runtime used by Ollama.

The export process includes:

1. Reading the trained PyTorch state dictionary.
2. Mapping model tensors to GGUF tensor names.
3. Preserving the model's Qwen3-compatible operations, including per-head Query/Key RMSNorm.
4. Preserving the tokenizer vocabulary and special-token metadata.
5. Exporting weights in F32 precision.
6. Writing the 128-token context configuration.
7. Checking tensor shapes, tokenizer behaviour and generation equivalence.

The improved GGUF artifact was reported at approximately 158 MB. Greedy generation was compared between PyTorch and llama.cpp/Ollama on matched prompts; the report found matching generated token sequences for the checked cases. This is evidence for the tested paths, not a proof of equivalence for every possible input.

### Why the conversion required debugging

The original Windows Ollama 0.33.2 build did not include the expected `llama-quantize` and `llama-server` helper executables. Compatible CPU helper binaries from llama.cpp release `b12503` were installed in the author's local Ollama runtime directory. The first conversion attempt also used an architecture mapping that omitted Q/K RMSNorm; this was replaced with a Qwen3-compatible export that preserves those tensors.

This runtime workaround is specific to the author's tested Windows installation and may not be necessary with other Ollama releases or builds. Use compatible binaries only; do not copy arbitrary runtime executables into an Ollama installation.

## 9. Running the model with Ollama

The improved model is registered locally under:

```text
air-quality-slm-improved:latest
```

The original v1 model is retained separately as:

```text
air-quality-slm:latest
```

### Prerequisites

- Windows 10/11 (the documented setup was tested on Windows 11).
- Ollama installed.
- The improved GGUF model file.
- The project's `Modelfile`.
- A compatible CPU runtime; the author's tested system used an AMD Ryzen 5 5600H and 8 GB RAM.

Check the installation:

```powershell
ollama --version
ollama list
```

### Obtain the model artifact

The GGUF file is approximately 158 MB. Large model files should be distributed through a model-artifact repository such as Hugging Face or Git LFS rather than committed as ordinary Git blobs.

> **Maintainer action before publishing:** Add the actual Hugging Face model-repository URL below and upload the improved GGUF and any checkpoint files intended for public release. The repository URL was not available when this README was prepared.

- Model files: **[ADD HUGGING FACE MODEL REPOSITORY URL]**
- GGUF filename: `air_quality_slm_finetuned.gguf` (confirm the final release filename and download path).

After downloading the GGUF, place it at the path referenced by the `FROM` line in the repository's `Modelfile`.

### Create the Ollama model

From the project root, with the GGUF file in the location referenced by `Modelfile`:

```powershell
ollama create air-quality-slm-improved -f Modelfile
```

Confirm registration:

```powershell
ollama list
```

Run the model from a terminal:

```powershell
ollama run air-quality-slm-improved
```

Or open **Ollama Desktop**, start a new chat and select:

```text
air-quality-slm-improved:latest
```

### Inference settings

The tested improved model uses a 128-token context. Its Modelfile uses deterministic/greedy-style generation settings and a bounded generation length to reduce random, malformed continuation. Keep prompts short enough to fit the context window. Use the repository's tested `Modelfile` rather than assuming that default Ollama sampling settings will behave the same way.

### Example prompt

The following follows the absolute-humidity instruction format used in training:

```text
Task: Predict the absolute humidity given the following parameters: Date: 08/12/2004, Time: 15.00.00, CO(GT): 1.2, NOx(GT): 179, NO2(GT): 82, Temperature: 16.7 C, Relative Humidity: 54.7 %. Response: The absolute humidity is
```

A previously observed v2 response was:

```text
The absolute humidity is 1.15 g/m3.
```

The corresponding dataset target is 1.0311 g/m³. This is an example of the model's output format, not a guarantee of exact prediction.

### Ollama API

Ollama exposes a local API (by default at `http://localhost:11434`). A basic chat request can be made with:

```powershell
$body = @{
  model = "air-quality-slm-improved:latest"
  messages = @(
    @{ role = "user"; content = "What is the absolute humidity?" }
  )
  stream = $false
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri "http://localhost:11434/api/chat" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

The API returns generated text from the local model. The same model limitations described above apply to API and desktop usage.

## 10. Training and evaluation scripts

The repository includes scripts for data preparation, initial training, continued training, evaluation, conversion and Ollama tests. The names below reflect the project files produced during development; check the actual repository tree before publishing if any file has been renamed or omitted.

| Script / artifact | Purpose |
|---|---|
| `src/preprocess_air_quality.py` | Preprocess the source air-quality dataset |
| `src/validate_dataset.py` | Validate the processed dataset |
| `src/prepare_training_data.py` | Convert records into instruction-style text examples |
| `src/train_slm.py` | Initial custom-model training implementation |
| `src/evaluate_slm.py` | Evaluate the original model |
| `src/continue_training.py` | Continue training from the original learned checkpoint |
| `src/evaluate_finetuned.py` | Evaluate the improved checkpoint |
| `src/export_gguf_from_checkpoint.py` | Export a PyTorch checkpoint to GGUF |
| `src/test_ollama_improved.py` | Test the improved model through Ollama |
| `notebooks/Air_Quality_SLM_Continue_Training_Colab.ipynb` | Colab-oriented continued-training workflow |
| `docs/EVALUATION_V2_REPORT.md` | Detailed full-split v1/v2 evaluation |
| `docs/OLLAMA_INTEGRATION.md` | GGUF conversion and Ollama integration notes |

### Local Python environment

Create a virtual environment from the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the dependencies listed in `requirements.txt` if that file is included in the repository:

```powershell
pip install -r requirements.txt
```

If no requirements file is present, the project maintainer should publish one with the exact tested package versions before expecting others to reproduce training. The main development stack includes PyTorch, NumPy, pandas, Transformers/tokenizers and the `gguf` package for export.

### Reproducing preprocessing and training

The intended workflow is:

```powershell
python src/preprocess_air_quality.py
python src/validate_dataset.py
python src/prepare_training_data.py
```

The initial training script creates the original v1 model. The improved v2 was produced by continuing from the saved v1 weights using `src/continue_training.py`; it is not equivalent to starting the initial script again. Inspect its supported arguments before running:

```powershell
python src/continue_training.py --help
```

For the prepared Colab workflow, open:

```text
notebooks/Air_Quality_SLM_Continue_Training_Colab.ipynb
```

Training is resource-intensive. The author's local machine had 8 GB RAM and no CUDA GPU. A compatible Colab GPU can be considered, but results and runtime will depend on available hardware and package compatibility.

## 11. Repository layout

The following is the intended high-level layout. Some generated evaluation files and model artifacts may be stored outside GitHub because of their size.

```text
.
├── README.md
├── Modelfile
├── requirements.txt                 # Add/pin tested dependencies before release
├── data/
│   ├── raw/                         # Original UCI CSV (if redistributed)
│   └── processed/                   # Prepared splits / instruction data
├── models/
│   ├── air_quality_slm_original/    # Preserved v1 backup
│   └── air_quality_slm_finetuned/   # Improved v2 checkpoint and config
├── src/
│   ├── preprocess_air_quality.py
│   ├── validate_dataset.py
│   ├── prepare_training_data.py
│   ├── train_slm.py
│   ├── evaluate_slm.py
│   ├── continue_training.py
│   ├── evaluate_finetuned.py
│   ├── export_gguf_from_checkpoint.py
│   └── test_ollama_improved.py
├── notebooks/
│   └── Air_Quality_SLM_Continue_Training_Colab.ipynb
└── docs/
    ├── EVALUATION_V2_REPORT.md
    ├── OLLAMA_INTEGRATION.md
    ├── loss_curve.csv
    ├── loss_curve.svg
    └── eval_v2/
```

## 12. Current limitations and future work

### Current limitations

- **Numerical prediction is weak:** v2 produces near-constant outputs for several tasks, even though its formatting and language-model loss improved.
- **Generalisation is limited:** outputs vary substantially when prompts differ from the formats used during training.
- **Historical lookup is not implemented:** the SLM does not query the original CSV to retrieve the true measurement for a requested timestamp.
- **Short context:** v2 uses 128 tokens. The full measurement-report template is longer than this and cannot be processed as one complete sequence.
- **Training exposure is limited:** the continued run covered approximately 0.202 epoch before validation plateau/overfitting signals.
- **Not a general-purpose assistant:** the model is specialised for the project's air-quality text templates.
- **Not real-time:** the training dataset contains historical readings from 2004–2005 and no live sensor connection.

### Planned improvements

1. Add a deterministic retrieval layer that looks up historical records by date/time in the source dataset.
2. Add a separately validated calculation component for supported derived values.
3. Keep retrieved/calculated values separate from free-form SLM-generated text so the model cannot silently alter the facts.
4. Expand context length and revise example packing/training design if justified by controlled experiments.
5. Evaluate on additional held-out data and report MAE/RMSE for numerical tasks.
6. Add a small local chat application that combines Ollama with verified retrieval and calculation results.

## 13. Reproducibility and model files

The original v1 checkpoint was backed up before continued training. The improved v2 checkpoint was saved separately. The reported original checkpoint SHA-256 began with `77BE6061` and ended with `B1172DA727`; consult the project's backup/integrity record for the full hash before verifying a downloaded copy.

Large files such as `.bin` checkpoints and `.gguf` exports should generally be hosted through Hugging Face or Git LFS. Do not commit large model files as ordinary Git blobs. Include the exact model download URL, file names, sizes and SHA-256 hashes in the release notes when publishing.

Before publishing this repository, the maintainer should:

- Add the actual GitHub repository URL to any external documentation.
- Add the actual Hugging Face model-repository URL in the download section.
- Add a tested `requirements.txt` with pinned versions.
- Confirm the final model artifact paths and filenames.
- Confirm the exact missing-value policy from the preprocessing code.
- Add a project software licence if one is intended. Do not imply that the project's code has the same licence as the UCI dataset.

## 14. Dataset reference and acknowledgement

This project uses the UCI Machine Learning Repository's **Air Quality** dataset:

- Dataset page: https://archive.ics.uci.edu/dataset/360/air+quality
- DOI: https://doi.org/10.24432/C59K5F

Suggested citation:

> Vito, S. (2008). *Air Quality* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C59K5F

Please follow the dataset page's current licence and usage terms when redistributing the raw data or derived files. The repository should retain appropriate dataset attribution.

---

## Project scope statement

This repository documents the development of a custom-trained, domain-specific language model and its local deployment through Ollama. The current v2 model demonstrates improved structured-text generation and successful local inference, but numerical prediction quality remains experimental. It should not be used as a source of authoritative air-quality measurements, environmental-health advice or real-world monitoring decisions.
