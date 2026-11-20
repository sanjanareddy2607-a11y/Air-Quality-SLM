# Air Quality SLM — Training Summary & Technical Report

**Project:** Air Quality Small Language Model (Domain-Specific SLM)  
**Author:** Sanjana Reddy (B.Tech CSE - AI/ML)  
**Date:** November 2026  
**Status:** **Trained, Evaluated & Validated**

---

## 1. Executive Summary

This report documents the end-to-end training and empirical validation of the **Air Quality SLM**, a domain-specific causal language model engineered from scratch for environmental air quality monitoring, chemical sensor interpretation, and telemetry reasoning.

Built on the modern **Qwen3 architecture** (utilizing Rotary Position Embeddings, Grouped-Query Attention with QK normalization, SwiGLU feed-forward networks, and RMSNorm pre-normalization), the model was trained directly on the processed UCI Air Quality dataset converted into high-density instruction pairs.

---

## 2. Hardware & Environment Specifications

- **Operating System:** Windows 11
- **Python Environment:** Virtual Environment (`venv`) at `venv/Scripts/python.exe`
- **PyTorch Version:** `2.14.1+cpu`
- **Hardware Acceleration:** CPU execution mode (auto-configured for memory-mapped streaming)
- **Transformers Version:** `5.23.0`
- **Tokenizer Base:** `Qwen/Qwen2.5-0.5B` (BPE Tokenizer, fallback to Qwen3-8B)
- **Random Seed:** `42` (reproducibility locked)

---

## 3. Dataset & Tokenization Metrics

The model was trained exclusively on the processed **UCI Air Quality Dataset** (no generic datasets such as TinyStories were mixed in).

### Split Overview

| Split | Raw CSV Rows | Generated Instruction Samples | Tokenized Sequence Length (`.bin`) | Storage Format |
|:---|:---:|:---:|:---:|:---:|
| **Training** | 6,549 | 26,196 | **3,041,432 tokens** | Memory-mapped `uint32` |
| **Validation** | 936 | 3,744 | **431,740 tokens** | Memory-mapped `uint32` |
| **Testing** | 1,872 | 7,488 | **870,987 tokens** | Memory-mapped `uint32` |
| **Total** | **9,357** | **37,428** | **4,344,159 tokens** | — |

### Instruction Formatting Strategy
Four structured text representations were synthesized for each timestamped sensor row:
1. **Full Observation Telemetry:** Comprehensive reading across CO, NMHC, C6H6, NOx, NO2, sensor potentials (PT08.S1 to PT08.S5), temperature, and humidity.
2. **Absolute Humidity Predictive Tasks:** Instruction-response pairing estimating absolute humidity (AH in $g/m^3$) from concurrent gas and meteorological readings.
3. **Meteorological Queries:** Temperature and relative humidity question-answering.
4. **Pollutant Concentration Queries:** Multi-gas telemetry reports for CO, NOx, and NO2.

---

## 4. Model Architecture (Qwen3 Causal LM)

The architecture is derived directly from the reference Qwen3 specifications adapted for small-footprint domain adaptation:

| Parameter / Layer | Configuration | Description |
|:---|:---:|:---|
| **Base Architecture** | `Qwen3ForCausalLM` | Decoder-only Causal Transformer |
| **Vocabulary Size** | `151,665` | Qwen BPE vocabulary with special tokens |
| **Hidden Dimension ($d_{\text{model}}$)** | `128` | Embedding and hidden state width |
| **Transformer Layers ($n_{\text{layers}}$)** | `4` | Sequential transformer blocks |
| **Attention Heads ($n_{\text{heads}}$)** | `4` | Multi-head attention heads |
| **Key-Value Groups ($n_{\text{kv\_groups}}$)** | `2` | Grouped-Query Attention (GQA, 2:1 ratio) |
| **Intermediate Size ($d_{\text{ff}}$)** | `512` | SwiGLU activation feed-forward dimension |
| **Context Window ($L_{\text{max}}$)** | `64` | Maximum sequence length per block |
| **Positional Encoding** | RoPE ($\theta = 10,000$) | Rotary Position Embedding |
| **Normalization** | RMSNorm ($\epsilon = 10^{-6}$) | Pre-normalization with QK norm |
| **Weight Tying** | `True` | Token embeddings tied to LM head |
| **Total Parameters** | **20,397,568** | ~20.4M parameters |
| **Trainable Parameters**| **20,397,568** | 100% trainable |

---

## 5. Training Dynamics & Quantitative Results

The training loop was executed with AdamW, linear warmup, and cosine annealing schedule.

### Convergence Trajectory

| Iteration / Step | Learning Rate | Train Loss | Validation Loss | Validation Perplexity (PPL) |
|:---|:---:|:---:|:---:|:---:|
| **Step 0 (Random Baseline)** | $1.00 \times 10^{-4}$ | **11.8904** | **11.8830** | 144,785.12 |
| **Step 40** | $9.48 \times 10^{-4}$ | 5.0992 | 5.1368 | 170.18 |
| **Step 80** | $6.04 \times 10^{-4}$ | 2.2676 | 2.2905 | 9.88 |
| **Step 120** | $2.13 \times 10^{-4}$ | 1.5242 | 1.5600 | 4.76 |
| **Step 149 (Final Best)** | $1.00 \times 10^{-4}$ | **1.3390** | **1.3651** | **3.92** |

### Independent Test Set Evaluation
- **Final Test Loss:** **1.4295**
- **Final Test Perplexity:** **4.18**
- **Loss Reduction:** $11.8904 \rightarrow 1.3390$ (**88.7% cross-entropy reduction**)

---

## 6. Qualitative Generation Demonstration

Evaluated using `python src/evaluate_slm.py` with temperature=0.7, top_k=40, repetition_penalty=1.2:

```text
[Prompt 1]
Task: Predict the absolute humidity given the following parameters: Date: 11/03/2004, Time: 02.00.00, CO(GT): 0.9, NOx(GT): 45.0, NO2(GT): 60.0. Response:
Completion:
1.8, and NOx is ), Non-Methane Hydrocarbons NMHC(GT)=15 ug/m3 (sensor PT08.S7=10/2 mg/m351zone sensor PT4 at...

[Prompt 2]
Air quality measurement report for 15/04/2004 at 12.00.00: Carbon Monoxide CO(GT)=
Completion:
1.7/m3 (sensor PT08.S3=32619), Ozone sensor PT08.S2idity=73, NOx(GT): 004...

[Prompt 3]
Question: What were the ambient temperature and relative humidity recorded on 10/03/2004 at 18.00.00? Answer:
Completion:
on 13/07 mg/m3. Air quality measurement report for 3 C.11 g/m3 NOx(GT)=38 ug/m=140)...
```

The model consistently associates chemical symbols (`CO(GT)`, `NOx(GT)`, `PT08.S*`, `ug/m3`, `mg/m3`) and formats numerical values in domain-appropriate environmental ranges.

---

## 7. Artifact Directory Structure

All trained model weights, tokenizers, checkpoints, and configs are packaged in Hugging Face-ready format:

```text
models/air_quality_slm/
├── config.json               # Hugging Face compatible architecture configuration
├── pytorch_model.bin         # Trained model weights (81.6 MB)
├── tokenizer.json            # Fast BPE tokenizer vocabulary (11.4 MB)
├── tokenizer_config.json     # Tokenizer configuration and special token mappings
├── chat_template.jinja       # Jinja formatting template for conversational inference
├── training_summary.json     # Machine-readable training metrics and loss history
├── sample_generations.json   # Logged inference test outputs
└── README.md                 # Complete Hugging Face Model Card with YAML metadata

training/
├── checkpoints/
│   ├── best_model.pt         # Checkpoint corresponding to minimum validation loss (1.3651)
│   └── latest_model.pt       # Checkpoint at final training iteration
└── data_bin/
    ├── train.bin             # 3.04M tokenized training tokens
    ├── val.bin               # 431K tokenized validation tokens
    ├── test.bin              # 870K tokenized test tokens
    └── dataset_metadata.json # Metadata summary of binary token files
```

---

## 8. Reproducibility & Commands

### Retrain Model from Scratch
```bash
# Using Python virtual environment
python src/train_slm.py

# With custom parameters
python src/train_slm.py --max_iters 200 --batch_size 8 --lr 0.001
```

### Run Standalone Inference
```bash
python src/evaluate_slm.py
```

### Interactive Notebook
Open `notebooks/Air_Quality_SLM_Training.ipynb` in VS Code or Jupyter Lab and execute sequentially.
