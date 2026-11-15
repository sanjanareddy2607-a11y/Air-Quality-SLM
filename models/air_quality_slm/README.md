---
language:
- en
license: cc-by-4.0
tags:
- air-quality
- slm
- qwen3
- time-series
- environmental-monitoring
- tabular-lm
datasets:
- sanju7/air-quality-uci
pipeline_tag: text-generation
model_type: air_quality_slm
---

# Air Quality Small Language Model (SLM)

A domain-specific Small Language Model (SLM) tailored for urban air quality telemetry, chemical sensor interpretation, and environmental time-series reasoning. The model architecture is built from scratch based on **Qwen3** (featuring Rotary Position Embeddings, Grouped-Query Attention with QK Normalization, SwiGLU FeedForward activations, and RMSNorm).

---

## 1. Model Summary

- **Model Name:** Air Quality SLM (Qwen3 Architecture)
- **Domain:** Urban Atmospheric Chemistry & Air Quality Telemetry
- **Primary Source Data:** UCI Machine Learning Repository – Air Quality Dataset (co-located reference analyzers and 5 metal-oxide chemical sensors)
- **Base Architecture:** Qwen3 Transformer Block (RMSNorm Pre-Norm, RoPE, Grouped-Query Attention, SwiGLU)
- **Tokenizer:** Qwen2.5 / Qwen3 BPE Tokenizer (`vocab_size = 151,665`)
- **Parameters:** ~20.4M parameters (optimized for resource-constrained edge devices and academic reproduction)
- **License:** CC BY 4.0

---

## 2. Architecture Specifications

| Component | Specification |
|:---|:---|
| **Base Family** | Qwen3 Causal LM |
| **Hidden Dimension ($d_{\text{model}}$)** | 128 |
| **Transformer Layers ($n_{\text{layers}}$)** | 4 |
| **Attention Heads ($n_{\text{heads}}$)** | 4 |
| **KV Groups ($n_{\text{kv\_groups}}$)** | 2 (Grouped-Query Attention, 2x sharing) |
| **Feed-Forward Dimension ($d_{\text{ff}}$)** | 512 (SwiGLU activation) |
| **Positional Encoding** | Rotary Position Embeddings (RoPE, $\theta = 10,000$) |
| **Normalization** | RMSNorm ($\epsilon = 10^{-6}$) with QK Normalization |
| **Context Window ($L_{\text{ctx}}$)** | 64 – 128 tokens |
| **Weight Tying** | Input embeddings tied with output linear head |

---

## 3. Training & Preprocessing

### Domain Formatting
The tabular dataset was converted into diverse natural-language instructional templates:
1. **Full Environmental Observations:** Hourly measurements across CO, NMHC, C6H6, NOx, NO2, sensor potentials (PT08.S1 to PT08.S5), temperature, and humidity.
2. **Predictive Tasks:** Calibrated estimation of absolute humidity and pollutant concentrations.
3. **Meteorological Queries:** Temperature and relative humidity reporting for historical timestamps.
4. **Missing Value Handling:** Sentinel values (`-200`) transformed into calibrated missingness tokens (`Unavailable`) to prevent hallucination.

### Optimization
- **Optimizer:** AdamW ($\beta_1 = 0.9, \beta_2 = 0.95, \text{weight decay} = 0.1, \epsilon = 10^{-9}$)
- **Learning Rate Schedule:** Linear Warmup followed by Cosine Annealing Decay.
- **Gradient Clipping:** Max norm of $0.5$.

---

## 4. Usage & Inference Example

You can load and run inference on this model directly with PyTorch and the Hugging Face tokenizer:

```python
import torch
import json
from transformers import AutoTokenizer
from src.train_slm import Qwen3Model

# 1. Load Tokenizer & Model Config
model_dir = "models/air_quality_slm"
tokenizer = AutoTokenizer.from_pretrained(model_dir)

with open(f"{model_dir}/config.json", "r") as f:
    cfg = json.load(f)

# 2. Instantiate and Load Weights
model = Qwen3Model(
    vocab_size=cfg["vocab_size"],
    d_model=cfg["hidden_size"],
    n_heads=cfg["num_attention_heads"],
    n_kv_groups=cfg["num_key_value_heads"],
    n_layers=cfg["num_hidden_layers"],
    d_ff=cfg["intermediate_size"],
    max_seq_len=cfg["max_position_embeddings"],
    tie_weights=cfg.get("tie_word_embeddings", True),
)

state_dict = torch.load(f"{model_dir}/pytorch_model.bin", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# 3. Generate Prediction
prompt = "Task: Predict the absolute humidity given the following parameters: Date: 11/03/2004, Time: 02.00.00, CO(GT): 0.9, NOx(GT): 45.0, NO2(GT): 60.0. Response:"
input_ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False)).unsqueeze(0)
output_ids = model.generate(input_ids, max_new_tokens=40, temperature=0.7, top_k=40)

print("Generated Response:")
print(tokenizer.decode(output_ids.squeeze().tolist())[len(prompt):].strip())
```

---

## 5. Intended Uses & Limitations

- **Intended Use:** Environmental sensor telemetry representation, air quality research, edge SLM exploration, and academic benchmark studies.
- **Limitations:** This SLM is trained specifically on the Italian urban air quality dataset (2004-2005) and should be validated with local sensor recalibration before critical field deployments.
