# Methodology

This document describes what was actually done, in the order it was done, with the file that
implements each step. Quantitative results belong to `EXPERIMENTAL_RESULTS.md`.

---

## 1. Pipeline at a glance

```
data/raw/AirQualityUCI.csv
    │  src/preprocess_air_quality.py       (parse, clean, split)
    ▼
data/processed/{train,validation,test}.csv
    │  src/prepare_training_data.py        (4 templates per row, tokenise)
    ▼
training/data_bin/{train,val,test}.bin    (uint32 token streams)
    │  src/train_slm.py                    (v1: 150 steps, block 64)
    ▼
models/air_quality_slm/                   (config, weights, tokenizer, summary)
    │  src/continue_training.py            (v2: 1,200 steps, block 128)
    ▼
models/air_quality_slm_finetuned/         (config, weights, best_model.pt, curve)
    │  src/convert_to_gguf.py
    │  src/export_gguf_from_checkpoint.py  (F32 GGUF, qwen3 arch)
    ▼
Ollama models (air-quality-slm, air-quality-slm-improved, air-quality-slm-v3)
    │  ollama_proxy.py + app.py + domain_guard.py + data_retrieval.py + humidity_calculator.py
    ▼
Assistant: model supplies language, Python supplies numbers
```

---

## 2. Dataset preparation

Detailed in `dataset/dataset_documentation.md`. Summary of the methodological choices that matter
for the paper:

* **No shuffling before splitting.** The split is contiguous in time (70/10/20), so validation and
  test contain *later* hours than training. This is the honest design for a monitoring record, and it
  is also what makes the numeric task a lookup rather than an interpolation.
* **Missing data kept as missing.** The `-200` sentinel becomes `NaN`, and in text becomes the word
  `Unavailable`; nothing is imputed. 16,701 cells are affected (1,683 in `CO(GT)`, 8,443 in
  `NMHC(GT)`).
* **No feature scaling.** Numbers are serialised verbatim into text, so the model sees raw physical
  values with their units written next to them.

## 3. Instruction data construction

`src/prepare_training_data.py::row_to_texts` emits four texts per record (full report,
absolute-humidity instruction, temperature/RH question, pollutant query — verbatim templates in
`dataset/dataset_documentation.md` §8). Each ends with the trained EOS token `<|endoftext|>`.

* Tokenizer: `Qwen/Qwen2.5-0.5B` BPE (fallbacks `Qwen2.5-7B`, `Qwen3-8B`, then `gpt2`), vocabulary
  151,665, `pad = eos = <|endoftext|>` (id 151,643), **no BOS**, `add_special_tokens=False`.
* Samples are concatenated into a single stream per split and stored as memory-mapped `uint32`.
* Measured sample lengths with this tokenizer (from `docs/EVALUATION_V2_REPORT.md` §2): report 193,
  absolute humidity 100 (86 prompt + 14 answer), temperature/RH 83 (36 + 47), pollutant 88 (43 + 45).

**The design defect this exposes.** v1 trained on `block_size = 64`. The *smallest complete* sample is
83 tokens, so no template ever appeared whole inside one training window; `Response:` and its answer
co-occurred roughly 2.6 times in the entire run. This is the mechanism behind v1's degenerate output,
and it is independent of how v1 is evaluated.

## 4. Model architecture

Implemented from scratch in `src/train_slm.py`; exported configuration in
`models/air_quality_slm/config.json` and `models/air_quality_slm_finetuned/config.json`.

| Component | Value |
|---|---|
| Family / HF class | Qwen3-style decoder, `Qwen3ForCausalLM` |
| Parameters | 20,397,568 trainable (39,810,688 stored, embeddings tied twice) |
| Hidden size `d_model` | 128 |
| Layers | 4 |
| Query heads / KV heads | 4 / 2 (GQA, 2:1) |
| Head dimension | 32 |
| Feed-forward | SwiGLU, intermediate 512 |
| Normalisation | RMSNorm, eps 1e-6, pre-norm |
| Attention extras | per-head Q/K RMSNorm applied before RoPE |
| Positional encoding | RoPE, theta 10,000, rotate-half |
| Context | v1 64 tokens; v2 128 tokens |
| Embedding tying | yes (`tie_word_embeddings: true`) |
| Input embedding scale | `tok_emb(x) * sqrt(128)`; output head unscaled |
| Vocabulary | 151,665 |
| Precision | float32, CPU |

Figure: `figures/architecture/fig04_model_architecture.png`.

## 5. Training

### 5.1 v1 — from random initialisation (`src/train_slm.py`)

| Setting | Value |
|---|---|
| Seed | 42 |
| Steps | 150 |
| Batch size | 8, no gradient accumulation (effective 8) |
| Block size | 64 |
| Optimiser | AdamW, lr 1e-3, betas (0.9, 0.95), weight decay 0.1, eps 1e-9 |
| Schedule | 10 % linear warmup + cosine decay to 0.1× |
| Gradient clipping | 0.5 |
| Eval cadence | every 50 steps, 15 batches |
| Device | CPU (no CUDA present) |
| Runtime | 279.01 s |
| Tokens seen | 150 × 8 × 64 = 76,800 ≈ **2.53 %** of the 3,041,432-token corpus |
| Checkpoints | `training/checkpoints/best_model.pt`, `latest_model.pt` (referenced; directory not present in the audited workspace) |
| Export | `models/air_quality_slm/` (weights, config, tokenizer, chat template, summary) |

Batch sampling in v1 is random **with replacement** (`torch.randint`), which is why its historical
validation figure (1.3651) is not exactly reproducible run-to-run.

### 5.2 v2 — continued training (`src/continue_training.py`)

Motivated by the measurement that no complete template fit a 64-token window.

| Setting | Value | Reason (as recorded in `docs/CONTINUED_TRAINING_REPORT.md`) |
|---|---|---|
| Initialisation | `models/air_quality_slm/pytorch_model.bin` | continue learned weights, never random init |
| Block size | **128** | every template except the 193-token report now fits whole |
| Steps attempted / completed | 2,400 target, stopped at 1,200 | early stopping, patience 6 evals |
| Batch × grad-accum | 2 × 2 (effective 4) | 8 GB RAM machine; logits at vocab 151,665 dominate memory |
| Learning rate | 2e-4 | resuming a converged model |
| Schedule | 20-step warmup + cosine to 0.1× | same shape as v1 |
| Sampler | one shuffled permutation per epoch | systematic coverage instead of sampling with replacement |
| Validation | 48 fixed evenly-spaced windows | identical batches for both models |
| Seed | 1234 | — |
| Threads | 6 of 12 | — |
| Device | CPU | no CUDA on the machine |
| Tokens seen | 614,400 = **0.202 epoch** (~8× v1 exposure) | — |
| Runtime | ≈31 min (1,864 s logged at step 1,200) | — |
| Best checkpoint | step 1,000 (val 0.4896) kept as `best_model.pt` | val rose to 0.4905 at 1,200 while train kept falling |

**Optimiser state was not resumed.** The exported v1 weights had no optimiser/scheduler state
attached (a bare `OrderedDict` of 47 tensors), so a fresh AdamW was initialised. This is documented
in the report rather than hidden; it means v2 is *continued training*, not a bit-exact resumption.

## 6. Evaluation methodology

One harness (`src/evaluate_v2_full.py`, summarised by `src/summarize_eval_v2.py`) scores **both**
checkpoints identically, so every difference attributed to a model is a model difference, not a
harness difference.

| Axis | Procedure |
|---|---|
| Full-split loss | non-overlapping windows over the entire token stream; every token scored exactly once (val 431,488 tokens / 3,371 windows; test 870,784 / 6,803); block 128 for both models, plus block 64 for v1 as a supplementary check |
| Teacher-forced answer loss | loss over answer tokens only, prompt given, per template |
| Generations | greedy decoding (temperature 0, top-k 1) from the four templates of `prepare_training_data.row_to_texts`; budget 50 new tokens |
| Numeric scoring | regex extraction of numbers from generated text, compared with the dataset's own label for that row; rows with missing labels dropped; "within 5 %" tolerance `abs(pred-gold) <= max(0.05, 0.05*abs(gold))` |
| Structure metrics | EOS emitted, ran to cap, repetitive, digits present, unit correct/wrong, malformed, hallucinated date |
| Memorisation | output verbatim equals a training answer; output timestamp matches an existing train row; date overlap between splits |
| Baselines | constant train-mean predictor per quantity (`docs/eval_v2/baseline.json`) |
| Cache correctness | cached decoding validated token-for-token against the uncached path (6 prompts, passed) |
| Determinism | greedy, so sampling variance is not characterised (stated as a limitation) |

The test split was used **for reporting only** — not for training, checkpoint selection or tuning of
the extractor; the extractor and tolerance were fixed on validation first.

## 7. Conversion to GGUF

`src/convert_to_gguf.py` (v1) and `src/export_gguf_from_checkpoint.py` (v2, parameterised). Both
write an **unquantised float32** GGUF with architecture `qwen3`:

* tensor mapping: `token_embd` (scaled by √128), `output` (unscaled, tied source), `output_norm`,
  and per block `attn_norm`, `attn_q/k/v/output`, `attn_q_norm`, `attn_k_norm`, `ffn_norm`,
  `ffn_gate/up/down`;
* tokenizer: full 151,665-token BPE vocabulary, 151,387 merges, `tokenizer.ggml.pre = qwen2`;
* chat template: replaced with a raw passthrough (`{% for message in messages %}{{ message['content'] }}{% endfor %}`)
  so no ChatML markers are injected into a model trained on raw text;
* context length: 64 (v1) / 128 (v2).

Verification (recorded, reproducible): llama.cpp reports `n_ctx_train` 64 / 128; 13/13 tokenizer
samples produce identical IDs; greedy Ollama output matches the PyTorch checkpoint token-for-token.
Details and checksums in `conversion/conversion_documentation.md`.

## 8. Deployment

`ollama create <name> -f <Modelfile>` registers each tag against the exported GGUF. Parameters were
chosen from measurement, not defaults: greedy decoding because the trained distribution is flat
(top-1 ≈ 0.18 on a weak prompt), `repeat_penalty` disabled because it distorted the answer,
`num_predict` 64 / 48 because the temperature/RH answer is 47 tokens (a 40-token cap truncated it),
and `num_ctx` matched to the trained window. Details in `ollama/deployment_documentation.md`.

## 9. Application layer

`app.py` (interactive shell) → `domain_guard.py` (greeting / out-of-domain / retrieval / humidity
routing) → `data_retrieval.py` (exact `(Date, Time)` lookup in the split CSVs) or
`humidity_calculator.py` (Tetens + dataset scale) → answer text. The language model is consulted only
for free-form continuation through `ollama_client.py`. `ollama_proxy.py` maintains the two-port
topology (11434 proxy, 11435 real server) that Ollama Desktop talks to, and `proxy_watchdog.py`
supervises both processes.

**The methodological rule this enforces:** every number shown to a user by the application comes
from Python. No output of `humidity_calculator.py` or `data_retrieval.py` is a neural prediction, and
no neural output is presented as a measurement.
