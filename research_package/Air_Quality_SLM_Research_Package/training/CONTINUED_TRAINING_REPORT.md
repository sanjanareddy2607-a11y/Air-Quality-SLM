# Air Quality SLM — Continued Training Report

**Original model:** `air-quality-slm:latest` — untouched, still registered, context 64
**Improved model:** `air-quality-slm-improved:latest` — context 128
**Original checkpoint SHA256:** `77BE6061B0CC249D881884922D388A67FAC4DF1D8604DAE7A7B157B1172DA727`
**Backup:** `models/air_quality_slm_original/` (byte-identical copy, plus `BACKUP_INFO.json`)

---

## 1. What was inspected before training

| Item | Finding |
|---|---|
| `src/train_slm.py` | **Cannot resume.** Always builds a fresh model with `torch.manual_seed(42)` and trains from random init. |
| Checkpoint contents | `pytorch_model.bin` is a bare `OrderedDict` of **47 tensors**, float32, no optimizer/scheduler state. 39,810,688 stored parameters (embeddings tied and stored twice; ~20.4 M unique). |
| Optimiser used originally | New `AdamW(lr=1e-3, betas=(0.9,0.95), weight_decay=0.1, eps=1e-9)`, LinearLR warmup 10 % + cosine decay to `lr*0.1`, grad-clip 0.5. |
| Saved state | `training/checkpoints/latest_model.pt` *does* contain `optimizer_state_dict` at step 149, but it is not paired with the exported weights, so resuming exactly is not possible. **A fresh optimiser was initialised — documented, not hidden.** |
| Batch sampling | `ix = torch.randint(len(data) - block_size - 1, (batch_size,))` → **random with replacement**, never systematic. |
| Original coverage | 150 iters × 8 × 64 = **76,800 tokens = 2.53 %** of the 3,041,432-token corpus. |
| Validation | Random windows each call, so the historical 1.3651 is not reproducible run-to-run. |
| Sample lengths | report **201**, absolute humidity **101**, temp/RH **83**, pollutant **90** tokens — all longer than `block_size = 64`. This is the structural defect: a template's instruction and its answer were rarely in the same window (measured: **≈2.6** exposures to `Response:` → answer in the whole original run). |

**Baseline re-measured** with deterministic evenly-spaced validation windows (`src/continue_training.py`):
val **1.41086** @ block 64 (historical random-window figure 1.3651), test **1.43349** @ block 64 (historical 1.4295).

---

## 2. Schedule actually used

Chosen from measurement, not guesswork. A CPU benchmark on this machine gave **723 tokens/s** at block 128.

| Setting | Value | Why |
|---|---|---|
| Init | `pytorch_model.bin` (original) | continue learned weights, never random |
| block_size | **128** | every template except the 201-token report now fits whole; the report boundary at token 34 is unaffected |
| batch × grad-accum | 2 × 2 | 8 GB RAM total, ~1.5 GB free — logits at vocab 151,665 dominate memory |
| lr | **2e-4** (was 1e-3) | resuming a converged model, not starting one |
| schedule | 20-step warmup + cosine to 0.1× | same shape as the original run |
| sampler | one shuffled permutation per epoch | systematic coverage instead of sampling with replacement |
| validation | 48 fixed, evenly-spaced windows | identical batches for original and new model |
| early stopping | patience 6 evals | overfitting guard |
| device | **CPU** (no CUDA on this machine) | 12 threads, 6 used |

Progress: 1200 steps, 614,400 tokens = **0.202 epoch** (8× the original exposure), 31 minutes.

**Stopped on plateau:** val loss 0.4896 @ step 1000 → 0.4905 @ step 1200 while train loss kept falling (0.4325 → 0.4229). That is the overfitting signature, so the best checkpoint (step 1000) was kept.

---

## 3. Loss results (same deterministic eval for both models)

| step | train (EMA) | val | val ppl | tokens | epochs |
|---|---|---|---|---|---|
| baseline | — | **1.4433** | 4.235 | 0 | — |
| 200 | 0.6425 | 0.6957 | 2.005 | 102,400 | 0.034 |
| 400 | 0.4955 | 0.5559 | 1.744 | 204,800 | 0.067 |
| 600 | 0.4699 | 0.5279 | 1.695 | 307,200 | 0.101 |
| 800 | 0.4470 | 0.5187 | 1.680 | 409,600 | 0.135 |
| **1000** | 0.4325 | **0.4896** | **1.632** | 512,000 | 0.168 |
| 1200 | 0.4229 | 0.4905 | 1.633 | 614,400 | 0.202 |

**Validation loss −66 %** (1.4433 → 0.4896), **perplexity 4.24 → 1.63**.
Curve: `docs/loss_curve.svg`, data `docs/loss_curve.csv`, raw log `docs/training_log_finetune.jsonl`.

Note: the block-128 baseline (1.4433) is higher than the block-64 figure (1.4109) because the model is being asked to predict positions 64–127 it had never trained on. The comparison above is like-for-like at block 128.

---

## 4. Generation: before vs after (identical greedy decoding)

Prompts are real templates from **held-out validation records**; both models get the same text.

| Template | Original `air-quality-slm` | Improved `air-quality-slm-improved` |
|---|---|---|
| **Absolute humidity** | `' What were the ambient temperature and relative humidity is 1.1 ug/m3.'` | **`' The absolute humidity is 1.15 g/m3.'`** ✅ correct sentence, correct unit, EOS |
| **Temp/RH** | `' 1.0.0.00, the ambient temperature and the ambient temperature and the ambient temperature and…'` | **`' On 04/04/2004 at 04.00.00, the ambient temperature was 21.2 degrees Celsius and the relative humidity was '`** |
| **Pollutant** | `' 1.00.0.0, NOx(GT)=112 ug/m3 (sensor PT08.S3=1111), Non-Methane Hydrocar'` | **`' Recorded CO is 2.2 mg/m3, NOx is 111 ppb, and NO2 is 111 ug/m3 with benzene level at 11'`** |
| **Report completion** | `'1.8 ug/m3 (sensor PT08.S3=1111), Non-Methane Hydrocarbons NMHC(GT)=1.8 ug/m3 …'` | `'1.2 mg/m3 (sensor PT08.S1=1021), Non-Methane Hydrocarbons NMHC(GT)=Unavailable ug/m3 (sensor PT08'` |

Rubric over 8 prompts (`src/evaluate_finetuned.py`): **improved 2, worse 0, tied 6**.
EOS emitted 0/8 → **2/8**; correct unit 4/8 → **6/8**; correct opening phrase 0/8 → **2/8**.

### Why the absolute-humidity case changed so much

Two independent causes were fixed:

1. **Context.** The template is 86 tokens; the old GGUF declared `context_length 64`, so Ollama silently cut every ≥64-token prompt down to its last **34** tokens — the model literally never saw the instruction. The new export declares **128**, and llama.cpp now confirms `prompt tokens evaluated: 86`.
2. **Training.** With a 128-token window, `Response:` and its answer finally co-occur, so `P(answer | Response:)` gets trained instead of seen ~2.6 times.

Confidence on the first token of the answer: original `P(' The') = 0.071`; improved `logprob −0.0101` (`P ≈ 0.99`).

---

## 5. Conversion verification

`src/export_gguf_from_checkpoint.py` reuses the exact mapping already proven equivalent
(`qwen3` arch, per-head Q/K RMSNorm before RoPE, `token_embd` × √128, unscaled tied `output`,
4 blocks, 151,665 tokens / 151,387 merges, passthrough chat template). Changes: paths and
`--context 128`.

Checked: tensor shapes asserted before writing; llama.cpp reports `n_ctx_train 128`; the
86-token prompt is evaluated in full; the GGUF's greedy output matches PyTorch token-for-token
(`' The absolute humidity is 1.15 g/m3.'`).

---

## 6. Using it

Ollama Desktop: pick **`air-quality-slm-improved:latest`** in the model selector.
The original **`air-quality-slm:latest`** is still there for comparison.

CLI:

```bash
ollama run air-quality-slm-improved "Task: Predict the absolute humidity given the following parameters: Date: 08/12/2004, Time: 15.00.00, CO(GT): 1.2, NOx(GT): 179, NO2(GT): 82, Temperature: 16.7 C, Relative Humidity: 54.7 %. Response:"
```

Rebuild from scratch if needed:

```bash
python src/export_gguf_from_checkpoint.py --checkpoint models/air_quality_slm_finetuned/pytorch_model.bin \
  --config models/air_quality_slm_finetuned/config.json --tokenizer models/air_quality_slm \
  --output models/air_quality_slm_finetuned/air_quality_slm_finetuned.gguf --context 128
ollama create air-quality-slm-improved -f Modelfile.improved
```

---

## 7. Honest limitations

- **Structure improved, numbers did not.** The model now writes the right sentence for each
  template, with the right field names and units, and terminates properly. The **values are
  still wrong**: humidity 1.15 vs the true 1.03; temp/RH invented a different date and
  temperature; pollutant values 2.2/111/111 vs true 1.2/179/82. It reproduces the *shape* of an
  observational report, not the observation. There is no retrieval mechanism, so it cannot
  recall a specific reading.
- **Only 0.2 epoch was completed** (614k of 3.04M tokens) in ~31 minutes on this CPU. A full
  epoch would take ~70 minutes locally; on a Colab T4 it is a few minutes — notebook provided at
  `notebooks/Air_Quality_SLM_Continue_Training_Colab.ipynb`.
- **Validation plateaued at ~0.49 nats/token (perplexity 1.63)** while train loss kept
  falling. Further identical training is unlikely to give much more; **numeric fidelity needs
  either much more training or a different approach** (e.g. a retrieval/table lookup for values,
  which this model does not have).
- **The 201-token report template still exceeds the 128 window**; it is trained across two
  windows.
- **20.4 M parameters** is small. It will not become a general chatbot, and it was never
  intended to be one. The target — a coherent, correctly-formatted air-quality SLM — is met for
  format, not for factual accuracy.
- Context 128 is what the model was trained at; do not raise `num_ctx` beyond 128 in the
  Modelfile without retraining at that length.
