# PyTorch → GGUF Conversion

Two scripts implement this, both writing an **unquantised float32 GGUF** with architecture `qwen3`:

| Script | Used for | Notes |
|---|---|---|
| `src/convert_to_gguf.py` | v1 | first implementation, context 64 |
| `src/export_gguf_from_checkpoint.py` | v2 | parameterised; reads `config.json` and writes context 128 |

There is no quantisation step anywhere in this project. Every deployed model is F32.

---

## 1. Why conversion was necessary

The trained artefact was a `pytorch_model.bin` (a bare `OrderedDict` of 47 tensors) plus a custom
`Qwen3Model` class in `src/train_slm.py`. That combination can only be run by code that has both files
and matching imports. Ollama does not execute PyTorch; it executes GGUF files through its bundled
llama.cpp (`b12503` in this install). GGUF is a single self-describing container holding:

* the architecture name and every structural hyper-parameter,
* all weights (here F32),
* the complete tokenizer vocabulary and merge list,
* a chat template,
* the trained context length.

Converting therefore moves the model from "a Python program must reconstruct it" to "any llama.cpp
binary can load it".

## 2. What the converter had to map

The custom model is not a stock HF Qwen3 module, so tensor names had to be translated by hand.

| PyTorch tensor | GGUF tensor | Transformation |
|---|---|---|
| `tok_emb.weight` | `token_embd.weight` | **multiplied by √128** (the model scales embeddings on input) |
| `lm_head.weight` (tied) | `output.weight` | written from the same storage, **unscaled** |
| `norm.weight` | `output_norm.weight` | — |
| `blocks.N.attn_norm.weight` | `blk.N.attn_norm.weight` | — |
| `blocks.N.attn.wq.weight` | `blk.N.attn_q.weight` | — |
| `blocks.N.attn.wk.weight` | `blk.N.attn_k.weight` | 4 query heads, 2 KV heads |
| `blocks.N.attn.wv.weight` | `blk.N.attn_v.weight` | — |
| `blocks.N.attn.wo.weight` | `blk.N.attn_output.weight` | — |
| `blocks.N.attn.q_norm.weight` | `blk.N.attn_q_norm.weight` | per-head Q RMSNorm, before RoPE |
| `blocks.N.attn.k_norm.weight` | `blk.N.attn_k_norm.weight` | per-head K RMSNorm, before RoPE |
| `blocks.N.ffn_norm.weight` | `blk.N.ffn_norm.weight` | — |
| `blocks.N.mlp.w_gate.weight` | `blk.N.ffn_gate.weight` | SwiGLU gate |
| `blocks.N.mlp.w_up.weight` | `blk.N.ffn_up.weight` | SwiGLU up |
| `blocks.N.mlp.w_down.weight` | `blk.N.ffn_down.weight` | SwiGLU down |

Scalar metadata written: `general.architecture = "qwen3"`, `qwen3.block_count = 4`,
`qwen3.embedding_length = 128`, `qwen3.feed_forward_length = 512`,
`qwen3.attention.head_count = 4`, `qwen3.attention.head_count_kv = 2`,
`qwen3.attention.layer_norm_rms_epsilon = 1e-6`, `qwen3.rope.freq_base = 10000.0`,
`qwen3.context_length = 64 | 128`.

## 3. Tokenizer preservation

The vocabulary is copied from the Qwen2.5 BPE tokenizer already saved next to the weights:

| Item | Value |
|---|---|
| `tokenizer.json` size | 11,421,892 B |
| Vocabulary | 151,665 tokens |
| Merges | 151,387 |
| `tokenizer.ggml.pre` | `qwen2` |
| `eos_token_id` / `pad_token_id` | 151,643 (`<|endoftext|>`) |
| BOS | none |

Verification performed at conversion time and reproducible:
`src/verify_tokenizer.py` encodes 13 sample strings through both the HF tokenizer and the GGUF vocabulary
and compares IDs — all 13 match.

## 4. Chat template

Both GGUF files carry a **raw passthrough** template, not ChatML:

```
{% for message in messages %}{{ message['content'] }}{% endfor %}
```

Reason: the model was trained on raw instruction text with no ` <|im_start|>` markers. Injecting ChatML
at inference inserts tokens the model has never seen in that position and degrades output. This is why
the Modelfiles additionally use `TEMPLATE "{{ .Prompt }}"` (v1, v2) or `{{ .System }}\n{{ .Prompt }}`
(v3) rather than relying on the tokenizer's chat template.

## 5. Commands

```bash
# v1 — context 64
./venv/Scripts/python.exe src/convert_to_gguf.py

# v2 — parameterised, context 128
./venv/Scripts/python.exe src/export_gguf_from_checkpoint.py \
    --checkpoint models/air_quality_slm_finetuned/pytorch_model.bin \
    --config     models/air_quality_slm_finetuned/config.json \
    --tokenizer  models/air_quality_slm_finetuned \
    --out        models/air_quality_slm_finetuned/air_quality_slm_finetuned.gguf \
    --context    128
```

`convert_to_gguf.py` also produced `air_quality_slm_llama_adapted.gguf`, an intermediate variant kept for
comparison. The file actually referenced by `Modelfile` is `air_quality_slm.gguf`.

## 6. Artefacts and checksums

| File | Size (B) | SHA-256 (prefix) | Context |
|---|---:|---|---:|
| `models/air_quality_slm/air_quality_slm.gguf` | 165,774,560 | `10bb0ead…` | 64 |
| `models/air_quality_slm/air_quality_slm_llama_adapted.gguf` | — | `5ad37e73…` | 64 |
| `models/air_quality_slm_finetuned/air_quality_slm_finetuned.gguf` | 165,774,528 | `ebf9037d…` | 128 |

Note that the v1 and v2 GGUF sizes differ by 32 bytes: the weights are the same architecture and dtype,
so the difference is metadata (context length, and the extra template bytes), not parameters.

## 7. Compatibility considerations

| Concern | Finding |
|---|---|
| llama.cpp support for `qwen3` arch | supported by the bundled build; `ollama show` and `api/ps` both report the model correctly |
| Context length honoured | `n_ctx_train` reported as 64 (v1) and 128 (v2) |
| Tokenizer parity | 13/13 identical token ID sequences |
| Numerical parity with PyTorch | greedy Ollama output matched the checkpoint token-for-token on the test prompts recorded in `docs/OLLAMA_INTEGRATION.md` |
| Float precision | F32 only; no `Q4_K_M` or other quantised variant was produced or benchmarked |
| Memory | 165 MB per model on disk; runs comfortably in the 7.35 GiB machine alongside the OS |

## 8. Limitations of the conversion

* **No quantisation was tested.** Any claim about a smaller/faster quantised build would be unsupported.
* **Architecture is hand-mapped.** A future change to `src/train_slm.py` (e.g. adding biases or a
  different norm placement) requires a matching change in the converter; nothing validates the mapping
  automatically beyond the parity checks above.
* **The embedding √d scaling is baked into `token_embd`.** A consumer that additionally applies its own
  scaling would double-apply it; llama.cpp's `qwen3` implementation does not.
* **Tied weights are stored once.** The `output` tensor is written from the same storage; if a future
  model unties them, the converter must be updated.
* **Only `qwen3` was exercised.** The script's metadata writer is not generic.
