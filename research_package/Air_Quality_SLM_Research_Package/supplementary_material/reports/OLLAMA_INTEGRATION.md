# Air Quality SLM — Ollama Integration & Garbled-Output Diagnosis

**Model name in Ollama / Ollama Desktop:** `air-quality-slm:latest`

This document records the export, the diagnosis of the garbled output, the fix applied, and
the honest limits of the trained model. The original training artefacts were not modified.

---

## 1. Root cause of the garbled output

**It was the generation settings, not the GGUF conversion and not the tokenizer.**

`Modelfile` asked for `temperature 0.7`, `top_k 40`, `repeat_penalty 1.2` and left the output
length unbounded. Measured on the real checkpoint, the model's next-token distribution is
extremely flat: on `what is the humidity?` the top token is only **p = 0.18**, and the next two
are 0.13 and 0.11. With `top_k 40` there are ~40 candidates of similar rank, most of them
unrelated Qwen-2.5 vocabulary fragments (`masks`, `_models`, `Bangladesh`, `treffen`, CJK
characters). Sampling from that set picks junk most of the time, and because the model then
conditions on its own junk the run never returns to the trained format — it drifts further out
of distribution. `repeat_penalty 1.2` made it worse by penalising *prompt* tokens the answer
must repeat (`Date`, `Temperature`, digits).

Reproduced exactly through the Desktop code path (`POST /api/chat`, no option overrides):

```
input : what is the humidity ?
output: '_models masksยังมี Bangladeshรัง.o\tcfFileSize kan回來 millennialsParameters…'
        (640 tokens, never emitted EOS, full of CJK/kana fragments)
```

## 2. Is the original PyTorch model affected?

**No. With greedy decoding the original checkpoint produces the same text as Ollama,
token for token.** The model itself is coherent on short prompts; only the sampling was wrong.

Direct comparison (`src/compare_pytorch_ollama.py`), identical prompts, `temperature 0 / top_k 1`:

| Prompt | PyTorch greedy | Ollama greedy |
|---|---|---|
| `what is the humidity?` | `' Result: 1.1 ug/m3.\n<|endoftext|>\nQuestion: What were the ambient temperature and relative humidity recorded on 11/08/2004 at 11.'` | `' Result: 1.1 ug/m3.\n'` (stops at EOS) |
| `What were the temperature and relative humidity?` | `' Result: 11.8/20004 at 1.0.0.00, NOx(GT)=111 ug/m3 (sensor PT08'` | identical |
| `Air quality measurement report for 15/04/2004 at 12.00.00:` | `' What were the ambient temperature and the ambient temperature and…'` | identical |
| `…: Carbon Monoxide CO(GT)=` (real training record) | `'1.8 ug/m3 (sensor PT08.S3=1111), Non-Methane Hydrocarbons NMHC(GT)=1.8 ug/m3 (sensor PT08.S2=111'` | identical |

The only difference is that Ollama *stops* at the EOS token while the PyTorch harness keeps
generating past it (the harness did not filter EOS). Beyond EOS both continue with the same
tokens. One digit differed in the `1.0` vs `1.1` case — a float32 tie-break on a near-equal
argmax, not a mapping error.

At the logit level the earlier verification stands (agreement to ~1e-4 on the same tokens),
but it was not the reason for the garble: a *sampled* decoder does not care about 1e-4.

## 3. What was checked, and ruled out

| Suspect | Result |
|---|---|
| **Tokenizer / vocabulary mapping** | Verified with `src/verify_tokenizer.py` against llama.cpp's own GGUF tokenizer: **13/13 samples, identical IDs**, covering dates, decimals, `PT08.S2(NMHC)`, `CO(GT)`, whitespace runs, accented Latin, CJK/Cyrillic/Arabic, and `<|endoftext|>` → 151643. Multilingual output was therefore *genuine model output*, not a mis-mapped vocabulary. |
| **Vocabulary ordering / merges / pre-tokenizer** | 151,665 tokens, 151,387 merges, `tokenizer.ggml.pre = qwen2`; full-vocabulary ID→token mapping identical. |
| **Special tokens** | `eos = <|endoftext|> = 151643`, `bos = None`, `pad = <|endoftext|>`. Verified `prompt_eval_count` equals the HF token count exactly (5/8/27/34) → **no BOS, no template wrapper is injected**. |
| **Chat template** | GGUF `tokenizer.chat_template` is a passthrough; `Modelfile` uses `TEMPLATE "{{ .Prompt }}"`. Ollama receives exactly the typed text. |
| **Tensors / architecture** | `qwen3` graph: `token_embd` (scaled by √128), `output` (unscaled, tied source), `attn_q/k/v/output`, `attn_q_norm`, `attn_k_norm`, `ffn_gate/up/down`, `attn_norm`, `ffn_norm`, `output_norm`, 4 blocks, vocab 151,665 × 128. Confirmed by the fact that Ollama reproduces PyTorch text exactly — any missing or mis-shaped tensor would diverge immediately. |
| **Embedding scale / QK-norm / GQA / RoPE / SwiGLU** | Present; no approximation is applied. Greedy equality across 40 tokens on 4 different prompts is a stronger check than selected log-probabilities. |

## 4. The fix

`Modelfile` only — **no re-export, no retraining, no quantisation, no Ollama reinstall.**

```diff
- PARAMETER temperature 0.7
- PARAMETER top_k 40
- PARAMETER repeat_penalty 1.2
+ PARAMETER temperature 0
+ PARAMETER top_k 1
+ PARAMETER repeat_penalty 1.0
+ PARAMETER repeat_last_n 64
+ PARAMETER num_predict 64
  PARAMETER num_ctx 64
  PARAMETER stop "<|endoftext|>"
```

Greedy decoding is the most faithful option available for a near-flat distribution: it always
takes the model's actual best guess instead of a random draw from a 40-token tail. It is also
what reproduces the PyTorch output exactly. `num_predict 64` bounds the response so a bad run
cannot dump hundreds of tokens. Re-registered with:

```bash
ollama create air-quality-slm -f Modelfile
```

The GGUF bytes are unchanged (`ollama create` reused the existing weights layer).

## 5. Before / after (`POST /api/chat`, the Desktop code path)

| # | Prompt | Before (temp 0.7, top_k 40, rp 1.2) | After (greedy) |
|---|---|---|---|
| 1 | `what is the humidity?` | `' Result: 41/04.2/20038.074 at 073 mg/m3.'` | **`' Result: 1.1 ug/m3.'`** (11 tokens, clean EOS) |
| 2 | `What were the temperature and relative humidity?` | `':42/201.8.S2=6060 C, Temperature=7.5 ug/m3, Absolute Humidity=.2 ppb (CO, NOx(GT): 12 g/m3.'` | `' Result: 11.8/20004 at 1.0.0.00, NOx(GT)=111 ug/m3 (sensor PT08.S2=1111), Non-Methane Hydrocarbons NMHC(GT)=11 ug/m'` |
| 3 | `Air quality measurement report for 15/04/2004 at 12.00.00:` | `' What are the ambient temperature was 3), Non-Methane Hydrocarbons NMHC(GT)==38 ug/m3 (sensor PT08.Sx is 647 O: 00 C, NOx(GT):2(GT)=1.8 ug/m3, Temperature: 12.7 g/m3.'` | `' What were the ambient temperature and the ambient temperature and the ambient temperature and…'` (degenerate loop — honest greedy behaviour, same in PyTorch) |
| 4 | `Air quality measurement report for 10/03/2004 at 18.00.00: Carbon Monoxide CO(GT)=` | `'67,/m3 (sensor PT0.18.S4=4485), Temperature3/28 C,: 15.08 ug/m9.3.'` | **`'1.8 ug/m3 (sensor PT08.S3=1111), Non-Methane Hydrocarbons NMHC(GT)=1.8 ug/m3 (sensor PT08.S2=1111), Benzene C66(GT)=1.8 ug/m3'`** |

Prompt 3 still fails: it loops. That is the trained weights' preference (the same loop appears
in PyTorch), not a decoding artefact, and no setting fixes it honestly.

## 6. Remaining limitations (properties of the weights)

- **Training was very short.** 150 iterations × batch 8 × block 64 ≈ **77k tokens seen**, for a
  20.4M-parameter model. Validation loss 1.365 (ppl 3.92).
- **Format dependence.** The model continues the four training templates well and drifts on
  free-form questions. `what is the humidity?` yields a well-formed but invented value.
- **Digits are unreliable.** Values are template-shaped, not recalled from the dataset. The
  model has no retrieval mechanism, so it cannot know the readings for a given date.
- **Near-ties cause loops.** A flat head makes greedy decoding repeat a phrase when the
  argmax lands just off the training path.
- **Context is 64 tokens.** Prompts longer than 64 are truncated by Ollama. The full
  absolute-humidity template (all six parameters) is 89 tokens and does not fit; a shortened
  form is OOD and performs badly. PyTorch itself cannot run >64 positions here — `apply_rope`
  slices a 64-position table and raises on longer input.
- **F32, unquantised.** No quantisation was applied.

## 7. Recommended prompts for a demonstration

Best (in-distribution report continuation — the model's strongest behaviour):

```
Air quality measurement report for 10/03/2004 at 18.00.00: Carbon Monoxide CO(GT)=
```

Good:

```
Query: What are the primary pollutant concentrations (CO, NOx, NO2) on 10/03/2004 at 18.00.00? Result:
Question: What is the absolute humidity? Answer:
what is the humidity?
What were the temperature and relative humidity recorded on 10/03/2004 at 18.00.00?
```

Avoid for the demo: `Air quality measurement report for 15/04/2004 at 12.00.00:` on its own
(loops), and the full six-parameter absolute-humidity sentence (89 tokens > 64).

Prompt screening data: `docs/prompt_screening.json`. Corrected-model raw output:
`docs/chat_results_after.json`. PyTorch-vs-Ollama comparison: `docs/compare_pytorch_ollama.json`.
Per-template capability probe: `docs/probe_capability.json`. Tokenizer proof:
`docs/tokenizer_verification.json`.

## 7b. Controlled diagnosis: the absolute-humidity template

Script: `src/diagnose_absolute_humidity.py`. Raw report: `docs/diagnose_absolute_humidity.json`.
The template is imported from `prepare_training_data.row_to_texts` (text #2), not retyped.

**Example used** — `data/processed/validation.csv` row 0
(`Date 08/12/2004, Time 15.00.00, CO(GT) 1.2, NOx(GT) 179, NO2(GT) 82, T 16.7 C, RH 54.7 %, AH 1.0311`):

```
input  : Task: Predict the absolute humidity given the following parameters: Date: 08/12/2004,
         Time: 15.00.00, CO(GT): 1.2, NOx(GT): 179, NO2(GT): 82, Temperature: 16.7 C,
         Relative Humidity: 54.7 %. Response:            <-- 86 tokens
expected: The absolute humidity is 1.03 g/m3.\n<|endoftext|>
```

### The template does not fit the model's own context

| quantity | value |
|---|---|
| prompt tokens | **86** |
| whole training sample | **101** |
| `block_size` / `max_position_embeddings` | **64** |

**Training side.** `prepare_training_data` concatenates all samples into one stream and
`get_batch` draws random 64-token windows. A 101-token sample can never appear whole in a
window, so `Response:` and its answer are almost never adjacent in any training input:

| statistic | value |
|---|---|
| corpus tokens | 3,041,432 |
| `' Response'` + `':'` occurrences | 6,549 |
| windows ending exactly on `Response:` | 0.215 % |
| windows actually drawn (150 iters × batch 8) | 1,200 |
| **expected times trained on P(answer \| `Response:`)** | **≈ 2.6** |
| tokens ever seen | 76,800 = **2.53 %** of the corpus |

**Deployment side.** The GGUF declares `qwen3.context_length = 64`, so llama.cpp logs
`the slot context (512) exceeds the training context of the model (64) - capping` and Ollama
likewise caps at 64. Measured Ollama behaviour on this prompt:

| prompt length | `prompt_eval_count` |
|---|---|
| 60, 61, 62, 63 tokens | 60, 61, 62, 63 (full) |
| **64 or more** | **34** |

The 86-token prompt is therefore reduced to its **last 34 tokens**;
the output is byte-identical to running that 34-token fragment alone. The model never receives
`Task: Predict the absolute humidity …` — only `179, NO2(GT): 82, Temperature: 16.7 C, … Response:`.

### Both engines agree, so the conversion is not at fault

Feeding PyTorch the *same* fragments Ollama actually evaluates:

| fragment | PyTorch greedy | Ollama | identical |
|---|---|---|---|
| last 34 tokens | `' What were the ambient temperature and relative humidity is 1.1 ug/m3.\n'` | same | **yes** (re-tokenised IDs equal) |
| first 34 tokens | `'00, CO(GT)=1.8 ug/m3, NOx(GT)=112 ug/m3 (sensor PT08.S3=1111), Non-M'` | same | **yes** |

With the full 86 tokens (PyTorch, RoPE widened — diagnostic only) the model returns
`' 11.1.0.0.0.0.0.0.0.0.0.0.0.08.04, NOx(GT)=1'`; first-token top-5 at `Response:` is
`' '` 0.097, `' What'` 0.096, `':'` 0.072, `' The'` 0.071, `' Recorded'` 0.041 — no confident answer.
Teacher-forcing the true completion gives mean log-prob **−1.26/token**
(`' absolute'` 0.32, `' humidity'` 0.61, `' is'` 0.28, `'1'` 0.31, `'.'` 0.24, `'0'` 0.088, `'3'` 0.043):
the weights know the shape of the answer but not well enough to produce it greedily.

### Expected answer and scientific correctness

| value | g/m3 |
|---|---|
| dataset `AH` column | 1.0311 |
| written into the training text | 1.03 (faithful) |
| Magnus formula from T=16.7 °C, RH=54.7 % | **7.7573** |

Across 804 validation rows the dataset column tracks the physics formula almost perfectly
(Pearson **r = 0.9998**) but on a different scale: `AH_column / AH_Magnus` median **0.1310**
(p10 0.1296, p90 0.1323), i.e. the UCI `AH` column is ~7.6× smaller than a standard
g/m³ calculation. Score the model against the dataset column; do not recompute AH from T and RH.

### Verdict

**Case 2 — both PyTorch and Ollama produce equally poor output.** The 86-token prompt is
unusable at a 64-token context, and the 101-token sample never fit the 64-token training window,
so the instruction→answer mapping for this template was effectively never trained (~2.6 exposures).
Neither the GGUF conversion nor the tokenizer is implicated: on identical inputs both engines emit
identical token IDs. **Do not modify the conversion.**

## 8. Environment note (unchanged from the earlier session)

`ollama create` needs `llama-quantize.exe` and inference needs `llama-server.exe`; Ollama 0.33.2
ships neither. llama.cpp `b12503` Windows CPU binaries are installed in
`%LOCALAPPDATA%\Programs\Ollama\lib\ollama\`. A copy is kept in `.llama_cpp/` and used by
`src/verify_tokenizer.py`.
