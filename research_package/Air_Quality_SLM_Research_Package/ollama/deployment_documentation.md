# Ollama Deployment

Three tags were registered from the files in this repository. They are listed in registration order,
which is also their version order. No tag was overwritten or deleted.

| Tag | Registered (file mtime) | GGUF | `num_ctx` | `num_predict` | Template | Model digest |
|---|---|---|---:|---:|---|---|
| `air-quality-slm:latest` | 2026-11-18 20:25 | `models/air_quality_slm/air_quality_slm.gguf` | 64 | 64 | `{{ .Prompt }}` | `21d7b363c2d4` |
| `air-quality-slm-improved:latest` | 2026-11-19 09:28 | `models/air_quality_slm_finetuned/air_quality_slm_finetuned.gguf` | 128 | 48 | `{{ .Prompt }}` | `f93c91b218b3` |
| `air-quality-slm-v3:latest` | 2026-11-19 18:56 | *same file as above* | 128 | 48 | `{{ .System }}\n{{ .Prompt }}` | `1ab6bee975c1` |

Verified with `OLLAMA_HOST=http://127.0.0.1:11435 ollama list` and `ollama --version` → `0.33.2`.

---

## 1. Why Ollama

| Requirement | How Ollama meets it |
|---|---|
| Run a custom GGUF without writing an inference server | `ollama create` + `ollama run` |
| Deterministic decoding controls | `temperature`, `top_k`, `top_p`, `repeat_penalty`, `num_predict`, `stop` are Modelfile parameters |
| Fixed context window | `num_ctx` is a parameter, and the GGUF carries the trained length |
| A desktop UI for demonstration | Ollama Desktop lists every registered tag |
| No GPU | CPU inference; the 20.4M-parameter model returns in well under a second |

## 2. What a Modelfile does

A Modelfile is a small build script for an Ollama model. It names the weights (`FROM`), fixes generation
defaults (`PARAMETER`), and defines how a prompt is assembled (`TEMPLATE`, `SYSTEM`). Registering a
Modelfile does not copy or alter the GGUF; Ollama records the file, the parameter set and the template,
and the resulting tag's digest changes when any of those change. That is exactly why v2 and v3 have
different digests while sharing byte-identical weights.

## 3. The three Modelfiles

### 3.1 `Modelfile` (v1)

```
FROM ./models/air_quality_slm/air_quality_slm.gguf
PARAMETER num_ctx 64
PARAMETER temperature 0
PARAMETER top_k 1
PARAMETER top_p 1.0
PARAMETER min_p 0.0
PARAMETER repeat_penalty 1.0
PARAMETER repeat_last_n 64
PARAMETER num_predict 64
PARAMETER stop "<|endoftext|>"
TEMPLATE """{{ .Prompt }}"""
```

The file's own comments record the measurement that forced greedy decoding: the trained next-token
distribution is flat (top-1 probability ≈ 0.18 even on in-distribution prompts), so sampling at
temperature 0.7 / top-k 40 selected low-probability fragments and produced garbled, multilingual text.
Greedy decoding reproduces the PyTorch checkpoint's exact output.

### 3.2 `Modelfile.improved` (v2)

Same parameters except `num_ctx 128` and `num_predict 48`. The context change is the substantive one: the
absolute-humidity prompt is 86 tokens, so under `num_ctx 64` Ollama silently truncated it to its last 34
tokens (`prompt_eval_count` measured as 34 for any prompt ≥ 64 tokens). The model never saw the
instruction. `num_predict 48` replaced 40 after measurement showed that a 40-token budget cut the
temperature/RH answer before the RH value was emitted (0/936 answers contained an RH number at 40 tokens;
936/936 at 50).

### 3.3 `Modelfile.v3` (v3)

Identical parameters to v2, but the template becomes

```
TEMPLATE "{{ .System }}\n{{ .Prompt }}"
```

so the application can inject a system instruction. `Modelfile.v3`'s comments document the system text
`app.py` is expected to send:

> "You are the Air Quality SLM Assistant. Provide concise natural-language explanations. Never invent
> numerical values - use Python-authoritative lookups. Return polite out-of-domain refusals for unsupported
> questions. Use deterministic decoding (temp=0, top_k=1)."

**Important:** the v3 weights are byte-identical to v2 (`FROM` the same GGUF). v3 was never evaluated
(`MISSING_EVIDENCE.md` §8). The system-prompt path changes what the model is *told*, not what it learned.

## 4. Registration commands

```bash
cd <project root>
ollama create air-quality-slm          -f Modelfile
ollama create air-quality-slm-improved -f Modelfile.improved
ollama create air-quality-slm-v3       -f Modelfile.v3
```

`FROM` is a relative path, so the command must be run from the project root. Registration takes a few
seconds (no conversion happens; Ollama imports the GGUF and writes its manifest).

## 5. This machine's two-port topology

Ollama Desktop talks to port **11434**. In this project that port is occupied by `ollama_proxy.py`, which
forwards to the real Ollama server on **11435**. Consequences that cost time if you do not know them:

```bash
ollama list                      # fails / hangs: 11434 is the proxy
OLLAMA_HOST=http://127.0.0.1:11435 ollama list   # works: real server
```

The proxy exists so that the application, the desktop UI and the CLI all reach the same model while the
proxy can log and supervise traffic. `start_ollama_proxy.bat`, `stop_ollama_proxy.bat`,
`check_ollama_proxy.bat`, `proxy_watchdog.py` and `scripts/integration_ctl.py` manage it; the operating
procedure is `docs/INTEGRATION_OPERATIONS.md`.

## 6. Verified model behaviour

Stored in `docs/ollama_before_after.json`, `docs/chat_results_after.json`, `docs/compare_pytorch_ollama.json`
and `docs/ollama_before_after.json`. Two facts matter:

1. **Greedy output matches PyTorch token-for-token** after the parameters above were fixed.
2. **The model is a text formatter, not a retrieval engine.** Given a pollutant prompt it emits a single
   constant sentence:
   `Recorded CO is 2.2 mg/m3, NOx is 111 ppb, and NO2 is 111 ug/m3 with benzene level at 11.2 ug/m3.`
   For every row. The true values for the example row are CO 1.2, NOx 179, NO2 82, C₆H₆ 4.6.

The polite refusals, the historical lookups and the humidity arithmetic that a user sees are produced by
`app.py` + `domain_guard.py` + `data_retrieval.py` + `humidity_calculator.py`. A bare
`ollama run air-quality-slm-v3:latest "When is Diwali in 2004?"` does **not** produce a refusal; it
continues text. See `MISSING_EVIDENCE.md` §7.

## 7. Installing on another computer

1. Install Ollama 0.30 or newer from https://ollama.com/download and confirm `ollama --version`.
2. Copy the project (the two `models/` directories with their `.gguf` files are required; ≈ 500 MB).
3. `cd` to the project root.
4. `ollama create air-quality-slm-v3 -f Modelfile.v3`
5. `ollama run air-quality-slm-v3:latest "Calculate absolute humidity for T = 16.7 C and RH = 54.7%"`
6. In Ollama Desktop the tag appears in the model selector as **air-quality-slm-v3:latest**.

No model file is publicly hosted by this project; the GGUF ships inside the repository. Do not assume a
`ollama pull air-quality-slm-v3` will work — it will not, because the tag is local-only.

## 8. Trouble-shooting

| Symptom | Cause | Fix |
|---|---|---|
| `ollama list` hangs or errors | 11434 is the proxy | set `OLLAMA_HOST=http://127.0.0.1:11435` |
| Output is garbled, multilingual, repetitive | sampling instead of greedy decoding | `temperature 0`, `top_k 1`, `repeat_penalty 1.0` |
| Answer stops before the humidity value | `num_predict` too small | 48 or more (the answer is 47 tokens) |
| Prompt appears to be ignored | `num_ctx 64` truncating an 86-token prompt | `num_ctx 128` with the v2 GGUF |
| `Error: model not found` | tag never registered, or `FROM` path wrong | register from the project root |
| `Error: failed to load model` | GGUF copied incompletely | re-check its SHA-256 against `FILE_INVENTORY.md` |
