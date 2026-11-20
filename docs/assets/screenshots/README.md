# Screenshots

**There is no screenshot in this package.** A recursive search of the project for `*.png`, `*.jpg`,
`*.jpeg` and `*.webp` outside `venv/` returns only two image fixtures inside the vendored `markitdown`
package (`markitdown/packages/markitdown/tests/test_files/test.jpg` and `test_llm.jpg`). No capture of the
Ollama Desktop session, the terminal assistant, or any earlier demonstration exists anywhere in the
workspace.

Replacement screenshots were **not** generated, because inventing one would misrepresent the project.

## What to capture

Run the proxy first (`start_ollama_proxy.bat`, or `python ollama_proxy.py`), open Ollama Desktop, select
**air-quality-slm-v3:latest**, and capture two screenshots.

### 1. `01_greeting.png` — greeting and capability introduction

Prompt:

```
Hey, what can you do?
```

Expected reply (exact text is `ollama_proxy.WELCOME_MESSAGE`, verified in `proxy_access.log` at
2026-11-20 01:20:08): the "Hello! Welcome to the Air Quality SLM Assistant." block listing historical
retrieval and absolute-humidity calculation with two example prompts.

### 2. `02_refusal_and_humidity.png` — out-of-domain refusal plus humidity calculation

Two turns in the same window:

```
When is Diwali in 2004?
Calculate absolute humidity for T = 16.7 C and RH = 54.7%.
```

Expected replies:

```
Sorry, this question is outside my current capabilities. I am an experimental Air Quality SLM
designed for historical air-quality measurements and supported humidity calculations.
Please ask me about pollutant concentrations, sensor readings, temperature or humidity instead.
```

```
Absolute Humidity Calculation
  Temperature:       16.7 C
  Relative Humidity: 54.7 %
  Absolute Humidity: 1.0284 g/m3
  Method: Dataset-calibrated absolute humidity
  Note: This is a dataset-calibrated estimate using the project's Tetens+scale formula,
        fitted on the training split only.
```

## What each screenshot would demonstrate — and what it would not

| Demonstrates | Does not demonstrate |
|---|---|
| Intent routing (greeting / refusal / calculation) | the neural network's quality |
| That the deployed tag answers coherently on these three intents | that the model can retrieve or compute anything |
| That the proxy path is live | that the model is correct on unseen data |

This is the single most important caveat in the whole documentation: **Ollama Desktop in this deployment
is answered by `ollama_proxy.py` in Python, not by the model.** The model's own behaviour is documented in
`docs/EVALUATION_REPORT.md` §7 (one constant sentence per template). Any caption that attributes these
answers to the neural network would be false.

If you want a screenshot of the *model itself*, bypass the proxy:

```bash
OLLAMA_HOST=http://127.0.0.1:11435 ollama run air-quality-slm-v3:latest "Calculate absolute humidity for T = 16.7 C and RH = 54.7%"
```

and expect a repeated template sentence rather than the block above. Caption it accordingly.

## Naming convention

Drop files here as `01_greeting.png` and `02_refusal_and_humidity.png`, then reference them from the
README and `docs/USER_MANUAL.md`. Both spots already have a placeholder comment marking where they belong.
