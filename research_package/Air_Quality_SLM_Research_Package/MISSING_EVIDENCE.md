# Missing Evidence and Unresolved Discrepancies

This file exists so that nothing in the package has to be guessed at. Each entry states **what is
absent or contradictory**, **how that was established**, **what it affects in the paper**, and **what
would close it**. Nothing here was resolved by assumption.

---

## 1. No screenshots exist anywhere in the project

**What is absent.** The task requires screenshots of the Ollama Desktop session (greeting and
capability introduction; domain refusal plus absolute-humidity calculation). A recursive search of the
project for `*.png`, `*.jpg`, `*.jpeg`, `*.webp` outside `venv/` returns only two files, both fixtures
inside the vendored `markitdown` package (`markitdown/packages/markitdown/tests/test_files/test.jpg`
and `test_llm.jpg`). **No project screenshot, in any format, exists in the workspace.**

**Effect.** The paper cannot show a captured deployment screen. Two honest options remain:

1. Capture them from the running model (commands in §2) and label them "captured for this report,
   <date>".
2. Omit them and describe the observed behaviour in text, citing
   `ollama/deployment_documentation.md` §6.

**To close.** With the proxy running, execute the four commands in §2 and screenshot the terminal or
the Ollama Desktop chat pane. The expected outputs are quoted there from stored logs, so the capture
is a reproduction, not an invention.

## 2. Commands that would produce the missing screenshots

```bash
OLLAMA_HOST=http://127.0.0.1:11435 ollama run air-quality-slm-v3:latest "Hey, what can you do?"
OLLAMA_HOST=http://127.0.0.1:11435 ollama run air-quality-slm-v3:latest "What was the NOx concentration on 08/12/2004 at 15:00?"
OLLAMA_HOST=http://127.0.0.1:11435 ollama run air-quality-slm-v3:latest "Calculate absolute humidity for T = 16.7 C and RH = 54.7%"
OLLAMA_HOST=http://127.0.0.1:11435 ollama run air-quality-slm-v3:latest "When is Diwali in 2004?"
```

Important: `ollama list` fails when `OLLAMA_HOST` is left at its default, because port 11434 is occupied
by `ollama_proxy.py`. Setting `OLLAMA_HOST=http://127.0.0.1:11435` addresses the real server directly.

**The model itself does not perform the retrieval, the humidity arithmetic or the refusal.** Those come
from `domain_guard.py`, `data_retrieval.py` and `humidity_calculator.py` inside `app.py`. A native
`ollama run` therefore shows only raw text continuation, which for out-of-domain prompts is *not* a
polite refusal. Any screenshot claiming otherwise would misrepresent the architecture — see §7.

## 3. `baseline.json` reports a validation/train date overlap of 1; the evaluation report says 0

**Evidence.** `docs/eval_v2/baseline.json` → `"val": { ..., "date_overlap_with_train": 1 }` and
`"test": { ..., "date_overlap_with_train": 0 }`.

**Cause.** The split is contiguous in time and the boundary date `08/12/2004` appears in both the train
and validation splits (train ends `08/12/2004 14:00`, validation begins `08/12/2004 15:00`). The
`date_overlap` counter compares *calendar dates*, not timestamps, so the boundary date counts once.

**Effect.** It is **not** record leakage: no `(Date, Time)` pair is in two splits, and the memorisation
table in `EXPERIMENTAL_RESULTS.md` §7 shows 0 timestamp collisions for both models. If the paper
mentions date overlap it must say "one shared boundary date, no shared timestamps", not "0".

**To close.** Recompute overlap on `(Date, Time)` pairs rather than `Date` alone; expected result 0/0.

## 4. `humidity_calculator.py` contradicts its own documentation on two points

### 4.1 Tetens denominator

* Module docstring and `magnus_saturation_vp` docstring: `e_s = 6.112 * exp(17.67 * T / (T + 243.5))`.
* Executed code: `return 6.112 * math.exp(17.67 * t_c / (t_c + 237.3))`.

The two constants are both in common use (243.5 over water in the Magnus form, 237.3 in the Tetens
form). The **code** value governs every number the application produced. The paper must quote 237.3 and
may note the docstring inconsistency.

### 4.2 The scale factor is measured from one row, not fitted

The docstring says "A scale factor was fitted on the training split: `scale ≈ 0.976` … This scale is
FIXED; it is NOT refit per run or per split."

The code (`get_scale_factor`) instead reads `data/processed/train.csv` at import time and computes

```
scale = AH[first valid train row] / AH_phys(T[first valid train row], RH[first valid train row]) * 1
```

i.e. a **single-row ratio**, cached in a module global. The measured value on this machine is
`scale = 0.12874776`, computed from the first valid training row
(`10/03/2004 18.00.00`, `T = 13.6 °C`, `RH = 48.9 %`, `AH = 0.7578`, and
`AH_phys(13.6, 48.9) = 5.8855 g/m³`), not from a regression over the split. The resulting application
output for the documentation's own worked example (`T = 16.7 °C`, `RH = 54.7 %`) is
**1.0284 g/m³**, whereas

* the pure Tetens value is ≈ 7.99 g/m³,
* the documented `0.976 ×` Tetens value is ≈ 7.80 g/m³,
* the dataset row itself is 0.7578 g/m³.

**Effect.** Three separate numbers are in play and none of them agree. The application's answer is
*the* answer a user sees, so it is the one to report if the paper describes the deployed behaviour;
the docstring's 0.976 is **not** what the code does. The `numeric_diagnostics.json` entry
`scale ≈ 0.1351678` is a fit over all training rows — also different from both.

**To close.** Decide which definition is intended, implement one, and recompute the worked example.
Until then, `MISSING_EVIDENCE` stands and the paper should present 1.0284 as "value returned by the
shipped code" with this caveat.

## 5. `app.py` labels CO and NOx with the wrong unit

`app.py::handle_retrieval` and `_format_sensor_values` print every pollutant as `ug/m3`, including
`CO(GT)` and `NOx(GT)`. In the UCI dataset:

| Column | True unit |
|---|---|
| `CO(GT)` | mg/m³ |
| `NOx(GT)` | ppb (parts per billion) |
| `NO2(GT)` | µg/m³ |
| `C6H6(GT)` | µg/m³ |
| `NMHC(GT)` | µg/m³ |

**Effect.** Any transcript of the application reporting "CO: 2.2 ug/m3" or "NOx: 179.0 ug/m3" is
unit-mislabelled. The *value* is the stored dataset value; only the unit string is wrong. The paper must
not reproduce the mislabel as if it were correct, and must not "fix" it silently either — report the
defect. Interestingly, v2's own generated text uses `mg/m3` and `ppb` correctly (see
`EXPERIMENTAL_RESULTS.md` §6), so the model and the application disagree.

**To close.** Change the unit map in `app.py`; the model weights stay untouched.

## 6. No dependency manifest, no dataset download script, no download timestamp

**Absent.** There is no `requirements.txt`, `pyproject.toml`, `environment.yml` or lockfile. The raw CSV
is present at `data/raw/air+quality/AirQualityUCI.csv` (checksummed in `FILE_INVENTORY.md`), but no
script and no recorded timestamp documents when or from which mirror it was fetched. The model card
names the Hugging Face dataset `sanju7/air-quality-uci`.

**Effect.** `ENVIRONMENT.md` §2.2 lists versions read from the live venv; that is the only dependency
record. Reproduction instructions must be written as "install these versions", not "run
`pip install -r`".

**To close.** Add a `requirements.txt` pinned from `pip freeze`, and record the CSV's source URL plus
its SHA-256 (`13277ae5…`, already computed).

## 7. The model does not implement retrieval, arithmetic or refusal

**Established by inspection**, not assumption:

* `domain_guard.py` is a set of regular expressions and keyword lists; it routes before any model call.
* `data_retrieval.py` does an exact `(Date, Time)` match in the split CSVs with pandas.
* `humidity_calculator.py` is closed-form arithmetic.
* `ollama_client.py` sends prompts to Ollama with `num_ctx 128`, `num_predict 48`, greedy decoding, and
  explicitly comments that "the model's numerical outputs are NEVER used as authoritative".

**Effect.** Every numeric answer the application gives is Python output. The neural network contributes
natural-language continuation only. Any paper sentence of the form "the SLM retrieves/predicts/refuses"
must be rewritten. This is the single most important honesty constraint in the package.

## 8. v3 has no evaluation of any kind

**Absent.** No metrics file, no generation dump, no report, no loss figure for
`air-quality-slm-v3:latest`. `Modelfile.v3`'s `FROM` line points at
`models/air_quality_slm_finetuned/air_quality_slm_finetuned.gguf` — **the same file v2 uses**. The v3
Ollama digest (`1ab6bee9…`) differs from v2's (`f93c91b2…`) only because the Modelfile differs
(template includes `{{ .System }}`; parameters are otherwise identical).

**Effect.** v3 cannot be described as better, worse or improved in any measured sense. It is an
interface variant. The paper may compare v1 with v2 and must say that v3's system-prompt path was never
measured.

**To close.** Run `src/evaluate_v2_full.py` against v3 and store the dump; the harness already accepts a
model name.

## 9. Historical figures in the repository's own documents disagree with the stored artefacts

Two examples, both of which the package resolves in favour of the JSON:

| Claim | Where | Stored artefact |
|---|---|---|
| v1 test loss 1.4295, PPL 4.18 | `models/air_quality_slm/training_summary.json` | 1.4474 / 4.252 under the block-128 harness (`loss_v1_128.json`) |
| v2 best val loss 0.4896 at step 1,000 | `docs/CONTINUED_TRAINING_REPORT.md` | same 0.4896, but the final full-split val loss is 0.4877 |

The differences come from different evaluation protocols (block 64 vs block 128; per-run vs full-split),
not from error. The package always states which harness produced a number.

## 10. Items that are genuinely fine and should not be flagged as missing

Recorded here so a reader does not go looking:

* `training/checkpoints/best_model.pt` (81,608,245 B) and `latest_model.pt` (244,826,949 B) **are**
  present, as are `training/data_bin/{train,val,test}.bin` and `dataset_metadata.json`.
* `training/data_bin/dataset_metadata.json` stores absolute Windows paths from the build machine; the
  paths are not portable but the files are present in the same directory.
* The 15 figures in `figures/` are regenerated from stored data by two committed scripts; none of their
  plotted values is hand-entered.
