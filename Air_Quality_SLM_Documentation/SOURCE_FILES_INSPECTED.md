# Source Files Inspected

Every file below was read, listed, or executed during the audit that produced this documentation. Paths are
relative to the project root. Nothing in the documentation rests on an unread file or on an earlier
conversation.

---

## 1. Application layer

| File | What was taken from it |
|---|---|
| `app.py` | routing order, welcome text, refusal text, date/time and T/RH parsing, unit strings (and their CO/NOx defect) |
| `domain_guard.py` | greeting regexes, out-of-domain keywords and logic, intent detection, reset phrases |
| `data_retrieval.py` | exact-match lookup, date/time normalisation, unit strings returned |
| `humidity_calculator.py` | Tetens coefficients (code vs docstring), `get_scale_factor()` behaviour, worked example |
| `ollama_client.py` | `MODEL_NAME`, default generation parameters, the "numbers are never authoritative" comment |
| `ollama_proxy.py` | two-port topology, `handle_message` routing, canned messages, correct unit map, access logging |
| `proxy_watchdog.py` | supervisor behaviour (restart policy, logs) |
| `start_ollama_proxy.bat`, `stop_ollama_proxy.bat`, `check_ollama_proxy.bat` | startup procedure |
| `scripts/integration_ctl.py` | operational commands |

## 2. Pipeline source

| File | What was taken from it |
|---|---|
| `src/preprocess_air_quality.py` | sentinel handling, ordering check, split ratios |
| `src/prepare_training_data.py` | the four templates verbatim, tokeniser fallbacks, `Unavailable` handling |
| `src/train_slm.py` | model class, parameter count, embedding scaling, training loop |
| `src/continue_training.py` | v2 hyper-parameters, sampler, early stopping, seeds |
| `src/convert_to_gguf.py` | tensor mapping, metadata keys, template replacement |
| `src/export_gguf_from_checkpoint.py` | parameterised export, context length |
| `src/verify_tokenizer.py` | 13-sample parity procedure |
| `src/evaluate_v2_full.py` | harness design, windows, extraction, structure metrics |
| `src/summarize_eval_v2.py` | aggregation into `summary.json` / `summary_tables.txt` |
| `src/diagnose_v2_numbers.py` | closed-form comparison, scale fit |
| `src/evaluate_slm.py`, `src/plot_loss.py`, `src/probe_capability.py`, `src/prompt_screening.py*` | earlier/auxiliary analysis scripts |
| `src/validate_dataset.py` | dataset checks reported in `docs/VALIDATION_REPORT.md` |

\* present as `src/probe_capability.py`; prompt screening exists as a result file under `docs/`.

## 3. Configuration and model artefacts

| File | What was taken from it |
|---|---|
| `training_config.json` | all CPU/GPU hyper-parameters, paths, inference defaults |
| `models/air_quality_slm/config.json` | v1 architecture, 64-token context |
| `models/air_quality_slm_finetuned/config.json` | v2 architecture, 128-token context |
| `models/air_quality_slm/training_summary.json` | v1 loss trace, runtime, final losses, parameter count |
| `models/air_quality_slm/README.md` | model card: licence, dataset mirror name, intended use |
| `models/air_quality_slm_finetuned/chat_template.jinja` | the Qwen ChatML template (not used at inference) |
| `models/air_quality_slm_original/BACKUP_INFO.json` | provenance of the pre-continued-training backup |
| `Modelfile`, `Modelfile.improved`, `Modelfile.v3` | every parameter and template, plus the rationale comments |
| `training/data_bin/dataset_metadata.json` | token and sample counts per split |

## 4. Results and logs

| File | What was taken from it |
|---|---|
| `docs/eval_v2/loss_v1_128.json`, `loss_v2_128.json`, `loss_v1_64.json` | loss/perplexity tables |
| `docs/eval_v2/baseline.json` | constant-predictor MAE, date-overlap field |
| `docs/eval_v2/numeric_diagnostics.json` | closed-form comparison, scale fit |
| `docs/eval_v2/summary.json`, `summary_tables.txt` | all structure, diversity and memorisation tables |
| `docs/eval_v2/metrics_v{1,2}_{val,test}.json` | per-row schema; MAE recomputation verification |
| `docs/EVALUATION_V2_REPORT.md` | narrative interpretation of v1 vs v2 |
| `docs/CONTINUED_TRAINING_REPORT.md` | why v2 exists; the `Response:` exposure measurement |
| `docs/OLLAMA_INTEGRATION.md` | garbled-output diagnosis; token-length measurements |
| `docs/VALIDATION_REPORT.md` | dataset validation results |
| `docs/INTEGRATION_OPERATIONS.md` | proxy start/stop/check procedure |
| `docs/loss_curve.csv`, `docs/training_log_finetune.jsonl` | v2 training trace |
| `docs/tokenizer_verification.json` | 13-sample parity result |
| `docs/compare_pytorch_ollama.json`, `comparison_finetune.json`, `ollama_before_after.json`, `chat_results_after.json`, `probe_capability.json`, `prompt_screening.json`, `diagnose_absolute_humidity.json` | runtime behaviour evidence |
| `training_summary.md`, `docs/README.md` | earlier project documentation |
| `proxy_access.log`, `proxy.log`, `watchdog.log`, `backend.log`, `recovery.log` | live service evidence and the quoted interaction records |

## 5. Data

| File | What was taken from it |
|---|---|
| `data/raw/air+quality/AirQualityUCI.csv` | every dataset statistic; row count; sentinel count; ranges |
| `data/raw/air+quality/AirQualityUCI.xlsx` | presence noted only |
| `data/processed/train.csv`, `validation.csv`, `test.csv` | split sizes, boundaries, first-valid-row values |
| `data/training/*_instructions.jsonl` | template text confirmation (quoted rows) |

## 6. Notebooks

| File | What was taken from it |
|---|---|
| `notebooks/Data-Convertion.ipynb` | exploration phase, 12 Nov |
| `notebooks/Air_Quality_SLM_Training.ipynb` | v1 training, prompt example |
| `notebooks/Air_Quality_SLM_Continue_Training_Colab.ipynb` | v2 plan |
| `notebooks/0_gpu_check.ipynb` | no-GPU environment confirmation |

## 7. Repository metadata

| Item | What was taken from it |
|---|---|
| `git log --oneline` (6 commits) | the git timeline in `docs/PROJECT_HISTORY.md` |
| file and directory modification times | the dated chronology |
| `.gitignore` | what is excluded from version control |

## 8. Executed during verification

These were run, not merely read:

```bash
# environment
./venv/Scripts/python.exe -c "import sys, platform; print(sys.version); print(platform.platform())"
./venv/Scripts/python.exe -m pip list
./venv/Scripts/python.exe -c "import torch; print(torch.cuda.is_available())"
ollama --version
OLLAMA_HOST=http://127.0.0.1:11435 ollama list

# data and model behaviour
./venv/Scripts/python.exe -c "... first valid row of train.csv ..."
./venv/Scripts/python.exe -c "import humidity_calculator as h; h.get_scale_factor(); h.calculate_absolute_humidity(16.7,54.7)"
./venv/Scripts/python.exe -c "import data_retrieval as d; d.find_record('08/12/2004','15:00','val')"
./venv/Scripts/python.exe -c "from ollama_proxy import _format_record; ..."
./venv/Scripts/python.exe -c "... inspect schema of metrics_v2_val.json ..."
```

## 9. Deliberately not touched

`models/**/*.bin`, `models/**/*.gguf`, `models/**/*.pt`, `training/data_bin/*.bin`, `training/checkpoints/*.pt`,
`data/raw/**`, `data/processed/*.csv`, and the three registered Ollama tags. No file in these groups was
modified, moved or re-generated.
