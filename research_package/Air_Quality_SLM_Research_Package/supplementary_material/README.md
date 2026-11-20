# Supplementary Material

Evidence that supports the narrative but is neither a primary result nor a source file.

| Directory | Contents |
|---|---|
| `notebooks/` | `Air_Quality_SLM_Training.ipynb` (v1 interactive training), `Air_Quality_SLM_Continue_Training_Colab.ipynb` (v2 Colab variant), `Data-Convertion.ipynb` (dataset exploration / conversion experiments), `0_gpu_check.ipynb` (environment probe) |
| `reports/` | Runtime probe JSONs and secondary reports, listed below |
| `scripts/` | Supporting scripts used during development but not on the main path |

## reports/

| File | What it records |
|---|---|
| `OLLAMA_INTEGRATION.md` | The garbled-output diagnosis: flat next-token distribution, why greedy decoding was required |
| `VALIDATION_REPORT.md` | Dataset validation pass |
| `ollama_before_after.json` | Model behaviour before and after the decoding-parameter fix |
| `chat_results_after.json` | Chat outputs after the fix |
| `compare_pytorch_ollama.json` | PyTorch vs Ollama token-for-token parity check |
| `comparison_finetune.json` | v1 vs v2 side-by-side generations |
| `diagnose_absolute_humidity.json` | Diagnostic of AH outputs and the closed-form comparison |
| `probe_capability.json` | Capability probe prompts and responses |
| `prompt_screening.json` | Prompt-screening pass used to pick representative examples |
| `tokenizer_verification.json` | 13-sample tokenizer parity between HF and GGUF |

## scripts/

`train_slm_backup.py`, `promote_checkpoint.py`, `plot_loss.py`, `probe_capability.py`,
`upload_to_huggingface.py`, `compare_pytorch_ollama.py`, `test_ollama_chat.py`,
`test_ollama_improved.py`, `package_dataset.py`.

## Not included

Model weights, GGUF files, token `.bin` streams, `best_model.pt` and `latest_model.pt` are omitted for
size. Their paths, sizes and SHA-256 values are in `../FILE_INVENTORY.md` and
`../conversion/conversion_documentation.md` §6.
