# Project History

A chronological account of how the Air Quality SLM was built. Every date and step below is taken from
one of three kinds of evidence: file modification timestamps in the workspace, git commits, or values
written into an artefact by the script that produced it. Where no evidence dates a step, the entry says
"undated" rather than inventing a date.

**There is no "Day 0 + n" numbering here, because the project was not built on a fixed schedule.** The
workspace records work on four separate days (15 October, 12 November, 15 November, 18–20 November
2026). Any paper timeline should use these dates.

---

## Phase 0 — Repository and notebook groundwork

| Date (evidence) | Event | Evidence |
|---|---|---|
| 2026-10-15 11:42 | Repository directory created; `venv/` created | directory mtime |
| 2026-10-15 12:13 | `.ipynb_checkpoints/` appears — notebook work begins | directory mtime |
| 2026-10-15 12:16 | `notebooks/0_gpu_check.ipynb` — an environment/GPU probe notebook | file mtime |
| 2026-09-18 (folder) | `markitdown/` vendored into the workspace | directory mtime |

The GPU-check notebook is the first artefact. It is consistent with the final outcome: the machine has no
CUDA device, so every run in this project is CPU-only.

## Phase 1 — Dataset acquisition, exploration and conversion (12 November 2026)

| Date | Event | Evidence |
|---|---|---|
| 2026-08-05 22:31 | `data/raw/air+quality/AirQualityUCI.csv` — last write of the raw file | file mtime |
| 2026-11-12 13:44 | `anaconda_projects/` — early experimentation | directory mtime |
| 2026-11-12 13:49 | `notebooks/Data-Convertion.ipynb` — dataset exploration and conversion experiments | file mtime |
| 2026-11-12 14:09 | `notebooks/Untitled.ipynb`, `research/Feature3.3.md` | file mtime |
| 2026-11-12 14:18 | `data/processed/{train,validation,test}.csv` written — 6,549 / 936 / 1,872 rows | file mtime |
| 2026-11-12 14:20 | `docs/VALIDATION_REPORT.md` | file mtime |
| 2026-11-12 15:03 | `data/` reorganised; `src/preprocess_air_quality.py` in place | directory mtime |
| 2026-11-12 15:08 | `research/` folder with air-quality reference PDFs | directory mtime |
| 2026-11-12 15:23 | `docs/README.md` — first substantial project documentation | file mtime |

The raw CSV predates every other artefact (5 August), but **no record exists of when or from where it
was downloaded**. That gap is carried into `UNAVAILABLE_INFORMATION.md`.

## Phase 2 — Instruction data and the v1 model (15 November 2026)

| Time | Event | Evidence |
|---|---|---|
| 22:31 | `training/data_bin/{train,val,test}.bin` + `dataset_metadata.json` — 3,041,432 / 431,740 / 870,987 tokens, 26,196 training texts | file mtimes, metadata |
| 22:32 | `scripts/train_slm.ps1` | file mtime |
| 22:33 | `train.bat`, `training/` tree | file mtimes |
| 22:43 | `training/checkpoints/{best_model.pt,latest_model.pt}` written by the v1 run | file mtimes |
| 23:19 | `training_config.json` | file mtime |
| 23:21 | `notebooks/Air_Quality_SLM_Training.ipynb` | file mtime |
| 23:27 | `README.md`, `training_summary.md`; `models/air_quality_slm/` exported | file mtimes |

The v1 training summary records **150 steps, 279.01 s, final train loss 1.3390, final val loss 1.3651,
test loss 1.4295, test perplexity 4.18**. Those are the values the run wrote itself.

A second checkpoint directory, `models/air_quality_slm_original/`, holds `BACKUP_INFO.json` and a copy of
the initial weights — the pre-continued-training state, kept for rollback.

## Phase 3 — First deployment, and the discovery that it was broken (18 November 2026)

| Time | Event | Evidence |
|---|---|---|
| 18:45 | `backup_pre_conversion/` created — a snapshot taken before the first GGUF conversion | directory mtime |
| 20:22 | `docs/compare_pytorch_ollama.json` — PyTorch vs Ollama output comparison | file mtime |
| 20:24 | `docs/probe_capability.json` — capability probe | file mtime |
| 20:25 | `Modelfile` (v1) written; `docs/chat_results_after.json` | file mtimes |
| 20:27 | `docs/tokenizer_verification.json` — 13-sample tokenizer parity check | file mtime |
| 20:28 | `docs/prompt_screening.json` | file mtime |
| 22:08 | `docs/OLLAMA_INTEGRATION.md` — the garbled-output diagnosis | file mtime |
| 22:22 | `models/` GGUF artefacts settled | directory mtime |
| 22:33 | `notebooks/Air_Quality_SLM_Continue_Training_Colab.ipynb` — the v2 plan | file mtime |
| 22:58 | `docs/loss_curve.csv`, `loss_curve.svg` — v2 training curve | file mtimes |
| 22:59 | `docs/comparison_finetune.json` | file mtime |
| 23:03 | `domain_guard.py` | file mtime |
| 23:07 | `docs/ollama_before_after.json` | file mtime |
| 23:10 | `docs/EVALUATION_V2_REPORT.md` begins | file mtime |
| 23:18 | `start_out.log` | file mtime |

**The central discovery of the project happened here.** Ollama's output was garbled and multilingual.
Two independent causes were established by measurement:

1. The trained next-token distribution is flat (top-1 probability ≈ 0.18 on in-distribution prompts), so
   sampling at temperature 0.7 / top-k 40 selected low-probability fragments. Greedy decoding reproduces
   the PyTorch checkpoint exactly.
2. The v1 GGUF declared `context_length 64`, but the shortest instruction template is 83 tokens. Ollama
   silently truncated the prompt to its last 34 tokens, so the model never received the instruction.

Cause 2 is a training-design defect, not a deployment defect: with `block_size = 64`, no complete
instruction/answer pair ever fitted inside one training window.

## Phase 4 — v2 continued training and full evaluation (18–19 November 2026)

| Time | Event | Evidence |
|---|---|---|
| 18 Nov 22:43–22:58 | v2 run: 1,200 steps completed, block 128, 614,400 tokens (0.202 epoch) | `loss_curve.csv`, `docs/CONTINUED_TRAINING_REPORT.md` |
| 18 Nov 23:38 | `docs/eval_v2/baseline.json` — constant-predictor baselines | file mtime |
| 19 Nov 00:16–00:53 | `metrics_v2_val.json` — first full generation dump | file mtimes |
| 19 Nov 01:01 | `loss_v1_128.json` | file mtime |
| 19 Nov 01:08 | `numeric_diagnostics.json` — closed-form AH comparison | file mtime |
| 19 Nov 01:14 | `metrics_v1_val.json` | file mtime |
| 19 Nov 09:28 | `Modelfile.improved` (v2): `num_ctx 128`, `num_predict 48` | file mtime |
| 19 Nov 09:29–09:30 | `loss_v2_128.json`, `loss_v1_64.json` | file mtimes |
| 19 Nov 09:42 | `metrics_v2_test.json` | file mtime |
| 19 Nov 10:05–10:06 | `metrics_v1_test.json`, `summary.json`, `summary_tables.txt` | file mtimes |

Two more deployment defects were found and fixed by measurement in this phase:

* `num_predict 40` cut the temperature/RH answer before the humidity value (0/936 answers contained an RH
  number). The answer is 47 tokens; the parameter became 48.
* The v1 GGUF's 64-token context was replaced by a 128-token export, and `prompt_eval_count` then equalled
  the true 86 tokens.

## Phase 5 — v3, the application layer, and service hardening (19–20 November 2026)

| Time | Event | Evidence |
|---|---|---|
| 19 Nov 18:01 | `data_retrieval.py`, `humidity_calculator.py`, `ollama_client.py` | file mtimes |
| 19 Nov 18:56 | `Modelfile.v3` — system-prompt template | file mtime |
| 19 Nov 20:22 | `app.py` — the interactive assistant | file mtime |
| 19 Nov 23:03 | `domain_guard.py` revised | file mtime |
| 19 Nov 23:18 | `start_out.log` | file mtime |
| 20 Nov 00:21 | `backend.err.log`, `proxy.err.log`, `watchdog.err.log` — service logs | file mtimes |
| 20 Nov 00:51 | `start_ollama_proxy.bat`, `stop_ollama_proxy.bat`, `check_ollama_proxy.bat` | file mtimes |
| 20 Nov 01:03 | `scripts/integration_ctl.py` | file mtime |
| 20 Nov 01:09 | `ollama_proxy.py` | file mtime |
| 20 Nov 01:15 | `proxy_watchdog.py` | file mtime |
| 20 Nov 01:22 | `docs/INTEGRATION_OPERATIONS.md`, `recovery.log` | file mtimes |
| 20 Nov 07:06–07:10 | `watchdog.log`, `backend.log`, `proxy_access.log` — live traffic | file mtimes |

`ollama create air-quality-slm-v3 -f Modelfile.v3` registered the final tag. **Its `FROM` line points at
the same GGUF as v2.** v3 is therefore an interface variant — a template that accepts `{{ .System }}` —
not a new model, and it was never evaluated.

## Phase 6 — This documentation package (20 November 2026)

The audit, the 15 figures, the research evidence package and this documentation package were produced
without retraining, without touching any checkpoint, and without modifying the dataset or the model tags.

---

## Git history

```
e967ef9  Harden the Ollama proxy integration with supervised startup and recovery
30edb50  Add SLM training scripts, notebooks, and config
b72c2ce  Merge branch 'main' of https://github.com/sanjanareddy2607-a11y/Air-Quality-SLM
244b306  Initial commit: Air Quality SLM project
688e9f2  Update README.md
ed2f93b  Initial commit
```

The commit messages are coarser than the file timestamps, so the timeline above is built primarily from
the artefacts themselves. Note that the first commit predates the workspace's own `README.md` by several
days, and that `e967ef9` (proxy hardening) is the most recent commit — the documentation and evaluation
work of 19–20 November is **not committed**.

## What the history does not contain

* No dated record of the dataset download.
* No record of any experiment between 12 and 15 November (the model work starts at 22:31 on the 15th).
* No v3 evaluation run.
* No screenshot capture at any point.
* No Hugging Face upload log, although `src/upload_to_huggingface.py` exists.
