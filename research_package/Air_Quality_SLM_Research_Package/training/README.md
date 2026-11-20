# Training

| Path | What it is |
|---|---|
| `training_config.json` | The full configuration: CPU and GPU branches, data paths, tokeniser, optimiser, output paths, inference defaults |
| `training_summary.md` | Human-readable summary of the v1 run |
| `scripts/train_slm.py` | v1: builds `Qwen3Model` from scratch and trains 150 steps at block 64 |
| `scripts/continue_training.py` | v2: resumes from the v1 weights, block 128, 1,200 steps, early stopping |
| `loss_curve.csv` | v2 evaluation trace: step, train EMA loss, val loss, val PPL, tokens seen, epochs, elapsed |
| `training_log_finetune.jsonl` | v2: one JSON object per evaluation |
| `EVALUATION_V2_REPORT.md` | The authoritative v1-vs-v2 evaluation report (also the source of the numbers in `../EXPERIMENTAL_RESULTS.md`) |
| `CONTINUED_TRAINING_REPORT.md` | Why v2 exists: the 64-token window never contained a complete instruction/answer pair |
| `checkpoints/dataset_metadata.json` | Token counts and sample counts per split, as written by `prepare_training_data.py` |

## What was actually run

| | v1 | v2 |
|---|---|---|
| Steps | 150 | 1,200 (target 2,400, early-stopped) |
| Block size | 64 | 128 |
| Batch × accumulation | 8 × 1 | 2 × 2 |
| Learning rate | 1e-3 | 2e-4 |
| Optimiser | AdamW (0.9, 0.95), wd 0.1 | same, fresh state |
| Runtime | 279.01 s | 1,864 s at step 1,200 |
| Tokens seen | 76,800 (2.5 % of corpus) | 614,400 (0.202 epoch) |

## Checkpoints

`training/checkpoints/best_model.pt` (81,608,245 B) and `latest_model.pt` (244,826,949 B) exist in the
project but are **not** included in this package. `best_model.pt` at step 1,000 (val 0.4896) is the
checkpoint the exported v2 weights correspond to.

Binaries not included: `training/data_bin/{train,val,test}.bin` (12.2 MB / 1.7 MB / 3.5 MB, `uint32`
token streams). They are regenerable with `dataset/prepare_training_data.py`.

## Reproduce

```bash
./venv/Scripts/python.exe src/train_slm.py
./venv/Scripts/python.exe src/continue_training.py
```
