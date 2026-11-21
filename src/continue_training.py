"""
src/continue_training.py
Continue training the existing Air Quality SLM from its trained checkpoint.

Why this script exists
----------------------
`src/train_slm.py` cannot resume: it always constructs a fresh model with
`torch.manual_seed(42)` and trains from random initialisation. It also draws random
64-token windows with `torch.randint`, and that window is shorter than every training
sample (83-201 tokens), so a template's instruction and its answer almost never appear
in the same window.

What this does
--------------
* Loads the learned weights from the original checkpoint (never overwritten).
* Rebuilds the same architecture with a larger context window (weights are unchanged;
  the RoPE cos/sin table is a non-persistent buffer, so a longer table is recomputed).
* Iterates the training stream systematically (one shuffled permutation per epoch)
  instead of sampling with replacement, so coverage is real.
* New AdamW optimiser + cosine decay at a reduced learning rate. The original run saved
  `optimizer_state_dict` only in `training/checkpoints/latest_model.pt`, not next to the
  exported weights, so continuing with a fresh optimiser is the documented choice.
* Deterministic validation (evenly spaced fixed windows) so the original checkpoint and the
  continued model are scored on exactly the same batches.
* Early stopping, best-checkpoint saving, JSONL log.

Nothing here writes to models/air_quality_slm/.
"""

import sys
import os
import json
import math
import time
import argparse
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC_DIR))
PROJECT_ROOT = _SRC_DIR.parent

import numpy as np
import torch
import torch.nn.functional as F

from train_slm import Qwen3Model

MODEL_DIR = PROJECT_ROOT / "models" / "air_quality_slm"
BIN_DIR = PROJECT_ROOT / "training" / "data_bin"
OUT_DIR = PROJECT_ROOT / "models" / "air_quality_slm_finetuned"
LOG_PATH = PROJECT_ROOT / "docs" / "training_log_finetune.jsonl"

CKPT = MODEL_DIR / "pytorch_model.bin"
CFG = MODEL_DIR / "config.json"


# --------------------------------------------------------------------------- data
def load_stream(split):
    return np.memmap(BIN_DIR / f"{split}.bin", dtype=np.uint32, mode="r")


def get_batch(data, starts, block):
    x = np.stack([np.asarray(data[s:s + block], dtype=np.int64) for s in starts])
    y = np.stack([np.asarray(data[s + 1:s + 1 + block], dtype=np.int64) for s in starts])
    return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def evaluate(model, data, block, n_windows=48, device="cpu"):
    """Deterministic validation: evenly spaced windows, identical for every model/run."""
    model.eval()
    n = len(data) - block - 2
    starts = [int(round(i * n / n_windows)) for i in range(n_windows)]
    total, count = 0.0, 0
    for i in range(0, len(starts), 8):
        x, y = get_batch(data, starts[i:i + 8], block)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total += float(loss) * len(starts[i:i + 8])
        count += len(starts[i:i + 8])
    model.train()
    return total / count


# --------------------------------------------------------------------------- setup
def build_model(block, dtype=torch.float32):
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    model = Qwen3Model(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["hidden_size"],
        n_heads=cfg["num_attention_heads"],
        n_kv_groups=cfg["num_key_value_heads"],
        n_layers=cfg["num_hidden_layers"],
        d_ff=cfg["intermediate_size"],
        max_seq_len=block,
        dtype=dtype,
        tie_weights=cfg.get("tie_word_embeddings", True),
    )
    sd = torch.load(CKPT, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=True), None
    return model, cfg, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min-lr-factor", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=40)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--benchmark", type=int, default=0, help="run N steps and exit without saving")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    device = "cuda" if (args.device in ("auto", "cuda")) and torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and device != "cuda":
        raise SystemExit("CUDA requested but not available")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train = load_stream("train")
    val = load_stream("val")

    model, cfg, sd = build_model(args.block)
    model.to(device).train()
    print(f"loaded {len(sd)} tensors from {CKPT}")

    # Baseline: score the ORIGINAL weights with this block size, for a like-for-like reference.
    base_val = evaluate(model, val, args.block, device=device)
    print(f"baseline val loss @ block {args.block}: {base_val:.4f} (ppl {math.exp(base_val):.3f})")
    if args.benchmark:
        tokens_per_step = args.batch * args.grad_accum * args.block
        t0 = time.time()
        for step in range(args.benchmark):
            x, y = get_batch(train, rng.integers(0, len(train) - args.block - 2, args.batch), args.block)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            loss.backward()
            model.zero_grad(set_to_none=True)
        dt = (time.time() - t0) / args.benchmark
        print(f"benchmark: {dt:.2f} s/step for {tokens_per_step} tokens "
              f"({tokens_per_step / dt:.0f} tok/s)")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = open(LOG_PATH, "w", encoding="utf-8")

    def emit(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()

    emit({"event": "start", "args": vars(args), "baseline_val": base_val,
          "baseline_ppl": math.exp(base_val)})

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1, eps=1e-9)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (0.1 + 0.9 * step / max(1, args.warmup))
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (args.min_lr_factor + (1 - args.min_lr_factor) * 0.5 * (1 + math.cos(math.pi * t)))

    n_windows = (len(train) - args.block - 2)
    order = rng.permutation(n_windows)
    cursor = 0
    tokens_per_step = args.batch * args.grad_accum * args.block
    best_val = base_val
    best_step = 0
    bad = 0
    t0 = time.time()
    ema = None

    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(args.grad_accum):
            if cursor + args.batch > n_windows:      # new epoch: reshuffle
                order = rng.permutation(n_windows)
                cursor = 0
            starts = order[cursor:cursor + args.batch]
            cursor += args.batch
            x, y = get_batch(train, starts, args.block)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            (loss / args.grad_accum).backward()
            step_loss += float(loss) / args.grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        ema = step_loss if ema is None else 0.9 * ema + 0.1 * step_loss
        tokens = (step + 1) * tokens_per_step
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            v = evaluate(model, val, args.block, device=device)
            rec = {"event": "eval", "step": step + 1, "train_loss_ema": round(ema, 4),
                   "val_loss": round(v, 5), "val_ppl": round(math.exp(v), 3),
                   "lr": round(lr_at(step), 7), "tokens_seen": tokens,
                   "epochs": round(tokens / len(train), 3),
                   "elapsed_s": round(time.time() - t0, 1)}
            print(json.dumps(rec))
            emit(rec)
            if v < best_val - 1e-4:
                best_val, best_step, bad = v, step + 1, 0
                torch.save({"step": step + 1, "model_state_dict": model.state_dict(),
                            "config": cfg, "block": args.block, "val_loss": v},
                           OUT_DIR / "best_model.pt")
                emit({"event": "saved_best", "step": step + 1, "val_loss": v})
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"early stop at step {step + 1} (no val improvement for {bad} evals)")
                    emit({"event": "early_stop", "step": step + 1})
                    break

    total = time.time() - t0
    emit({"event": "done", "steps": step + 1, "best_val": best_val, "best_step": best_step,
          "baseline_val": base_val, "seconds": round(total, 1),
          "tokens_seen": (step + 1) * tokens_per_step})
    print(f"\ndone: {step + 1} steps, {total:.0f}s, {(step + 1) * tokens_per_step:,} tokens, "
          f"best val {best_val:.4f} at step {best_step} (baseline {base_val:.4f})")
    log.close()


if __name__ == "__main__":
    main()
