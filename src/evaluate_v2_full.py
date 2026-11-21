"""
src/evaluate_v2_full.py
Comprehensive evaluation of the original (v1) and continued-training (v2) checkpoints
on the COMPLETE validation (936 records) and test (1,872 records) splits.

Both models are scored with exactly the same code path, prompts, tokenizer, context
window (128) and greedy decoding. Nothing is written back to either checkpoint.

Phases
------
  baseline : constant-mean predictors, split leakage checks (no model needed)
  loss     : full-split next-token loss over every token of val.bin / test.bin
             (non-overlapping 128-token windows, so every token is scored once)
  metrics  : per-sample teacher-forced answer loss + full-split greedy generation,
             numerical extraction/accuracy, structure, units, dates, repetition

Outputs JSON under docs/eval_v2/.
"""

import sys
import os
import re
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

from train_slm import Qwen3Model, apply_rope

EOS = 151643
BLOCK = 128
OUT = PROJECT_ROOT / "docs" / "eval_v2"

WEIGHTS = {
    "v1": PROJECT_ROOT / "models" / "air_quality_slm_original" / "pytorch_model.bin",
    "v2": PROJECT_ROOT / "models" / "air_quality_slm_finetuned" / "pytorch_model.bin",
}
CFG_PATH = PROJECT_ROOT / "models" / "air_quality_slm" / "config.json"
TOK_DIR = PROJECT_ROOT / "models" / "air_quality_slm"
BIN_DIR = PROJECT_ROOT / "training" / "data_bin"
SPLITS = {
    "val": PROJECT_ROOT / "data" / "processed" / "validation.csv",
    "test": PROJECT_ROOT / "data" / "processed" / "test.csv",
}

# which template index belongs to which kind (see prepare_training_data.row_to_texts)
KINDS = ["REPORT", "AH", "TEMP_RH", "POLLUTANT"]
SPLITTERS = {"AH": "Response:", "TEMP_RH": "Answer:", "POLLUTANT": "Result:"}


# --------------------------------------------------------------------------- model
def build(weights, block=BLOCK):
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    m = Qwen3Model(
        vocab_size=cfg["vocab_size"], d_model=cfg["hidden_size"],
        n_heads=cfg["num_attention_heads"], n_kv_groups=cfg["num_key_value_heads"],
        n_layers=cfg["num_hidden_layers"], d_ff=cfg["intermediate_size"],
        max_seq_len=block, tie_weights=cfg.get("tie_word_embeddings", True),
    )
    sd = torch.load(weights, map_location="cpu")
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m


def causal_mask(seq_len, cache={}):
    m = cache.get(seq_len)
    if m is None:
        m = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        cache[seq_len] = m
    return m


@torch.no_grad()
def body(model, ids):
    """Transformer body + final norm (identical math to Qwen3Model.forward)."""
    b, seq = ids.shape
    x = model.tok_emb(ids) * (model.d_model ** 0.5)
    mask = causal_mask(seq)
    for blk in model.blocks:
        x = blk(x, mask, model.cos, model.sin)
    return model.final_norm(x)


def _rope_at(x, cos, sin, pos):
    hd = x.shape[-1]
    c = cos[pos:pos + 1].unsqueeze(0).unsqueeze(0)
    s = sin[pos:pos + 1].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
    rot = torch.cat((-x2, x1), dim=-1)
    return x * c + rot * s


class CachedLM:
    """Greedy decoder with a per-layer key/value cache.

    Math is identical to src/train_slm.Qwen3Model.forward + apply_rope; only the
    recomputation is avoided. Verified against the uncached greedy path by
    --verify-cache.
    """

    def __init__(self, model):
        self.m = model
        self.k = None
        self.v = None

    @torch.no_grad()
    def prefill(self, ids):
        m = self.m
        b, L = ids.shape
        x = m.tok_emb(ids) * (m.d_model ** 0.5)
        mask = causal_mask(L)
        self.k, self.v = [], []
        for blk in m.blocks:
            att = blk.attention
            xn = blk.norm1(x)
            q = att.W_query(xn).view(b, L, att.num_heads, att.head_dim).transpose(1, 2)
            k = att.W_key(xn).view(b, L, att.num_kv_groups, att.head_dim).transpose(1, 2)
            v = att.W_value(xn).view(b, L, att.num_kv_groups, att.head_dim).transpose(1, 2)
            q = apply_rope(att.q_norm(q), m.cos, m.sin)
            k = apply_rope(att.k_norm(k), m.cos, m.sin)
            self.k.append(k)
            self.v.append(v)
            scores = (q * att.scaling) @ k.repeat_interleave(att.group_size, dim=1).transpose(2, 3)
            scores = scores.masked_fill(mask, -torch.inf)
            w = torch.softmax(scores, dim=-1)
            o = (w @ v.repeat_interleave(att.group_size, dim=1)).transpose(1, 2).reshape(b, L, att.d_out)
            x = x + att.out_proj(o)
            x = x + blk.feed_forward(blk.norm2(x))
        return m.out_head(m.final_norm(x)[:, -1, :])

    @torch.no_grad()
    def step(self, ids, pos):
        m = self.m
        b = ids.shape[0]
        x = m.tok_emb(ids) * (m.d_model ** 0.5)
        for i, blk in enumerate(m.blocks):
            att = blk.attention
            xn = blk.norm1(x)
            q = att.W_query(xn).view(b, 1, att.num_heads, att.head_dim).transpose(1, 2)
            k = att.W_key(xn).view(b, 1, att.num_kv_groups, att.head_dim).transpose(1, 2)
            v = att.W_value(xn).view(b, 1, att.num_kv_groups, att.head_dim).transpose(1, 2)
            q = _rope_at(att.q_norm(q), m.cos, m.sin, pos)
            k = _rope_at(att.k_norm(k), m.cos, m.sin, pos)
            self.k[i] = torch.cat([self.k[i], k], dim=2)
            self.v[i] = torch.cat([self.v[i], v], dim=2)
            scores = (q * att.scaling) @ self.k[i].repeat_interleave(att.group_size, dim=1).transpose(2, 3)
            w = torch.softmax(scores, dim=-1)
            o = (w @ self.v[i].repeat_interleave(att.group_size, dim=1)).transpose(1, 2).reshape(b, 1, att.d_out)
            x = x + att.out_proj(o)
            x = x + blk.feed_forward(blk.norm2(x))
        return m.out_head(m.final_norm(x)[:, -1, :])

    @torch.no_grad()
    def generate(self, batch_ids, max_new=40):
        """batch_ids: list of equal-length int lists. Returns list of new-token ids."""
        b = len(batch_ids)
        L = len(batch_ids[0])
        ids = torch.tensor(batch_ids, dtype=torch.long)
        logits = self.prefill(ids)
        nxt = logits.argmax(-1)                       # [b]
        outs = [[] for _ in range(b)]
        done = [False] * b
        pos = L
        for _ in range(max_new):
            for i in range(b):
                if not done[i]:
                    outs[i].append(int(nxt[i]))
                    if int(nxt[i]) == EOS:
                        done[i] = True
            if all(done) or pos >= BLOCK:
                break
            ids = torch.tensor([[nxt[i].item()] for i in range(b)], dtype=torch.long)
            logits = self.step(ids, pos)
            nxt = logits.argmax(-1)
            pos += 1
        return outs


# --------------------------------------------------------------------------- data
def load_rows(split):
    import pandas as pd
    df = pd.read_csv(SPLITS[split])
    return df.to_dict("records")


def template_texts(row):
    from prepare_training_data import row_to_texts
    return [t.strip() for t in row_to_texts(row)]


def split_prompt(kind, text):
    if kind == "REPORT":
        return None, text
    marker = SPLITTERS[kind]
    cut = text.index(marker) + len(marker)
    return text[:cut], text[cut:]


# --------------------------------------------------------------------------- metrics helpers
NUM = re.compile(r"-?\d+(?:\.\d+)?")
DATE = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")
TIME = re.compile(r"\b\d{2}\.\d{2}\.\d{2}\b")


def fnum(x):
    """Float or None; NaN/absent dataset labels become None (131 of 936 val rows have gaps)."""
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def extract_numbers(kind, text, prompt):
    """Pull the model's numeric claims out of a generated completion."""
    out = {}
    t = text
    if kind == "AH":
        m = re.search(r"absolute humidity is\s*(" + NUM.pattern + ")", t)
        if m:
            out["AH"] = fnum(m.group(1))
        else:
            m = re.search(NUM.pattern, t)
            out["AH"] = fnum(m.group(0)) if m else None
    elif kind == "TEMP_RH":
        m = re.search(r"temperature was\s*(" + NUM.pattern + ")", t)
        out["T"] = fnum(m.group(1)) if m else None
        m = re.search(r"humidity was\s*(" + NUM.pattern + ")", t)
        out["RH"] = fnum(m.group(1)) if m else None
    elif kind == "POLLUTANT":
        for key, pat in (("CO", r"CO is\s*(" + NUM.pattern + ")"),
                         ("NOx", r"NOx is\s*(" + NUM.pattern + ")"),
                         ("NO2", r"NO2 is\s*(" + NUM.pattern + ")"),
                         ("C6H6", r"benzene level at\s*(" + NUM.pattern + ")")):
            m = re.search(pat, t)
            out[key] = fnum(m.group(1)) if m else None
    return out


def gold_values(kind, row):
    if kind == "AH":
        return {"AH": fnum(row.get("AH"))}
    if kind == "TEMP_RH":
        return {"T": fnum(row.get("T")), "RH": fnum(row.get("RH"))}
    if kind == "POLLUTANT":
        return {"CO": fnum(row.get("CO(GT)")), "NOx": fnum(row.get("NOx(GT)")),
                "NO2": fnum(row.get("NO2(GT)")), "C6H6": fnum(row.get("C6H6(GT)"))}
    return {}


def close(pred, gold, tol=0.05):
    if pred is None or gold is None:
        return False
    return abs(pred - gold) <= max(tol, tol * abs(gold))


def structure_metrics(kind, prompt, text, gold_answer, expected_unit):
    low = text.lower()
    checks = {
        "emitted_eos": "<|endoftext|>" in text or "\ufffd" in text,
        "ran_to_cap": "<|endoftext|>" not in text,
        "has_replacement_char": "\ufffd" in text,
        "empty": len(text.strip()) == 0,
        "has_digits": bool(NUM.search(text)),
        "unit_ok": expected_unit in text,
        "unit_wrong": any(u in text for u in ("mg/m3", "ug/m3", "ppb", "g/m3", "%"))
                      and expected_unit not in text,
        "has_extra_text": len(text.split("\n")) > 2,
    }
    # repetition: any 3+ token-ish chunk repeated
    words = text.split()
    rep = False
    for n in (3, 5):
        grams = [" ".join(words[i:i + n]) for i in range(max(0, len(words) - n + 1))]
        if grams and len(grams) != len(set(grams)):
            rep = True
    checks["repetitive"] = rep
    # dates / times invented or copied
    d = DATE.search(text)
    ti = TIME.search(text)
    checks["date_in_output"] = d.group(0) if d else None
    checks["time_in_output"] = ti.group(0) if ti else None
    qd = DATE.search(prompt) if prompt else None
    qt = TIME.search(prompt) if prompt else None
    checks["queried_date"] = qd.group(0) if qd else None
    checks["queried_time"] = qt.group(0) if qt else None
    checks["date_matches_query"] = bool(d and qd and d.group(0) == qd.group(0))
    checks["date_hallucinated"] = bool(d and qd and d.group(0) != qd.group(0))
    checks["open_with_gold_prefix"] = text.strip().startswith(gold_answer.strip()[:12])
    return checks


# --------------------------------------------------------------------------- phases
def phase_baseline():
    import pandas as pd
    out = {}
    train = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "train.csv")
    train_tokens = set()
    for split in SPLITS:
        df = pd.read_csv(SPLITS[split])
        gaps = {}
        for col in ("AH", "T", "RH", "CO(GT)", "NOx(GT)", "NO2(GT)", "C6H6(GT)"):
            g = df[col].dropna().to_numpy(dtype=float)
            mu = float(train[col].dropna().mean())
            if col == "AH":
                sig = float(train[col].dropna().std())
            else:
                sig = float(train[col].dropna().std())
            gaps[col] = {
                "n": int(len(g)), "gold_mean": float(g.mean()), "gold_std": float(g.std()),
                "train_mean": mu, "train_std": sig,
                "constant_mae": float(np.mean(np.abs(g - mu))),
                "constant_rmse": float(np.sqrt(np.mean((g - mu) ** 2))),
                "min": float(g.min()), "max": float(g.max()),
            }
        # date overlap with train -> would indicate leakage
        tr_dates = set(train["Date"].astype(str))
        sp_dates = set(df["Date"].astype(str))
        out[split] = {"rows": int(len(df)), "by_quantity": gaps,
                      "date_overlap_with_train": len(tr_dates & sp_dates)}
    train_dates = set(train["Date"].astype(str))
    for split in SPLITS:
        out[split]["train_date_pool"] = len(train_dates)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "baseline.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


def phase_loss(model_name, block=BLOCK, threads=12, max_windows=None):
    torch.set_num_threads(threads)
    m = build(WEIGHTS[model_name], block=block)
    res = {"model": model_name, "block": block}
    for split in ("val", "test"):
        data = np.memmap(BIN_DIR / f"{split}.bin", dtype=np.uint32, mode="r")
        n = (len(data) - block - 1) // block          # non-overlapping: every token scored once
        starts = [i * block for i in range(n)]
        if max_windows:
            starts = starts[:max_windows]
        tot, cnt, t0 = 0.0, 0, time.time()
        for i in range(0, len(starts), 8):
            st = starts[i:i + 8]
            x = torch.stack([torch.from_numpy(np.asarray(data[s:s + block], dtype=np.int64)) for s in st])
            y = torch.stack([torch.from_numpy(np.asarray(data[s + 1:s + 1 + block], dtype=np.int64)) for s in st])
            with torch.no_grad():
                _, loss = m(x, y)
            tot += float(loss) * len(st)
            cnt += len(st)
            if i % 160 == 0:
                print(f"[{model_name} {split}@{block}] {cnt}/{len(starts)} windows "
                      f"loss {tot / cnt:.4f} {time.time() - t0:.0f}s", flush=True)
        mean = tot / cnt
        res[split] = {"tokens_scored": cnt * block, "windows": cnt, "loss": mean,
                      "ppl": math.exp(mean)}
        print(f"[{model_name}] {split}@{block}: loss {mean:.5f} ppl {math.exp(mean):.4f}", flush=True)
    (OUT / f"loss_{model_name}_{block}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


@torch.no_grad()
def phase_metrics(model_name, split, batch=24, max_new=50, gen_limit=None, threads=12):
    from transformers import AutoTokenizer
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    m = build(WEIGHTS[model_name])
    rows = load_rows(split)

    samples = []
    for idx, row in enumerate(rows):
        texts = template_texts(row)
        for k, kind in enumerate(KINDS):
            text = texts[k]
            prompt, answer = split_prompt(kind, text)
            samples.append({"row": idx, "kind": kind, "text": text,
                            "prompt": prompt, "answer": answer,
                            "gold": gold_values(kind, row),
                            "row_ref": row})
    print(f"[{model_name} {split}] {len(samples)} samples "
          f"({sum(1 for s in samples if s['kind'] != 'REPORT')} generatable)", flush=True)

    # ---- teacher-forced answer loss (full split, all 4 templates where they fit)
    tf = []
    buf = []
    t0 = time.time()
    for s in samples:
        ids = tok.encode(s["text"], add_special_tokens=False)
        pl = len(tok.encode(s["prompt"], add_special_tokens=False)) if s["prompt"] else 0
        s["ids"] = ids
        s["plen"] = pl
        if len(ids) > BLOCK:
            s["tf_skipped"] = f"text {len(ids)} tokens > {BLOCK}"
            continue
        if len(ids) - pl < 1:
            s["tf_skipped"] = "no answer tokens"
            continue
        buf.append(s)
        if len(buf) == batch:
            tf.extend(_tf_batch(m, buf))
            buf = []
            print(f"  tf {len(tf)}/{len(samples)} {time.time() - t0:.0f}s", flush=True)
    if buf:
        tf.extend(_tf_batch(m, buf))
    for s, rec in zip([s for s in samples if "tf_skipped" not in s], tf):
        s["tf_loss"] = rec["loss"]
        s["tf_argmax"] = rec["argmax_text"]
        s["tf_top1_hit"] = rec["top1_hit"]
    print(f"  teacher-forced pass done {time.time() - t0:.0f}s", flush=True)

    # ---- full-split greedy generation for the 3 instruction templates
    gen = [s for s in samples if s["kind"] != "REPORT"]
    if gen_limit:
        gen = gen[:gen_limit]
    by_len = {}
    for s in gen:
        ids = tok.encode(s["prompt"], add_special_tokens=False)
        s["prompt_ids"] = ids
        if len(ids) >= BLOCK:
            s["gen_skipped"] = f"prompt {len(ids)} >= {BLOCK}"
            continue
        by_len.setdefault(len(ids), []).append(s)
    t0 = time.time()
    done = 0
    for L in sorted(by_len):
        group = by_len[L]
        for i in range(0, len(group), batch):
            chunk = group[i:i + batch]
            lm = CachedLM(m)
            outs = lm.generate([s["prompt_ids"] for s in chunk], max_new=min(max_new, BLOCK - L))
            for s, o in zip(chunk, outs):
                s["new_ids"] = o
                s["output"] = tok.decode(o, skip_special_tokens=False)
            done += len(chunk)
        print(f"  gen {done}/{len(gen)} {time.time() - t0:.0f}s", flush=True)

    # ---- score
    per_sample = []
    for s in samples:
        rec = {"row": s["row"], "kind": s["kind"], "gold": s["gold"]}
        if "tf_loss" in s:
            rec["tf_loss"] = s["tf_loss"]
            rec["tf_top1_hit"] = s["tf_top1_hit"]
        if "output" in s:
            out = s["output"]
            rec["output"] = out[:200]
            rec["n_new"] = len(s["new_ids"])
            pred = extract_numbers(s["kind"], out, s["prompt"])
            rec["pred"] = pred
            rec["struct"] = structure_metrics(s["kind"], s["prompt"], out,
                                              s["answer"], {
                                                  "AH": "g/m3", "TEMP_RH": "degrees Celsius",
                                                  "POLLUTANT": "mg/m3"}[s["kind"]])
            for k, gv in s["gold"].items():
                rec[f"ok_{k}"] = close(pred.get(k), gv)
        per_sample.append(rec)

    (OUT / f"metrics_{model_name}_{split}.json").write_text(
        json.dumps(per_sample, indent=1), encoding="utf-8")
    print(f"[{model_name} {split}] wrote {len(per_sample)} records", flush=True)


def _tf_batch(model, batch):
    B = len(batch)
    maxlen = max(len(s["ids"]) for s in batch)
    X = torch.full((B, maxlen), EOS, dtype=torch.long)
    for i, s in enumerate(batch):
        X[i, :len(s["ids"])] = torch.tensor(s["ids"])
    h = body(model, X)
    out = []
    for i, s in enumerate(batch):
        ids, pl = s["ids"], s["plen"]
        pos = torch.arange(pl - 1, len(ids) - 1)
        logits = model.out_head(h[i, pos, :])
        tgt = torch.tensor(ids[pl:])
        loss = F.cross_entropy(logits, tgt, reduction="mean")
        am = logits.argmax(-1).tolist()
        from transformers import AutoTokenizer
        out.append({"loss": float(loss), "argmax": am,
                    "argmax_text": None, "top1_hit": sum(a == b for a, b in zip(am, ids[pl:])) / max(1, len(ids) - pl)})
    return out


@torch.no_grad()
def verify_cache(model_name):
    """Prove CachedLM.generate matches the uncached greedy decoder token-for-token."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    m = build(WEIGHTS[model_name])
    rows = load_rows("val")[:2]
    ok = True
    for r, row in enumerate(rows):
        texts = template_texts(row)
        for k, kind in enumerate(KINDS):
            if kind == "REPORT":
                continue
            prompt, _ = split_prompt(kind, texts[k])
            ids = tok.encode(prompt, add_special_tokens=False)
            # uncached reference (same code as src/evaluate_finetuned.py)
            idx = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            ref = []
            for _ in range(40):
                logits, _ = m(idx)
                nxt = int(logits[0, -1].argmax())
                ref.append(nxt)
                if nxt == EOS:
                    break
                idx = torch.cat([idx, torch.tensor([[nxt]])], dim=1)
            lm = CachedLM(m)
            got = lm.generate([ids], max_new=40)[0]
            same = ref == got
            ok &= same
            print(f"  {kind} row{r}: cache==uncached {same} tokens={len(got)}")
    print("CACHE VERIFICATION", "PASSED" if ok else "FAILED")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["baseline", "loss", "metrics", "verify-cache", "suite"])
    ap.add_argument("--suite", default="metrics", choices=["metrics", "loss"])
    ap.add_argument("--model", default="v2", choices=["v1", "v2"])
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--block", type=int, default=BLOCK)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--gen-limit", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=50, help="new tokens per completion (capped by context)")
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()

    if args.phase == "suite":
        t0 = time.time()
        if args.suite == "metrics":
            for split in ("val", "test"):
                for model in ("v2", "v1"):
                    print(f"### metrics {model} {split} start", flush=True)
                    phase_metrics(model, split, args.batch, max_new=args.max_new,
                                  threads=args.threads)
                    print(f"### metrics {model} {split} done {time.time() - t0:.0f}s", flush=True)
        else:
            phase_loss("v1", 128, args.threads)
            phase_loss("v2", 128, args.threads)
            phase_loss("v1", 64, args.threads, 400)
        print(f"### suite {args.suite} done in {time.time() - t0:.0f}s", flush=True)
    elif args.phase == "baseline":
        phase_baseline()
    elif args.phase == "loss":
        phase_loss(args.model, args.block, args.threads, args.max_windows or None)
    elif args.phase == "verify-cache":
        verify_cache(args.model)
    else:
        phase_metrics(args.model, args.split, args.batch, max_new=args.max_new,
                      gen_limit=args.gen_limit or None, threads=args.threads)
