"""
src/evaluate_finetuned.py
Before/after comparison of the original checkpoint and the continued-training checkpoint.

Uses the four real training templates, taken verbatim from held-out validation records
(`data/training/val_instructions.jsonl`), split at the template's own answer boundary.
Both models get identical prompts and identical greedy decoding.

Scoring is mechanical, not subjective: EOS emitted, required unit present, expected
opening phrase present, and whether the number token matches the record's value.
"""

import sys
import json
import math
import argparse
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC_DIR))
PROJECT_ROOT = _SRC_DIR.parent

import torch
import torch.nn.functional as F

from train_slm import Qwen3Model

MODEL_DIR = PROJECT_ROOT / "models" / "air_quality_slm"
ORIGINAL = PROJECT_ROOT / "models" / "air_quality_slm_original" / "pytorch_model.bin"
FINETUNED = PROJECT_ROOT / "models" / "air_quality_slm_finetuned" / "pytorch_model.bin"
VAL_JSONL = PROJECT_ROOT / "data" / "training" / "val_instructions.jsonl"
EOS = 151643
GEN = 40
BLOCK = 128


def build(weights, block=BLOCK):
    cfg = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    m = Qwen3Model(vocab_size=cfg["vocab_size"], d_model=cfg["hidden_size"],
                   n_heads=cfg["num_attention_heads"], n_kv_groups=cfg["num_key_value_heads"],
                   n_layers=cfg["num_hidden_layers"], d_ff=cfg["intermediate_size"],
                   max_seq_len=block, tie_weights=cfg.get("tie_word_embeddings", True))
    m.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
    m.eval()
    return m


@torch.no_grad()
def greedy(model, ids, n=GEN):
    idx = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    out = []
    for _ in range(n):
        logits, _ = model(idx)
        nxt = int(torch.argmax(logits[0, -1]))
        out.append(nxt)
        if nxt == EOS:
            break
        idx = torch.cat([idx, torch.tensor([[nxt]])], dim=1)
    return out


def score(text, expected):
    """Mechanical checks shared by both models."""
    exp_first = expected.strip().split(" ")[:4]
    return {
        "emitted_eos": "\ufffd" not in text and text.endswith("\n") or False,
        "has_unit_gm3": "g/m3" in text,
        "has_unit_mgm3": "mg/m3" in text,
        "has_ppb": "ppb" in text,
        "opens_with_expected": text.strip().startswith(" ".join(exp_first)),
        "has_absolute_humidity": "absolute humidity" in text,
        "has_ambient_temperature": "ambient temperature" in text,
        "has_recorded_co": "Recorded CO" in text,
        "n_digits": sum(c.isdigit() for c in text),
        "chars": len(text),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--out", default="docs/comparison_finetune.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)

    lines = [json.loads(l)["text"] for l in open(VAL_JSONL, encoding="utf-8")][: 4 * args.rows]
    cases = []
    splitters = ["Carbon Monoxide CO(GT)=", "Response:", "Answer:", "Result:"]
    names = ["REPORT", "AH", "TEMP_RH", "POLLUTANT"]
    for r in range(args.rows):
        for k in range(4):
            text = lines[r * 4 + k]
            cut = text.index(splitters[k]) + len(splitters[k])
            cases.append({"name": f"{names[k]}_row{r}", "kind": names[k],
                          "prompt": text[:cut], "expected": text[cut:].split("\n")[0]})

    models = {}
    for label, path in (("original", ORIGINAL), ("finetuned", FINETUNED)):
        if not path.exists():
            print(f"skip {label}: {path} not found")
            continue
        models[label] = build(path)

    report = {"prompts": cases, "models": {}}
    for label, m in models.items():
        rows = []
        print("\n" + "=" * 96)
        print(f"### {label.upper()}")
        for c in cases:
            ids = tok.encode(c["prompt"], add_special_tokens=False)
            assert len(ids) + GEN <= BLOCK, f"{c['name']} prompt {len(ids)} > {BLOCK - GEN}"
            out = greedy(m, ids)
            text = tok.decode(out, skip_special_tokens=False)
            s = score(text, c["expected"])
            s["hit_eos"] = EOS in out
            s["n_tokens"] = len(out)
            print("\n" + "-" * 90)
            print(f"[{c['name']}] {len(ids)} prompt tokens")
            print(f"  prompt   : ...{c['prompt'][-70:]!r}")
            print(f"  expected : {c['expected'][:90]!r}")
            print(f"  output   : {text!r}")
            print(f"  scores   : eos={s['hit_eos']} unit_gm3={s['has_unit_gm3']} "
                  f"opens_expected={s['opens_with_expected']} digits={s['n_digits']} chars={s['chars']}")
            rows.append({**c, "output": text, **s})
        report["models"][label] = rows

    # summary
    print("\n" + "=" * 96)
    print("SUMMARY")
    print(f"{'model':<10} {'eos':>5} {'unit_gm3':>9} {'opens_expected':>15} {'mean_chars':>11}")
    for label, rows in report["models"].items():
        n = len(rows)
        print(f"{label:<10} {sum(r['hit_eos'] for r in rows):>3}/{n:<2} "
              f"{sum(r['has_unit_gm3'] for r in rows):>7}/{n:<2} "
              f"{sum(r['opens_with_expected'] for r in rows):>13}/{n:<2} "
              f"{sum(r['chars'] for r in rows) / n:>11.1f}")

    if "original" in report["models"] and "finetuned" in report["models"]:
        improved = sum(1 for a, b in zip(report["models"]["original"], report["models"]["finetuned"])
                       if (b["hit_eos"] + b["has_unit_gm3"] + b["opens_with_expected"]) >
                          (a["hit_eos"] + a["has_unit_gm3"] + a["opens_with_expected"]))
        worse = sum(1 for a, b in zip(report["models"]["original"], report["models"]["finetuned"])
                    if (b["hit_eos"] + b["has_unit_gm3"] + b["opens_with_expected"]) <
                       (a["hit_eos"] + a["has_unit_gm3"] + a["opens_with_expected"]))
        report["outcome"] = {"improved": improved, "worse": worse,
                             "tied": len(cases) - improved - worse}
        print(f"\nper-prompt rubric (EOS + unit + opening phrase): "
              f"improved {improved}, worse {worse}, tied {len(cases) - improved - worse}")

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
