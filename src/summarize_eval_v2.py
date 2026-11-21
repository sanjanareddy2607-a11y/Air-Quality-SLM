"""
src/summarize_eval_v2.py
Aggregate docs/eval_v2/*.json into one comparison table (v1 vs v2, val vs test).

Reports: full-split loss/perplexity, per-template teacher-forced loss, numeric
extraction coverage, MAE/RMSE/within-tolerance accuracy per quantity, prediction
spread and correlation, structural failure rates, date hallucination and
memorisation checks.
"""

import json
import re
import math
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / "docs" / "eval_v2"

QUANTS = {
    "AH": ["AH"],
    "TEMP_RH": ["T", "RH"],
    "POLLUTANT": ["CO", "NOx", "NO2", "C6H6"],
}
TOL = 0.05


def load(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def date_time(s):
    return s


def train_lookup():
    import pandas as pd
    df = pd.read_csv(ROOT / "data" / "processed" / "train.csv")
    lut = {}
    for _, r in df.iterrows():
        lut[(str(r["Date"]), str(r["Time"]))] = {
            "AH": r["AH"], "T": r["T"], "RH": r["RH"], "CO": r["CO(GT)"],
            "NOx": r["NOx(GT)"], "NO2": r["NO2(GT)"], "C6H6": r["C6H6(GT)"],
        }
    dates = set(df["Date"].astype(str))
    # answer strings actually trained on, for verbatim-memorisation checks
    answers = set()
    for line in open(ROOT / "data" / "training" / "train_instructions.jsonl", encoding="utf-8"):
        t = json.loads(line)["text"]
        for marker in ("Response:", "Answer:", "Result:"):
            if marker in t:
                answers.add(t.split(marker, 1)[1].split("\n")[0].strip())
    return lut, dates, answers


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def finite(*vals):
    for v in vals:
        if v is None or not math.isfinite(float(v)):
            return False
    return True


def stats(pairs):
    """pairs: list of (pred, gold); non-finite pairs are dropped (missing dataset labels)."""
    pairs = [(a, b) for a, b in pairs if finite(a, b)]
    if not pairs:
        return {"n": 0}
    p = np.array([a for a, _ in pairs], dtype=float)
    g = np.array([b for _, b in pairs], dtype=float)
    err = p - g
    out = {
        "n": int(len(p)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
        "pred_mean": float(p.mean()),
        "pred_std": float(p.std()),
        "gold_mean": float(g.mean()),
        "gold_std": float(g.std()),
        "within_5pct": float(np.mean(np.abs(err) <= np.maximum(TOL, TOL * np.abs(g)))),
        "within_10pct": float(np.mean(np.abs(err) <= np.maximum(0.10, 0.10 * np.abs(g)))),
        "exact": float(np.mean(err == 0)),
    }
    if p.std() > 1e-9 and g.std() > 1e-9:
        out["pearson_r"] = float(np.corrcoef(p, g)[0, 1])
    else:
        out["pearson_r"] = None
    return out


def summarise(model, split, lut, dates, answers):
    recs = load(EV / f"metrics_{model}_{split}.json")
    if recs is None:
        return None
    base = load(EV / "baseline.json")

    by_kind = defaultdict(list)
    for r in recs:
        by_kind[r["kind"]].append(r)

    out = {"model": model, "split": split, "n_records": len(recs)}

    # ---- teacher-forced answer loss
    tf = {}
    for kind, rows in by_kind.items():
        ls = [r["tf_loss"] for r in rows if "tf_loss" in r]
        hits = [r["tf_top1_hit"] for r in rows if "tf_top1_hit" in r]
        tf[kind] = {
            "n": len(ls),
            "answer_loss": float(np.mean(ls)) if ls else None,
            "answer_ppl": float(math.exp(np.mean(ls))) if ls else None,
            "top1_token_acc": float(np.mean(hits)) if hits else None,
        }
    all_ls = [r["tf_loss"] for r in recs if "tf_loss" in r]
    tf["ALL"] = {"n": len(all_ls),
                 "answer_loss": float(np.mean(all_ls)) if all_ls else None,
                 "answer_ppl": float(math.exp(np.mean(all_ls))) if all_ls else None}
    out["teacher_forced"] = tf

    # ---- numeric accuracy per quantity
    numeric = {}
    for kind, quants in QUANTS.items():
        rows = by_kind.get(kind, [])
        for q in quants:
            pairs, preds_all = [], []
            for r in rows:
                if "pred" not in r:
                    continue
                pv = num(r["pred"].get(q))
                gv = num(r["gold"].get(q))
                preds_all.append((pv, gv))
                if pv is not None and gv is not None:
                    pairs.append((pv, gv))
            gens = [r for r in rows if "pred" in r]
            parsed = [r for r in gens if num(r["pred"].get(q)) is not None]
            gold_missing = [r for r in gens if num(r["gold"].get(q)) is None]
            s = stats(pairs)
            s["generated"] = len(gens)
            s["parsed"] = len(parsed)
            s["prediction_parsed_rate"] = (len(parsed) / len(gens)) if gens else 0.0
            s["gold_missing"] = len(gold_missing)
            s["scored"] = len(pairs)
            s["extraction_rate"] = (len(pairs) / len(gens)) if gens else 0.0
            if base:
                bq = base[split]["by_quantity"]
                key = {"AH": "AH", "T": "T", "RH": "RH", "CO": "CO(GT)",
                       "NOx": "NOx(GT)", "NO2": "NO2(GT)", "C6H6": "C6H6(GT)"}[q]
                s["constant_mae"] = bq[key]["constant_mae"]
                s["constant_rmse"] = bq[key]["constant_rmse"]
                s["mae_vs_constant"] = (s.get("mae", 0) / bq[key]["constant_mae"]
                                        if bq[key]["constant_mae"] else None)
            numeric[f"{kind}:{q}"] = s
    out["numeric"] = numeric

    # ---- output diversity per template (a constant speaker has 1 distinct output)
    diversity = {}
    for kind, rows in by_kind.items():
        outs = [r["output"].strip() for r in rows if "output" in r]
        if not outs:
            continue
        from collections import Counter
        c = Counter(outs)
        top, n = c.most_common(1)[0]
        diversity[kind] = {"generated": len(outs), "distinct": len(c),
                           "modal_share": n / len(outs), "modal_output": top[:120]}
    out["diversity"] = diversity

    # ---- structural / behavioural (aggregate and per template)
    struct = {}
    struct_by_kind = {}
    for kind, rows in by_kind.items():
        g = [r for r in rows if "struct" in r]
        if not g:
            continue
        struct_by_kind[kind] = {
            "generated": len(g),
            "eos_rate": float(np.mean([r["struct"]["emitted_eos"] for r in g])),
            "repetitive_rate": float(np.mean([r["struct"]["repetitive"] for r in g])),
            "unit_ok_rate": float(np.mean([r["struct"]["unit_ok"] for r in g])),
            "wrong_unit_rate": float(np.mean([r["struct"]["unit_wrong"] for r in g])),
            "date_in_output_rate": float(np.mean([bool(r["struct"]["date_in_output"]) for r in g])),
            "date_matches_query_rate": float(np.mean([r["struct"]["date_matches_query"] for r in g])),
            "date_hallucinated_rate": float(np.mean([r["struct"]["date_hallucinated"] for r in g])),
        }
    out["structure_by_kind"] = struct_by_kind
    gens = [r for r in recs if "struct" in r]
    if gens:
        def rate(fn):
            return float(np.mean([1.0 if fn(r) else 0.0 for r in gens]))
        struct = {
            "generated": len(gens),
            "eos_rate": rate(lambda r: r["struct"]["emitted_eos"]),
            "ran_to_cap_rate": rate(lambda r: r["struct"]["ran_to_cap"]),
            "repetitive_rate": rate(lambda r: r["struct"]["repetitive"]),
            "no_digits_rate": rate(lambda r: not r["struct"]["has_digits"]),
            "unit_ok_rate": rate(lambda r: r["struct"]["unit_ok"]),
            "wrong_unit_rate": rate(lambda r: r["struct"]["unit_wrong"]),
            "replacement_char_rate": rate(lambda r: r["struct"]["has_replacement_char"]),
            "empty_rate": rate(lambda r: r["struct"]["empty"]),
            "extra_text_rate": rate(lambda r: r["struct"]["has_extra_text"]),
            "gold_prefix_rate": rate(lambda r: r["struct"]["open_with_gold_prefix"]),
            "date_in_output_rate": rate(lambda r: bool(r["struct"]["date_in_output"])),
            "date_matches_query_rate": rate(lambda r: r["struct"]["date_matches_query"]),
            "date_hallucinated_rate": rate(lambda r: r["struct"]["date_hallucinated"]),
        }
        # invent dates that never occur anywhere in the dataset
        invented, train_date, in_train_lut = 0, 0, 0
        for r in gens:
            d = r["struct"].get("date_in_output")
            if not d:
                continue
            if d in dates:
                train_date += 1
            else:
                invented += 1
            t = r["struct"].get("time_in_output")
            if t and (d, t) in lut:
                in_train_lut += 1
        struct["dates_seen_in_train"] = train_date
        struct["dates_not_in_any_split"] = invented
        struct["outputs_matching_a_train_row_key"] = in_train_lut
        malformed = [r for r in gens
                     if r["struct"]["empty"] or r["struct"]["ran_to_cap"]
                     or not r["struct"]["has_digits"] or r["struct"]["unit_wrong"]
                     or r["struct"]["has_replacement_char"]]
        struct["malformed_rate"] = len(malformed) / len(gens)
    out["structure"] = struct

    # ---- memorisation: verbatim trained answer, and record borrowing
    verbatim = 0
    borrow_err = []
    for r in gens:
        if "output" not in r:
            continue
        txt = r["output"].split("<|endoftext|>")[0].strip()
        if txt in answers:
            verbatim += 1
        d, t = r["struct"].get("date_in_output"), r["struct"].get("time_in_output")
        if d and t and (d, t) in lut:
            row = lut[(d, t)]
            errs = []
            for q, pv in (r.get("pred") or {}).items():
                pv, gv = num(pv), num(row.get(q))
                if pv is not None and gv is not None:
                    errs.append(abs(pv - gv))
            if errs:
                borrow_err.append(float(np.mean(errs)))
    out["memorisation"] = {
        "verbatim_training_answer_rate": (verbatim / len(gens)) if gens else 0.0,
        "verbatim_count": verbatim,
        "outputs_with_train_row_key": len(borrow_err),
        "mean_abs_err_vs_that_train_row": float(np.mean(borrow_err)) if borrow_err else None,
    }
    return out


def compare(v1, v2):
    """Paired improvement counts on the same generated prompts."""
    out = {}
    for split in ("val", "test"):
        a = load(EV / f"metrics_v1_{split}.json")
        b = load(EV / f"metrics_v2_{split}.json")
        if not a or not b:
            continue
        per_q = defaultdict(lambda: {"better": 0, "worse": 0, "tie": 0, "n": 0})
        for ra, rb in zip(a, b):
            if "pred" not in ra or "pred" not in rb:
                continue
            for q, gv in ra["gold"].items():
                gv = num(gv)
                pa, pb = num(ra["pred"].get(q)), num(rb["pred"].get(q))
                if gv is None or pa is None or pb is None:
                    continue
                ea, eb = abs(pa - gv), abs(pb - gv)
                d = ea - eb
                key = q
                per_q[key]["n"] += 1
                if d > 1e-9:
                    per_q[key]["better"] += 1
                elif d < -1e-9:
                    per_q[key]["worse"] += 1
                else:
                    per_q[key]["tie"] += 1
        out[split] = dict(per_q)
    return out


def main():
    lut, dates, answers = train_lookup()
    result = {}
    for split in ("val", "test"):
        for model in ("v1", "v2"):
            s = summarise(model, split, lut, dates, answers)
            if s:
                result[f"{model}_{split}"] = s
        for model in ("v1", "v2"):
            l = load(EV / f"loss_{model}_128.json")
            if l:
                result.setdefault(f"{model}_loss", {})[split] = l.get(split)
    for model in ("v1", "v2"):
        l = load(EV / f"loss_{model}_64.json")
        if l:
            result.setdefault(f"{model}_loss64", {}).update(l)
    result["paired"] = compare(None, None)
    result["baseline"] = load(EV / "baseline.json")

    (EV / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    # ---- console tables
    def f(x, n=4):
        return "n/a" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))

    print("=" * 100)
    print("FULL-SPLIT LOSS / PERPLEXITY (block 128, every token scored once)")
    print(f"{'model':<5}{'split':<7}{'tokens':>12}{'loss':>9}{'ppl':>10}")
    for model in ("v1", "v2"):
        for split in ("val", "test"):
            d = result.get(f"{model}_loss", {}).get(split)
            if d:
                print(f"{model:<5}{split:<7}{d['tokens_scored']:>12,}{d['loss']:>9.4f}{d['ppl']:>10.3f}")
    for model in ("v1", "v2"):
        d = result.get(f"{model}_loss64", {})
        for split in ("val", "test"):
            if split in d:
                print(f"{model:<5}{split+'@64':<7}{d[split]['tokens_scored']:>12,}"
                      f"{d[split]['loss']:>9.4f}{d[split]['ppl']:>10.3f}")

    print("\n" + "=" * 100)
    print("TEACHER-FORCED ANSWER LOSS (tokens after the prompt only)")
    print(f"{'model':<5}{'split':<7}{'template':<11}{'n':>6}{'loss':>9}{'ppl':>10}{'top1':>8}")
    for model in ("v1", "v2"):
        for split in ("val", "test"):
            d = result.get(f"{model}_{split}")
            if not d:
                continue
            for kind, v in d["teacher_forced"].items():
                print(f"{model:<5}{split:<7}{kind:<11}{v['n']:>6}{f(v['answer_loss']):>9}"
                      f"{f(v['answer_ppl'],3):>10}{f(v.get('top1_token_acc'),3):>8}")

    print("\n" + "=" * 100)
    print("NUMERIC ACCURACY (gold = dataset label; MAE/RMSE; constant = train-mean predictor)")
    print(f"{'quantity':<12}{'model':<5}{'split':<7}{'ext%':>7}{'MAE':>10}{'RMSE':>10}"
          f"{'±5%':>7}{'r':>7}{'pred_std':>10}{'const_MAE':>11}{'ratio':>7}")
    for kind, quants in QUANTS.items():
        for q in quants:
            for model in ("v1", "v2"):
                for split in ("val", "test"):
                    d = result.get(f"{model}_{split}")
                    if not d:
                        continue
                    s = d["numeric"].get(f"{kind}:{q}", {})
                    if not s.get("n"):
                        print(f"{q:<12}{model:<5}{split:<7}{'0':>7}  (no extractable predictions)")
                        continue
                    print(f"{q:<12}{model:<5}{split:<7}{s['prediction_parsed_rate']*100:>7.1f}"
                          f"{s['mae']:>10.4f}{s['rmse']:>10.4f}{s['within_5pct']*100:>7.1f}"
                          f"{f(s.get('pearson_r'),3):>7}{s['pred_std']:>10.3f}"
                          f"{s.get('constant_mae',float('nan')):>11.3f}"
                          f"{f(s.get('mae_vs_constant'),3):>7}"
                          f"   n={s['scored']} (gold missing {s['gold_missing']})")

    print("\n" + "=" * 100)
    print("STRUCTURE / BEHAVIOUR (fraction of generated outputs)")
    keys = ["eos_rate", "ran_to_cap_rate", "repetitive_rate", "no_digits_rate",
            "unit_ok_rate", "wrong_unit_rate", "date_hallucinated_rate",
            "date_in_output_rate", "malformed_rate"]
    print(f"{'model':<5}{'split':<7}" + "".join(f"{k[:13]:>15}" for k in keys))
    for model in ("v1", "v2"):
        for split in ("val", "test"):
            d = result.get(f"{model}_{split}")
            if not d or not d["structure"]:
                continue
            print(f"{model:<5}{split:<7}" + "".join(f"{d['structure'].get(k, 0):>15.3f}" for k in keys))

    print("\n" + "=" * 100)
    print("OUTPUT DIVERSITY AND PER-TEMPLATE STRUCTURE")
    for model in ("v1", "v2"):
        for split in ("val", "test"):
            d = result.get(f"{model}_{split}")
            if not d:
                continue
            for kind, v in d.get("diversity", {}).items():
                s = d.get("structure_by_kind", {}).get(kind, {})
                print(f"{model} {split} {kind:<10} gen={v['generated']:>5} distinct={v['distinct']:>5} "
                      f"modal={v['modal_share']*100:>5.1f}% eos={s.get('eos_rate', 0):.3f} "
                      f"unit_ok={s.get('unit_ok_rate', 0):.3f} "
                      f"date_halluc={s.get('date_hallucinated_rate', 0):.3f}")
                print(f"      modal: {v['modal_output']!r}")

    print("\n" + "=" * 100)
    print("MEMORISATION")
    for model in ("v1", "v2"):
        for split in ("val", "test"):
            d = result.get(f"{model}_{split}")
            if d:
                m = d["memorisation"]
                print(f"{model} {split}: verbatim trained answer {m['verbatim_count']} "
                      f"({m['verbatim_training_answer_rate']*100:.2f}%), "
                      f"outputs keyed to a train row {m['outputs_with_train_row_key']}, "
                      f"mean |err| vs that train row {f(m['mean_abs_err_vs_that_train_row'])}")
                s = d["structure"]
                print(f"        dates seen in dataset {s.get('dates_seen_in_train')}, "
                      f"dates not in any split {s.get('dates_not_in_any_split')}")

    print("\n" + "=" * 100)
    print("PAIRED v1 -> v2 (per predicted number, |err| comparison on identical prompts)")
    for split, d in result["paired"].items():
        print(f"{split}: " + ", ".join(
            f"{q} better {v['better']} / worse {v['worse']} / tie {v['tie']} (n={v['n']})"
            for q, v in d.items()))
    print(f"\nwrote {EV / 'summary.json'}")


if __name__ == "__main__":
    main()
