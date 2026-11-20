"""
src/diagnose_v2_numbers.py
Two questions for the v2 report, answered from the dataset itself:

1. Is the AH value recoverable from what the AH prompt contains (T and RH)?
   The UCI AH column is not standard g/m3: it is ~0.131x the Magnus-formula value.
   If a fixed linear rescaling of Magnus(T, RH) predicts the column well, then the
   information needed is present in the prompt and the model's failure is arithmetic,
   not missing input. The scale factor is fitted on TRAIN rows only.
2. Is T / RH / CO / NOx / NO2 / C6H6 recoverable from the prompt, which contains only
   date and time? That is a lookup, so we report how much of the variance a constant
   leaves behind (constant MAE) as the no-retrieval floor.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "eval_v2" / "numeric_diagnostics.json"


def magnus_ah(t, rh):
    """Absolute humidity in g/m3 from temperature (C) and relative humidity (%)."""
    es = 6.112 * np.exp(17.67 * t / (t + 243.5))       # hPa
    return 216.7 * (es * rh / 100.0) / (t + 273.15)


def mae(p, g):
    return float(np.mean(np.abs(p - g)))


def rmse(p, g):
    return float(np.sqrt(np.mean((p - g) ** 2)))


def main():
    tr = pd.read_csv(ROOT / "data" / "processed" / "train.csv")
    splits = {s: pd.read_csv(ROOT / "data" / "processed" / f"{s}.csv")
              for s in ("train", "validation", "test")}
    name = {"train": "train", "validation": "val", "test": "test"}

    # fit the dataset column's relation to physics on TRAIN only
    m = tr.dropna(subset=["T", "RH", "AH"])
    phys = magnus_ah(m["T"].to_numpy(float), m["RH"].to_numpy(float))
    scale = float(np.sum(phys * m["AH"].to_numpy(float)) / np.sum(phys ** 2))

    res = {"ah_scale_from_train": scale, "splits": {}}
    for s, df in splits.items():
        d = df.dropna(subset=["T", "RH", "AH"])
        if len(d) == 0:
            continue
        ph = magnus_ah(d["T"].to_numpy(float), d["RH"].to_numpy(float)) * scale
        g = d["AH"].to_numpy(float)
        entry = {
            "n": int(len(d)),
            "ah_column_mean": float(g.mean()), "ah_column_std": float(g.std()),
            "physics_scaled_mae": mae(ph, g), "physics_scaled_rmse": rmse(ph, g),
            "physics_scaled_r": float(np.corrcoef(ph, g)[0, 1]),
            "constant_mae": mae(np.full_like(g, tr["AH"].dropna().mean()), g),
        }
        # no-retrieval floor for quantities that the prompt does not contain
        for col in ("T", "RH", "CO(GT)", "NOx(GT)", "NO2(GT)", "C6H6(GT)"):
            v = df[col].dropna().to_numpy(float)
            if len(v):
                entry[f"{col}_std"] = float(v.std())
                entry[f"{col}_constant_mae"] = mae(np.full_like(v, tr[col].dropna().mean()), v)
                if entry[f"{col}_constant_mae"] > 0:
                    entry[f"{col}_rel_noise"] = float(v.std() / abs(v.mean()))
        res["splits"][name[s]] = entry

    # what the model actually produced, from the per-sample metric dumps
    for model in ("v1", "v2"):
        for split in ("val", "test"):
            p = ROOT / "docs" / "eval_v2" / f"metrics_{model}_{split}.json"
            if not p.exists():
                continue
            recs = json.loads(p.read_text(encoding="utf-8"))
            gens = [r for r in recs if "output" in r]
            distinct = len({r["output"].strip() for r in gens})
            all_wrong = 0
            any_ok = 0
            for r in gens:
                oks = [v for k, v in r.items() if k.startswith("ok_")]
                if oks:
                    all_wrong += not any(oks)
                    any_ok += any(oks)
            res.setdefault("model_outputs", {})[f"{model}_{split}"] = {
                "generated": len(gens), "distinct_outputs": distinct,
                "all_numbers_wrong_rate": all_wrong / len(gens) if gens else None,
                "any_number_correct_rate": any_ok / len(gens) if gens else None,
            }

    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
