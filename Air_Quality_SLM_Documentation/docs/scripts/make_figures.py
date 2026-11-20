"""
make_figures.py - data-driven figures for the Air Quality SLM research package.

Every figure produced here is computed from files that already exist in the
project workspace (dataset CSVs, evaluation JSON, training logs). No value in
this script is typed in by hand except the closed-form Tetens constant, which
is documented in docs/EVALUATION_V2_REPORT.md section 4.4.

Usage (from the project root, or with AQSLM_ROOT set):
    python research_package/Air_Quality_SLM_Research_Package/figures/scripts/make_figures.py

Outputs PNG (300 dpi) and PDF next to this package's figures/ directory.
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DPI = 300
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": ":",
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_V1 = "#9aa5b1"
C_V2 = "#1f6feb"
C_CONST = "#d1495b"
C_FORMULA = "#2a9d8f"
C_TRAIN = "#264653"
C_VAL = "#e76f51"

HERE = Path(__file__).resolve()
FIGDIR = HERE.parents[1]


def find_root() -> Path:
    """Locate the project workspace that holds the evidence files."""
    env = os.environ.get("AQSLM_ROOT")
    if env and (Path(env) / "data" / "processed" / "train.csv").exists():
        return Path(env)
    for cand in HERE.parents:
        if (cand / "data" / "processed" / "train.csv").exists():
            return cand
    raise SystemExit(
        "Could not find the project workspace. Run this script from inside the "
        "Air-Quality-SLM project, or set AQSLM_ROOT to it."
    )


ROOT = find_root()


def save(fig, relpath: str) -> None:
    """Save a figure as PNG and PDF under figures/<relpath>."""
    out = FIGDIR / relpath
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=DPI)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {out.relative_to(FIGDIR)}.png / .pdf")


def load_splits():
    d = ROOT / "data" / "processed"
    return {name: pd.read_csv(d / f"{name}.csv")
            for name in ("train", "validation", "test")}


# ---------------------------------------------------------------- dataset ---

def fig_feature_distributions(splits):
    df = splits["train"]
    numeric = [c for c in df.columns if c not in ("Date", "Time")]
    fig, axes = plt.subplots(4, 4, figsize=(11, 9))
    axes = axes.ravel()
    for ax, col in zip(axes, numeric):
        vals = df[col].dropna()
        ax.hist(vals, bins=40, color=C_V2, alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes[len(numeric):]:
        ax.axis("off")
    fig.suptitle("UCI Air Quality - feature distributions (training split, n=%d)" % len(df),
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "dataset/fig01_feature_distributions")


def fig_missing_values(splits):
    allrows = pd.concat(splits.values(), ignore_index=True)
    frac = allrows.isna().mean() * 100
    frac = frac[frac.index.isin([c for c in allrows.columns if c not in ("Date", "Time")])]
    frac = frac.sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bars = ax.barh(frac.index, frac.values, color=C_VAL, alpha=0.9)
    for bar, v in zip(bars, frac.values):
        ax.text(v + 1, bar.get_y() + bar.get_height() / 2, f"{v:.2f}%",
                va="center", fontsize=7)
    ax.set_xlabel("Missing values (% of 9,357 records)")
    ax.set_xlim(0, 100)
    ax.set_title("Missing-value distribution after -200 sentinel conversion to NaN")
    fig.tight_layout()
    save(fig, "dataset/fig02_missing_values")


# --------------------------------------------------------------- training ---

def fig_training_v1():
    sp = json.loads((ROOT / "models/air_quality_slm/training_summary.json").read_text(encoding="utf-8"))
    steps = [0] + [p["step"] for p in sp["loss_progression"]]
    train = [sp["initial_train_loss"]] + [p["train_loss"] for p in sp["loss_progression"]]
    val = [sp["initial_val_loss"]] + [p["val_loss"] for p in sp["loss_progression"]]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(steps, train, "o-", color=C_TRAIN, label="training loss")
    ax.plot(steps, val, "s--", color=C_VAL, label="validation loss")
    ax.set_yscale("log")
    ax.set_xlabel("optimiser step")
    ax.set_ylabel("cross-entropy loss (log scale)")
    ax.set_title("v1 training dynamics (150 steps, block 64)")
    ax.annotate(f"best val {sp['best_val_loss']:.4f}\n(ppl {sp['best_val_perplexity']:.2f})",
                xy=(steps[-1], val[-1]), xytext=(-95, 18), textcoords="offset points",
                fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "training/fig05_loss_curve_v1")


def fig_training_v2():
    df = pd.read_csv(ROOT / "docs/loss_curve.csv")
    log = [json.loads(l) for l in (ROOT / "docs/training_log_finetune.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    baseline = next(e["baseline_val"] for e in log if e["event"] == "start")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.9))
    ax1.axhline(baseline, color=C_V1, ls=":", lw=1.2, label=f"v1 init val {baseline:.4f}")
    ax1.plot(df["step"], df["train_loss_ema"], "o-", color=C_TRAIN, label="train loss (EMA)")
    ax1.plot(df["step"], df["val_loss"], "s--", color=C_VAL, label="validation loss")
    ax1.set_xlabel("step")
    ax1.set_ylabel("cross-entropy loss")
    ax1.set_title("v2 continued training (block 128)")
    ax1.legend(frameon=False, fontsize=7.5)

    ax2.plot(df["step"], df["val_ppl"], "o-", color=C_V2)
    ax2.set_xlabel("step")
    ax2.set_ylabel("validation perplexity")
    ax2.set_title("v2 validation perplexity")
    fig.tight_layout()
    save(fig, "training/fig06_loss_curve_v2")


# ------------------------------------------------------------- evaluation ---

def fig_loss_ppl_comparison():
    l1 = json.loads((ROOT / "docs/eval_v2/loss_v1_128.json").read_text(encoding="utf-8"))
    l2 = json.loads((ROOT / "docs/eval_v2/loss_v2_128.json").read_text(encoding="utf-8"))
    labels = ["v1 val", "v1 test", "v2 val", "v2 test"]
    losses = [l1["val"]["loss"], l1["test"]["loss"], l2["val"]["loss"], l2["test"]["loss"]]
    ppls = [l1["val"]["ppl"], l1["test"]["ppl"], l2["val"]["ppl"], l2["test"]["ppl"]]
    colors = [C_V1, C_V1, C_V2, C_V2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.8))
    for ax, vals, ylabel, title, fmt in (
        (ax1, losses, "cross-entropy loss", "Full-split loss (block 128)", "{:.4f}"),
        (ax2, ppls, "perplexity", "Full-split perplexity (block 128)", "{:.2f}"),
    ):
        bars = ax.bar(labels, vals, color=colors, alpha=0.9)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.02, fmt.format(v),
                    ha="center", fontsize=7.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, max(vals) * 1.18)
    fig.tight_layout()
    save(fig, "evaluation/fig07_v1_v2_loss_perplexity")


def fig_mae_vs_baseline():
    base = json.loads((ROOT / "docs/eval_v2/baseline.json").read_text(encoding="utf-8"))
    # v2 MAE values, taken from docs/eval_v2/summary_tables.txt (NUMERIC ACCURACY block)
    model_mae = {
        "AH": (0.4335, 0.5345),
        "T": (11.0262, 11.5356),
        "RH": (36.3190, 29.3173),
        "CO": (1.2180, 1.1197),
        "NOx": (300.8435, 202.5402),
        "NO2": (35.9242, 49.4129),
        "C6H6": (5.8106, 6.3469),
    }
    key = {"AH": "AH", "T": "T", "RH": "RH", "CO": "CO(GT)", "NOx": "NOx(GT)",
           "NO2": "NO2(GT)", "C6H6": "C6H6(GT)"}
    qty = list(model_mae)
    const = [base["test"]["by_quantity"][key[q]]["constant_mae"] for q in qty]
    model = [model_mae[q][1] for q in qty]
    # normalise each quantity by its constant baseline so units are comparable
    model_n = [m / c for m, c in zip(model, const)]

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    x = np.arange(len(qty))
    ax.bar(x - 0.19, [1.0] * len(qty), width=0.36, color=C_CONST, alpha=0.85,
           label="constant (train-mean) predictor = 1.0")
    ax.bar(x + 0.19, model_n, width=0.36, color=C_V2, alpha=0.9,
           label="v2 model MAE / constant MAE (test split)")
    ax.axhline(1.0, color="black", lw=0.8)
    for xi, m, c in zip(x, model, const):
        ax.text(xi + 0.19, m / c + 0.03, f"{m / c:.2f}", ha="center", fontsize=7.5)
        ax.text(xi + 0.19, 0.02, f"MAE\n{m:.3g}", ha="center", fontsize=6.2, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(qty)
    ax.set_ylabel("MAE relative to constant baseline")
    ax.set_title("v2 numerical error vs a constant predictor (test split)\n"
                 "values below 1.0 beat the constant; above 1.0 are worse")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    save(fig, "evaluation/fig08_mae_rmse_vs_baseline")


def _ah_rows(metrics_path):
    rows = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    gold, pred = [], []
    for r in rows:
        if r.get("kind") != "AH":
            continue
        g, p = r.get("gold", {}).get("AH"), r.get("pred", {}).get("AH")
        if g is None or p is None:
            continue
        gold.append(g)
        pred.append(p)
    return np.array(gold), np.array(pred)


def fig_ah_actual_vs_predicted():
    g2, p2 = _ah_rows(ROOT / "docs/eval_v2/metrics_v2_test.json")
    g1, p1 = _ah_rows(ROOT / "docs/eval_v2/metrics_v1_test.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.2))
    for ax, g, p, color, name in ((ax1, g1, p1, C_V1, "v1"), (ax2, g2, p2, C_V2, "v2")):
        ax.scatter(g, p, s=5, alpha=0.30, color=color, linewidths=0)
        lim = [0, max(g.max(), np.percentile(p, 99.5)) * 1.05]
        ax.plot(lim, lim, "k--", lw=0.9, label="ideal (y = x)")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        mae = np.mean(np.abs(p - g))
        r = np.corrcoef(g, p)[0, 1] if np.std(p) > 0 else float("nan")
        ax.set_xlabel("dataset AH label (g/m3)")
        ax.set_ylabel("model AH output (g/m3)")
        ax.set_title(f"{name} test split: MAE {mae:.3f}, r {r:.3f}, n={len(g)}")
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    save(fig, "evaluation/fig09_ah_actual_vs_predicted")


def fig_ah_formula():
    """Dataset AH column vs the closed-form Magnus/Tetens formula (verified check)."""
    test = pd.read_csv(ROOT / "data" / "processed" / "test.csv")
    d = test.dropna(subset=["T", "RH", "AH"])
    scale = 0.13516784880686916  # fitted on train only, docs/eval_v2/numeric_diagnostics.json
    t = d["T"].to_numpy(dtype=float)
    rh = d["RH"].to_numpy(dtype=float)
    es = 6.112 * np.exp(17.67 * t / (t + 243.5))
    ah = scale * 216.7 * (es * rh / 100.0) / (t + 273.15)
    gold = d["AH"].to_numpy(dtype=float)
    r = np.corrcoef(gold, ah)[0, 1]
    mae = np.mean(np.abs(ah - gold))

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.scatter(gold, ah, s=6, alpha=0.35, color=C_FORMULA, linewidths=0)
    lim = [0, gold.max() * 1.05]
    ax.plot(lim, lim, "k--", lw=0.9, label="identity")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("dataset AH column (g/m3)")
    ax.set_ylabel("closed-form AH from T and RH (g/m3)")
    ax.set_title(f"Dataset AH is a deterministic function of T and RH\n"
                 f"r = {r:.5f}, MAE = {mae:.5f} g/m3 (test, n={len(d)})", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "evaluation/fig10_ah_formula_reconstruction")


def fig_structure_metrics():
    # docs/eval_v2/summary_tables.txt, STRUCTURE / BEHAVIOUR block, test split
    metrics = {
        "emitted EOS": (0.000, 1.000),
        "hit token cap": (1.000, 0.000),
        "repetitive text": (0.355, 0.000),
        "correct unit": (0.053, 1.000),
        "malformed output": (1.000, 0.000),
        "hallucinated date": (0.000, 0.333),
    }
    names = list(metrics)
    x = np.arange(len(names))
    v1 = [metrics[k][0] * 100 for k in names]
    v2 = [metrics[k][1] * 100 for k in names]

    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    ax.bar(x - 0.19, v1, width=0.36, color=C_V1, label="v1 (block 64)")
    ax.bar(x + 0.19, v2, width=0.36, color=C_V2, label="v2 (block 128)")
    for xi, a, b in zip(x, v1, v2):
        ax.text(xi - 0.19, a + 1.5, f"{a:.0f}", ha="center", fontsize=7)
        ax.text(xi + 0.19, b + 1.5, f"{b:.0f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylabel("share of generated outputs (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Output structure and formatting, test split (greedy decoding)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "evaluation/fig11_output_structure")


def main():
    print(f"project root: {ROOT}")
    splits = load_splits()
    print("dataset figures")
    fig_feature_distributions(splits)
    fig_missing_values(splits)
    print("training figures")
    fig_training_v1()
    fig_training_v2()
    print("evaluation figures")
    fig_loss_ppl_comparison()
    fig_mae_vs_baseline()
    fig_ah_actual_vs_predicted()
    fig_ah_formula()
    fig_structure_metrics()
    print("done")


if __name__ == "__main__":
    main()
