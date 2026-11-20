"""
src/plot_loss.py
Turn docs/training_log_finetune.jsonl into a CSV and a dependency-free SVG loss curve.

No plotting library is installed in this project, and adding one just for a chart is not
worth a new dependency, so the SVG is written directly.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG = PROJECT_ROOT / "docs" / "training_log_finetune.jsonl"
CSV = PROJECT_ROOT / "docs" / "loss_curve.csv"
SVG = PROJECT_ROOT / "docs" / "loss_curve.svg"

W, H = 760, 420
PAD_L, PAD_R, PAD_T, PAD_B = 64, 24, 28, 46


def main():
    rows, events = [], []
    for line in open(LOG, encoding="utf-8"):
        r = json.loads(line)
        if r.get("event") == "eval":
            rows.append(r)
        events.append(r)

    baseline = next((e.get("baseline_val") for e in events if e.get("event") == "start"), None)
    if not rows:
        raise SystemExit("no eval records in log")

    with open(CSV, "w", encoding="utf-8") as f:
        f.write("step,train_loss_ema,val_loss,val_ppl,lr,tokens_seen,epochs,elapsed_s\n")
        for r in rows:
            f.write(f"{r['step']},{r['train_loss_ema']},{r['val_loss']},{r['val_ppl']},"
                    f"{r['lr']},{r['tokens_seen']},{r['epochs']},{r['elapsed_s']}\n")

    xs = [r["step"] for r in rows]
    series = {
        "val loss": ([r["val_loss"] for r in rows], "#1f77b4"),
        "train loss (ema)": ([r["train_loss_ema"] for r in rows], "#ff7f0e"),
    }
    if baseline is not None:
        series["baseline val (original checkpoint)"] = (None, "#888888")

    vals = [v for v, _ in series.values() if v]
    ymin = min(min(v) for v in vals)
    ymax = max(max(v) for v in vals)
    if baseline is not None:
        ymin, ymax = min(ymin, baseline), max(ymax, baseline)
    pad = (ymax - ymin) * 0.12 or 0.1
    ymin, ymax = ymin - pad, ymax + pad
    xmax = max(xs) or 1

    def px(x):
        return PAD_L + (x / xmax) * (W - PAD_L - PAD_R)

    def py(y):
        return H - PAD_B - ((y - ymin) / (ymax - ymin)) * (H - PAD_T - PAD_B)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="Segoe UI,Arial,sans-serif" font-size="12">',
             f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
             f'<text x="{PAD_L}" y="18" font-size="14" font-weight="600">'
             f'Air Quality SLM — continued training loss</text>']

    # y grid
    for i in range(6):
        y = ymin + (ymax - ymin) * i / 5
        Y = py(y)
        parts.append(f'<line x1="{PAD_L}" y1="{Y:.1f}" x2="{W - PAD_R}" y2="{Y:.1f}" '
                     f'stroke="#e8e8e8"/>')
        parts.append(f'<text x="{PAD_L - 8}" y="{Y + 4:.1f}" text-anchor="end" '
                     f'fill="#666">{y:.2f}</text>')
    # x ticks
    for i in range(6):
        x = xmax * i / 5
        X = px(x)
        parts.append(f'<text x="{X:.1f}" y="{H - PAD_B + 18}" text-anchor="middle" '
                     f'fill="#666">{int(x)}</text>')
    parts.append(f'<text x="{(PAD_L + W - PAD_R) / 2:.0f}" y="{H - 8}" text-anchor="middle" '
                 f'fill="#444">training step</text>')

    if baseline is not None:
        Y = py(baseline)
        parts.append(f'<line x1="{PAD_L}" y1="{Y:.1f}" x2="{W - PAD_R}" y2="{Y:.1f}" '
                     f'stroke="#888" stroke-width="1.5" stroke-dasharray="6 4"/>')

    for label, (ys, color) in series.items():
        if not ys:
            continue
        pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in zip(xs, ys):
            parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3" fill="{color}"/>')

    # legend
    lx, ly = PAD_L + 8, PAD_T + 6
    for label, (ys, color) in series.items():
        dash = ' stroke-dasharray="6 4"' if ys is None else ""
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 22}" y2="{ly}" stroke="{color}" '
                     f'stroke-width="2.5"{dash}/>')
        parts.append(f'<text x="{lx + 30}" y="{ly + 4}" fill="#333">{label}</text>')
        ly += 18

    parts.append("</svg>")
    SVG.write_text("\n".join(parts), encoding="utf-8")

    print(f"{'step':>6} {'train':>8} {'val':>8} {'ppl':>7} {'tokens':>10} {'epoch':>7}")
    for r in rows:
        print(f"{r['step']:>6} {r['train_loss_ema']:>8.4f} {r['val_loss']:>8.4f} "
              f"{r['val_ppl']:>7.3f} {r['tokens_seen']:>10,} {r['epochs']:>7.3f}")
    if baseline is not None:
        print(f"\nbaseline (original checkpoint, same eval): {baseline:.4f}")
    print(f"\nwrote {CSV}\nwrote {SVG}")


if __name__ == "__main__":
    main()
