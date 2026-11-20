"""
make_diagrams.py - schematic diagrams for the Air Quality SLM research package.

These are drawing-only schematics: every box and arrow corresponds to a step that
exists in the project source code (named in the figure caption / FIGURE_INDEX.md).
They contain no measured data, so they must not be cited as experimental results.

Usage:
    python research_package/Air_Quality_SLM_Research_Package/figures/scripts/make_diagrams.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

DPI = 300
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})

FIGDIR = Path(__file__).resolve().parents[1]

FILL = "#eaf2fb"
EDGE = "#1f6feb"
FILL_ALT = "#eef7f2"
EDGE_ALT = "#2a9d8f"
FILL_WARN = "#fdf1e7"
EDGE_WARN = "#e76f51"


def save(fig, relpath):
    out = FIGDIR / relpath
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=DPI)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {out.relative_to(FIGDIR)}.png / .pdf")


def box(ax, x, y, w, h, text, fill=FILL, edge=EDGE, fontsize=8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.2, facecolor=fill, edgecolor=edge))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, p1, p2, color="#4a5568", style="-|>", lw=1.2, ls="-", rad=0.0, text=None,
          text_offset=(0, 0.012), fontsize=7):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=11,
                                 linewidth=lw, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if text:
        mx, my = (p1[0] + p2[0]) / 2 + text_offset[0], (p1[1] + p2[1]) / 2 + text_offset[1]
        ax.text(mx, my, text, ha="center", va="center", fontsize=fontsize, color="#2d3748")


def canvas(w=10, h=5.6):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


# ------------------------------------------------------------------ figures ---

def fig_preprocessing():
    fig, ax = canvas(11, 5.2)
    y = 0.72
    h = 0.17
    steps = [
        (0.02, "AirQualityUCI.csv\nraw UCI export\n9,471 x 17"),
        (0.215, "Parse European format\nsep=';'  decimal=','"),
        (0.41, "Drop empty artifact\ncols + 114 blank rows\n9,471 -> 9,357"),
        (0.605, "Sentinel -200 -> NaN\n13 numeric cols\n16,701 NaN cells"),
        (0.80, "Chronological sort\n+ monotonicity check"),
    ]
    for x, text in steps:
        box(ax, x, y, 0.18, h, text, fontsize=7.5)
    for i in range(len(steps) - 1):
        arrow(ax, (steps[i][0] + 0.18, y + h / 2), (steps[i + 1][0], y + h / 2))

    splits = [
        (0.06, "train.csv\n6,549 rows (69.99%)\n10/03/2004 18:00 - 08/12/2004 14:00", FILL_ALT, EDGE_ALT),
        (0.38, "validation.csv\n936 rows (10.00%)\n08/12/2004 15:00 - 16/01/2005 14:00", FILL_ALT, EDGE_ALT),
        (0.70, "test.csv\n1,872 rows (20.01%)\n16/01/2005 15:00 - 04/04/2005 14:00", FILL_ALT, EDGE_ALT),
    ]
    for x, text, f, e in splits:
        box(ax, x, 0.30, 0.26, 0.20, text, fill=f, edge=e, fontsize=7.5)
        arrow(ax, (0.89, y), (x + 0.13, 0.50), color="#4a5568", rad=0.0)

    box(ax, 0.06, 0.05, 0.90, 0.14,
        "Instruction construction (src/prepare_training_data.py): 4 text templates per row "
        "(full report / absolute-humidity task / temperature-RH question / pollutant query)\n"
        "37,428 samples -> Qwen2.5-0.5B BPE tokenizer -> 4,344,159 tokens in uint32 memmap files",
        fill=FILL_WARN, edge=EDGE_WARN, fontsize=7.5)
    ax.text(0.5, 0.965, "Data preprocessing workflow (schematic)", ha="center", fontsize=10.5,
            fontweight="bold")
    ax.text(0.5, 0.92, "counts and date ranges verified from data/processed/*.csv and docs/VALIDATION_REPORT.md",
            ha="center", fontsize=7, color="#4a5568")
    save(fig, "architecture/fig03_preprocessing_workflow")


def fig_model_architecture():
    fig, ax = canvas(10, 5.4)
    box(ax, 0.03, 0.80, 0.20, 0.13, "input text\n<= 128 tokens", fill="#f7f7fa", edge="#6b7280")
    box(ax, 0.27, 0.80, 0.24, 0.13, "Qwen BPE tokenizer\nvocab 151,665")
    box(ax, 0.55, 0.80, 0.19, 0.13, "token embedding\n151,665 x 128\n(x sqrt(128))")
    box(ax, 0.78, 0.80, 0.19, 0.13, "RoPE positions\n(dim 32, theta 1e4)")

    arrow(ax, (0.23, 0.865), (0.27, 0.865))
    arrow(ax, (0.51, 0.865), (0.55, 0.865))
    arrow(ax, (0.74, 0.865), (0.78, 0.865))

    # transformer stack
    box(ax, 0.06, 0.26, 0.44, 0.48, "", fill="#ffffff", edge="#1f6feb")
    ax.text(0.28, 0.705, "Transformer block  x 4", ha="center", fontsize=8.5, fontweight="bold")
    inner = [
        (0.555, "RMSNorm -> GQA attention\n4 query / 2 KV heads, head_dim 32\nper-head Q/K RMSNorm before RoPE"),
        (0.455, "residual add"),
        (0.315, "RMSNorm -> SwiGLU feed-forward\n512 intermediate\nresidual add"),
    ]
    for yy, text in inner:
        box(ax, 0.09, yy, 0.38, 0.075, text, fontsize=7, fill=FILL)
    arrow(ax, (0.28, 0.80), (0.28, 0.635))
    arrow(ax, (0.28, 0.555), (0.28, 0.535), lw=0.9)
    arrow(ax, (0.28, 0.455), (0.28, 0.395), lw=0.9)

    box(ax, 0.56, 0.50, 0.20, 0.12, "final RMSNorm\n(eps 1e-6)")
    box(ax, 0.56, 0.30, 0.20, 0.12, "LM head\n(tied embeddings)")
    box(ax, 0.80, 0.40, 0.17, 0.12, "next-token\ndistribution", fill=FILL_ALT, edge=EDGE_ALT)
    arrow(ax, (0.50, 0.40), (0.56, 0.56))
    arrow(ax, (0.66, 0.50), (0.66, 0.42))
    arrow(ax, (0.76, 0.36), (0.80, 0.46))

    ax.text(0.5, 0.20,
            "20,397,568 parameters (39.81 M stored in PyTorch, embeddings tied)   |   "
            "d_model 128   |   4 layers   |   4 query / 2 KV heads   |   d_ff 512",
            ha="center", fontsize=8)
    ax.text(0.5, 0.12,
            "Training objective: next-token cross-entropy   |   AdamW (lr 1e-3, betas 0.9/0.95, wd 0.1), "
            "warmup 10% + cosine decay, grad-clip 0.5",
            ha="center", fontsize=7.5, color="#2d3748")
    ax.text(0.5, 0.05, "v1 context 64 tokens; v2 continued training at 128 tokens (same architecture)",
            ha="center", fontsize=7.5, color="#4a5568")
    ax.text(0.5, 0.965, "Model architecture (schematic, from config.json and src/train_slm.py)",
            ha="center", fontsize=10.5, fontweight="bold")
    save(fig, "architecture/fig04_model_architecture")


def fig_conversion():
    fig, ax = canvas(11, 4.6)
    top = [
        (0.015, "pytorch_model.bin\n81,608,145 bytes\nSHA256 77BE6061..."),
        (0.215, "config.json\n128 hidden / 4 layers /\n4 heads / 2 KV / block 64"),
        (0.415, "tokenizer.json\n151,665 tokens\n151,387 merges"),
        (0.615, "export script\nconvert_to_gguf.py\n(or export_gguf_from_checkpoint.py)"),
        (0.815, "GGUF F32\nqwen3 arch\ncontext 64 / 128"),
    ]
    for x, text in top:
        box(ax, x, 0.66, 0.175, 0.19, text, fontsize=7.2)
    for i in range(4):
        arrow(ax, (top[i][0] + 0.175, 0.755), (top[i + 1][0], 0.755))

    box(ax, 0.05, 0.38, 0.40, 0.16,
        "tensor mapping\nattn_q/k/v/output, attn_q_norm, attn_k_norm,\nffn_gate/up/down, norms\n"
        "token_embd x sqrt(128); output unscaled", fontsize=7.2)
    box(ax, 0.55, 0.38, 0.40, 0.16,
        "chat template replaced with raw passthrough\n(no ChatML markers, model trained on raw text)",
        fontsize=7.2, fill=FILL_WARN, edge=EDGE_WARN)
    arrow(ax, (0.90, 0.66), (0.90, 0.54))
    arrow(ax, (0.70, 0.66), (0.70, 0.54), lw=1.0)

    box(ax, 0.05, 0.10, 0.40, 0.15,
        "verification: llama.cpp reports n_ctx_train, tokenizer IDs\n13/13 identical, greedy output == PyTorch",
        fontsize=7.2, fill=FILL_ALT, edge=EDGE_ALT)
    box(ax, 0.55, 0.10, 0.40, 0.15,
        "ollama create <name> -f Modelfile\n(no quantisation step was performed: F32 kept)",
        fontsize=7.2, fill=FILL_ALT, edge=EDGE_ALT)
    arrow(ax, (0.25, 0.38), (0.25, 0.25))
    arrow(ax, (0.75, 0.38), (0.75, 0.25))
    ax.text(0.5, 0.965, "PyTorch-to-GGUF conversion workflow (schematic)",
            ha="center", fontsize=10.5, fontweight="bold")
    save(fig, "architecture/fig12_conversion_workflow")


def fig_deployment():
    fig, ax = canvas(10.5, 4.8)
    box(ax, 0.03, 0.55, 0.24, 0.26,
        "Ollama Desktop / API client\n127.0.0.1:11434", fill="#f7f7fa", edge="#6b7280")
    box(ax, 0.34, 0.55, 0.30, 0.26,
        "Python proxy (ollama_proxy.py)\ndomain routing, retrieval,\nhumidity calculation",
        fill=FILL, edge=EDGE, bold=False)
    box(ax, 0.71, 0.55, 0.26, 0.26,
        "ollama serve\n127.0.0.1:11435\nair-quality-slm-v1/v2/v3", fill=FILL_ALT, edge=EDGE_ALT)
    arrow(ax, (0.27, 0.72), (0.34, 0.72), text="request", text_offset=(0, 0.03))
    arrow(ax, (0.34, 0.62), (0.27, 0.62), text="response + X-AQ-Proxy", text_offset=(0, -0.03))
    arrow(ax, (0.64, 0.72), (0.71, 0.72), text="/api/generate", text_offset=(0, 0.03))

    box(ax, 0.03, 0.14, 0.20, 0.20, "data/processed/*.csv\n(9,357 records)", fontsize=7.5)
    box(ax, 0.27, 0.14, 0.20, 0.20, "humidity_calculator.py\nTetens + fitted scale", fontsize=7.5)
    box(ax, 0.51, 0.14, 0.20, 0.20, "domain_guard.py\nintent / OOD routing", fontsize=7.5)
    box(ax, 0.75, 0.14, 0.22, 0.20, "proxy_watchdog.py\nsupervision + recovery", fontsize=7.5)
    for x in (0.13, 0.37, 0.61):
        arrow(ax, (x, 0.34), (x, 0.55), ls="--", lw=0.9)
    ax.text(0.5, 0.965, "Ollama deployment topology (schematic, as running on 2026-11-19)",
            ha="center", fontsize=10.5, fontweight="bold")
    ax.text(0.5, 0.05, "Ports are localhost-only. 11434 is the proxy; Ollama's own server is moved to 11435.",
            ha="center", fontsize=7.5, color="#4a5568")
    save(fig, "architecture/fig13_ollama_deployment")


def fig_routing():
    fig, ax = canvas(9.5, 5.6)
    box(ax, 0.30, 0.86, 0.40, 0.10, "user message", fill="#f7f7fa", edge="#6b7280", bold=True)
    box(ax, 0.30, 0.68, 0.40, 0.11, "domain_guard.is_greeting()", fontsize=8)
    box(ax, 0.30, 0.50, 0.40, 0.11, "domain_guard.is_out_of_domain()", fontsize=8)
    box(ax, 0.30, 0.32, 0.40, 0.11, "domain_guard.detect_intent()", fontsize=8)
    arrow(ax, (0.50, 0.86), (0.50, 0.79))
    arrow(ax, (0.50, 0.68), (0.50, 0.61))
    arrow(ax, (0.50, 0.50), (0.50, 0.43))

    box(ax, 0.02, 0.66, 0.22, 0.13, "MODE A\ngreeting -> welcome text\n(no model call)",
        fontsize=7.2, fill=FILL_ALT, edge=EDGE_ALT)
    box(ax, 0.02, 0.46, 0.22, 0.13, "MODE C\nout-of-domain -> refusal\n(no model call)",
        fontsize=7.2, fill=FILL_WARN, edge=EDGE_WARN)
    box(ax, 0.02, 0.20, 0.30, 0.16,
        "MODE B1: retrieval\ndata_retrieval.find_record()\nexact date + time match",
        fontsize=7.2, fill=FILL)
    box(ax, 0.38, 0.20, 0.30, 0.16,
        "MODE B2: AH calculation\nhumidity_calculator\n.calculate_absolute_humidity()",
        fontsize=7.2, fill=FILL)
    box(ax, 0.72, 0.20, 0.26, 0.16,
        "MODE B0: model call\nollama_client.generate\n(raw template completion)",
        fontsize=7.2, fill="#f7f7fa", edge="#6b7280")

    arrow(ax, (0.30, 0.725), (0.24, 0.725), text="yes", text_offset=(-0.01, 0.03))
    arrow(ax, (0.30, 0.525), (0.24, 0.525), text="yes", text_offset=(-0.01, 0.03))
    arrow(ax, (0.40, 0.32), (0.32, 0.30), text="retrieval", text_offset=(0.0, 0.035))
    arrow(ax, (0.50, 0.32), (0.53, 0.36), text="ah_calculation", text_offset=(0.07, 0.0))
    arrow(ax, (0.70, 0.32), (0.85, 0.36), text="unknown /\nmodel path", text_offset=(0.05, 0.02))

    ax.text(0.5, 0.965, "User query routing (schematic, from domain_guard.py and app.py)",
            ha="center", fontsize=10.5, fontweight="bold")
    ax.text(0.5, 0.08,
            "Numbers returned by B1 and B2 come from Python (dataset lookup and closed-form formula).\n"
            "They are not neural-network predictions and must not be reported as such.",
            ha="center", fontsize=7.5, color="#8a1c1c")
    save(fig, "architecture/fig14_query_routing")


def fig_system_architecture():
    fig, ax = canvas(10.5, 6.0)
    # layer bands
    for y, label in ((0.79, "presentation"), (0.52, "application"), (0.16, "data and artefacts")):
        ax.add_patch(plt.Rectangle((0.01, y - 0.015), 0.98, 0.235, facecolor="#fbfcfe",
                                   edgecolor="#d7dde5", lw=0.8, zorder=0))
        ax.text(0.025, y + 0.19, label, fontsize=7.5, color="#6b7280", style="italic")

    box(ax, 0.06, 0.84, 0.26, 0.13, "Ollama Desktop\n(model selector)", fontsize=7.5)
    box(ax, 0.37, 0.84, 0.26, 0.13, "terminal CLI / API client\ncurl /api/chat", fontsize=7.5)
    box(ax, 0.68, 0.84, 0.26, 0.13, "app.py interactive shell\nchat_loop()", fontsize=7.5)

    box(ax, 0.06, 0.58, 0.41, 0.14, "ollama_proxy.py  :11434\nHTTP routing + X-AQ-Proxy header", fontsize=7.5)
    box(ax, 0.53, 0.58, 0.41, 0.14, "ollama_client.py\nHTTP client to :11435", fontsize=7.5)

    box(ax, 0.06, 0.37, 0.17, 0.13, "domain_guard.py\nrouting", fontsize=7)
    box(ax, 0.25, 0.37, 0.17, 0.13, "data_retrieval.py\nCSV lookup", fontsize=7)
    box(ax, 0.44, 0.37, 0.19, 0.13, "humidity_calculator.py\nTetens + scale", fontsize=7)
    box(ax, 0.65, 0.37, 0.29, 0.13, "ollama serve :11435\nqwen3 GGUF, ctx 64/128", fontsize=7,
        fill=FILL_ALT, edge=EDGE_ALT)

    for x in (0.19, 0.50, 0.81):
        arrow(ax, (x, 0.84), (x, 0.72))
    arrow(ax, (0.47, 0.65), (0.53, 0.65), text="forward", text_offset=(0, 0.02))
    arrow(ax, (0.74, 0.58), (0.74, 0.50), text="/api/generate", text_offset=(0.12, 0.0))

    box(ax, 0.06, 0.06, 0.20, 0.13, "data/raw\nAirQualityUCI.csv", fontsize=7)
    box(ax, 0.28, 0.06, 0.20, 0.13, "data/processed\ntrain/val/test.csv", fontsize=7)
    box(ax, 0.50, 0.06, 0.20, 0.13, "models/\nPyTorch + GGUF", fontsize=7)
    box(ax, 0.72, 0.06, 0.22, 0.13, "docs/\neval JSON + logs", fontsize=7)
    arrow(ax, (0.34, 0.19), (0.34, 0.37), ls="--", lw=0.9)
    ax.text(0.5, 0.975, "Complete system architecture (schematic)", ha="center",
            fontsize=10.5, fontweight="bold")
    save(fig, "architecture/fig15_system_architecture")


def main():
    print("diagrams")
    fig_preprocessing()
    fig_model_architecture()
    fig_conversion()
    fig_deployment()
    fig_routing()
    fig_system_architecture()
    print("done")


if __name__ == "__main__":
    main()
