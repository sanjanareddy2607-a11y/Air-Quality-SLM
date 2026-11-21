"""
src/test_ollama_chat.py
Exercise the registered Ollama model through the SAME endpoint Ollama Desktop uses
(/api/chat), with the model's own Modelfile defaults (no option overrides).

Usage:  python src/test_ollama_chat.py <label>
Writes docs/chat_results_<label>.json
"""

import sys
import json
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OLLAMA = "http://127.0.0.1:11434"
MODEL = "air-quality-slm:latest"

PROMPTS = [
    "what is the humidity?",
    "What were the temperature and relative humidity?",
    "Air quality measurement report for 15/04/2004 at 12.00.00:",
    "Air quality measurement report for 10/03/2004 at 18.00.00: Carbon Monoxide CO(GT)=",
]


def chat(prompt):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "stream": False}
    req = urllib.request.Request(OLLAMA + "/api/chat", data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    out = {}
    for i, p in enumerate(PROMPTS, 1):
        d = chat(p)
        c = d["message"]["content"]
        cjk = sum(1 for ch in c if ord(ch) > 0x2E80)
        print("\n" + "=" * 80)
        print(f"[{label} | prompt {i}] {p}")
        print(f"  chars={len(c)}  eval_count={d.get('eval_count')}  done={d.get('done_reason')}"
              f"  prompt_eval={d.get('prompt_eval_count')}  CJK_chars={cjk}")
        print(f"  OUTPUT: {c[:400]!r}")
        out[f"prompt_{i}"] = {
            "prompt": p, "output": c, "chars": len(c),
            "eval_count": d.get("eval_count"), "done_reason": d.get("done_reason"),
            "prompt_eval_count": d.get("prompt_eval_count"), "cjk_chars": cjk,
        }
    path = PROJECT_ROOT / "docs" / f"chat_results_{label}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
