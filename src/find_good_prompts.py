"""
src/find_good_prompts.py
Screen candidate demonstration prompts: how many tokens each costs (must fit num_ctx 64)
and what the registered Ollama model actually returns under greedy decoding.
"""

import json
import urllib.request
from pathlib import Path

import sys
_SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC_DIR))
PROJECT_ROOT = _SRC_DIR.parent

from transformers import AutoTokenizer

MODEL_DIR = PROJECT_ROOT / "models" / "air_quality_slm"
OLLAMA = "http://127.0.0.1:11434"

CANDIDATES = [
    "Task: Predict the absolute humidity given the following parameters: Date: 11/03/2004, Time: 02.00.00, CO(GT): 0.9, NOx(GT): 45.0, NO2(GT): 60.0. Response:",
    "Task: Predict the absolute humidity given the following parameters: Temperature: 13.6 C, Relative Humidity: 48.9 %. Response:",
    "Task: Predict the absolute humidity: Date: 11/03/2004, Time: 02.00.00, CO(GT): 0.9, NOx(GT): 45.0, NO2(GT): 60.0. Response:",
    "Question: What is the absolute humidity? Answer:",
    "Absolute Humidity= ? given Temperature=13.6 C, Relative Humidity=48.9 %.",
    "Air quality measurement report for 10/03/2004 at 18.00.00: Carbon Monoxide CO(GT)=",
    "Air quality measurement report for 10/03/2004 at 18.00.00: Temperature=",
    "Question: What were the ambient temperature and relative humidity recorded on 10/03/2004 at 18.00.00? Answer:",
    "What were the temperature and relative humidity recorded on 10/03/2004 at 18.00.00?",
    "Query: What are the primary pollutant concentrations (CO, NOx, NO2) on 10/03/2004 at 18.00.00? Result:",
]


def chat(prompt):
    body = {"model": "air-quality-slm:latest", "messages": [{"role": "user", "content": prompt}],
            "stream": False}
    req = urllib.request.Request(OLLAMA + "/api/chat", data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    rows = []
    for p in CANDIDATES:
        n = len(tok.encode(p, add_special_tokens=False))
        d = chat(p)
        c = d["message"]["content"]
        print("\n" + "=" * 84)
        print(f"tokens={n:>3}  fits64={n <= 64}  eval={d.get('eval_count')}  done={d.get('done_reason')}")
        print(f"PROMPT: {p}")
        print(f"OUTPUT: {c!r}")
        rows.append({"prompt": p, "tokens": n, "fits": n <= 64, "output": c,
                     "eval_count": d.get("eval_count")})
    with open(PROJECT_ROOT / "docs" / "prompt_screening.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
