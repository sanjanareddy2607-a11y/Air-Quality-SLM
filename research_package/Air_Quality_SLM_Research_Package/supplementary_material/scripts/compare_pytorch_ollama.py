"""
src/compare_pytorch_ollama.py
Side-by-side comparison of the ORIGINAL PyTorch checkpoint and the registered Ollama
model, using identical prompts and identical generation settings.

Purpose: separate "conversion bug" from "training limitation" from "bad generation config".
Nothing here modifies weights, tokenizer or the registered model.
"""

import sys
import json
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import torch
import torch.nn.functional as F
import urllib.request

PROJECT_ROOT = _SRC_DIR.parent
OLLAMA = "http://127.0.0.1:11434"

from evaluate_slm import load_model_and_tokenizer

MAX_SEQ = 64

# --- case A: pure greedy, exactly what the model believes most likely -----------------
GREEDY = dict(temperature=0.0, top_k=1, repeat_penalty=1.0)
# --- case B: the settings currently in the Modelfile ----------------------------------
CURRENT = dict(temperature=0.7, top_k=40, repeat_penalty=1.2)
# --- case C: greedy with a light anti-repetition guard for loop collapse --------------
GREEDY_RP = dict(temperature=0.0, top_k=1, repeat_penalty=1.1)

SETTINGS = [("A greedy", GREEDY), ("B Modelfile", CURRENT), ("C greedy+rp1.1", GREEDY_RP)]


def pytorch_generate(model, tokenizer, prompt, max_new_tokens, **kw):
    ids = tokenizer.encode(prompt, add_special_tokens=False)[-MAX_SEQ:]
    idx = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    # temperature 0 == argmax == top_k 1
    out = model.generate(idx, max_new_tokens=max_new_tokens,
                         temperature=max(kw["temperature"], 1e-5),
                         top_k=kw["top_k"], repetition_penalty=kw["repeat_penalty"])
    new = out[0].tolist()[len(ids):]
    return new, tokenizer.decode(new, skip_special_tokens=False)


def ollama_generate(prompt, max_new_tokens, **kw):
    body = {
        "model": "air-quality-slm:latest",
        "prompt": prompt,
        "raw": True,
        "stream": False,
        "options": {
            "num_ctx": 64,
            "num_predict": max_new_tokens,
            "seed": 42,
            "temperature": kw["temperature"],
            "top_k": kw["top_k"],
            "repeat_penalty": kw["repeat_penalty"],
            "top_p": 1.0,
            "min_p": 0.0,
            "repeat_last_n": 64,
        },
    }
    req = urllib.request.Request(OLLAMA + "/api/generate",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def main():
    model, tokenizer, device = load_model_and_tokenizer()

    prompts = [
        ("1 humidity", "what is the humidity?"),
        ("2 temp/rh", "What were the temperature and relative humidity?"),
        ("3 report", "Air quality measurement report for 15/04/2004 at 12.00.00:"),
        ("4 train AH", "Task: Predict the absolute humidity given the following parameters: "
                       "Date: 11/03/2004, Time: 02.00.00, CO(GT): 0.9, NOx(GT): 45.0, "
                       "NO2(GT): 60.0, Temperature: 13.6 C, Relative Humidity: 48.9 %. Response:"),
        ("5 train T/RH", "Question: What were the ambient temperature and relative humidity "
                         "recorded on 10/03/2004 at 18.00.00? Answer:"),
    ]

    results = {}
    for pname, prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        print("\n" + "#" * 84)
        print(f"### [{pname}] {len(ids)} tokens (fits in 64: {len(ids) <= MAX_SEQ})")
        print(f"### PROMPT: {prompt}")
        print("### IDs:", ids[:20], "..." if len(ids) > 20 else "")

        entry = {"prompt": prompt, "num_tokens": len(ids), "settings": {}}
        for sname, kw in SETTINGS:
            torch.manual_seed(42)
            pt_ids, pt_text = pytorch_generate(model, tokenizer, prompt, 40, **kw)
            o = ollama_generate(prompt, 40, **kw)
            o_text = o.get("response", "")
            print(f"\n--- {sname} ------------------------------------------------------")
            print(f"  PyTorch ids : {pt_ids[:14]}")
            print(f"  PyTorch text: {pt_text!r}")
            print(f"  Ollama  text: {o_text!r}")
            print(f"  Ollama  eval_count={o.get('eval_count')} done={o.get('done_reason')}")
            entry["settings"][sname] = {
                "pytorch_ids": pt_ids,
                "pytorch_text": pt_text,
                "ollama_text": o_text,
                "ollama_eval_count": o.get("eval_count"),
                "ollama_done_reason": o.get("done_reason"),
            }
        results[pname] = entry

    out = PROJECT_ROOT / "docs" / "compare_pytorch_ollama.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
