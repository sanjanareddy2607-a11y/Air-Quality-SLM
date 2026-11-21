"""
src/probe_capability.py
Probe what the trained weights can actually reproduce, per training template, using
greedy decoding on real training records. Also measures the runaway-sampling tail that
produces the multilingual garbage.

Read-only: no weights or tokenizer are modified.
"""

import sys
import json
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import torch
import urllib.request

PROJECT_ROOT = _SRC_DIR.parent
from evaluate_slm import load_model_and_tokenizer

MAX_SEQ = 64
OLLAMA = "http://127.0.0.1:11434"


def pt_greedy(model, tokenizer, prompt, n=40, rp=1.0):
    ids = tokenizer.encode(prompt, add_special_tokens=False)[-MAX_SEQ:]
    idx = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    out = model.generate(idx, max_new_tokens=n, temperature=1.0, top_k=1, repetition_penalty=rp)
    new = out[0].tolist()[len(ids):]
    return tokenizer.decode(new, skip_special_tokens=False)


def ollama(prompt, n=40, **opts):
    body = {"model": "air-quality-slm:latest", "prompt": prompt, "raw": True, "stream": False,
            "options": {"num_ctx": 64, "num_predict": n, "seed": 42, "top_p": 1.0, "min_p": 0.0, **opts}}
    req = urllib.request.Request(OLLAMA + "/api/generate", data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["response"]


def main():
    model, tokenizer, device = load_model_and_tokenizer()

    # Real training records, split at a natural cut point per template.
    rows = []
    with open(PROJECT_ROOT / "data" / "training" / "train_instructions.jsonl", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line)["text"])
            if len(rows) >= 60:
                break

    def first_starting(prefix, n=0):
        c = [r for r in rows if r.startswith(prefix)]
        return c[n]

    cases = []

    t1 = first_starting("Air quality measurement report")
    cases.append(("T1 full record", t1.split("Carbon Monoxide CO(GT)=")[0] + "Carbon Monoxide CO(GT)=",
                  t1, "1. record after CO(GT)="))

    t2 = first_starting("Task: Predict the absolute humidity")
    # shorten params so the prompt fits inside 64 positions
    import re
    d = re.search(r"Date: ([^,]+), Time: ([^,]+)", t2)
    temp = re.search(r"Temperature: ([^,]+) C, Relative Humidity: ([^ ]+) %\.", t2)
    p2 = (f"Task: Predict the absolute humidity given the following parameters: Date: {d.group(1)}, "
          f"Time: {d.group(2)}, Temperature: {temp.group(1)} C, Relative Humidity: {temp.group(2)} %. Response:")
    cases.append(("T2 absolute humidity", p2, t2, "2. AH from shortened params"))

    t3 = first_starting("Question: What were the ambient temperature")
    cases.append(("T3 temp/RH Q&A", t3.split("Answer:")[0] + "Answer:", t3, "3. answer"))

    t4 = first_starting("Query: What are the primary pollutant")
    cases.append(("T4 pollutant query", t4.split("Result:")[0] + "Result:", t4, "4. result"))

    results = {}
    for name, prompt, truth, note in cases:
        ntok = len(tokenizer.encode(prompt, add_special_tokens=False))
        print("\n" + "=" * 88)
        print(f"[{name}] prompt tokens = {ntok} (fits 64: {ntok <= MAX_SEQ})")
        print(f"PROMPT : {prompt}")
        print(f"TRUTH  : {truth[len(prompt):] if truth.startswith(prompt) else truth[-160:]}")
        g = pt_greedy(model, tokenizer, prompt, 48)
        print(f"PT-greedy: {g!r}")
        print(f"OLLAMA   : {ollama(prompt, 48, temperature=0.0, top_k=1, repeat_penalty=1.0)!r}")
        results[name] = {"prompt": prompt, "tokens": ntok, "truth_tail": truth[-220:],
                         "pytorch_greedy": g}

    # Demonstrate the runaway tail: current Modelfile sampling, long generation.
    print("\n" + "=" * 88)
    print("[RUNAWAY] temp 0.7 / top_k 40 / repeat 1.2, 220 tokens, prompt 'what is the humidity?'")
    long_run = ollama("what is the humidity?", 220, temperature=0.7, top_k=40, repeat_penalty=1.2)
    print(repr(long_run))
    non_ascii = sorted({c for c in long_run if ord(c) > 0x2E80})
    print(f"non-ASCII CJK/kana chars present: {non_ascii[:40]}")
    results["runaway"] = {"text": long_run, "cjk": non_ascii}

    print("\n" + "=" * 88)
    print("[RUNAWAY-GREEDY] temp 0 / num_predict 220, same prompt")
    long_g = ollama("what is the humidity?", 220, temperature=0.0, top_k=1, repeat_penalty=1.0)
    print(repr(long_g))
    print(f"CJK chars: {sorted({c for c in long_g if ord(c) > 0x2E80})}")

    out = PROJECT_ROOT / "docs" / "probe_capability.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
