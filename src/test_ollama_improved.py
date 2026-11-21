"""Compare the two registered Ollama models on the same real training-template prompts."""
import json
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OLLAMA = "http://127.0.0.1:11434"

lines = [json.loads(l)["text"] for l in
         open(PROJECT_ROOT / "data/training/val_instructions.jsonl", encoding="utf-8")][:4]
splitters = ["Carbon Monoxide CO(GT)=", "Response:", "Answer:", "Result:"]
names = ["REPORT", "AH", "TEMP_RH", "POLLUTANT"]

cases = []
for k in range(4):
    t = lines[k]
    cut = t.index(splitters[k]) + len(splitters[k])
    cases.append((names[k], t[:cut], t[cut:].split("\n")[0]))


def chat(model, prompt, num_ctx):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False,
            "options": {"num_ctx": num_ctx, "num_predict": 40, "seed": 42, "temperature": 0,
                        "top_k": 1, "repeat_penalty": 1.0, "top_p": 1.0, "min_p": 0.0}}
    req = urllib.request.Request(OLLAMA + "/api/chat", data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))


out = {}
for name, prompt, expected in cases:
    print("\n" + "=" * 92)
    print(f"[{name}]  expected: {expected[:80]!r}")
    out[name] = {"prompt": prompt, "expected": expected, "models": {}}
    for label, model, ctx in (("original", "air-quality-slm:latest", 64),
                              ("improved", "air-quality-slm-improved:latest", 128)):
        d = chat(model, prompt, ctx)
        txt = d["message"]["content"]
        print(f"  {label:<9} prompt_eval={d.get('prompt_eval_count'):<4} "
              f"eval={d.get('eval_count'):<3} {txt!r}")
        out[name]["models"][label] = {"text": txt,
                                      "prompt_eval_count": d.get("prompt_eval_count"),
                                      "eval_count": d.get("eval_count"),
                                      "done": d.get("done_reason")}

path = PROJECT_ROOT / "docs" / "ollama_before_after.json"
json.dump(out, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"\nWrote {path}")
