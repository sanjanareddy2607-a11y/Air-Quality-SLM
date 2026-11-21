"""
src/diagnose_absolute_humidity.py
Controlled diagnosis of ONE training template: absolute-humidity prediction.

The template is taken verbatim from src/prepare_training_data.py (row_to_texts, text #2)
by importing that function -- it is not re-typed or approximated.

An actual validation row is converted to its training text, split into the exact input
(everything up to and including "Response:") and the exact expected completion.
That identical input string is then sent to:
  * the original PyTorch model loaded from pytorch_model.bin
  * the converted GGUF, through llama.cpp (same engine Ollama uses) -- tokens + logprobs
  * the registered Ollama model

Read-only: nothing is retrained, re-exported or overwritten.
"""

import sys
import json
import math
import subprocess
import urllib.request
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC_DIR))
PROJECT_ROOT = _SRC_DIR.parent

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from prepare_training_data import row_to_texts
from evaluate_slm import load_model_and_tokenizer
from train_slm import compute_rope_params

LLAMA_PORT = 8091
OLLAMA = "http://127.0.0.1:11434"
GEN_TOKENS = 40


# ----------------------------------------------------------------------------- template
def get_ah_example(split="validation", row_idx=0):
    """Exact training text for one real dataset row, via the real formatter."""
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / f"{split}.csv")
    row = df.iloc[row_idx]
    texts = row_to_texts(row)
    ah_text = [t for t in texts if t.startswith("Task: Predict the absolute humidity")][0]
    return row, ah_text


def physical_ah(t_celsius, rh_percent):
    """Absolute humidity g/m3 from temperature and relative humidity (Magnus formula)."""
    es = 6.112 * math.exp((17.62 * t_celsius) / (243.12 + t_celsius))   # hPa
    e = es * rh_percent / 100.0
    return 216.7 * e / (273.15 + t_celsius)


# ----------------------------------------------------------------------------- runners
def _widen_rope(model, ctx=1024):
    """The checkpoint stores a 64-row cos/sin table, so it cannot even generate one token
    past position 64 without this. Diagnostic only: positions >= 64 are untrained."""
    cos, sin = compute_rope_params(head_dim=32, context_length=ctx)
    model.cos, model.sin = cos, sin


def pt_forward_logits(model, ids, extend_rope_to=1024):
    if extend_rope_to is not None:
        _widen_rope(model, extend_rope_to)
    idx = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(idx)
    return logits[0, -1, :]


def pt_greedy(model, ids, n, extend_rope_to=1024):
    if extend_rope_to is not None:
        _widen_rope(model, extend_rope_to)
    idx = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    out = []
    with torch.no_grad():
        for _ in range(n):
            logits, _ = model(idx)
            nxt = int(torch.argmax(logits[0, -1]))
            out.append(nxt)
            idx = torch.cat([idx, torch.tensor([[nxt]])], dim=1)
    return out


def llama_completion(prompt, n_predict=GEN_TOKENS, n_probs=10):
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "top_k": 1,
            "repeat_penalty": 1.0, "n_probs": n_probs, "seed": 42, "cache_prompt": False}
    req = urllib.request.Request(f"http://127.0.0.1:{LLAMA_PORT}/completion",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def ollama_chat(prompt, num_ctx, num_predict=GEN_TOKENS, temperature=0.0, top_k=1):
    body = {"model": "air-quality-slm:latest",
            "messages": [{"role": "user", "content": prompt}], "stream": False,
            "options": {"num_ctx": num_ctx, "num_predict": num_predict, "seed": 42,
                        "temperature": temperature, "top_k": top_k,
                        "repeat_penalty": 1.0, "top_p": 1.0, "min_p": 0.0}}
    req = urllib.request.Request(OLLAMA + "/api/chat", data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def main():
    report = {}

    # ------------------------------------------------------------------ 1. template
    print("=" * 90)
    print("1. EXACT TEMPLATE (verbatim from src/prepare_training_data.py, row_to_texts text #2)")
    print("=" * 90)
    print('''f"Task: Predict the absolute humidity given the following parameters: "
f"Date: {date}, Time: {time}, CO(GT): {co}, NOx(GT): {nox}, NO2(GT): {no2}, "
f"Temperature: {t} C, Relative Humidity: {rh} %. "
f"Response: The absolute humidity is {ah} g/m3.\\n<|endoftext|>\\n"''')

    # ------------------------------------------------------------------ 2. example
    row, ah_text = get_ah_example("validation", 0)
    print("\n" + "=" * 90)
    print("2. REAL VALIDATION EXAMPLE (data/processed/validation.csv row 0)")
    print("=" * 90)
    print(f"row      : Date={row['Date']} Time={row['Time']} CO(GT)={row['CO(GT)']} "
          f"NOx(GT)={row['NOx(GT)']} NO2(GT)={row['NO2(GT)']} T={row['T']} RH={row['RH']} AH={row['AH']}")
    print(f"FULL TRAINING TEXT:\n{ah_text!r}")

    marker = "Response:"
    idx = ah_text.index(marker) + len(marker)
    prompt = ah_text[:idx]
    expected = ah_text[idx:]
    print(f"\nINPUT (sent to both models):\n{prompt!r}")
    print(f"\nEXPECTED COMPLETION:\n{expected!r}")

    dataset_ah = float(row["AH"])
    computed_ah = physical_ah(float(row["T"]), float(row["RH"]))
    print(f"\nAH from dataset column        : {dataset_ah:.4f} g/m3")
    print(f"AH computed from T and RH     : {computed_ah:.4f} g/m3  (Magnus formula)")
    print(f"AH written into training text : {expected.split('is ')[1].split(' g/m3')[0]} g/m3")

    # Is the dataset's AH column consistent with the standard formula at all?
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "validation.csv").dropna(subset=["T", "RH", "AH"])
    phys = df.apply(lambda r: physical_ah(float(r["T"]), float(r["RH"])), axis=1)
    ratio = (df["AH"] / phys).astype(float)
    corr = float(np.corrcoef(df["AH"].astype(float), phys.astype(float))[0, 1])
    print(f"\ndataset AH column vs Magnus formula over {len(df)} validation rows:")
    print(f"  Pearson r              : {corr:.4f}")
    print(f"  ratio AH_col/AH_phys   : median {ratio.median():.4f}  "
          f"p10 {ratio.quantile(0.1):.4f}  p90 {ratio.quantile(0.9):.4f}")
    print(f"  dataset AH range       : {df['AH'].min():.3f} .. {df['AH'].max():.3f} g/m3")
    print(f"  Magnus  AH range       : {phys.min():.3f} .. {phys.max():.3f} g/m3")
    print("  => the AH column scales with T/RH but is ~"
          f"{1 / ratio.median():.1f}x smaller than a standard absolute-humidity calculation.")

    # ------------------------------------------------------------------ 3. lengths
    model, tokenizer, device = load_model_and_tokenizer()
    p_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(ah_text, add_special_tokens=False)
    print("\n" + "=" * 90)
    print("3. DOES THE TEMPLATE FIT THE TRAINED CONTEXT?")
    print("=" * 90)
    print(f"input tokens            : {len(p_ids)}")
    print(f"full sample tokens      : {len(full_ids)}")
    print(f"model block_size        : {model.max_seq_len}   (cfg max_position_embeddings=64)")
    print(f"input fits in 64?       : {len(p_ids) <= 64}")
    print(f"full sample fits in 64? : {len(full_ids) <= 64}")

    # How often did training actually see the "Response:" -> answer boundary?
    train = np.memmap(PROJECT_ROOT / "training" / "data_bin" / "train.bin", dtype=np.uint32, mode="r")
    n = len(train)
    # Use the exact token pair that ends the prompt -- "Response" in context is " Response".
    resp_id, colon_id = p_ids[-2], p_ids[-1]
    print(f"boundary token pair: {resp_id} {tokenizer.decode([resp_id])!r} + "
          f"{colon_id} {tokenizer.decode([colon_id])!r}")
    arr = np.asarray(train)
    occ = int(np.sum((arr[:-1] == resp_id) & (arr[1:] == colon_id)))
    n_windows = n - 64
    windows_seen = 150 * 8
    print(f"\ntrain corpus tokens     : {n:,}")
    print(f"'Response:' occurrences : {occ:,}  ({occ / n * 100:.3f}% of positions)")
    print(f"possible 64-token windows:{n_windows:,}")
    print(f"windows a random 64-token sample ends on 'Response:' : {occ / n_windows * 100:.3f}%")
    print(f"training windows actually drawn (150 iters x batch 8): {windows_seen:,}")
    print(f"=> expected times the model was trained on P(answer | 'Response:') : "
          f"{windows_seen * occ / n_windows:.1f}")
    print(f"=> total tokens the model ever saw: {windows_seen * 64:,} "
          f"({windows_seen * 64 / n * 100:.2f}% of the corpus)")

    report["template_lengths"] = {"input_tokens": len(p_ids), "full_tokens": len(full_ids),
                                  "block_size": 64,
                                  "response_occurrences": occ,
                                  "expected_boundary_windows": windows_seen * occ / n_windows,
                                  "corpus_coverage_pct": windows_seen * 64 / n * 100}

    # ------------------------------------------------------------------ 4. PyTorch
    print("\n" + "=" * 90)
    print("4. ORIGINAL PYTORCH MODEL (pytorch_model.bin), greedy, 40 new tokens")
    print("=" * 90)

    # (a) exactly as trained context allows: truncate to the last 64 tokens
    trunc = p_ids[-64:]
    pt_a = pt_greedy(model, trunc, GEN_TOKENS)
    print(f"(a) truncated to last 64 tokens ({len(trunc)}):")
    print(f"    ids  : {pt_a}")
    print(f"    text : {tokenizer.decode(pt_a)!r}")

    # (b) diagnostic only: full 89-token input with an extended RoPE table
    logits = pt_forward_logits(model, p_ids, extend_rope_to=512)
    lp = F.log_softmax(logits, dim=-1)
    pr = F.softmax(logits, dim=-1)
    vals, idxs = torch.topk(pr, 10)
    top10 = [{"id": int(i), "token": tokenizer.decode([int(i)]),
              "prob": round(float(v), 6), "logprob": round(float(lp[i]), 6)}
             for v, i in zip(vals, idxs)]
    print(f"\n(b) full {len(p_ids)}-token input, RoPE extended to 512 (DIAGNOSTIC ONLY):")
    print("    first-token top-10:")
    for t in top10:
        print(f"      {t['id']:>7} p={t['prob']:<9.6f} logp={t['logprob']:<11.6f} {t['token']!r}")
    pt_b = pt_greedy(model, p_ids, GEN_TOKENS, extend_rope_to=512)
    print(f"    ids  : {pt_b}")
    print(f"    text : {tokenizer.decode(pt_b)!r}")

    # teacher-forced probability of the TRUE answer tokens
    true_ids = tokenizer.encode(expected, add_special_tokens=False)
    _widen_rope(model, 1024)
    with torch.no_grad():
        seq = torch.tensor([p_ids + true_ids[:-1]], dtype=torch.long)
        lg, _ = model(seq)
    print(f"\n    teacher-forced on the true completion {true_ids}:")
    tot = 0.0
    detail = []
    for k, tid in enumerate(true_ids):
        row_logits = lg[0, len(p_ids) - 1 + k, :]
        row_lp = F.log_softmax(row_logits, dim=-1)
        p = float(row_lp[tid])
        tot += p
        detail.append({"id": tid, "token": tokenizer.decode([tid]), "logprob": round(p, 6)})
        print(f"      {tid:>7} {tokenizer.decode([tid])!r:<18} logp={p:>9.4f} p={math.exp(p):.5f}")
    print(f"    total logprob of true answer: {tot:.3f}  (mean per token {tot / len(true_ids):.3f})")

    report["pytorch"] = {"truncated_ids": pt_a, "truncated_text": tokenizer.decode(pt_a),
                         "full_top10": top10, "full_ids": pt_b,
                         "full_text": tokenizer.decode(pt_b),
                         "true_answer_ids": true_ids, "true_answer_logprobs": detail,
                         "true_answer_total_logprob": tot}

    # ------------------------------------------------------------------ 5. GGUF / llama.cpp
    print("\n" + "=" * 90)
    print("5. CONVERTED GGUF via llama.cpp (same engine Ollama uses) — tokens + probabilities")
    print("=" * 90)
    try:
        lc_ctx64 = llama_completion(prompt)
        print("(a) n_ctx 64 (deployed setting):")
        print(f"    prompt tokens evaluated: {lc_ctx64.get('tokens_evaluated')}")
        print(f"    generated tokens       : {lc_ctx64.get('tokens')}")
        print(f"    text                   : {lc_ctx64.get('content')!r}")
        print(f"    first-token candidates :")
        for c in (lc_ctx64.get("completion_probabilities") or [{}])[0].get("top_logprobs", []):
            print(f"      {c['id']:>7} logp={c['logprob']:>10.6f} {c['token']!r}")

        lc_full = llama_completion(prompt)
        report["llamacpp_ctx64"] = {"evaluated": lc_ctx64.get("tokens_evaluated"),
                                    "ids": lc_ctx64.get("tokens"),
                                    "text": lc_ctx64.get("content")}
        # Same prompt with a context large enough to hold all 89 tokens
        body = {"prompt": prompt, "n_predict": GEN_TOKENS, "n_probs": 10, "temperature": 0.0,
                "top_k": 1, "repeat_penalty": 1.0, "seed": 42, "cache_prompt": False,
                "n_ctx": 512}
        req = urllib.request.Request(f"http://127.0.0.1:{LLAMA_PORT}/completion",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            lc_big = json.load(r)
        print(f"\n(b) n_ctx 512 (diagnostic): prompt tokens evaluated = {lc_big.get('tokens_evaluated')}")
        print(f"    generated tokens       : {lc_big.get('tokens')}")
        print(f"    text                   : {lc_big.get('content')!r}")
        for c in (lc_big.get("completion_probabilities") or [{}])[0].get("top_logprobs", []):
            print(f"      {c['id']:>7} logp={c['logprob']:>10.6f} {c['token']!r}")
        report["llamacpp_ctx512"] = {"evaluated": lc_big.get("tokens_evaluated"),
                                     "ids": lc_big.get("tokens"),
                                     "text": lc_big.get("content"),
                                     "top_logprobs": (lc_big.get("completion_probabilities") or [{}])[0].get("top_logprobs")}
    except Exception as e:
        print(f"llama.cpp unavailable: {e}")
        report["llamacpp_error"] = str(e)

    # ------------------------------------------------------------------ 6. Ollama
    print("\n" + "=" * 90)
    print("6. REGISTERED OLLAMA MODEL")
    print("=" * 90)
    for ctx in (64, 512):
        d = ollama_chat(prompt, num_ctx=ctx)
        c = d["message"]["content"]
        print(f"\n(num_ctx {ctx})")
        print(f"    prompt_eval_count : {d.get('prompt_eval_count')}   "
              f"(input is {len(p_ids)} tokens)")
        print(f"    eval_count        : {d.get('eval_count')}   done={d.get('done_reason')}")
        print(f"    text              : {c!r}")
        report[f"ollama_ctx{ctx}"] = {"prompt_eval_count": d.get("prompt_eval_count"),
                                      "text": c, "done": d.get("done_reason")}

    # ------------------------------------------------------------------ 7. verdict
    print("\n" + "=" * 90)
    print("7. TERMINAL OBSERVATION")
    print("=" * 90)
    print(f"expected answer token sequence: {tokenizer.encode(expected, add_special_tokens=False)} "
          f"= {expected.split(chr(10))[0]!r}")
    print(f"PyTorch (64-truncated)        : {tokenizer.decode(pt_a)!r}")
    print(f"PyTorch (full, extended RoPE) : {tokenizer.decode(pt_b)!r}")
    print(f"Ollama (num_ctx 64)           : {report.get('ollama_ctx64', {}).get('text')!r}")
    print(f"Ollama (num_ctx 512)          : {report.get('ollama_ctx512', {}).get('text')!r}")

    out = PROJECT_ROOT / "docs" / "diagnose_absolute_humidity.json"
    report["prompt"] = prompt
    report["expected"] = expected
    report["dataset_ah"] = dataset_ah
    report["computed_ah"] = computed_ah
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
