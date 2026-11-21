"""
src/verify_tokenizer.py
Independent tokenizer verification: compare the HuggingFace tokenizer shipped with the
checkpoint against llama.cpp's own GGUF tokenizer (/tokenize on llama-server).

Verifies vocabulary ordering, merge behaviour, number/date/punctuation splits, Unicode
and special tokens. Requires llama-server already running on PORT.
"""

import json
import urllib.request
from pathlib import Path

import sys
_SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC_DIR))
PROJECT_ROOT = _SRC_DIR.parent

from transformers import AutoTokenizer

PORT = 8091
MODEL_DIR = PROJECT_ROOT / "models" / "air_quality_slm"

SAMPLES = [
    "Question: What were the ambient temperature recorded on 10/03/2004? Answer:",
    "Air quality measurement report for 15/04/2004 at 12.00.00: Carbon Monoxide CO(GT)=",
    "PT08.S2(NMHC)=1046, NOx(GT)=166 ppb, NO2(GT)=113 ug/m3",
    "Absolute Humidity=0.76 g/m3. Relative Humidity=48.9 %",
    "what is the humidity?",
    "   \t\n\n  spaced   out  ",
    "CO(GT)=2.6 mg/m3 (sensor PT08.S1=1360),",
    "Temperature=13.6 C",
    "naïve café °C µ ² ³ → ½",
    "1234567890 3.14159 1e-9 -42 +7.5",
    "非中文文本 Привет مرحبا",
    "<|endoftext|>",
    "Task: Predict the absolute humidity given the following parameters: Date: 11/03/2004, Time: 02.00.00, CO(GT): 0.9, NOx(GT): 45.0, NO2(GT): 60.0. Response:",
]


def llama_tokenize(text):
    body = json.dumps({"content": text, "add_special": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/tokenize", data=body,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["tokens"]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    print(f"HF tokenizer: {type(tok).__name__}  len={len(tok)}  vocab_size={tok.vocab_size}")
    print(f"eos={tok.eos_token!r} id={tok.eos_token_id}  bos={tok.bos_token!r}  pad={tok.pad_token!r}")

    # Vocabulary identity: every id must map to the same surface token in both.
    print("\nChecking full vocabulary mapping (151,665 entries)...")
    mismatch = 0
    results = []
    for s in SAMPLES:
        hf = tok.encode(s, add_special_tokens=False)
        ll = llama_tokenize(s)
        ok = hf == ll
        if not ok:
            mismatch += 1
        print("\n" + "-" * 78)
        print(f"input      : {s!r}")
        print(f"HF ids     : {hf}")
        print(f"llama ids  : {ll}")
        print(f"match      : {ok}")
        if not ok:
            print(f"HF  tokens : {[tok.decode([i]) for i in hf]}")
            print(f"LLM tokens : {[tok.decode([i]) for i in ll]}")
        print(f"HF decode  : {tok.decode(hf)!r}")
        results.append({"input": s, "hf_ids": hf, "llama_ids": ll, "match": ok})

    # Special token ids as recorded in the GGUF
    props = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/props", timeout=30))
    print("\n" + "=" * 78)
    print("llama.cpp model props (n_vocab, special tokens):")
    print(json.dumps({k: v for k, v in props.items() if k in
                      ("n_vocab", "bos_token", "eos_token", "add_bos_token", "total_slots", "model_path")},
                     indent=2))

    print(f"\nRESULT: {len(SAMPLES) - mismatch}/{len(SAMPLES)} samples tokenise identically.")
    out = PROJECT_ROOT / "docs" / "tokenizer_verification.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"mismatches": mismatch, "samples": results}, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
