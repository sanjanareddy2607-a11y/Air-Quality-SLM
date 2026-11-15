"""
evaluate_slm.py
Interactive and Batch Evaluation Script for the trained Air Quality Small Language Model (SLM).
Allows generating predictions and evaluating loss on test sets.
"""

import sys
import json
import argparse
from pathlib import Path
import torch

# Ensure src/ is on the path so train_slm can be imported from any CWD
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

PROJECT_ROOT = _SRC_DIR.parent
MODEL_DIR = PROJECT_ROOT / "models" / "air_quality_slm"

def load_model_and_tokenizer(model_dir=MODEL_DIR, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoTokenizer
    from train_slm import Qwen3Model

    print(f"Loading tokenizer from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    cfg_path = model_dir / "config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    print(f"Instantiating Qwen3 SLM ({cfg['hidden_size']}d, {cfg['num_hidden_layers']}L, {cfg['num_attention_heads']}H)...")
    model = Qwen3Model(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["hidden_size"],
        n_heads=cfg["num_attention_heads"],
        n_kv_groups=cfg["num_key_value_heads"],
        n_layers=cfg["num_hidden_layers"],
        d_ff=cfg["intermediate_size"],
        max_seq_len=cfg["max_position_embeddings"],
        tie_weights=cfg.get("tie_word_embeddings", True),
    )

    weights_path = model_dir / "pytorch_model.bin"
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    return model, tokenizer, device

def generate_response(model, tokenizer, prompt, device, max_new_tokens=50, temperature=0.7, top_k=40):
    input_ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False), dtype=torch.long).unsqueeze(0).to(device)
    output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    full_text = tokenizer.decode(output_ids.squeeze().tolist(), skip_special_tokens=False)
    completion = full_text[len(prompt):].strip()
    return completion

def main():
    parser = argparse.ArgumentParser(description="Evaluate Air Quality SLM")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt to test")
    parser.add_argument("--max_tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    args = parser.parse_args()

    model, tokenizer, device = load_model_and_tokenizer()

    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = [
            "Task: Predict the absolute humidity given the following parameters: Date: 11/03/2004, Time: 02.00.00, CO(GT): 0.9, NOx(GT): 45.0, NO2(GT): 60.0. Response:",
            "Air quality measurement report for 15/04/2004 at 12.00.00: Carbon Monoxide CO(GT)=",
            "Question: What were the ambient temperature and relative humidity recorded on 10/03/2004 at 18.00.00? Answer:",
            "Query: What are the primary pollutant concentrations (CO, NOx, NO2) on 12/03/2004 at 08.00.00? Result:",
        ]

    print("\n" + "=" * 65)
    print("       AIR QUALITY SLM - INFERENCE GENERATION RESULTS")
    print("=" * 65)
    for i, p in enumerate(prompts, 1):
        ans = generate_response(model, tokenizer, p, device, max_new_tokens=args.max_tokens, temperature=args.temperature)
        print(f"\n[Test {i}]")
        print(f"Prompt:     {p}")
        print(f"Completion: {ans}")
    print("\n" + "=" * 65)

if __name__ == "__main__":
    main()
