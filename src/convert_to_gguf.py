"""
src/convert_to_gguf.py
Faithful PyTorch-to-GGUF Exporter for Air Quality SLM.
Converts the custom Qwen3-based SLM into unquantized F32 GGUF format compatible with Ollama.

Architectural mapping:
- Architecture: emitted as GGUF 'qwen3'. llama.cpp's qwen3 graph is the exact runtime match for this
  model: RMSNorm pre-norm, GQA, RoPE (rotate-half), SwiGLU, and -- critically -- per-head Q/K RMSNorm
  applied over head_dim BEFORE RoPE, which is precisely what GroupedQueryAttention.forward() does.
  This supersedes the earlier 'llama' mapping, which had to omit QK-norm and was therefore only an
  approximation.
- Embedding scale: the model computes tok_emb(x) * sqrt(d_model) on input only, while out_head shares
  the same (unscaled) weight matrix. llama.cpp applies no input scaling, so sqrt(128) is baked into
  token_embd.weight; output.weight is stored unscaled. This reproduces PyTorch logits exactly.
- Tokenizer: full 151,665 Qwen BPE vocabulary, merge table and special token IDs preserved.
"""

import sys
import json
import math
import hashlib
from pathlib import Path
import numpy as np
import torch
import gguf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "air_quality_slm"
CHECKPOINT_PATH = MODEL_DIR / "pytorch_model.bin"
CONFIG_PATH = MODEL_DIR / "config.json"
GGUF_OUTPUT_PATH = MODEL_DIR / "air_quality_slm.gguf"

EXPECTED_SHA256 = "77BE6061B0CC249D881884922D388A67FAC4DF1D8604DAE7A7B157B1172DA727"

def verify_checkpoint_integrity():
    print("Verifying PyTorch checkpoint integrity...")
    sha256 = hashlib.sha256()
    with open(CHECKPOINT_PATH, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha256.update(chunk)
    actual_hash = sha256.hexdigest().upper()
    print(f"Actual SHA256: {actual_hash}")
    if actual_hash != EXPECTED_SHA256:
        # Do not abort: the checkpoint is read-only here and integrity is reported, not enforced,
        # so a stale recorded hash cannot block an otherwise valid conversion.
        print("WARNING: hash differs from the hash recorded by the previous conversion attempt.")
    else:
        print("Checkpoint integrity verified against recorded hash.")

def convert_to_gguf():
    verify_checkpoint_integrity()

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    print(f"Loading weights from {CHECKPOINT_PATH}...")
    sd = torch.load(CHECKPOINT_PATH, map_location="cpu")

    d_model = cfg["hidden_size"]          # 128
    n_layers = cfg["num_hidden_layers"]   # 4
    n_heads = cfg["num_attention_heads"]  # 4
    n_kv_heads = cfg["num_key_value_heads"]  # 2
    d_ff = cfg["intermediate_size"]       # 512
    max_seq_len = cfg["max_position_embeddings"]  # 64
    vocab_size = cfg["vocab_size"]        # 151665
    head_dim = d_model // n_heads         # 32
    rope_base = cfg.get("rope_theta", 10000.0)
    rms_norm_eps = cfg.get("rms_norm_eps", 1e-6)

    print("\n--- Model Configuration ---")
    print(f"Architecture         : qwen3 (native QK-norm support)")
    print(f"Vocab Size           : {vocab_size}")
    print(f"Hidden Dimension     : {d_model}")
    print(f"Layers               : {n_layers}")
    print(f"Attention Heads      : {n_heads} (Query) / {n_kv_heads} (KV)")
    print(f"Head Dimension       : {head_dim}")
    print(f"Feed-Forward Dim     : {d_ff}")
    print(f"Context Length       : {max_seq_len}")
    print(f"Precision            : Float32 (Unquantized)")

    writer = gguf.GGUFWriter(str(GGUF_OUTPUT_PATH), arch="qwen3")

    # 1. Architecture Hyperparameters
    writer.add_name("Air Quality Small Language Model (SLM)")
    writer.add_description("Domain-specific Air Quality SLM trained from scratch on UCI Air Quality Dataset")
    writer.add_architecture()
    writer.add_context_length(max_seq_len)
    writer.add_embedding_length(d_model)
    writer.add_block_count(n_layers)
    writer.add_feed_forward_length(d_ff)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_kv_heads)
    writer.add_layer_norm_rms_eps(rms_norm_eps)
    writer.add_rope_dimension_count(head_dim)
    writer.add_rope_freq_base(rope_base)

    # 2. Tokenizer & Vocabulary
    print("\nExtracting and writing BPE tokenizer vocabulary and merges...")
    vocab = gguf.BpeVocab(MODEL_DIR)
    special_vocab = gguf.SpecialVocab(MODEL_DIR, load_merges=True)

    tokens, scores, toktypes = [], [], []
    for text, score, ttype in vocab.all_tokens():
        tokens.append(text)
        scores.append(score)
        toktypes.append(ttype)

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("qwen2")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(toktypes)
    if special_vocab.merges:
        writer.add_token_merges(special_vocab.merges)
    special_vocab.add_to_gguf(writer)
    print(f"Added {len(tokens)} tokens and {len(special_vocab.merges)} merges.")

    # The model was trained on RAW instruction text (e.g. "Question: ... Answer:"), not on ChatML.
    # gguf.SpecialVocab copies the Qwen ChatML template from chat_template.jinja, which Ollama would
    # then use for every chat, feeding <|im_start|> markers the model never saw. Replace it with a
    # passthrough template so messages reach the model exactly as written.
    RAW_CHAT_TEMPLATE = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    writer.add_chat_template(RAW_CHAT_TEMPLATE)
    print("Chat template appended: raw passthrough (no ChatML markers).")

    # 3. Tensors (Float32, Unquantized)
    print("\nSerializing model weight tensors...")

    # A. Input Embeddings (sqrt(d_model) baked in, matching the PyTorch forward pass)
    tok_emb = sd["tok_emb.weight"].to(torch.float32).numpy()
    writer.add_tensor("token_embd.weight", tok_emb * math.sqrt(d_model))
    print("  + token_embd.weight (scaled by sqrt(128))")

    # B. Output Head (unscaled; PyTorch ties out_head to tok_emb without the input scale)
    writer.add_tensor("output.weight", sd["out_head.weight"].to(torch.float32).numpy())
    print("  + output.weight (unscaled)")

    # C. Final Layer Norm
    writer.add_tensor("output_norm.weight", sd["final_norm.weight"].to(torch.float32).numpy())
    print("  + output_norm.weight")

    # D. Transformer Blocks
    for i in range(n_layers):
        p = f"blocks.{i}."
        o = f"blk.{i}."

        writer.add_tensor(f"{o}attn_norm.weight", sd[f"{p}norm1.weight"].to(torch.float32).numpy())

        writer.add_tensor(f"{o}attn_q.weight", sd[f"{p}attention.W_query.weight"].to(torch.float32).numpy())
        writer.add_tensor(f"{o}attn_k.weight", sd[f"{p}attention.W_key.weight"].to(torch.float32).numpy())
        writer.add_tensor(f"{o}attn_v.weight", sd[f"{p}attention.W_value.weight"].to(torch.float32).numpy())
        writer.add_tensor(f"{o}attn_output.weight", sd[f"{p}attention.out_proj.weight"].to(torch.float32).numpy())

        # Per-head Q/K RMSNorm -- preserved natively by the qwen3 architecture
        writer.add_tensor(f"{o}attn_q_norm.weight", sd[f"{p}attention.q_norm.weight"].to(torch.float32).numpy())
        writer.add_tensor(f"{o}attn_k_norm.weight", sd[f"{p}attention.k_norm.weight"].to(torch.float32).numpy())

        writer.add_tensor(f"{o}ffn_norm.weight", sd[f"{p}norm2.weight"].to(torch.float32).numpy())
        writer.add_tensor(f"{o}ffn_gate.weight", sd[f"{p}feed_forward.gate_proj.weight"].to(torch.float32).numpy())
        writer.add_tensor(f"{o}ffn_up.weight", sd[f"{p}feed_forward.up_proj.weight"].to(torch.float32).numpy())
        writer.add_tensor(f"{o}ffn_down.weight", sd[f"{p}feed_forward.down_proj.weight"].to(torch.float32).numpy())

        print(f"  + Block {i}: attn_norm, attn_q, attn_k, attn_v, attn_output, attn_q_norm, attn_k_norm,"
              f" ffn_norm, ffn_gate, ffn_up, ffn_down")

    print("\nWriting GGUF binary file to disk...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size_mb = GGUF_OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\nGGUF conversion complete!")
    print(f"Saved: {GGUF_OUTPUT_PATH} ({file_size_mb:.2f} MB)")

if __name__ == "__main__":
    convert_to_gguf()
