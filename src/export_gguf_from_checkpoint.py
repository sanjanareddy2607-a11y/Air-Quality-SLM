"""
src/export_gguf_from_checkpoint.py
Parameterised GGUF exporter.

Same mapping as src/convert_to_gguf.py (verified numerically equivalent to PyTorch), but it
takes paths and the declared context length as arguments so the continued-training checkpoint
can be exported with its new 128-token context instead of the original 64.

Mapping preserved exactly:
  qwen3 architecture (per-head Q/K RMSNorm before RoPE), token_embd scaled by sqrt(128),
  output.weight unscaled (tied source), all 4 blocks, final norm, full 151,665-token vocab,
  151,387 merges, raw passthrough chat template.

Usage:
  python src/export_gguf_from_checkpoint.py \
      --checkpoint models/air_quality_slm_finetuned/pytorch_model.bin \
      --config     models/air_quality_slm_finetuned/config.json \
      --tokenizer  models/air_quality_slm \
      --output     models/air_quality_slm_finetuned/air_quality_slm_finetuned.gguf \
      --context 128
"""

import argparse
import json
import math
from pathlib import Path

import torch
import gguf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CHAT_TEMPLATE = "{% for message in messages %}{{ message['content'] }}{% endfor %}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--context", type=int, required=True)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    tok_dir = Path(args.tokenizer)
    out = Path(args.output)
    if out.exists():
        out.unlink()   # rewriting over a locked file fails on this filesystem

    d_model = cfg["hidden_size"]
    n_layers = cfg["num_hidden_layers"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    d_ff = cfg["intermediate_size"]
    vocab = cfg["vocab_size"]
    head_dim = d_model // n_heads
    ctx = args.context

    print(f"checkpoint : {ckpt}")
    print(f"config     : {cfg}")
    print(f"context    : {ctx}")

    sd = torch.load(ckpt, map_location="cpu")
    for k, v in (("tok_emb.weight", (vocab, d_model)), ("out_head.weight", (vocab, d_model)),
                 ("final_norm.weight", (d_model,))):
        assert tuple(sd[k].shape) == v, f"{k} shape {tuple(sd[k].shape)} != {v}"
    print(f"validated tensor shapes (vocab={vocab}, d_model={d_model})")

    writer = gguf.GGUFWriter(str(out), arch="qwen3")
    writer.add_name("Air Quality SLM (continued training)")
    writer.add_description("Domain-specific Air Quality SLM, continued from the original checkpoint")
    writer.add_architecture()
    writer.add_context_length(ctx)
    writer.add_embedding_length(d_model)
    writer.add_block_count(n_layers)
    writer.add_feed_forward_length(d_ff)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_kv)
    writer.add_layer_norm_rms_eps(cfg.get("rms_norm_eps", 1e-6))
    writer.add_rope_dimension_count(head_dim)
    writer.add_rope_freq_base(cfg.get("rope_theta", 10000.0))

    vocab_obj = gguf.BpeVocab(tok_dir)
    special = gguf.SpecialVocab(tok_dir, load_merges=True)
    tokens, scores, types = [], [], []
    for text, score, ttype in vocab_obj.all_tokens():
        tokens.append(text)
        scores.append(score)
        types.append(ttype)
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("qwen2")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(types)
    if special.merges:
        writer.add_token_merges(special.merges)
    special.add_to_gguf(writer)
    writer.add_chat_template(RAW_CHAT_TEMPLATE)
    print(f"tokenizer: {len(tokens)} tokens, {len(special.merges)} merges, raw chat template")

    writer.add_tensor("token_embd.weight", (sd["tok_emb.weight"].float().numpy() * math.sqrt(d_model)))
    writer.add_tensor("output.weight", sd["out_head.weight"].float().numpy())
    writer.add_tensor("output_norm.weight", sd["final_norm.weight"].float().numpy())

    for i in range(n_layers):
        p, o = f"blocks.{i}.", f"blk.{i}."
        pairs = [
            ("attn_norm.weight", "norm1.weight"),
            ("attn_q.weight", "attention.W_query.weight"),
            ("attn_k.weight", "attention.W_key.weight"),
            ("attn_v.weight", "attention.W_value.weight"),
            ("attn_output.weight", "attention.out_proj.weight"),
            ("attn_q_norm.weight", "attention.q_norm.weight"),
            ("attn_k_norm.weight", "attention.k_norm.weight"),
            ("ffn_norm.weight", "norm2.weight"),
            ("ffn_gate.weight", "feed_forward.gate_proj.weight"),
            ("ffn_up.weight", "feed_forward.up_proj.weight"),
            ("ffn_down.weight", "feed_forward.down_proj.weight"),
        ]
        for gguf_name, pt_name in pairs:
            writer.add_tensor(o + gguf_name, sd[p + pt_name].float().numpy())

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {out} ({out.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
