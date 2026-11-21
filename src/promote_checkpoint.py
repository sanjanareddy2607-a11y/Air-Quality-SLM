"""Promote the best continued-training checkpoint to a standalone model directory."""
import json, shutil, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
out = ROOT / "models" / "air_quality_slm_finetuned"
orig = ROOT / "models" / "air_quality_slm"

ck = torch.load(out / "best_model.pt", map_location="cpu")
torch.save(ck["model_state_dict"], out / "pytorch_model.bin")
print("promoted step", ck["step"], "val_loss", round(ck["val_loss"], 5), "block", ck.get("block"))

cfg = json.loads((orig / "config.json").read_text(encoding="utf-8"))
cfg["max_position_embeddings"] = int(ck.get("block", 128))
(out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]:
    shutil.copy(orig / f, out / f)

print("config max_position_embeddings ->", cfg["max_position_embeddings"])
print(sorted(p.name for p in out.iterdir()))
