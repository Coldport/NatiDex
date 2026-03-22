"""
NatiDex — PyTorch → ONNX export script
Run from the project root (e:/ML_projects/NatiDex):

    .venv/Scripts/python mobile/export_onnx.py

Then copy the required data assets:

    cp class_labels.json  mobile/assets/
    cp common_names.json  mobile/assets/
    cp -r wiki            mobile/assets/wiki

Output: mobile/assets/model.onnx  (~13 MB)
"""

import os
import sys

import torch
import torch.nn as nn
import torchvision

# ── Locate project root ───────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "mobile", "assets", "model.onnx")

# ── Load checkpoint ───────────────────────────────────────────────────
ckpt_path = None
for name in ("best_model.pth", "species_model.pth"):
    p = os.path.join(ROOT, name)
    if os.path.isfile(p):
        ckpt_path = p
        break

if ckpt_path is None:
    print("ERROR: No model checkpoint found. Train the model first.")
    sys.exit(1)

print(f"Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
num_classes = ckpt["num_classes"]
print(f"  num_classes = {num_classes}")

# ── Reconstruct model (must match LIB/train.py and LIB/infer.py) ─────
model = torchvision.models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.last_channel, num_classes),  # last_channel = 1280
)
model.load_state_dict(ckpt["state_dict"])
model.eval()
print("  Model loaded and set to eval mode.")

# ── Export to ONNX ────────────────────────────────────────────────────
# dummy input matches the preprocessing in LIB/infer.py:
#   Resize(224, 224) + Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
#   → float32 CHW tensor, batch size 1
dummy = torch.randn(1, 3, 224, 224)

os.makedirs(os.path.dirname(OUT), exist_ok=True)

print(f"Exporting to ONNX: {OUT}")
torch.onnx.export(
    model,
    dummy,
    OUT,
    export_params=True,
    opset_version=17,           # onnxruntime-web supports up to opset 17
    do_constant_folding=True,   # folds BatchNorm into Conv weights
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input":  {0: "batch_size"},
        "output": {0: "batch_size"},
    },
)

size_mb = os.path.getsize(OUT) / 1_048_576
print(f"Done. model.onnx = {size_mb:.1f} MB")
print()
print("Next steps:")
print("  1. Copy assets into mobile/assets/:")
print("       cp class_labels.json  mobile/assets/")
print("       cp common_names.json  mobile/assets/")
print("       cp -r wiki            mobile/assets/wiki")
print("  2. cd mobile && npm install")
print("  3. npx cap add android   (and/or: npx cap add ios)")
print("  4. npx cap sync")
print("  5. npx cap open android  (opens Android Studio)")
