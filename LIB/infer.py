import io
import json

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model  = None
_labels: dict[str, str] = {}
_common: dict[str, str] = {}

_IMG_SIZE = 320

_preprocess = T.Compose([
    T.Resize((_IMG_SIZE, _IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _build_model(num_classes: int) -> nn.Module:
    """Must match the architecture used in train.py exactly."""
    m = torchvision.models.efficientnet_b4(weights=None)
    in_features = m.classifier[1].in_features  # 1792

    m.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 1024),
        nn.BatchNorm1d(1024),
        nn.SiLU(inplace=True),
        nn.Dropout(0.35),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.SiLU(inplace=True),
        nn.Dropout(0.25),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.SiLU(inplace=True),
        nn.Dropout(0.15),
        nn.Linear(256, num_classes),
    )
    return m


def _load():
    global _model, _labels, _common
    if _model is not None:
        return

    ckpt = None
    for path in ("best_model.pth", "species_model.pth"):
        try:
            ckpt = torch.load(path, map_location=device, weights_only=True)
            break
        except FileNotFoundError:
            continue
    if ckpt is None:
        raise FileNotFoundError("No trained model found. Train first.")

    num_classes = ckpt["num_classes"]
    m = _build_model(num_classes)
    m.load_state_dict(ckpt["state_dict"])

    with open("class_labels.json") as f:
        _labels = json.load(f)

    m.to(device, memory_format=torch.channels_last).eval()
    _model = m


def predict(image_bytes: bytes, top_k: int = 5) -> list[dict]:
    """
    Returns top_k predictions as:
      [{"species": "Homo sapiens", "common_name": "Human", "confidence": 0.94}, ...]
    """
    global _common
    _load()
    try:
        with open("common_names.json", encoding="utf-8") as f:
            _common = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x   = _preprocess(img).unsqueeze(0).to(device, memory_format=torch.channels_last)

    with torch.no_grad(), torch.amp.autocast(device.type):
        probs = torch.softmax(_model(x), dim=1)[0].cpu().float().numpy()

    top_indices = np.argsort(probs)[::-1][:top_k]
    return [
        {
            "species":     _labels[str(i)].replace("_", " "),
            "common_name": _common.get(_labels[str(i)], ""),
            "confidence":  round(float(probs[i]), 4),
        }
        for i in top_indices
    ]
