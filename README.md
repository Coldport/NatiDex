# NatiDex

A local species identification app powered by a custom-trained EfficientNet-B4 model. Download wildlife photos from iNaturalist, train a classifier on your GPU, and identify species from photos via a web UI — all running on your own machine.

---

## Features

- **Automated dataset download** — pulls research-grade observation photos from iNaturalist for animals and plants, with resume support (restarts pick up exactly where they stopped)
- **GPU-accelerated training** — EfficientNet-B4 backbone with a deep 4-layer classifier head, mixed-precision (AMP), TF32, channels-last layout, cosine LR scheduling, and early stopping
- **Live training dashboard** — real-time loss/accuracy charts, GPU/CPU load meters, and per-epoch metrics streamed over WebSocket
- **Species identification** — upload a photo and get top-K predictions with confidence scores
- **Offline Wikipedia cards** — species descriptions, facts (venomous, dangerous, size, mass, speed, lifespan, habitat), and thumbnail images cached locally

---

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA 12.x (tested on RTX 5080)
- ~20 GB free disk space for a typical 400-species dataset

---

## Installation

**1. Clone the repo**

```bash
git clone https://github.com/Coldport/NatiDex
cd NatiDex
```

**2. Create a virtual environment**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

**3. Install PyTorch with CUDA**

Go to [pytorch.org](https://pytorch.org/get-started/locally/) and pick the right command for your CUDA version. Example for CUDA 12.4:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**4. Install the remaining dependencies**

```bash
pip install -r requirements.txt
```

---

## Running the app

```bash
python server.py
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

The server runs on port **8000** by default.

---

## Workflow

### Step 1 — Download dataset

In the UI, go to the **Download** tab and configure:

| Setting | Default | Description |
|---|---|---|
| Species limit | 400 | Total number of species to collect (split evenly between animals and plants) |
| Photos per species | 50 | Number of research-grade photos per species |

Click **Start Download**. Progress is shown live per-species and overall.

**Resume support:** if the download is interrupted, restart it with the same settings — it will load the species list from `download_state.json` and jump straight to where it stopped without re-fetching anything from the API.

### Step 2 — Train

Go to the **Train** tab and click **Start Training**. Training runs in two phases:

| Phase | What trains | Epochs |
|---|---|---|
| 1 — Head only | Classifier MLP only (backbone frozen) | up to 200 (early stopping) |
| 2 — Fine-tune | Last 3 EfficientNet-B4 feature blocks + head | up to 60 (early stopping) |

The best checkpoint (lowest validation loss) is saved automatically to `best_model.pth` during training. The final model is saved to `species_model.pth` when training completes.

You can **Pause** and **Resume** training at any time without losing progress.

### Step 3 — Identify

Go to the **Identify** tab, upload a photo, and get the top predictions ranked by confidence.

---

## Model architecture

| Component | Detail |
|---|---|
| Backbone | EfficientNet-B4 (pretrained on ImageNet) |
| Input size | 320 × 320 px |
| Classifier head | Linear 1792→1024→512→256→N with BatchNorm + SiLU + Dropout |
| Precision | Mixed (AMP) with TF32 on Ampere/Blackwell |
| Memory layout | Channels-last (NHWC) for faster NVIDIA conv kernels |
| Loss | CrossEntropyLoss with label smoothing 0.1 |
| Optimizer | AdamW with weight decay 1e-4 |
| LR schedule | CosineAnnealingWarmRestarts |
| Gradient clipping | max norm 1.0 |

---

## Project structure

```
NatiDex/
├── server.py              # FastAPI server — all HTTP + WebSocket endpoints
├── main.py                # (legacy) headless CLI entry point
├── requirements.txt       # Python dependencies
│
├── LIB/
│   ├── train.py           # Model definition, training loop, GPU setup
│   ├── downloader.py      # iNaturalist photo downloader + Wikipedia fetcher
│   └── infer.py           # Inference: loads model, runs prediction
│
├── ui/
│   └── index.html         # React-based single-page web UI
│
├── data/                  # Downloaded images — one subfolder per species
│   └── Homo_sapiens/
│       ├── 12345.jpg
│       └── ...
│
├── wiki/                  # Cached Wikipedia data
│   ├── Homo_sapiens.json
│   └── img/
│       └── Homo_sapiens.jpg
│
├── best_model.pth         # Best checkpoint saved during training
├── species_model.pth      # Final model saved after training completes
├── class_labels.json      # Maps class index → species folder name
├── common_names.json      # Maps species name → common name
└── download_state.json    # Resume checkpoint (deleted on completion)
```

---

## API reference

All endpoints are served at `http://localhost:8000`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/status` | Full server state (training progress, download progress) |
| `WS` | `/ws` | WebSocket — receives all live update events |
| `POST` | `/train/start` | Start training |
| `POST` | `/train/pause` | Pause training |
| `POST` | `/train/resume` | Resume training |
| `POST` | `/train/stop` | Stop training |
| `POST` | `/download/start` | Start download (`species_limit`, `photos_per_species` in body) |
| `POST` | `/download/stop` | Stop download |
| `POST` | `/infer` | Identify species from uploaded image (`multipart/form-data`, `top_k` query param) |
| `GET` | `/wiki_data/{species}` | Offline Wikipedia JSON for a species |
| `GET` | `/wiki_img/{species}` | Offline Wikipedia thumbnail |
| `POST` | `/wiki_refresh/{species}` | Force-refresh Wikipedia data for one species |
| `POST` | `/wiki_refresh_all/start` | Refresh Wikipedia data for all downloaded species |
| `POST` | `/wiki_refresh_all/stop` | Stop the refresh-all job |

---

## GPU tips

- TF32 is enabled automatically on Ampere and Blackwell GPUs — gives roughly 2-3× conv/matmul throughput over FP32 with no accuracy loss
- Channels-last (NHWC) memory format is used for both the model and input tensors, which is natively faster for NVIDIA convolution kernels
- AMP (automatic mixed precision) keeps most of the model in FP16/BF16 during the forward pass
- If you run out of VRAM, reduce `batch_size` in `train.py` (currently 256)
- The DataLoader uses 8 workers with `persistent_workers=True` and `prefetch_factor=4` to keep the GPU fed
