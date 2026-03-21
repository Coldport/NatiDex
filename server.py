import asyncio
import json
import re
import sys
import os
import threading
from contextlib import asynccontextmanager


sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from LIB.train import TrainingController
from LIB.downloader import DownloadController
from LIB.infer import predict

_connected: set[WebSocket] = set()
_loop: asyncio.AbstractEventLoop | None = None

trainer = TrainingController()
downloader = DownloadController()

# ── Server-side state (survives page refresh) ────────────────────────
_state = {
    "train": {
        "status": "idle",
        "phase": 1,
        "epoch": 0,
        "total_epochs": 50,
        "loss": None, "accuracy": None,
        "val_loss": None, "val_accuracy": None,
        "history": [],          # list of train_metrics dicts
        "device": None, "device_name": None,
    },
    "download": {
        "status": "idle",
        "overall_pct": 0,
        "overall_downloaded": 0,
        "overall_total": 0,
        "active_idx": None,
        "species": [],          # [{name, pct, downloaded, total, done, skipped}]
    },
}

def _update_state(data: dict):
    t = data.get("type", "")

    if t == "train_info":
        _state["train"]["device"]      = data.get("device")
        _state["train"]["device_name"] = data.get("device_name")

    elif t == "train_metrics":
        s = _state["train"]
        s.update({k: data[k] for k in ("phase","epoch","total_epochs","loss","accuracy","val_loss","val_accuracy")})
        s["history"].append({k: data[k] for k in ("epoch","loss","accuracy","val_loss","val_accuracy")})
        s["history"] = s["history"][-100:]  # keep last 100 epochs

    elif t == "train_status":
        _state["train"]["status"] = data["status"]
        if "phase" in data:
            _state["train"]["phase"] = data["phase"]

    elif t == "download_init":
        photos = data.get("photos_per_species", 50)
        _state["download"]["status"] = "running"
        _state["download"]["species"] = [
            {"name": n, "pct": 0, "downloaded": 0, "total": photos, "done": False, "skipped": False}
            for n in data["species"]
        ]

    elif t == "download_species_start":
        _state["download"]["active_idx"] = data["species_idx"]

    elif t == "download_progress":
        dl = _state["download"]
        dl["overall_pct"]         = data["overall_pct"]
        dl["overall_downloaded"]  = data["overall_downloaded"]
        dl["overall_total"]       = data["overall_total"]
        idx = data["species_idx"]
        if idx < len(dl["species"]):
            dl["species"][idx]["pct"]        = data["species_pct"]
            dl["species"][idx]["downloaded"] = data["downloaded"]

    elif t == "download_species_done":
        dl = _state["download"]
        dl["overall_pct"]        = data["overall_pct"]
        dl["overall_downloaded"] = data.get("overall_downloaded", dl["overall_downloaded"])
        dl["overall_total"]      = data.get("overall_total", dl["overall_total"])
        dl["active_idx"] = None
        idx = data["species_idx"]
        if idx < len(dl["species"]):
            dl["species"][idx].update({"pct": 100, "downloaded": data["downloaded"],
                                       "done": True, "skipped": data["skipped"]})

    elif t == "download_status":
        _state["download"]["status"] = data["status"]
        if data["status"] in ("stopped", "done"):
            _state["download"]["active_idx"] = None


# ── WebSocket broadcast ──────────────────────────────────────────────

async def _broadcast(data: dict):
    msg = json.dumps(data)
    dead: set[WebSocket] = set()
    for ws in _connected:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    _connected.difference_update(dead)


def sync_broadcast(data: dict):
    """Called from training/download threads — updates state then broadcasts."""
    _update_state(data)
    if _loop:
        asyncio.run_coroutine_threadsafe(_broadcast(data), _loop)


@asynccontextmanager
async def lifespan(_app):
    global _loop
    _loop = asyncio.get_running_loop()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _connected.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        _connected.discard(ws)


# ── Status route (called on page load to restore state) ─────────────

@app.get("/status")
async def get_status():
    return _state


# ── Training routes ──────────────────────────────────────────────────

@app.post("/train/start")
async def train_start():
    if trainer.is_running():
        return {"status": "already_running"}
    threading.Thread(target=trainer.start, args=(sync_broadcast,), daemon=True).start()
    return {"status": "started"}


@app.post("/train/pause")
async def train_pause():
    trainer.pause()
    sync_broadcast({"type": "train_status", "status": "paused"})
    return {"status": "paused"}


@app.post("/train/resume")
async def train_resume():
    trainer.resume()
    sync_broadcast({"type": "train_status", "status": "training"})
    return {"status": "resumed"}


@app.post("/train/stop")
async def train_stop():
    trainer.stop()
    return {"status": "stopped"}


# ── Download routes ──────────────────────────────────────────────────

class DownloadConfig(BaseModel):
    species_limit: int = 400
    photos_per_species: int = 50

@app.post("/download/start")
async def download_start(cfg: DownloadConfig):
    if downloader.is_running():
        return {"status": "already_running"}
    threading.Thread(
        target=downloader.start,
        args=(sync_broadcast,),
        kwargs={"species_limit": cfg.species_limit, "photos_per_species": cfg.photos_per_species},
        daemon=True,
    ).start()
    return {"status": "started"}


@app.post("/download/stop")
async def download_stop():
    downloader.stop()
    return {"status": "stopped"}


# ── Inference route ──────────────────────────────────────────────────

@app.post("/infer")
async def infer(file: UploadFile = File(...), top_k: int = 5):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    data = await file.read()
    try:
        results = predict(data, top_k=top_k)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No trained model found. Train first.")
    return {"predictions": results}


# ── Wikipedia offline routes ─────────────────────────────────────────

_SAFE_SPECIES = re.compile(r'^[A-Za-z0-9_]+$')

@app.get("/wiki_data/{species}")
async def wiki_data(species: str):
    if not _SAFE_SPECIES.match(species):
        raise HTTPException(status_code=400, detail="Invalid species name")
    path = os.path.join("wiki", f"{species}.json")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Wiki data not available offline")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/wiki_img/{species}")
async def wiki_img(species: str):
    if not _SAFE_SPECIES.match(species):
        raise HTTPException(status_code=400, detail="Invalid species name")
    path = os.path.join("wiki", "img", f"{species}.jpg")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Image not available offline")
    return FileResponse(path, media_type="image/jpeg")


# ── Serve React UI ───────────────────────────────────────────────────

app.mount("/", StaticFiles(directory="ui", html=True), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
