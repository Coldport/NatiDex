import json
import subprocess
import threading

import psutil
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

torch.backends.cudnn.benchmark = True   # optimise kernels for fixed input size
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 matmul (Ampere/Blackwell)
torch.backends.cudnn.allow_tf32       = True  # TF32 convolutions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Input resolution — EfficientNet-B4 was trained at 380; 320 is a good
# balance between accuracy and VRAM / throughput.
IMG_SIZE = 320


class TrainingController:
    def __init__(self):
        self._pause   = threading.Event()
        self._stop    = threading.Event()
        self._running = False
        self._pause.set()  # not paused initially

    def is_running(self): return self._running
    def pause(self):  self._pause.clear()
    def resume(self): self._pause.set()
    def stop(self):   self._stop.set(); self._pause.set()

    def start(self, on_update, data_dir="data"):
        self._running = True
        self._pause.set()
        self._stop.clear()
        try:
            _train(on_update, data_dir, self._pause, self._stop)
        except Exception as e:
            import traceback
            traceback.print_exc()
            on_update({"type": "train_status", "status": "error", "message": str(e)})
        finally:
            self._running = False


# ── Hardware monitor ─────────────────────────────────────────────────

def _query_gpu_util() -> int | None:
    """Return GPU compute utilisation % via nvidia-smi (no extra packages needed)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        return int(out.stdout.strip().split("\n")[0])
    except Exception:
        return None


def _hw_monitor(on_update, stop_event):
    """Background thread: broadcasts CPU% and GPU% every 2 s while training."""
    _nvml_handle = None
    try:
        import pynvml
        pynvml.nvmlInit()
        _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        pass

    psutil.cpu_percent()  # prime the counter (first call always returns 0)

    while not stop_event.wait(2.0):
        cpu = psutil.cpu_percent()

        gpu = None
        if _nvml_handle is not None:
            try:
                import pynvml
                gpu = pynvml.nvmlDeviceGetUtilizationRates(_nvml_handle).gpu
            except Exception:
                pass
        elif device.type == "cuda":
            gpu = _query_gpu_util()

        on_update({"type": "hw_stats", "cpu": round(cpu, 1), "gpu": gpu})


# ── Model factory ────────────────────────────────────────────────────

def _build_model(num_classes: int) -> nn.Module:
    """EfficientNet-B4 backbone with a deep, wide 3-layer MLP head."""
    m = torchvision.models.efficientnet_b4(
        weights=torchvision.models.EfficientNet_B4_Weights.IMAGENET1K_V1
    )
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
    # channels_last (NHWC) layout is significantly faster on modern NVIDIA GPUs
    # for conv-heavy models like EfficientNet.
    m = m.to(device, memory_format=torch.channels_last)
    return m


# ── Single epoch (train or validate) ────────────────────────────────

def _run_epoch(model, loader, criterion, optimizer, scaler, use_amp, amp_dtype,
               pause_event, stop_event, on_update, phase, epoch):
    """Returns (avg_loss, accuracy) or (None, None) if stopped mid-epoch."""
    is_train      = optimizer is not None
    total_batches = len(loader)
    update_every  = max(1, total_batches // 50)  # ~50 UI updates per epoch
    model.train(is_train)

    total_loss = 0.0
    correct    = 0
    total      = 0

    for batch_idx, (inputs, labels) in enumerate(loader):
        pause_event.wait()
        if stop_event.is_set():
            return None, None

        inputs = inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                out  = model(inputs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            # Gradient clipping prevents exploding gradients in the deep head
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                out  = model(inputs)
                loss = criterion(out, labels)

        # Print VRAM usage after the very first training batch so it's visible in logs
        if batch_idx == 0 and is_train and epoch == 1 and use_amp and device.type == "cuda":
            used = torch.cuda.memory_reserved() // (1024 ** 2)
            peak = torch.cuda.max_memory_allocated() // (1024 ** 2)
            print(f"[NatiDex] VRAM reserved={used}MB  peak_allocated={peak}MB")

        total_loss += loss.item() * labels.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        total      += labels.size(0)

        if batch_idx % update_every == 0 or batch_idx == total_batches - 1:
            on_update({
                "type":          "train_batch",
                "phase":         phase,
                "epoch":         epoch,
                "batch":         batch_idx + 1,
                "total_batches": total_batches,
                "is_train":      is_train,
                "running_loss":  round(total_loss / total, 4) if total else 0,
                "running_acc":   round(correct    / total, 4) if total else 0,
            })

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


# ── One training phase with early stopping + cosine LR ──────────────

def _phase(on_update, model, train_loader, val_loader, criterion, optimizer,
           scheduler, pause_event, stop_event, phase, max_epochs, num_classes):

    use_amp   = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else None  # BF16 is native+faster on Blackwell
    scaler    = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_val_loss = float('inf')
    best_state    = None
    es_wait       = 0    # early-stopping counter

    for epoch in range(max_epochs):
        if stop_event.is_set():
            break

        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, scaler, use_amp, amp_dtype,
            pause_event, stop_event, on_update, phase, epoch + 1)
        if train_loss is None:
            break

        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, None, scaler, use_amp, amp_dtype,
            pause_event, stop_event, on_update, phase, epoch + 1)
        if val_loss is None:
            break

        if scheduler is not None:
            scheduler.step()

        on_update({
            "type":         "train_metrics",
            "phase":        phase,
            "epoch":        epoch + 1,
            "total_epochs": max_epochs,
            "loss":         round(train_loss, 4),
            "accuracy":     round(train_acc,  4),
            "val_loss":     round(val_loss,   4),
            "val_accuracy": round(val_acc,    4),
        })

        # ── Checkpoint (save best) ────────────────────────────────────
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            # unwrap compiled model for portable state dict
            raw = getattr(model, "_orig_mod", model)
            best_state = {k: v.cpu().clone() for k, v in raw.state_dict().items()}
            torch.save({"state_dict": best_state, "num_classes": num_classes},
                       "best_model.pth")

        # ── Early stopping ────────────────────────────────────────────
        if improved:
            es_wait = 0
        else:
            es_wait += 1
            if es_wait >= 7:
                if best_state:
                    raw = getattr(model, "_orig_mod", model)
                    raw.load_state_dict(best_state)
                break


# ── Main training entry point ────────────────────────────────────────

def _train(on_update, data_dir, pause_event, stop_event):
    on_update({"type": "train_status", "status": "loading_data"})

    train_tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(p=0.1),
        # RandomAffine with degrees only is ~2× faster than RandomRotation
        # because it can skip the full affine matrix when no translate/scale is set.
        T.RandomAffine(degrees=20, fill=0),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.RandomGrayscale(p=0.05),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])
    val_tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load dataset twice (different transforms), split with the same indices
    ds_train = ImageFolder(data_dir, transform=train_tf)
    ds_val   = ImageFolder(data_dir, transform=val_tf)

    n_total = len(ds_train)
    n_val   = max(1, int(n_total * 0.2))
    n_train = n_total - n_val
    g       = torch.Generator().manual_seed(42)
    idx     = torch.randperm(n_total, generator=g).tolist()

    use_cuda = device.type == "cuda"
    # VRAM at batch=128 was only 6/16 GB — double again to fill the headroom.
    nw = 4 if use_cuda else 0
    BATCH = 256
    # pin_memory_device="" tells PyTorch to pin directly to the CUDA device,
    # saving one CPU→GPU memcpy per batch on Windows.
    pin_dev = "cuda" if use_cuda else ""
    train_loader = DataLoader(
        Subset(ds_train, idx[:n_train]),
        batch_size=BATCH, shuffle=True, drop_last=True,
        num_workers=nw, pin_memory=use_cuda, pin_memory_device=pin_dev,
        persistent_workers=(nw > 0), prefetch_factor=(2 if nw > 0 else None),
    )
    val_loader = DataLoader(
        Subset(ds_val, idx[n_train:]),
        batch_size=BATCH, shuffle=False, drop_last=True,
        num_workers=nw, pin_memory=use_cuda, pin_memory_device=pin_dev,
        persistent_workers=(nw > 0), prefetch_factor=(2 if nw > 0 else None),
    )

    num_classes = len(ds_train.classes)
    device_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    vram_total  = torch.cuda.get_device_properties(0).total_memory // (1024**2) if use_cuda else 0
    print(f"[NatiDex] device={device}  gpu={device_name}  vram={vram_total}MB  "
          f"classes={num_classes}  train={n_train}  val={n_val}  "
          f"batch={BATCH}  workers={nw}")
    on_update({"type": "train_info", "num_classes": num_classes,
               "device": device.type.upper(), "device_name": device_name})

    # Start hardware monitor
    hw_stop = threading.Event()
    threading.Thread(target=_hw_monitor, args=(on_update, hw_stop), daemon=True).start()

    # Save index → class name mapping for inference
    idx_to_class = {str(v): k for k, v in ds_train.class_to_idx.items()}
    with open("class_labels.json", "w") as f:
        json.dump(idx_to_class, f)

    model     = _build_model(num_classes)
    # torch.compile with reduce-overhead uses CUDA graphs (no Triton needed on Windows)
    # — typically 15-30% faster after the first warm-up batch.
    try:
        model = torch.compile(model, backend="cudagraphs")
        print("[NatiDex] torch.compile enabled (cudagraphs backend, no Triton needed)")
    except Exception as e:
        print(f"[NatiDex] torch.compile unavailable, running eager: {e}")
    # Label smoothing regularises the deep head and reduces overconfidence
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Phase 1: Train the classification head only ───────────────────
    raw = getattr(model, "_orig_mod", model)
    for param in raw.features.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-5
    )

    on_update({"type": "train_status", "status": "training", "phase": 1})
    _phase(on_update, model, train_loader, val_loader, criterion, optimizer,
           scheduler, pause_event, stop_event, phase=1, max_epochs=200,
           num_classes=num_classes)

    if stop_event.is_set():
        hw_stop.set()
        on_update({"type": "train_status", "status": "stopped"})
        return

    # ── Phase 2: Fine-tune last 3 feature blocks ──────────────────────
    # EfficientNet-B4 has features[0..8]; unfreeze 6, 7, 8
    for param in raw.features.parameters():
        param.requires_grad = True
    for i, block in enumerate(raw.features):
        if i < 6:
            for param in block.parameters():
                param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )

    on_update({"type": "train_status", "status": "training", "phase": 2})
    _phase(on_update, model, train_loader, val_loader, criterion, optimizer,
           scheduler, pause_event, stop_event, phase=2, max_epochs=60,
           num_classes=num_classes)

    hw_stop.set()  # stop hardware monitor

    if stop_event.is_set():
        on_update({"type": "train_status", "status": "stopped"})
    else:
        raw = getattr(model, "_orig_mod", model)
        torch.save({"state_dict": raw.state_dict(), "num_classes": num_classes},
                   "species_model.pth")
        on_update({"type": "train_status", "status": "done"})
