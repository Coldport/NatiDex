import json
import threading

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

torch.backends.cudnn.benchmark = True  # optimise kernels for fixed input size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# ── Model factory ────────────────────────────────────────────────────

def _build_model(num_classes: int) -> nn.Module:
    m = torchvision.models.mobilenet_v2(
        weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
    )
    m.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(m.last_channel, num_classes),
    )
    return m.to(device)


# ── Single epoch (train or validate) ────────────────────────────────

def _run_epoch(model, loader, criterion, optimizer, scaler, use_amp,
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

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out  = model(inputs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out  = model(inputs)
                loss = criterion(out, labels)

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


# ── One training phase with early stopping + LR-on-stall ────────────

def _phase(on_update, model, train_loader, val_loader, criterion, optimizer,
           pause_event, stop_event, phase, max_epochs, max_lr, num_classes):

    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_val_loss = float('inf')
    best_state    = None
    es_wait       = 0    # early-stopping counter
    lr_wait       = 0    # LR-boost counter
    lr_best_acc   = -1.0

    for epoch in range(max_epochs):
        if stop_event.is_set():
            break

        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, scaler, use_amp,
            pause_event, stop_event, on_update, phase, epoch + 1)
        if train_loss is None:
            break

        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, None, scaler, use_amp,
            pause_event, stop_event, on_update, phase, epoch + 1)
        if val_loss is None:
            break

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
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"state_dict": best_state, "num_classes": num_classes},
                       "best_model.pth")

        # ── Early stopping ────────────────────────────────────────────
        if improved:
            es_wait = 0
        else:
            es_wait += 1
            if es_wait >= 5:
                if best_state:
                    model.load_state_dict(best_state)
                break

        # ── Increase LR when val_accuracy stalls ──────────────────────
        if val_acc > lr_best_acc + 1e-4:
            lr_best_acc = val_acc
            lr_wait     = 0
        else:
            lr_wait += 1
            if lr_wait >= 3:
                for pg in optimizer.param_groups:
                    pg['lr'] = min(pg['lr'] * 1.5, max_lr)
                lr_wait = 0


# ── Main training entry point ────────────────────────────────────────

def _train(on_update, data_dir, pause_event, stop_event):
    on_update({"type": "train_status", "status": "loading_data"})

    train_tf = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = T.Compose([
        T.Resize((224, 224)),
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

    use_cuda   = device.type == "cuda"
    # Parallel workers keep the GPU fed; persistent_workers avoids re-spawning each epoch
    nw = 16 if use_cuda else 0
    train_loader = DataLoader(
        Subset(ds_train, idx[:n_train]),
        batch_size=128, shuffle=True,
        num_workers=nw, pin_memory=use_cuda,
        persistent_workers=(nw > 0), prefetch_factor=(2 if nw > 0 else None),
    )
    val_loader = DataLoader(
        Subset(ds_val, idx[n_train:]),
        batch_size=128, shuffle=False,
        num_workers=nw, pin_memory=use_cuda,
        persistent_workers=(nw > 0), prefetch_factor=(2 if nw > 0 else None),
    )

    num_classes = len(ds_train.classes)
    device_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    print(f"[NatiDex] device={device}  gpu={device_name}  "
          f"classes={num_classes}  train={n_train}  val={n_val}  workers={nw}")
    on_update({"type": "train_info", "num_classes": num_classes,
               "device": device.type.upper(), "device_name": device_name})

    # Save index → class name mapping for inference
    idx_to_class = {str(v): k for k, v in ds_train.class_to_idx.items()}
    with open("class_labels.json", "w") as f:
        json.dump(idx_to_class, f)

    model     = _build_model(num_classes)
    criterion = nn.CrossEntropyLoss()

    # ── Phase 1: Train the classification head ────────────────────────
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)

    on_update({"type": "train_status", "status": "training", "phase": 1})
    _phase(on_update, model, train_loader, val_loader, criterion, optimizer,
           pause_event, stop_event, phase=1, max_epochs=50, max_lr=1e-3,
           num_classes=num_classes)

    if stop_event.is_set():
        on_update({"type": "train_status", "status": "stopped"})
        return

    # ── Phase 2: Fine-tune the last 5 feature blocks ──────────────────
    for param in model.features.parameters():
        param.requires_grad = True
    for i, block in enumerate(model.features):
        if i < 14:   # freeze first 14 of 19 blocks (~equivalent to TF's [:-30])
            for param in block.parameters():
                param.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5
    )

    on_update({"type": "train_status", "status": "training", "phase": 2})
    _phase(on_update, model, train_loader, val_loader, criterion, optimizer,
           pause_event, stop_event, phase=2, max_epochs=50, max_lr=1e-4,
           num_classes=num_classes)

    if stop_event.is_set():
        on_update({"type": "train_status", "status": "stopped"})
    else:
        torch.save({"state_dict": model.state_dict(), "num_classes": num_classes},
                   "species_model.pth")
        on_update({"type": "train_status", "status": "done"})
