import torch
import torch.nn as nn
import numpy as np
import os
import itertools
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from src.features.extraction import MAX_SPEED_MPS
from src.utils.paths import get_dataset_paths

_ce    = nn.CrossEntropyLoss(label_smoothing=0.05)
_huber = nn.SmoothL1Loss(beta=0.5)

def compute_loss(p_hat, s_hat, d_hat, p_gt, s_gt, d_gt, max_dist):
    loss_path  = _ce(p_hat, p_gt)
    loss_speed = _huber(s_hat, s_gt / MAX_SPEED_MPS)
    loss_dist  = _huber(d_hat, torch.log1p(d_gt) / np.log1p(max_dist))
    return loss_path + 2.5 * loss_speed + 1.5 * loss_dist

def save_ckpt(path, model, optimizer, scheduler,
              epoch, batch_idx, running_loss, best_combined):
    torch.save({
        "model":         model.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "scheduler":     scheduler.state_dict(),
        "epoch":         epoch,
        "batch_idx":     batch_idx,
        "running_loss":  running_loss,
        "best_combined": best_combined,
    }, path)

def load_ckpt(path, model, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], ckpt["batch_idx"], ckpt["running_loss"], ckpt.get("best_combined", float("inf"))

def build_epoch_loader(dataset, batch_size, epoch_idx, device="cpu"):
    g = torch.Generator()
    g.manual_seed(42 + epoch_idx * 997)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      generator=g, num_workers=0, pin_memory=(device == "cuda"))

def run_validation(model, val_loader, device, max_dist):
    model.eval()
    correct, speed_errs, dist_errs = 0, [], []
    with torch.no_grad():
        for x, p_gt, s_gt, d_gt, _ in val_loader:
            x = x.to(device)
            p_hat, s_hat, d_hat = model(x)
            pred_path  = p_hat.argmax(1).item()
            pred_speed = s_hat.item() * MAX_SPEED_MPS
            pred_dist  = max(0.0, torch.expm1(d_hat * np.log1p(max_dist)).item())
            correct   += int(pred_path == p_gt.item())
            speed_errs.append(abs(pred_speed - s_gt.item()))
            dist_errs.append( abs(pred_dist  - d_gt.item()))
    
    n = max(1, len(val_loader))
    path_acc = 100.0 * correct / n
    s_mae    = float(np.mean(speed_errs))
    d_mae    = float(np.mean(dist_errs))
    combined = s_mae / MAX_SPEED_MPS + d_mae / max_dist + (1.0 - path_acc / 100.0)
    return path_acc, s_mae, d_mae, combined

def train_model(model_name, model, train_files, val_files, DatasetCls, ckpt_path, config, device="cpu"):
    epochs = config.get("epochs", 300)
    batch_size = config.get("batch_size", 16)
    lr = config.get("lr", 3e-4)
    save_every = config.get("save_every", 20)
    val_every = config.get("val_every", 20)
    
    dataset_info = get_dataset_paths()
    max_dist = dataset_info["max_dist"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6)

    start_epoch   = 0
    resume_batch  = 0
    resume_loss   = 0.0
    best_combined = float("inf")

    if os.path.exists(ckpt_path):
        start_epoch, resume_batch, resume_loss, best_combined = load_ckpt(
            ckpt_path, model, optimizer, scheduler, device=device)
        if start_epoch >= epochs:
            print(f"[{model_name}] Already trained for {epochs} epochs — skipping.")
            return model
        print(f"[{model_name}] Resumed from epoch {start_epoch + 1}, batch {resume_batch}")
    else:
        print(f"[{model_name}] Starting fresh training")

    train_ds   = DatasetCls(train_files, augment=True)
    val_ds     = DatasetCls(val_files,   augment=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    model.to(device)

    for epoch in range(start_epoch, epochs):
        loader    = build_epoch_loader(train_ds, batch_size, epoch, device=device)
        n_batches = len(loader)
        midpoint  = n_batches // 2
        running   = resume_loss  if epoch == start_epoch else 0.0
        skip      = resume_batch if epoch == start_epoch else 0
        mid_saved = skip > midpoint
        model.train()

        _iter = tqdm(itertools.islice(loader, skip, None),
                      total=n_batches - skip,
                      desc=f"[{model_name}] Ep {epoch+1:03d}/{epochs}",
                      unit="batch", leave=False)

        for i, (x, p, s, d, _) in enumerate(_iter, start=skip):
            x = x.to(device); p = p.to(device)
            s = s.to(device); d = d.to(device)
            optimizer.zero_grad()
            p_hat, s_hat, d_hat = model(x)
            loss = compute_loss(p_hat, s_hat, d_hat, p, s, d, max_dist)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            running += loss.item()
            _iter.set_postfix(loss=f"{running/(i+1):.4f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")

            if i == midpoint and not mid_saved:
                save_ckpt(ckpt_path, model, optimizer, scheduler, epoch, i + 1, running, best_combined)
                mid_saved = True
            elif (i + 1) % save_every == 0:
                save_ckpt(ckpt_path, model, optimizer, scheduler, epoch, i + 1, running, best_combined)

        avg_loss     = running / max(1, n_batches)
        is_val_epoch = ((epoch + 1) % val_every == 0 or epoch == epochs - 1)

        if is_val_epoch:
            path_acc, s_mae, d_mae, combined = run_validation(model, val_loader, device, max_dist)
            print(f"[{model_name}] Epoch {epoch+1:02d}/{epochs} train_loss {avg_loss:.4f} VAL path {path_acc:.1f}% speed MAE {s_mae:.2f} m/s dist MAE {d_mae:.2f} m combined {combined:.4f}")
            if combined < best_combined:
                best_combined = combined
                save_ckpt(ckpt_path, model, optimizer, scheduler, epoch + 1, 0, 0.0, best_combined)
            else:
                save_ckpt(ckpt_path, model, optimizer, scheduler, epoch + 1, 0, 0.0, best_combined)
        else:
            save_ckpt(ckpt_path, model, optimizer, scheduler, epoch + 1, 0, 0.0, best_combined)

        scheduler.step(epoch + 1)
        resume_batch = 0
        resume_loss  = 0.0

    print(f"[{model_name}] Training complete. Best combined: {best_combined:.4f}")
    return model
