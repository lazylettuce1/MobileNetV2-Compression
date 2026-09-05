"""
train.py — training logic for the CIFAR-10 MobileNetV2 baseline.

Two ways to use this:
  1. CLI / subprocess:  python train.py --epochs 150 --out-dir runs/baseline
     (or `!python train.py --args` from a notebook cell)
  2. Imported directly into a notebook cell running on the Colab kernel:
         from train import TrainConfig, run_training
         cfg = TrainConfig(epochs=150, out_dir="runs/baseline")
         history, model = run_training(cfg)
     Runs in-process — you keep the trained `model` and per-epoch `history`
     as live Python objects for the next cell to plot/inspect.

Both paths call the same `run_training`, so there's exactly one training
implementation regardless of how you launch it.
"""
import argparse
import csv
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn

from data import get_dataloaders
from model import MobileNetV2CIFAR
from utils import AverageMeter, accuracy, save_checkpoint, set_seed


@dataclass
class TrainConfig:
    data_dir: str = "./data"
    out_dir: str = "runs/baseline"
    epochs: int = 150
    batch_size: int = 128
    lr: float = 0.1
    weight_decay: float = 5e-4
    warmup_epochs: int = 5
    label_smoothing: float = 0.1
    width_mult: float = 1.0
    dropout: float = 0.2
    pretrained: bool = False
    num_workers: int = 2          # Colab: few CPU cores, keep this low
    seed: int = 42
    amp: bool = True
    use_wandb: bool = False
    resume: Optional[str] = None


def build_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)).item())

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, scheduler, criterion, scaler, device):
    model.train()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        top1, = accuracy(outputs, targets, topk=(1,))
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(top1, images.size(0))
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        top1, = accuracy(outputs, targets, topk=(1,))
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(top1, images.size(0))
    return loss_meter.avg, acc_meter.avg


def run_training(cfg: TrainConfig):
    """Runs the full training loop and returns (history, model)."""
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(cfg.out_dir, exist_ok=True)
    log_path = os.path.join(cfg.out_dir, "log.csv")
    if not (cfg.resume and os.path.exists(log_path)):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "train_acc1", "test_loss", "test_acc1", "lr", "epoch_time_s"]
            )

    wandb_run = None
    if cfg.use_wandb:
        import wandb
        wandb_run = wandb.init(project="mobilenetv2-cifar10", config=asdict(cfg))

    train_loader, test_loader = get_dataloaders(
        cfg.data_dir, batch_size=cfg.batch_size, num_workers=cfg.num_workers
    )

    model = MobileNetV2CIFAR(
        num_classes=10, width_mult=cfg.width_mult, dropout=cfg.dropout,
        pretrained=cfg.pretrained,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg.lr, momentum=0.9, nesterov=True,
        weight_decay=cfg.weight_decay,
    )
    scheduler = build_scheduler(optimizer, cfg.warmup_epochs, cfg.epochs, len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device == "cuda")

    start_epoch, best_acc = 1, 0.0
    if cfg.resume:
        print(f"Resuming from {cfg.resume}")
        ckpt = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        best_acc = ckpt["best_acc"]
        start_epoch = ckpt["epoch"] + 1
        for _ in range(ckpt["global_step"]):
            scheduler.step()
        print(f"Resumed at epoch {start_epoch}, best_acc so far {best_acc:.2f}%")

    history = []
    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, scaler, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        epoch_time = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        row = {"epoch": epoch, "train_loss": train_loss, "train_acc1": train_acc,
               "test_loss": test_loss, "test_acc1": test_acc, "lr": current_lr,
               "epoch_time_s": epoch_time}
        history.append(row)

        print(f"epoch {epoch:3d}/{cfg.epochs} | "
              f"train_loss {train_loss:.4f} train_acc {train_acc:.2f} | "
              f"test_loss {test_loss:.4f} test_acc {test_acc:.2f} | "
              f"lr {current_lr:.5f} | {epoch_time:.1f}s")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(list(row.values()))
        if wandb_run is not None:
            wandb_run.log(row)

        is_best = test_acc > best_acc
        best_acc = max(best_acc, test_acc)
        checkpoint = {
            "epoch": epoch, "global_step": epoch * len(train_loader),
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_acc": best_acc, "config": asdict(cfg),
        }
        save_checkpoint(checkpoint, cfg.out_dir, filename="last.pth")
        if is_best:
            save_checkpoint(checkpoint, cfg.out_dir, filename="best.pth")

    print(f"Best test top-1 accuracy: {best_acc:.2f}%")
    if wandb_run is not None:
        wandb_run.summary["best_test_acc1"] = best_acc
        wandb_run.finish()

    model.eval()
    return history, model


def _parse_args() -> TrainConfig:
    """CLI entrypoint only — builds a TrainConfig from sys.argv."""
    d = TrainConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default=d.data_dir)
    p.add_argument("--out-dir", type=str, default=d.out_dir)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--warmup-epochs", type=int, default=d.warmup_epochs)
    p.add_argument("--label-smoothing", type=float, default=d.label_smoothing)
    p.add_argument("--width-mult", type=float, default=d.width_mult)
    p.add_argument("--dropout", type=float, default=d.dropout)
    p.add_argument("--pretrained", action="store_true", default=d.pretrained)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--amp", action="store_true", default=d.amp)
    p.add_argument("--use-wandb", action="store_true", default=d.use_wandb)
    p.add_argument("--resume", type=str, default=d.resume)
    return TrainConfig(**vars(p.parse_args()))


if __name__ == "__main__":
    run_training(_parse_args())