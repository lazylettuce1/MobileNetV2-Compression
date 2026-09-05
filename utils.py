"""
utils.py — seeding for reproducibility, running-average meter, top-k accuracy,
and checkpoint save/load helpers.
"""
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set all relevant RNG seeds for reproducibility (see README for caveats
    about full determinism vs. cuDNN performance)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic=True trades some speed for exact reproducibility; flip to
    # False if you need max throughput and can tolerate minor run-to-run noise
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Tracks a running average of a scalar (loss, accuracy, ...)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes top-k accuracy for the given logits/targets."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        results.append((correct_k * (100.0 / batch_size)).item())
    return results


def save_checkpoint(state: dict, out_dir: str, filename: str = "checkpoint.pth"):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(state, os.path.join(out_dir, filename))
