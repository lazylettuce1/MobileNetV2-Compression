# MobileNetV2 CIFAR-10 Baseline

## Environment
```
python >= 3.9
torch >= 2.0
torchvision >= 0.15
```
Install:
```bash
pip install torch torchvision
# optional, only needed if you pass --use-wandb
pip install wandb
```

## Architecture
Stock torchvision MobileNetV2 downsamples 32x total, which collapses a 32x32
CIFAR-10 image to a 1x1 feature map. `model.py` removes two of the five
stride-2 steps (the stem conv and the first block of the 24-channel stage),
bringing total downsampling to 8x -> a 4x4 final feature map. See the
docstring in `model.py` for the exact layers touched and the reasoning.

## Run
```bash
python train.py \
  --epochs 150 \
  --batch-size 128 \
  --lr 0.1 \
  --weight-decay 5e-4 \
  --warmup-epochs 5 \
  --label-smoothing 0.1 \
  --width-mult 1.0 \
  --dropout 0.2 \
  --seed 42 \
  --out-dir runs/baseline
```
Add `--pretrained` to initialize from ImageNet weights (shape-compatible with
the stride-modified stem/stage2). Add `--use-wandb` to log to Weights & Biases.

## Reproducibility
- Seed is fixed via `utils.set_seed` (Python, NumPy, torch, and cuDNN
  determinism flags). `--seed 42` is used for the reported results.
- Note: `cudnn.deterministic=True` trades some throughput for exact
  reproducibility; set it to `False` in `utils.py` if you need max speed and
  can tolerate small run-to-run noise.
- Data pipeline, hyperparameters, and checkpoints are all logged to
  `runs/<name>/log.csv` and `runs/<name>/{last,best}.pth`.

## Outputs
- `runs/baseline/log.csv` — per-epoch train/test loss & top-1 accuracy, for
  the loss/accuracy curves required in Q1(c).
- `runs/baseline/best.pth` — checkpoint with the best test top-1 accuracy.

## Next step
This baseline (`model.py` / `train.py`) is the fixed reference point for the
Q2–Q4 compression pipeline — compression will be applied on top of
`best.pth` without touching this training recipe.
