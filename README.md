
# MobileNetV2 CIFAR-10 Compression

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

## Model Architecture
Stock torchvision MobileNetV2 downsamples 32x total, which collapses a 32x32
CIFAR-10 image to a 1x1 feature map. `model.py` removes two of the five
stride-2 steps (the stem conv and the first block of the 24-channel stage),
bringing total downsampling to 8x -> a 4x4 final feature map. See the
docstring in `model.py` for the exact layers touched and the reasoning.


**Global, magnitude-based pruning and Quantization are the two compression techniques implemented in this project.**
1. Magnitude based pruning seems to be working better than Hessian-based, need to look into why,
2. Batch Norm parameters are generally expected to be more sensitive, but did not show any particular during quantization.


## Quantize Option
```bash
python quantize.py \
  --ckpt-path ./outputs/baseline/best.pth \
  --data-dir ./data \
  --weight-bits 6 \
  --bias-bits 2 \
  --act-bits 8 \
  --calib-batches 40 \
  --sparsity 0.85 \
  --pruning-method magnitude
```
Equivalent Python API:
```python
from quantize import run_quantization_pipeline
baseline_ckpt = "/kaggle/working/MobileNetV2-Compression/outputs/baseline/best.pth"

quant_rpt = run_quantization_pipeline(
        ckpt_path=baseline_ckpt,
        data_dir=DATA_DIR,
        weight_bits=6,
        bias_bits=2,
        act_bits=8,
        calib_batches=40,
        sparsity=0.85,
        pruning_method='magnitude'
)
```

**`run_quantization_pipeline` parameters**:
| Parameter | Default | Description |
|-------|---------|-------------|
| `ckpt_path` | *required* | Path to a baseline checkpoint (`state_dict` or wrapped dict — see `load_state_dict_flexible`) |
| `data_dir` | *required* | Path to CIFAR-10 data directory |
| `weight_bits` | `8` | Weight bit-width (e.g. `4` for INT4), per-channel symmetric. Applied uniformly to Conv2d, Linear, **and BatchNorm2d's `weight`/gamma** — no per-layer-type exception |
| `act_bits` | `8` | Activation bit-width (e.g. `6` for ACT6), per-tensor asymmetric, fake-quant only |
| `calib_batches` | `40` | Number of training-set batches used to calibrate activation ranges |
| `sparsity` | `0.3` | Fraction of Conv2d/Linear weights zeroed before quantization, per `pruning_method` |
| `pruning_method` | `"magnitude"` | `"magnitude"` — global unstructured pruning by `\|w\|`. `"hessian"` — Optimal Brain Damage saliency (`0.5 * H_ii * w_i^2`), where `H_ii` is an empirical-Fisher diagonal estimated from `calib_batches` of real training data (`compute_fisher_diagonal` + `apply_hessian_pruning`) — costs one extra forward+backward pass per calibration batch, but ranks a weight by how much removing it actually moves the loss, not just its magnitude |
| `download` | `False` | If set, downloads CIFAR-10 into `data_dir` if not already present |



**Pipeline steps** (`run_quantization_pipeline`):
1. Loads the baseline checkpoint (`state_dict` or wrapped dict) and moves the model to `device`
2. Prunes weights to the target `sparsity` via `pruning_method` (`magnitude` or `hessian`)
3. Swaps Conv2d/Linear/BatchNorm2d layers for quantized variants (`QuantConv2d`/`QuantLinear`/`QuantBatchNorm2d`) and wraps ReLU/ReLU6 with activation fake-quant
4. Attaches activation quantizers to `InvertedResidual` block outputs (`attach_block_output_quant`) — needed because linear-bottleneck/residual outputs aren't reachable by the isinstance-based swap in step 3
5. Moves the newly created quantized submodules to `device` (they're constructed on CPU in step 3, so this second `.to(device)` is required — see Device handling below)
6. Calibrates activation ranges on `calib_batches` batches of **training** data
7. Freezes all quantization parameters (`freeze_all`)
8. Evaluates on the test set and reports accuracy, size, and compression ratios

## Running Locally vs. Kaggle

- **Device handling needs no changes.** Both `train.py` and `quantize.py`
  compute `device = "cuda" if torch.cuda.is_available() else "cpu"` themselves
  and move the model to it at every point that matters — see the Device
  handling keynote below for the specifics of `quantize.py`'s two `.to(device)`
  calls. Checkpoints are loaded with `map_location=device` (train) /
  `map_location="cpu"` (quantize), so loading a GPU-trained checkpoint on a
  CPU-only machine works without editing anything.
- **First run needs `--download` once.** On Kaggle the dataset was usually
  already cached in `data_dir`. On a fresh local clone, pass `--download` the
  first time you call `train.py`, and again for `quantize.py` if you point it
  at a `--data-dir` that doesn't already have CIFAR-10 in it (see the
  Quantize section below — its default is `download=False`, since normally
  it reuses the directory training already populated).
- **Replace Kaggle-absolute paths.** Anywhere you see
  `/kaggle/working/MobileNetV2-Compression/...` in old notebook cells,
  substitute a local relative path such as `./outputs/baseline/best.pth`.


## Train Option
Run training via CLI:
```bash
python train.py \
  --epochs 150 \
  --out-dir runs/baseline \
  --data-dir ./data \
  --batch-size 128 \
  --lr 0.1 \
  --weight-decay 5e-4 \
  --warmup-epochs 5 \
  --label-smoothing 0.1 \
  --width-mult 1.0 \
  --dropout 0.2 \
  --seed 42 \
  --amp \
  --use-wandb \
  --resume /path/to/checkpoint.pth \
  --download
```
`--download` and `--use-wandb` are on/off switches — just include the flag to
turn them on, don't pass a value after them (e.g. `--download`, not
`--download True`). `--amp` accepts `--no-amp` to turn it off explicitly
(it's on by default, but only on CUDA — on CPU it's a no-op either way).

**TrainConfig fields** (also settable via CLI or Python):
| Field | Default | Description |
|-------|---------|-------------|
| `data_dir` | `./data` | Path to CIFAR-10 data directory |
| `out_dir` | `runs/baseline` | Output directory for checkpoints and logs |
| `epochs` | `150` | Total training epochs |
| `batch_size` | `128` | Batch size per GPU |
| `lr` | `0.1` | Initial learning rate. For fine-tuning, use `lr=0.001` |
| `weight_decay` | `5e-4` | SGD weight decay |
| `warmup_epochs` | `5` | Linear warmup epochs. For fine-tuning, use `warmup_epochs=1` |
| `label_smoothing` | `0.1` | Label smoothing for CrossEntropyLoss |
| `width_mult` | `1.0` | Width multiplier |
| `dropout` | `0.2` | Dropout probability |
| `pretrained` | `False` | If set, initializes from ImageNet weights (shape-compatible with stride-modified model) |
| `num_workers` | `0` | DataLoader workers. Use `0` to avoid worker-teardown noise in notebooks; raise it for faster local data loading if your machine has spare CPU cores |
| `seed` | `42` | Random seed for reproducibility |
| `amp` | `True` | Automatic mixed precision. Only takes effect on CUDA — auto-disabled on CPU regardless of this flag. Use `--no-amp` to disable explicitly |
| `use_wandb` | `False` | Log to Weights & Biases (requires `pip install wandb` and being logged in via `wandb login`) |
| `resume` | `None` | Path to checkpoint `.pth` to fine-tune from. Loads model weights only and restarts the epoch/LR schedule at epoch 1 (not a training-crash resume) |
| `download` | `False` | If set, downloads CIFAR-10 into `data_dir` if not already present |

**Fine-tuning overrides** (pass on CLI or set in `TrainConfig`):
```python
from train import TrainConfig, run_training

cfg = TrainConfig(
    data_dir="./data",
    out_dir="./outputs/finetune",
    epochs=50,
    use_wandb=True,
    resume="./outputs/baseline/best.pth",
    download=False,
    num_workers=0,
    # Add these overrides for fine-tuning:
    lr=0.001,          # Lower learning rate (default is 0.1)
    warmup_epochs=1,   # Shorter warmup (default is 5)
)
history, model = run_training(cfg)
```
Equivalent CLI:
```bash
python train.py --epochs 50 --out-dir ./outputs/finetune \
  --resume ./outputs/baseline/best.pth --lr 0.001 --warmup-epochs 1
```

Checkpoints are written to `<out_dir>/best.pth` (highest test top-1 so far)
and `<out_dir>/last.pth` (most recent epoch) — e.g. `./outputs/baseline/best.pth`.

