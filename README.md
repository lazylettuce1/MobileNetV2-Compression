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

## Architecture
Stock torchvision MobileNetV2 downsamples 32x total, which collapses a 32x32
CIFAR-10 image to a 1x1 feature map. `model.py` removes two of the five
stride-2 steps (the stem conv and the first block of the 24-channel stage),
bringing total downsampling to 8x -> a 4x4 final feature map. See the
docstring in `model.py` for the exact layers touched and the reasoning.

1. Magnitude based pruning seems to be working better than Hessian-based, need to look into why,
2. Batch Norm parameters are quantized seperately, since they are more sensitive, but FP works worse than INT, need to look into why,


## Quantize Option
```bash
python quantize.py \
  --ckpt-path ./outputs/baseline/best.pth \
  --data-dir ./data \
  --weight-bits 4 \
  --act-bits 6 \
  --bn-mode fp8 \
  --bn-bits 8 \
  --calib-batches 40 \
  --sparsity 0.85 \
  --pruning-method magnitude
```
Equivalent Python API:
```python
from quantize import run_quantization_pipeline

baseline_ckpt = "./outputs/baseline/best.pth"
DATA_DIR = "./data"

quant_rpt = run_quantization_pipeline(
    ckpt_path=baseline_ckpt,
    data_dir=DATA_DIR,
    weight_bits=4,
    act_bits=6,
    bn_mode="fp8",
    bn_bits=8,
    calib_batches=40,
    sparsity=0.85,
    pruning_method='magnitude',
)
```

**`run_quantization_pipeline` parameters**:
| Parameter | Default | Description |
|-------|---------|-------------|
| `ckpt_path` | *required* | Path to a baseline checkpoint (`state_dict` or wrapped dict — see `load_state_dict_flexible`) |
| `data_dir` | *required* | Path to CIFAR-10 data directory |
| `weight_bits` | `8` | Weight bit-width (e.g. `4` for INT4), per-channel symmetric |
| `act_bits` | `8` | Activation bit-width (e.g. `6` for ACT6), per-tensor asymmetric, fake-quant only |
| `calib_batches` | `40` | Number of training-set batches used to calibrate activation ranges |
| `sparsity` | `0.3` | Fraction of Conv2d/Linear weights zeroed before quantization |
| `pruning_method` | `"magnitude"` | `"magnitude"` — global unstructured pruning by `\|w\|`. `"hessian"` — Optimal Brain Damage saliency (`0.5 * H_ii * w_i^2`), where `H_ii` is an empirical-Fisher diagonal estimated from `calib_batches` of real training data (`compute_fisher_diagonal` + `apply_hessian_pruning`) — costs one extra forward+backward pass per calibration batch, but ranks a weight by how much removing it actually moves the loss, not just its magnitude |
| `bn_mode` | `"fp16"` | Precision used to fake-quantize each BatchNorm layer's `weight`/`bias`. One of `"fp16"` (true half-precision round-trip), `"fp8"` (8-bit float, E4M3: 1 sign + 4 exponent + 3 mantissa bits), `"fp4"` (4-bit float, E2M1: 1 sign + 2 exponent + 1 mantissa bit, max representable magnitude 6.0 — expect visible accuracy loss), `"int"` (fixed-point at `bn_bits`, per-parameter scale), or `"none"` (skip; BN stays fp32) |
| `bn_bits` | `16` | Bit-width for fixed-point BN quantization — only consulted when `bn_mode="int"` |
| `download` | `False` | If set, downloads CIFAR-10 into `data_dir` if not already present |

fp8/fp4 are implemented as a generic minifloat fake-quantizer
(`quantize_to_minifloat`, per-element: pick the representable power-of-two
exponent, then round the mantissa to the target bit-width) rather than
relying on hardware fp8/fp4 dtypes, so `--bn-mode fp8`/`fp4` behave
identically on CPU and GPU regardless of torch version.

**Pipeline steps** (`run_quantization_pipeline`):
1. Loads the baseline checkpoint (`state_dict` or wrapped dict) and moves the model to `device`
2. Prunes weights to the target `sparsity` via `pruning_method` (`magnitude` or `hessian`)
3. Swaps Conv2d/Linear layers for quantized variants (`QuantConv2d`/`QuantLinear`) and wraps ReLU/ReLU6 with activation fake-quant
4. Attaches activation quantizers to `InvertedResidual` block outputs (`attach_block_output_quant`) — needed because linear-bottleneck/residual outputs aren't reachable by the isinstance-based swap in step 3
5. Fake-quantizes BatchNorm `weight`/`bias` per `bn_mode` (`quantize_batchnorm`)
6. Moves the newly created quantized submodules to `device` (they're constructed on CPU in step 3, so this second `.to(device)` is required — see Device handling below)
7. Calibrates activation ranges on `calib_batches` batches of **training** data
8. Freezes all quantization parameters (`freeze_all`)
9. Evaluates on the test set and reports accuracy, size, and compression ratios

## Running Locally vs. Kaggle
This repo was originally driven from a Kaggle notebook (cloning the repo into
`/kaggle/working/...` and calling `run_training` / `run_quantization_pipeline`
directly from notebook cells). Both scripts also work as plain CLIs — that's
now the primary way to run them locally:

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

## General Keynotes
- **Checkpoint format**: Checkpoints saved by `train.py` contain `state_dict`, `optimizer_state_dict`, `scaler_state_dict`, `best_acc`, and `config`. `load_state_dict_flexible()` in `quantize.py` tolerates both this wrapped-dict format and a raw `state_dict`.
- **Quantization is fake-quant only**: weights and activations are rounded/dequantized in-place to *simulate* quantization; compute still runs in FP32 — there are no custom INT kernels, so quantized runs are not faster, only smaller (as reported by `compression_ratio_report`).
- **Data-dir reuse between train and quantize**: `quantize.py` defaults to `download=False` because it's normally pointed at the same `data_dir` a prior `train.py --download` run already populated. Point both scripts at the same `--data-dir` unless you want CIFAR-10 downloaded twice.
- **`--use-wandb`**: requires `pip install wandb` and `wandb login` (or `WANDB_API_KEY` set) before running, or `train.py` will raise on `wandb.init(...)`.
- **Reproducibility**: seed is fixed via `utils.set_seed`; `cudnn.deterministic=True` trades throughput for exact reproducibility (safe to run even without a GPU present).
- **Outputs**:
  - `<out_dir>/log.csv` — per-epoch train/test loss & top-1 accuracy
  - `<out_dir>/best.pth` — best checkpoint by test top-1
  - `<out_dir>/last.pth` — most recent epoch's checkpoint
