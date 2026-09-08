import itertools
import wandb
from quantize import run_quantization_pipeline


def run_wandb_sweep(ckpt_path, data_dir,
                     bit_widths=(8, 6, 4, 2),
                     sparsities=(0.75, 0.85, 0.9),
                     calib_batches=40,
                     project="mobilenetv2-cifar10-quant"):
    """
    Runs run_quantization_pipeline once per (weight_bits, sparsity) combo,
    each as its own wandb run — this is what lets the Parallel Coordinates
    panel show every combination on one chart, with accuracy as an axis
    alongside the compression knobs.
    """
    for bits, sparsity in itertools.product(bit_widths, sparsities):
        run = wandb.init(
            project=project, reinit=True,
            name=f"w{bits}a{bits}_sparsity{sparsity}",
            config={"weight_bits": bits, "act_bits": bits,
                    "sparsity": sparsity, "calib_batches": calib_batches},
        )
        try:
            result = run_quantization_pipeline(
                ckpt_path=ckpt_path, data_dir=data_dir,
                weight_bits=bits, act_bits=bits,
                calib_batches=calib_batches, sparsity=sparsity,
            )
            
        # flatten for logging: numeric values as-is, percentage strings
            # (e.g. "30.00%" from overall_weight_sparsity) parsed to float so
            # they can be a plottable axis, not just a label
            log_dict = {}
            for k, v in result.items():
                if isinstance(v, (int, float)):
                    log_dict[k] = v
                elif isinstance(v, str) and v.endswith("%"):
                    log_dict[k] = float(v.rstrip("%"))
            wandb.log(log_dict)

        except Exception as e:
            print(f"FAILED at weight_bits={bits}, sparsity={sparsity}: {e}")
            wandb.log({"failed": 1})
        finally:
            run.finish()

    print("Sweep complete.")
    print("wandb -> your project -> Workspace -> Add panel -> Parallel Coordinates")
    print("Suggested axes, in order: weight_bits, sparsity, "
          "overall_weight_sparsity, weights_compression_ratio, accuracy")


    run_wandb_sweep(
    ckpt_path="/kaggle/working/MobileNetV2-Compression/outputs/baseline/best.pth",
    data_dir=DATA_DIR,
    bit_widths=(8, 6, 4, 2),
    sparsities=(0.75, 0.85, 0.9),
)