"""
sweep.py — Run grid sweeps over weight bits, activation bits, and sparsities,
logging model size, accuracy, and compression metrics to Weights & Biases.
"""
import argparse
import itertools
import wandb
from quantize import run_quantization_pipeline


def run_wandb_sweep(
    ckpt_path,
    data_dir,
    weight_bits=(8, 6, 4, 2),
    act_bits=(8, 6, 4, 2),
    sparsities=(0.75, 0.85, 0.9),
    bias_bits=2,
    pruning_method="magnitude",
    calib_batches=40,
    project="mobilenetv2-cifar10-pruneplusquantv2",
):
    """
    Runs run_quantization_pipeline once per (weight_bits, act_bits, sparsity) combo,
    logging all outputs to W&B for Parallel Coordinates visualization.
    """
    for wbits, abits, sparsity in itertools.product(weight_bits, act_bits, sparsities):
        run = wandb.init(
            project=project,
            reinit="finish_previous",
            name=f"w{wbits}a{abits}_s{sparsity}_{pruning_method}",
            config={
                "weight_bits": wbits,
                "act_bits": abits,
                "bias_bits": bias_bits,
                "sparsity": sparsity,
                "calib_batches": calib_batches,
                "pruning_method": pruning_method,
            },
        )
        try:
            result = run_quantization_pipeline(
                ckpt_path=ckpt_path,
                data_dir=data_dir,
                weight_bits=wbits,
                act_bits=abits,
                bias_bits=bias_bits,
                calib_batches=calib_batches,
                sparsity=sparsity,
                pruning_method=pruning_method,
            )

            # Parse percentage string (e.g., "75.00%") if returned as string
            raw_sparsity = result.get("overall_weight_sparsity", sparsity)
            sparsity_pct = (
                float(raw_sparsity.rstrip("%"))
                if isinstance(raw_sparsity, str)
                else raw_sparsity * 100
            )

            wandb.log({
                "weight_bits": wbits,
                "act_bits": abits,
                "target_sparsity": sparsity,
                "overall_compression_ratio": result["overall_compression_ratio"],
                "weights_compression_ratio": result["weights_compression_ratio"],
                "activation_compression_ratio": result["activation_compression_ratio"],
                "model_size_mb": result["quantized_sparse_total_mb"],
                "accuracy": result["accuracy"],
            })

        except Exception as e:
            print(f"FAILED at weight_bits={wbits}, act_bits={abits}, sparsity={sparsity}: {e}")
            wandb.log({"failed": 1})
        finally:
            run.finish()

    print("Sweep complete.")
    print("wandb -> your project -> Workspace -> Add panel -> Parallel Coordinates")
    print(
        "Suggested axes: weight_bits, act_bits, target_sparsity, "
        "overall_compression_ratio, weights_compression_ratio, accuracy"
    )


def _parse_args():
    parser = argparse.ArgumentParser(description="Run W&B sweep for pruning and quantization.")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to baseline checkpoint (.pth)")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--weight-bits", type=int, nargs="+", default=[8, 6, 4, 2])
    parser.add_argument("--act-bits", type=int, nargs="+", default=[8, 6, 4, 2])
    parser.add_argument("--sparsities", type=float, nargs="+", default=[0.75, 0.85, 0.9])
    parser.add_argument("--bias-bits", type=int, default=16, help="Bit width for derived bias quantization")
    parser.add_argument("--pruning-method", type=str, default="magnitude", choices=["magnitude", "hessian"])
    parser.add_argument("--calib-batches", type=int, default=40)
    parser.add_argument("--project", type=str, default="mobilenetv2-cifar10-pruneplusquantv2")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_wandb_sweep(
        ckpt_path=args.ckpt_path,
        data_dir=args.data_dir,
        weight_bits=args.weight_bits,
        act_bits=args.act_bits,
        sparsities=args.sparsities,
        bias_bits=args.bias_bits,
        pruning_method=args.pruning_method,
        calib_batches=args.calib_batches,
        project=args.project,
    )