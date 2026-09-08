import itertools
import wandb
from quantize import run_quantization_pipeline


def run_wandb_sweep(ckpt_path, data_dir,
                     bit_widths=(8, 6, 4, 2),
                     sparsities=(0.75, 0.85, 0.9),
                     calib_batches=40,
                     project="mobilenetv2-cifar10-pruneplusquant"):
    """
    Runs run_quantization_pipeline once per (weight_bits, sparsity) combo,
    each as its own wandb run — this is what lets the Parallel Coordinates
    panel show every combination on one chart, with accuracy as an axis
    alongside the compression knobs.
    """
    for wbits, abits, sparsity in itertools.product(bit_widths, bit_widths, sparsities):
        run = wandb.init(
            project=project, reinit="finish_previous",
            name=f"w{wbits}a{abits}_sparsity{sparsity}",
            config={"weight_bits": wbits, "act_bits": abits,
                    "sparsity": sparsity, "calib_batches": calib_batches},
        )
        try:
            result = run_quantization_pipeline(
                ckpt_path=ckpt_path, data_dir=data_dir,
                weight_bits=wbits, act_bits=abits,
                calib_batches=calib_batches, sparsity=sparsity,
            )

             # Only log the quantities we actually want in W&B
            wandb.log({
                "weight_bits": wbits,
                "act_bits": abits,
                "compression_ratio": result["overall_compression_ratio"],
                "model_size_mb": result["quantized_sparse_total_mb"],
                "accuracy": result["accuracy"],
            })

        except Exception as e:
            print(f"FAILED at weight_bits={wbits}, sparsity={sparsity}: {e}")
            wandb.log({"failed": 1})
        finally:
            run.finish()

    print("Sweep complete.")
    print("wandb -> your project -> Workspace -> Add panel -> Parallel Coordinates")
    print("Suggested axes, in order: weight_bits, sparsity, "
          "overall_weight_sparsity, weights_compression_ratio, accuracy")

if __name__ == "__main__":
    run_wandb_sweep(
        ckpt_path="/kaggle/working/MobileNetV2-Compression/outputs/baseline/best.pth",
        data_dir=DATA_DIR,
        bit_widths=(8, 6, 4, 2),
        sparsities=(0.75, 0.85, 0.9),
    )

## this is quantize-sweep for a sparsity vs quantization sweep
# import itertools
# from quantize import run_quantization_pipeline

# weight_bits_list = [8, 6, 4]
# act_bits_list = [8, 6, 4]
# sparsity_list = [0.70, 0.80, 0.85, 0.90]

# all_reports = {}

# for w_bits, a_bits, sparsity in itertools.product(weight_bits_list, act_bits_list, sparsity_list):
#     print(f"\n==================================================================")
#     print(f" Running: Weight Bits = {w_bits} | Act Bits = {a_bits} | Sparsity = {int(sparsity * 100)}%")
#     print(f"==================================================================")
    
#     quant_rpt = run_quantization_pipeline(
#         ckpt_path=baseline_ckpt,
#         data_dir=DATA_DIR,
#         weight_bits=w_bits,
#         act_bits=a_bits,
#         calib_batches=40,
#         sparsity=sparsity
#     )
    
#     all_reports[(w_bits, a_bits, sparsity)] = quant_rpt