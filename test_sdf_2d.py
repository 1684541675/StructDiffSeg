"""Evaluate a selected DiffUNet checkpoint on the entire held-out test split.

This script never trains or selects a checkpoint. It only accepts a checkpoint
from --mode train and evaluates *all* case IDs in checkpoint-matched test.txt.
It writes three separate PNGs per case, metrics.csv and summary.json.

Important: metrics are binary foreground semantic segmentation scores on the
resized image grid, NOT GlaS instance-level challenge metrics. Your local
train/val/test.txt protocol may differ from the official GlaS 85/80 split.

Example:
  python test_sdf_2d.py --checkpoint runs/glas_gcf_seed42/best.pt \
      --data-path "D:/code/Python/MedSeg/glas" --device cuda

Only load checkpoints created by your own training script (torch.load uses
Python deserialization for the trainer's RNG/optimizer state).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt

from datasets.rgb_binary import RGBBinaryDataset
from preview_sdf_2d import save_case_images
from train_sdf_2d import binary_case_metrics, build_model, split_hashes


def symmetric_hd95_px(prediction, reference):
    """Symmetric 95th percentile of pooled surface distances, in resized pixels.

    Connectivity: 4-connected binary erosion defines the 1-pixel boundary.
    Both-empty masks: 0. Exactly one empty: None (undefined; not zero).
    This is not a physical-distance metric and not a gland-instance metric.
    """
    prediction = np.asarray(prediction, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    if prediction.shape != reference.shape or prediction.ndim != 2:
        raise ValueError("HD95 expects two same-shaped 2D masks")
    if not prediction.any() and not reference.any():
        return 0.0
    if not prediction.any() or not reference.any():
        return None

    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    pred_border = prediction & ~binary_erosion(
        prediction, structure=structure, border_value=0
    )
    ref_border = reference & ~binary_erosion(
        reference, structure=structure, border_value=0
    )
    pred_to_ref = distance_transform_edt(~ref_border)[pred_border]
    ref_to_pred = distance_transform_edt(~pred_border)[ref_border]
    return float(np.percentile(np.concatenate((pred_to_ref, ref_to_pred)), 95))


@torch.inference_mode()
def run_test(checkpoint, *, data_path=None, device="cuda", output_dir=None):
    """Run a one-time, full-split binary evaluation with labels used only for metrics."""
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if checkpoint.name != "best.pt":
        raise ValueError("Use the validation-selected best.pt checkpoint for testing")

    # Our trusted trainer checkpoint includes non-tensor optimizer/RNG states.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    if config["mode"] != "train":
        raise ValueError("Formal test requires a --mode train checkpoint, not --mode overfit")
    root = Path(data_path if data_path is not None else config["data_path"]).resolve()
    actual_hashes = split_hashes(root, config.get("list_dir"))
    if actual_hashes != config["split_sha256"]:
        raise ValueError("Dataset split lists differ from the checkpoint configuration")

    dataset = RGBBinaryDataset(
        root, split="test", list_dir=config.get("list_dir"),
        output_size=(int(config["size"]), int(config["size"])), augment=False
    )
    model = build_model(config["variant"], config["ddim_steps"]).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    directory = (Path(output_dir).resolve() if output_dir
                 else checkpoint.parent / "test_results")
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Test output directory is not empty: " + str(directory))
    directory.mkdir(parents=True, exist_ok=True)

    rows = []
    cuda_devices = [torch.cuda.current_device()] if device == "cuda" else []
    for index in range(len(dataset)):
        item = dataset[index]
        image = item["image"].unsqueeze(0).to(device=device, dtype=torch.float32)
        label = item["label"].unsqueeze(0)

        # Same deterministic, GT-free DDIM invocation as validation.
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(config["seed"]) + 100000 + index)
            logits = model(image, ddim=True)
        if logits.shape != (1, 2, *image.shape[2:]):
            raise ValueError("Unexpected DDIM segmentation output shape")

        prediction = logits.argmax(dim=1).cpu()
        dice, iou = binary_case_metrics(prediction, label)
        gt = label[0].numpy().astype(bool)
        pred = prediction[0].numpy().astype(bool)
        rgb = np.clip(np.rint(
            item["image"].permute(1, 2, 0).numpy() * 255.0
        ), 0, 255).astype(np.uint8)

        stem = f"{index:03d}_{item['case_name']}"
        names = save_case_images(directory, stem, rgb, gt, pred)
        tp = int((gt & pred).sum())
        fp = int((~gt & pred).sum())
        fn = int((gt & ~pred).sum())
        hd95 = symmetric_hd95_px(pred, gt)
        row = {
            "case_name": item["case_name"],
            "dice": float(dice[0]),
            "iou": float(iou[0]),
            "hd95_px": hd95,
            "tp": tp, "fp": fp, "fn": fn,
            **names,
        }
        rows.append(row)
        hd95_str = "undefined" if hd95 is None else f"{hd95:.2f}px"
        print(f"TEST [{index + 1}/{len(dataset)}] {row['case_name']}: "
              f"Dice={row['dice']:.4f} IoU={row['iou']:.4f} HD95={hd95_str}",
              flush=True)

    valid_hd95 = [row["hd95_px"] for row in rows if row["hd95_px"] is not None]
    # Avoid a misleading average that silently excludes undefined failures.
    complete_hd95 = len(valid_hd95) == len(rows)
    summary = {
        "split": "test",
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(state["epoch"]),
        "checkpoint_validation_best_dice": float(state["best_dice"]),
        "variant": config["variant"],
        "dataset": config["dataset"],
        "evaluation_grid": f"{int(config['size'])}x{int(config['size'])}",
        "cases_evaluated": len(rows),
        "mean_dice": float(np.mean([row["dice"] for row in rows])),
        "mean_iou": float(np.mean([row["iou"] for row in rows])),
        "mean_hd95_px": (float(np.mean(valid_hd95)) if complete_hd95 else None),
        "mean_hd95_defined_only_px": (
            float(np.mean(valid_hd95)) if valid_hd95 else None
        ),
        "hd95_defined_cases": len(valid_hd95),
        "hd95_undefined_cases": len(rows) - len(valid_hd95),
        "hd95_definition": (
            "Symmetric pooled 95th percentile of 4-connected 1-pixel "
            "boundary distances in resized-image pixels. Both empty: 0; "
            "exactly one empty: undefined. No physical pixel spacing."
        ),
        "protocol_note": (
            "Evaluated the complete checkpoint-matched test.txt, which may be "
            "a custom split rather than official GlaS 85/80. "
            "GlaS instance labels were binarized; scores are foreground "
            "semantic segmentation metrics, not instance challenge scores."
        ),
        "cases": rows,
    }
    with (directory / "metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8"
    )
    return {
        "output_dir": str(directory), "cases_evaluated": len(rows),
        "mean_dice": summary["mean_dice"],
        "mean_iou": summary["mean_iou"],
        "mean_hd95_px": summary["mean_hd95_px"],
        "hd95_undefined_cases": summary["hd95_undefined_cases"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    report = run_test(
        args.checkpoint, data_path=args.data_path,
        device=args.device, output_dir=args.output_dir
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
