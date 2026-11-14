"""Evaluate a conventional U-Net checkpoint on the complete held-out test split.

Requires a validation-selected best.pt produced by train_unet_2d.py in
--mode train. Does not train or choose model weights. Uses the exact same
binary Dice, IoU and resized-pixel HD95 definitions as test_sdf_2d.py.
Saves independent RGB input, ground-truth mask, prediction mask PNGs per case.

Example:
  python test_unet_2d.py --checkpoint runs/cvc_unet_64_bs4/best.pt \
      --data-path "D:/code/Python/MedSeg/cvc" --device cuda

Load only checkpoints produced by trusted local runs (torch.load with
weights_only=False is needed for trainer RNG/optimizer metadata).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from datasets.rgb_binary import RGBBinaryDataset
from models.UNet.unet import UNet
from preview_sdf_2d import save_case_images
from test_sdf_2d import symmetric_hd95_px
from train_sdf_2d import binary_case_metrics, split_hashes


@torch.inference_mode()
def run_test(checkpoint, *, data_path=None, device="cuda", output_dir=None):
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if checkpoint.name != "best.pt":
        raise ValueError("Use a validation-selected best.pt, not last.pt")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = saved["config"]
    if config.get("architecture") != "UNet":
        raise ValueError("Checkpoint is not from train_unet_2d.py")
    if config.get("mode") != "train":
        raise ValueError("Formal test requires --mode train, not --mode overfit")

    root = Path(data_path if data_path is not None else config["data_path"]).resolve()
    hashes = split_hashes(root, config.get("list_dir"))
    if hashes != config["split_sha256"]:
        raise ValueError("Dataset split lists differ from checkpoint configuration")

    dataset = RGBBinaryDataset(
        root, split="test", list_dir=config.get("list_dir"),
        output_size=(int(config["size"]), int(config["size"])), augment=False
    )
    model = UNet(in_channels=3, num_classes=2,
                 base_channels=int(config["base_channels"])).to(device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    folder = (Path(output_dir).resolve() if output_dir
              else checkpoint.parent / "test_results")
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Test output directory is not empty: {folder}")
    folder.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(len(dataset)):
        batch = dataset[i]
        image = batch["image"].unsqueeze(0).to(device=device, dtype=torch.float32)
        label = batch["label"].unsqueeze(0)
        logits = model(image)  # U-Net is deterministic: no DDIM or GT input.
        if logits.shape != (1, 2, *image.shape[2:]):
            raise ValueError("Unexpected UNet output shape")
        pred_tensor = logits.argmax(dim=1).cpu()
        dices, ious = binary_case_metrics(pred_tensor, label)

        gt = label[0].numpy().astype(bool)
        pred = pred_tensor[0].numpy().astype(bool)
        image_rgb = np.clip(np.rint(
            batch["image"].permute(1, 2, 0).numpy() * 255.0
        ), 0, 255).astype(np.uint8)
        stem = f"{i:03d}_{batch['case_name']}"
        filenames = save_case_images(folder, stem, image_rgb, gt, pred)
        hd95 = symmetric_hd95_px(pred, gt)
        row = {
            "case_name": batch["case_name"],
            "dice": float(dices[0]),
            "iou": float(ious[0]),
            "hd95_px": hd95,
            "tp": int((gt & pred).sum()),
            "fp": int((~gt & pred).sum()),
            "fn": int((gt & ~pred).sum()),
            **filenames,
        }
        rows.append(row)
        hd_text = "undefined" if hd95 is None else f"{hd95:.2f}px"
        print(f"TEST [{i+1}/{len(dataset)}] {row['case_name']}: "
              f"Dice={row['dice']:.4f} IoU={row['iou']:.4f} HD95={hd_text}",
              flush=True)

    defined = [row["hd95_px"] for row in rows if row["hd95_px"] is not None]
    summary = {
        "split": "test",
        "architecture": "UNet",
        "dataset": config["dataset"],
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(saved["epoch"]),
        "checkpoint_validation_best_dice": float(saved["best_dice"]),
        "evaluation_grid": f"{int(config['size'])}x{int(config['size'])}",
        "cases_evaluated": len(rows),
        "mean_dice": float(np.mean([row["dice"] for row in rows])),
        "mean_iou": float(np.mean([row["iou"] for row in rows])),
        "mean_hd95_px": (
            float(np.mean(defined)) if len(defined) == len(rows) else None
        ),
        "mean_hd95_defined_only_px": (
            float(np.mean(defined)) if defined else None
        ),
        "hd95_defined_cases": len(defined),
        "hd95_undefined_cases": len(rows) - len(defined),
        "hd95_definition": (
            "Symmetric pooled 95th percentile, 4-connected boundary, resized pixels; "
            "both-empty=0, exactly-one-empty=undefined"
        ),
        "protocol_note": (
            "Uses checkpoint-matched test.txt. GlaS instance IDs are converted "
            "to binary semantic foreground; this is not GlaS instance challenge "
            "evaluation or necessarily its official 85/80 split."
        ),
        "cases": rows,
    }
    with (folder / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (folder / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8"
    )
    return {
        "output_dir": str(folder),
        "cases_evaluated": len(rows),
        "mean_dice": summary["mean_dice"],
        "mean_iou": summary["mean_iou"],
        "mean_hd95_px": summary["mean_hd95_px"],
        "hd95_undefined_cases": summary["hd95_undefined_cases"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    result = run_test(args.checkpoint, data_path=args.data_path,
                      device=args.device, output_dir=args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
