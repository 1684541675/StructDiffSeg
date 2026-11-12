"""Visualize a saved DiffUNet checkpoint without further training.

Saves three independent PNGs per case (RGB input, GT mask, prediction mask),
plus metrics.csv and summary.json.
This is a *preview*, not official held-out TEST evaluation.

Example (after 30-epoch two-case overfit run):
    python preview_sdf_2d.py \
      --checkpoint runs/glas_gcf_overfit_30ep/best.pt \
      --data-path "D:/code/Python/MedSeg/glas" \
      --split train --limit 2 --device cuda

Use --split val to *inspect* a held-out example, NOT to claim a fair
generalization score from the two-case overfit experiment.
Do not load checkpoints from untrusted sources (torch.load deserializes data).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from datasets.rgb_binary import RGBBinaryDataset
from train_sdf_2d import binary_case_metrics, build_model, split_hashes


def save_case_images(directory, stem, image_rgb, ground_truth, prediction):
    """Save three independent PNGs: RGB image and 0/255 grayscale masks."""
    image_rgb = np.asarray(image_rgb)
    ground_truth = np.asarray(ground_truth)
    prediction = np.asarray(prediction)
    if (image_rgb.ndim != 3 or image_rgb.shape[-1] != 3
            or image_rgb.dtype != np.uint8):
        raise ValueError("image_rgb must be an RGB uint8 array")
    if (ground_truth.shape != image_rgb.shape[:2]
            or prediction.shape != image_rgb.shape[:2]):
        raise ValueError("image and mask spatial shapes must match")

    names = {
        "image_png": stem + "_image.png",
        "gt_png": stem + "_gt.png",
        "prediction_png": stem + "_pred.png",
    }
    Image.fromarray(image_rgb, mode="RGB").save(directory / names["image_png"])
    Image.fromarray(
        ground_truth.astype(np.uint8) * 255, mode="L"
    ).save(directory / names["gt_png"])
    Image.fromarray(
        prediction.astype(np.uint8) * 255, mode="L"
    ).save(directory / names["prediction_png"])
    return names


@torch.inference_mode()
def run_preview(checkpoint, *, data_path=None, split="train", limit=2,
                device="cuda", output_dir=None):
    """Read a trusted checkpoint, predict without GT, save visualizations."""
    if split not in ("train", "val"):
        raise ValueError("Preview only accepts train or val; never test")
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    # Restricted to checkpoint files from our own trainer; not arbitrary URLs.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    root = Path(data_path if data_path is not None else config["data_path"]).resolve()
    hashes = split_hashes(root, config.get("list_dir"))
    if hashes != config["split_sha256"]:
        raise ValueError("Dataset split lists differ from checkpoint training configuration")

    dataset = RGBBinaryDataset(
        root, split=split, list_dir=config.get("list_dir"),
        output_size=(int(config["size"]), int(config["size"])),
        augment=False
    )
    if split == "train" and config["mode"] == "overfit":
        limit = min(limit, int(config["overfit_samples"]))
    count = min(limit, len(dataset))
    if count == 0:
        raise ValueError("No samples to preview")
    model = build_model(config["variant"], config["ddim_steps"]).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    directory = (Path(output_dir).resolve() if output_dir
                 else checkpoint.parent / ("preview_" + split))
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Preview output directory is not empty: " + str(directory))
    directory.mkdir(parents=True, exist_ok=True)

    cases = []
    cuda_devices = [torch.cuda.current_device()] if device == "cuda" else []
    for i in range(count):
        batch = dataset[i]
        image = batch["image"].unsqueeze(0).to(device)
        label = batch["label"].unsqueeze(0)
        # Same per-image seed and DDIM path as the trainer's evaluate().
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(config["seed"]) + 100000 + i)
            logits = model(image, ddim=True)

        pred = logits.argmax(dim=1).cpu()
        dice, iou = binary_case_metrics(pred, label)
        gt_np = label[0].numpy().astype(bool)
        pred_np = pred[0].numpy().astype(bool)
        image_np = np.clip(
            np.rint(batch["image"].permute(1, 2, 0).numpy() * 255.0),
            0, 255
        ).astype(np.uint8)
        stem = f"{i:03d}_{batch['case_name']}"
        image_files = save_case_images(directory, stem, image_np, gt_np, pred_np)
        tp = int((gt_np & pred_np).sum())
        fp = int((~gt_np & pred_np).sum())
        fn = int((gt_np & ~pred_np).sum())
        row = {
            "case_name": batch["case_name"],
            "dice": float(dice[0]),
            "iou": float(iou[0]),
            "tp": tp, "fp": fp, "fn": fn,
            **image_files,
        }
        cases.append(row)
        print(f"{row['case_name']}: Dice={row['dice']:.4f} IoU={row['iou']:.4f} -> {stem}_*.png",
              flush=True)

    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(state["epoch"]),
        "checkpoint_best_dice": float(state["best_dice"]),
        "variant": config["variant"],
        "checkpoint_training_mode": config["mode"],
        "preview_split": split,
        "cases_evaluated": len(cases),
        "mean_dice": float(np.mean([row["dice"] for row in cases])),
        "mean_iou": float(np.mean([row["iou"] for row in cases])),
        "warning": ("Training-subset fit only; NOT a test-set score"
                    if split == "train" else
                    "Exploratory validation preview; NOT official test-set evaluation"),
        "cases": cases,
    }
    with (directory / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"output_dir": str(directory), "checkpoint_epoch": int(state["epoch"]),
            "mean_dice": summary["mean_dice"], "mean_iou": summary["mean_iou"],
            "cases_evaluated": len(cases), "split": split}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    result = run_preview(args.checkpoint, data_path=args.data_path,
                         split=args.split, limit=args.limit, device=args.device,
                         output_dir=args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
