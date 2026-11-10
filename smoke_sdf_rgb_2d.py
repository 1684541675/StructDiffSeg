"""GlaS/CVC binary-data check for the SDF-enabled DiffUNet.

No checkpoints, validation, epochs, or automatic training.
Examples (from repository root):
    python smoke_sdf_rgb_2d.py --dataset glas --data-path D:/data/GlaS --mode data
    python smoke_sdf_rgb_2d.py --dataset glas --data-path D:/data/GlaS --mode forward --device cuda
    python smoke_sdf_rgb_2d.py --dataset cvc --data-path D:/data/CVC --mode step --device cuda
Expected layout: images/<ID>.png, masks/<ID>.png, train.txt / val.txt / test.txt.
"""

import argparse
import json

import torch
from torch.utils.data import DataLoader

from datasets.rgb_binary import RGBBinaryDataset
from models.DiffUNet3201.diffunet.diffunet_model import DiffUNet
from train_diffunet_sdf_step import train_step_sdf
from utils import CeDiceLoss


def run_smoke(dataset, data_path, *, list_dir=None, split="train", mode="data",
              device="cpu", size=256, seed=42):
    """Inspect one batch, optionally run forward or exactly one optimizer update."""
    if dataset not in ("glas", "cvc"):
        raise ValueError("dataset must be glas or cvc")
    if mode not in ("data", "forward", "step"):
        raise ValueError("mode must be data, forward, or step")
    if mode == "step" and split != "train":
        raise ValueError("single training update is only allowed for split=train")
    if size < 32 or size % 32 != 0:
        raise ValueError("size must be a positive multiple of 32 (at least 32)")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this environment")

    torch.manual_seed(seed)
    dataset_obj = RGBBinaryDataset(
        data_path, split=split, list_dir=list_dir,
        output_size=(size, size)
    )
    loader = DataLoader(dataset_obj, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    image, label = batch["image"], batch["label"]
    if image.shape != (1, 3, size, size) or label.shape != (1, size, size):
        raise ValueError("Unexpected binary RGB batch shape")
    if not torch.isfinite(image).all().item() or not ((label == 0) | (label == 1)).all().item():
        raise ValueError("Image or mask contains invalid values")

    report = {
        "dataset": dataset,
        "split": split,
        "samples": len(dataset_obj),
        "case_name": batch["case_name"][0],
        "device": str(device),
        "mode": mode,
        "image_shape": list(image.shape),
        "label_shape": list(label.shape),
        "foreground_pixels": int(label.sum().item()),
        "image_range": [float(image.min()), float(image.max())],
    }
    if mode == "data":
        return report

    model = DiffUNet(3, 2, ddim_steps=3, enable_sdf=True).to(device)
    criterion = CeDiceLoss(2, [1, 1]).to(device)

    if mode == "forward":
        model.eval()
        with torch.no_grad():
            out = model(image.to(device), gt=label.to(device), return_sdf=True)
        pred_diff, pred_geo, uncertainty, pred_sdf = out
        report.update({
            "diffusion_shape": list(pred_diff.shape),
            "geometry_shape": list(pred_geo.shape),
            "uncertainty_shape": list(uncertainty.shape),
            "sdf_shape": list(pred_sdf.shape),
            "sdf_range": [float(pred_sdf.min()), float(pred_sdf.max())],
        })
        return report

    # Exactly one in-memory optimization step; never writes weights or results.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    report.update(train_step_sdf(
        batch, model, criterion, optimizer,
        epoch=1, epochs=100, lambda_sdf=0.25, sdf_radius=16
    ))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("glas", "cvc"))
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--list-dir")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--mode", default="data", choices=("data", "forward", "step"))
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--size", default=256, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    report = run_smoke(args.dataset, args.data_path, list_dir=args.list_dir,
                       split=args.split, mode=args.mode, device=args.device,
                       size=args.size, seed=args.seed)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
