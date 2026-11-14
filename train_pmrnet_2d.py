"""Independent training for the *unmodified uploaded* PMRNet architecture.

Uses the existing GlaS/CVC RGB binary loader and the same CE+Dice criterion
as the DiffUNet experiments; no diffusion or SDF is involved.

  python train_pmrnet_2d.py --dataset cvc --data-path "D:/code/Python/MedSeg/cvc" \
      --mode train --epochs 100 --size 64 --batch-size 4 --device cuda \
      --output-dir runs/cvc_pmrnet_64_bs4

  python train_pmrnet_2d.py --dataset glas --data-path "D:/code/Python/MedSeg/glas" \
      --mode train --epochs 100 --size 256 --batch-size 1 --device cuda \
      --output-dir runs/glas_pmrnet_256

The default run validates every epoch; best.pt is selected by mean validation
foreground Dice. Overfit mode is only for a training-subset smoke check.
Load only checkpoints from your own trusted training runs.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets.rgb_binary import RGBBinaryDataset
from models.PMRNet.pmrnet import PMRNet
from train_sdf_2d import (
    binary_case_metrics, capture_rng, restore_rng, seed_everything, split_hashes
)
from utils import CeDiceLoss


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("glas", "cvc"), required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--list-dir", default=None)
    parser.add_argument("--mode", choices=("train", "overfit"), default="train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--overfit-samples", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def validate_args(args):
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("epochs and batch-size must be positive; num-workers nonnegative")
    if args.size < 32 or args.size % 8:
        raise ValueError("size must be a multiple of 8 and at least 32")
    if args.val_interval < 1 or args.overfit_samples < 1:
        raise ValueError("val-interval and overfit-samples must be positive")
    if not 0 <= args.seed < 2 ** 32:
        raise ValueError("seed must fit uint32")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("lr must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight-decay must be finite and nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if args.resume and not Path(args.resume).is_file():
        raise FileNotFoundError(args.resume)


def get_config(args, hashes):
    return {
        "architecture": "PMRNet",
        "dataset": args.dataset,
        "data_path": str(Path(args.data_path).resolve()),
        "list_dir": str(Path(args.list_dir).resolve()) if args.list_dir else None,
        "split_sha256": hashes,
        "mode": args.mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "size": args.size,
        "overfit_samples": args.overfit_samples,
        "num_workers": args.num_workers,
        "val_interval": args.val_interval,
        "seed": args.seed,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }


def make_loaders(args):
    train_data = RGBBinaryDataset(
        args.data_path, "train", list_dir=args.list_dir,
        output_size=(args.size, args.size), augment=args.mode == "train"
    )
    if args.mode == "overfit":
        if args.overfit_samples > len(train_data):
            raise ValueError("overfit-samples exceeds available train images")
        small = Subset(train_data, list(range(args.overfit_samples)))
        training, validation = small, small
    else:
        training = train_data
        validation = RGBBinaryDataset(
            args.data_path, "val", list_dir=args.list_dir,
            output_size=(args.size, args.size), augment=False
        )
    kwargs = dict(
        num_workers=args.num_workers, pin_memory=(args.device == "cuda")
    )
    train_loader = DataLoader(
        training, batch_size=args.batch_size, shuffle=args.mode == "train",
        drop_last=False, **kwargs
    )
    val_loader = DataLoader(validation, batch_size=1, shuffle=False, **kwargs)
    return train_loader, val_loader


@torch.inference_mode()
def validate(model, loader, device):
    model.eval()
    dice_scores, iou_scores = [], []
    for batch in loader:
        images = batch["image"].to(device=device, dtype=torch.float32)
        labels = batch["label"].to(device=device, dtype=torch.long)
        logits = model(images)
        if logits.shape != (images.shape[0], 2, *images.shape[2:]):
            raise ValueError("Unexpected PMRNet output shape")
        predicted = logits.argmax(dim=1)
        case_dice, case_iou = binary_case_metrics(predicted, labels)
        dice_scores.extend(case_dice)
        iou_scores.extend(case_iou)
    if not dice_scores:
        raise ValueError("Cannot evaluate an empty validation loader")
    return {
        "mean_dice": float(np.mean(dice_scores)),
        "mean_iou": float(np.mean(iou_scores)),
        "cases_evaluated": len(dice_scores),
    }


def run(args):
    validate_args(args)
    seed_everything(args.seed)
    hashes = split_hashes(args.data_path, args.list_dir)
    config = get_config(args, hashes)
    folder = Path(args.output_dir).resolve()
    if args.resume is None and folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = make_loaders(args)

    device = torch.device(args.device)
    model = PMRNet(in_channels=3, num_classes=2).to(device)
    criterion = CeDiceLoss(2, [1, 1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=min(1e-6, args.lr * 0.01)
    )

    start_epoch, best_dice = 1, -1.0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["config"] != config:
            raise ValueError("Resume config or dataset hashes differ")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_dice = float(checkpoint["best_dice"])
        restore_rng(checkpoint["rng"])

    config_file = folder / "config.json"
    if args.resume:
        if (not config_file.is_file() or
                json.loads(config_file.read_text(encoding="utf-8")) != config):
            raise ValueError("Cannot resume into another run or configuration")
    else:
        config_file.write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(f"PMRNet | {args.dataset} | {args.mode} | "
          f"train={len(train_loader.dataset)}, validation={len(val_loader.dataset)}",
          flush=True)
    if args.mode == "overfit":
        print("OVERFIT: evaluation is on training samples, not held-out validation",
              flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        tic = time.perf_counter()
        losses = []
        model.train()
        for batch in train_loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            labels = batch["label"].to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().item())
        if not losses:
            raise ValueError("Empty training loader")
        train_seconds = time.perf_counter() - tic
        scheduler.step()

        perform_validation = (
            epoch == 1 or epoch % args.val_interval == 0 or epoch == args.epochs
        )
        start_val = time.perf_counter()
        metrics = (validate(model, val_loader, device) if perform_validation
                   else {"mean_dice": None, "mean_iou": None, "cases_evaluated": 0})
        validation_seconds = time.perf_counter() - start_val
        better = perform_validation and metrics["mean_dice"] > best_dice
        if better:
            best_dice = metrics["mean_dice"]

        checkpoint = {
            "epoch": epoch, "best_dice": best_dice, "config": config,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "rng": capture_rng(),
        }
        torch.save(checkpoint, folder / "last.pt")
        if better:
            torch.save(checkpoint, folder / "best.pt")
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": float(time.perf_counter() - tic),
            "train_seconds": float(train_seconds),
            "validation_seconds": float(validation_seconds),
            "did_validate": bool(perform_validation),
            **metrics,
        }
        with (folder / "history.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return {"output_dir": str(folder), "best_val_dice": best_dice,
            "last_epoch": args.epochs, "mode": args.mode}


if __name__ == "__main__":
    parsed = make_parser().parse_args()
    print(json.dumps(run(parsed), ensure_ascii=False, indent=2))
