"""Reproducible binary DiffUNet experiments: baseline / SDF / SDF+GCF.

GlaS/CVC layout: root/images/<ID>.png, root/masks/<ID>.png,
train.txt, val.txt and test.txt. Test IDs are checked for overlap, but
no test images/masks are loaded or evaluated.

Two intentionally distinct modes:
  overfit: fixed subset of *training* images, no augmentation, evaluate that
           same subset (diagnostic only, not a generalization estimate).
  train:   augmented train.txt, select checkpoint on val.txt mean case Dice.

Commands (run from repository root):
  python train_sdf_2d.py --dataset glas --data-path "D:/code/Python/MedSeg/glas" \
      --variant gcf --mode overfit --epochs 2 --overfit-samples 2 \
      --output-dir runs/glas_gcf_overfit_check --device cuda
  python train_sdf_2d.py --dataset glas --data-path "D:/code/Python/MedSeg/glas" \
      --variant baseline --mode train --epochs 100 \
      --output-dir runs/glas_baseline_seed42 --device cuda

Each output directory is exclusive to one run. Resume only from its own
last.pt using --resume <path>. Use only trusted local checkpoints.
"""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets.rgb_binary import RGBBinaryDataset
from models.DiffUNet3201.diffunet.diffunet_model import DiffUNet
from train_diffunet_sdf_step import train_step_sdf
from utils import CeDiceLoss


VARIANTS = ("baseline", "sdf", "gcf")


def build_model(variant, ddim_steps=3, shared_seed=None):
    """Initialize shared baseline weights identically in all three variants.

    Additional SDF/GCF parameters are initialized independently. Constructing
    a model with an extra head otherwise shifts the denoiser's random weights.
    """
    if variant not in VARIANTS:
        raise ValueError("variant must be baseline, sdf or gcf")

    def construct(which):
        return DiffUNet(
            3, 2, ddim_steps=ddim_steps,
            enable_sdf=(which != "baseline"),
            enable_gcf=(which == "gcf"),
            gcf_stage=2,
        )

    if shared_seed is None:
        return construct(variant)
    with torch.random.fork_rng(devices=[]):
        # CPU-only initialization: avoid altering global train/DDIM RNG.
        torch.random.default_generator.manual_seed(shared_seed)
        reference = construct("baseline")
        if variant == "baseline":
            return reference
        torch.random.default_generator.manual_seed(shared_seed + 1)
        variant_model = construct(variant)
        loaded = variant_model.load_state_dict(reference.state_dict(), strict=False)
        expected_prefixes = ("edge_model.sdf_head.", "denoise_model.encoder.gcf.")
        if loaded.unexpected_keys or any(
                not key.startswith(expected_prefixes)
                for key in loaded.missing_keys):
            raise RuntimeError("Unexpected shared-weight mismatch: " + str(loaded))
        return variant_model


def baseline_update(batch, model, criterion, optimizer, *, epoch, epochs):
    """Original two-loss update; matches the SDF step with lambda_sdf=0."""
    images = batch["image"].to(next(model.parameters()).device, dtype=torch.float32)
    labels = batch["label"].to(images.device, dtype=torch.long)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    pred, pred_edge, uncertainty = model(images, gt=labels)
    scale = math.exp(-10.0 * (1.0 - epoch / epochs) ** 2)
    loss_diff = criterion(pred, labels, uncertainty.clamp(0, 1), scale)
    loss_geo = criterion(pred_edge, labels)
    loss = loss_diff + loss_geo
    if not torch.isfinite(loss).item():
        raise FloatingPointError("Non-finite baseline loss")
    loss.backward()
    optimizer.step()
    return {
        "loss_total": loss.detach().item(),
        "loss_diffusion": loss_diff.detach().item(),
        "loss_geometry": loss_geo.detach().item(),
        "loss_sdf": 0.0,
    }


def update_batch(batch, model, criterion, optimizer, *, variant, epoch, epochs,
                 lambda_sdf, sdf_radius):
    if variant == "baseline":
        return baseline_update(batch, model, criterion, optimizer,
                               epoch=epoch, epochs=epochs)
    return train_step_sdf(
        batch, model, criterion, optimizer, epoch=epoch, epochs=epochs,
        lambda_sdf=lambda_sdf, sdf_radius=sdf_radius
    )


def binary_case_metrics(prediction, target):
    """Per-image foreground Dice/IoU; both-empty masks score 1.0."""
    if prediction.shape != target.shape:
        raise ValueError("Prediction/target shapes do not match")
    prediction, target = prediction.bool(), target.bool()
    if prediction.ndim != 3:
        raise ValueError("Expected [B,H,W] masks")
    tp = (prediction & target).flatten(1).sum(dim=1).double()
    pred_n = prediction.flatten(1).sum(dim=1).double()
    target_n = target.flatten(1).sum(dim=1).double()
    dice = torch.where(pred_n + target_n > 0,
                       2 * tp / (pred_n + target_n).clamp_min(1), 1.0)
    union = pred_n + target_n - tp
    iou = torch.where(union > 0, tp / union.clamp_min(1), 1.0)
    return dice.cpu().tolist(), iou.cpu().tolist()


@torch.inference_mode()
def evaluate(model, loader, device, *, seed):
    """True label-free DDIM prediction, reproducible noise per case."""
    model.eval()
    dice_values, iou_values = [], []
    cuda_devices = ([device.index if device.index is not None
                     else torch.cuda.current_device()]
                    if device.type == "cuda" else [])
    for index, batch in enumerate(loader):
        images = batch["image"].to(device, dtype=torch.float32)
        labels = batch["label"].to(device, dtype=torch.long)
        # Fix DDIM noise while preserving the training RNG sequence.
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed + index)
            logits = model(images, ddim=True)
        if logits.shape != (images.shape[0], 2, *images.shape[2:]):
            raise ValueError("Unexpected DDIM prediction shape")
        predicted = logits.argmax(dim=1)
        case_dice, case_iou = binary_case_metrics(predicted, labels)
        dice_values.extend(case_dice)
        iou_values.extend(case_iou)
    if not dice_values:
        raise ValueError("Cannot evaluate an empty loader")
    return {
        "mean_dice": float(np.mean(dice_values)),
        "mean_iou": float(np.mean(iou_values)),
        "cases_evaluated": len(dice_values),
    }


def split_hashes(root, list_dir):
    directory = Path(list_dir) if list_dir is not None else Path(root)
    result = {}
    splits = {}
    for split in ("train", "val", "test"):
        path = directory / (split + ".txt")
        if not path.is_file():
            raise FileNotFoundError(str(path))
        raw = path.read_bytes()
        result[split] = hashlib.sha256(raw).hexdigest()
        lines = [s.strip() for s in raw.decode("utf-8-sig").splitlines() if s.strip()]
        if len(lines) != len(set(lines)):
            raise ValueError("Duplicate sample IDs in " + str(path))
        splits[split] = set(lines)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = splits[a] & splits[b]
        if overlap:
            raise ValueError("Overlapping " + a + "/" + b + " split IDs")
    return result


def make_loaders(args):
    source = RGBBinaryDataset(
        args.data_path, "train", list_dir=args.list_dir,
        output_size=(args.size, args.size), augment=args.mode == "train"
    )
    if args.mode == "overfit":
        if not 1 <= args.overfit_samples <= len(source):
            raise ValueError("overfit_samples exceeds available train images")
        chosen = Subset(source, list(range(args.overfit_samples)))
        training, validation = chosen, chosen
    else:
        training = source
        validation = RGBBinaryDataset(
            args.data_path, "val", list_dir=args.list_dir,
            output_size=(args.size, args.size), augment=False
        )
    params = dict(num_workers=args.num_workers, pin_memory=(args.device == "cuda"))
    train_loader = DataLoader(training, batch_size=args.batch_size,
                              shuffle=args.mode == "train", drop_last=False, **params)
    eval_loader = DataLoader(validation, batch_size=1, shuffle=False, **params)
    return train_loader, eval_loader


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def validate_args(args):
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("epochs/batch_size must be positive; num_workers nonnegative")
    if args.size < 64 or args.size % 32:
        raise ValueError("size must be a multiple of 32 (at least 64)")
    if args.ddim_steps < 1 or args.overfit_samples < 1:
        raise ValueError("ddim_steps and overfit_samples must be positive")
    if not 0 <= args.seed < 2 ** 32:
        raise ValueError("seed must fit uint32")
    if not math.isfinite(args.lambda_sdf) or args.lambda_sdf < 0:
        raise ValueError("lambda_sdf must be finite and nonnegative")
    if not math.isfinite(args.sdf_radius) or args.sdf_radius <= 0:
        raise ValueError("sdf_radius must be finite and positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("lr must be finite and positive")
    if not 0 <= args.weight_decay < float("inf"):
        raise ValueError("weight_decay must be finite and nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if args.resume and not Path(args.resume).is_file():
        raise FileNotFoundError(args.resume)


def experiment_config(args, split_sha):
    return {
        "dataset": args.dataset,
        "data_path": str(Path(args.data_path).resolve()),
        "list_dir": str(Path(args.list_dir).resolve()) if args.list_dir else None,
        "split_sha256": split_sha,
        "variant": args.variant,
        "mode": args.mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "size": args.size,
        "ddim_steps": args.ddim_steps,
        "overfit_samples": args.overfit_samples,
        "seed": args.seed,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lambda_sdf": args.lambda_sdf if args.variant != "baseline" else 0.0,
        "sdf_radius": args.sdf_radius if args.variant != "baseline" else None,
    }


def run(args):
    validate_args(args)
    seed_everything(args.seed)
    sha = split_hashes(args.data_path, args.list_dir)
    config = experiment_config(args, sha)
    output_dir = Path(args.output_dir).resolve()
    if args.resume is None and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Output directory is not empty: " + str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, eval_loader = make_loaders(args)
    device = torch.device(args.device)
    model = build_model(args.variant, args.ddim_steps, shared_seed=args.seed).to(device)
    criterion = CeDiceLoss(2, [1, 1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=min(1e-6, args.lr * 0.01)
    )

    epoch_start, best_dice = 1, -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ckpt["config"] != config:
            raise ValueError("Resume config/split hashes do not match current run")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        epoch_start, best_dice = int(ckpt["epoch"]) + 1, float(ckpt["best_dice"])
        restore_rng(ckpt["rng"])

    config_file = output_dir / "config.json"
    if args.resume:
        if not config_file.is_file() or json.loads(config_file.read_text(encoding="utf-8")) != config:
            raise ValueError("Cannot resume into a different output directory/config")
    else:
        config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("MODE:", args.mode, "VARIANT:", args.variant,
          "TRAIN IMAGES:", len(train_loader.dataset),
          "EVAL IMAGES:", len(eval_loader.dataset), flush=True)
    print("EVALUATION:", "train-subset fit Dice (not validation)" if args.mode == "overfit"
          else "val.txt DDIM Dice", flush=True)

    history_file = output_dir / "history.jsonl"
    for epoch in range(epoch_start, args.epochs + 1):
        start = time.perf_counter()
        losses = []
        model.train()
        for batch in train_loader:
            report = update_batch(
                batch, model, criterion, optimizer, variant=args.variant,
                epoch=epoch, epochs=args.epochs,
                lambda_sdf=args.lambda_sdf, sdf_radius=args.sdf_radius
            )
            losses.append(report["loss_total"])
        if not losses:
            raise ValueError("Empty training loader")
        scheduler.step()
        metrics = evaluate(model, eval_loader, device, seed=args.seed + 100000)
        row = {
            "epoch": epoch, "train_loss": float(np.mean(losses)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": float(time.perf_counter() - start),
            **metrics,
        }
        improved = metrics["mean_dice"] > best_dice
        if improved:
            best_dice = metrics["mean_dice"]

        payload = {
            "epoch": epoch, "best_dice": best_dice, "config": config,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "rng": capture_rng(),
        }
        torch.save(payload, output_dir / "last.pt")
        if improved:
            torch.save(payload, output_dir / "best.pt")
        with history_file.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    return {"output_dir": str(output_dir), "last_epoch": args.epochs,
            "best_dice": best_dice, "mode": args.mode}


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("glas", "cvc"), required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--list-dir", default=None)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--mode", choices=("overfit", "train"), default="overfit")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--overfit-samples", type=int, default=2)
    parser.add_argument("--ddim-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--lambda-sdf", type=float, default=0.25)
    parser.add_argument("--sdf-radius", type=float, default=16.0)
    return parser


if __name__ == "__main__":
    options = make_parser().parse_args()
    print(json.dumps(run(options), ensure_ascii=False, indent=2))
