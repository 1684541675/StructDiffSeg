"""Paired synthetic timing; measures SDF overhead without ACDC training.

Run from repo root: python tests/benchmark_geo_label.py --output path.json
Use --production for the unchanged six-stage denoiser; default is test widths.
"""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from test_geo_label import labels, small_model
from models.DiffUNet3201.diffunet.nnunet2d_denoise import get_nnunet2d_denoise
from utils import CeDiceLoss


def benchmark(device, size, batch, rounds, production):
    model = (get_nnunet2d_denoise(1, 4, lambda_geo=0)
             if production else small_model(0)).to(device)
    mask = labels(batch=batch, size=size, device=device)
    image = torch.randn(batch, 1, size, size, device=device)
    criterion = CeDiceLoss(4, [1, 1])

    def synchronize():
        if device == "cuda":
            torch.cuda.synchronize()

    def run(kind):
        if kind == "encode":
            with torch.no_grad():
                model.encode_mask(mask)
        elif kind == "train_forward_backward":
            model.train()
            model.zero_grad(set_to_none=True)
            pred, uncertainty = model(image, mask)
            criterion(pred, mask, uncertainty, scale=0.1).backward()
        else:
            model.eval()
            model(image, ddim=True)

    rows = []
    for kind in ("encode", "train_forward_backward", "ddim_three_steps"):
        samples = {0: [], 0.25: []}
        for weight in samples:
            model.lambda_geo = weight
            for _ in range(2):
                run(kind)
            synchronize()
        for round_index in range(rounds):
            # Alternate order to reduce warm-up / temperature bias.
            for weight in ((0, 0.25) if round_index % 2 == 0 else (0.25, 0)):
                model.lambda_geo = weight
                torch.manual_seed(100 + round_index)
                synchronize()
                start = time.perf_counter()
                run(kind)
                synchronize()
                samples[weight].append((time.perf_counter() - start) * 1000)
        baseline = statistics.median(samples[0])
        geometry = statistics.median(samples[0.25])
        rows.append({"operation": kind, "baseline_ms": baseline,
                     "geometry_ms": geometry, "extra_ms": geometry - baseline,
                     "extra_percent": 100 * (geometry / baseline - 1),
                     "baseline_samples_ms": samples[0],
                     "geometry_samples_ms": samples[0.25]})
    return {"device": device, "size": size, "batch": batch, "rounds": rounds,
            "architecture": "production_six_stage_denoiser" if production else "reduced_test_denoiser",
            "timings": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda", "all"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.size < 32 or args.batch < 1 or args.rounds < 1:
        parser.error("size >= 32, batch >= 1 and rounds >= 1 required")
    if args.production and (args.size < 64 or args.size % 32):
        parser.error("production size must be >=64 and divisible by 32")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    devices = (["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
               if args.device == "all" else [args.device])
    result = {"torch": torch.__version__, "python": platform.python_version(),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "threads": torch.get_num_threads(), "warmup_per_mode": 2,
              "scope": "Synthetic denoiser only; excludes image encoder, optimizer step and data loading.",
              "results": []}
    for device in devices:
        row = benchmark(device, args.size, args.batch, args.rounds, args.production)
        result["results"].append(row)
        print(json.dumps(row, indent=2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
