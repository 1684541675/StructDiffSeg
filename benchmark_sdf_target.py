"""Measure binary SDF target-generation cost without modifying training code.

Example (from repository root):
  python benchmark_sdf_target.py --dataset glas --data-path ../glas --size 256 --batch-size 16 --device cuda --repeats 5

Use --dataset synthetic for a self-contained benchmark. Reading/resizing masks
and initially moving them to the GPU happen BEFORE timing: actual training
calls binary_sdf_target with labels already on the model device.

The breakdown duplicates the original SDF calculation, with CUDA synchronization
for stage timing. Exact parity with the original function is checked. As
instrumentation adds overhead, use the original function's separately measured
elapsed time as the primary cost estimate.
"""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from datasets.rgb_binary import RGBBinaryDataset
from models.DiffUNet3201.diffunet.sdf_target import binary_sdf_target


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def instrumented_sdf_target(mask, *, radius=16.0, dtype=torch.float32):
    """Numerically identical diagnostic implementation with per-stage timings."""
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.ndim != 3 or any(v == 0 for v in mask.shape):
        raise ValueError("Expected nonempty masks [B,H,W] or [B,1,H,W]")
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite")

    device = mask.device
    synchronize(device)
    start = time.perf_counter()
    labels = mask.detach().cpu().numpy()
    synchronize(device)
    read_end = time.perf_counter()

    if not np.isfinite(labels).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError("mask must contain only finite binary values 0 and 1")
    validate_end = time.perf_counter()

    targets = np.empty((labels.shape[0], 1, *labels.shape[1:]), dtype=np.float32)
    for i, label in enumerate(labels):
        foreground = label == 1
        if not foreground.any():
            targets[i, 0].fill(-1.0)
        elif foreground.all():
            targets[i, 0].fill(1.0)
        else:
            distance = (distance_transform_edt(foreground)
                        - distance_transform_edt(~foreground))
            targets[i, 0] = np.clip(distance, -radius, radius) / radius
    edt_end = time.perf_counter()

    output = torch.from_numpy(targets).to(device=device, dtype=dtype)
    synchronize(device)
    finish = time.perf_counter()
    times_ms = {
        "mask_device_to_cpu": (read_end - start) * 1000,
        "mask_validation": (validate_end - read_end) * 1000,
        "cpu_edt_and_target": (edt_end - validate_end) * 1000,
        "target_cpu_to_device": (finish - edt_end) * 1000,
        "total": (finish - start) * 1000,
    }
    return output, times_ms


def synthetic_masks(*, samples, size, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.ogrid[:size, :size]
    masks = []
    for i in range(samples):
        if i % 17 == 0:
            mask = np.zeros((size, size), dtype=np.int64)
        elif i % 19 == 0:
            mask = np.ones((size, size), dtype=np.int64)
        else:
            cy, cx = rng.uniform(0.25, 0.75, size=2) * size
            ry, rx = rng.uniform(0.08, 0.3, size=2) * size
            mask = ((((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1).astype(np.int64)
        masks.append(torch.from_numpy(mask))
    return masks


def prepare_batches(args, device):
    if args.dataset == "synthetic":
        masks = synthetic_masks(samples=args.samples, size=args.size, seed=args.seed)
    else:
        if not args.data_path:
            raise ValueError("--data-path is required for GlaS/CVC")
        data = RGBBinaryDataset(args.data_path, args.split,
                                output_size=(args.size, args.size), augment=False)
        # Reuse original mask preprocessing, but exclude it from timed SDF calls.
        masks = [data[i]["label"] for i in range(len(data))]
    batches = [torch.stack(masks[i:i + args.batch_size]).to(device)
               for i in range(0, len(masks), args.batch_size)]
    synchronize(device)
    return batches, len(masks)


def summarize(values):
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def benchmark(batches, *, radius=16.0, repeats=5, warmup=3):
    if not batches or repeats < 1 or warmup < 0:
        raise ValueError("Need nonempty batches, repeats >= 1, warmup >= 0")
    device = batches[0].device
    for i in range(warmup):
        binary_sdf_target(batches[i % len(batches)], radius=radius)
    synchronize(device)

    original_epoch_ms = []
    original_batch_ms = []
    for _ in range(repeats):
        per_epoch = 0.0
        for batch in batches:
            synchronize(device)
            t0 = time.perf_counter()
            binary_sdf_target(batch, radius=radius)
            synchronize(device)
            elapsed = (time.perf_counter() - t0) * 1000
            original_batch_ms.append(elapsed)
            per_epoch += elapsed
        original_epoch_ms.append(per_epoch)

    # Stage breakdown is a separate pass (not nested in the original timing).
    # Verify equivalence for every batch size, including a short final batch.
    breakdown_epoch = []
    stage_names = ("mask_device_to_cpu", "mask_validation",
                   "cpu_edt_and_target", "target_cpu_to_device", "total")
    stage_epoch = {name: [] for name in stage_names}
    for repeat in range(repeats):
        sum_by_stage = {name: 0.0 for name in stage_names}
        for batch_index, batch in enumerate(batches):
            result, times = instrumented_sdf_target(batch, radius=radius)
            if repeat == 0:
                expected = binary_sdf_target(batch, radius=radius)
                if not torch.equal(result, expected):
                    raise AssertionError(f"Instrumented result differs for batch {batch_index}")
            for name in stage_names:
                sum_by_stage[name] += times[name]
        breakdown_epoch.append(sum_by_stage["total"])
        for name in stage_names:
            stage_epoch[name].append(sum_by_stage[name])

    return {
        "device": str(device),
        "batches_per_epoch": len(batches),
        "samples_per_epoch": sum(len(b) for b in batches),
        "repeats": repeats,
        "warmup_calls": warmup,
        "original_sdf_ms_per_batch": summarize(original_batch_ms),
        "original_sdf_ms_per_epoch": summarize(original_epoch_ms),
        "instrumented_ms_per_epoch": summarize(breakdown_epoch),
        "breakdown_ms_per_epoch": {
            name: summarize(stage_epoch[name]) for name in stage_names
        },
        "exact_output_match": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("glas", "cvc", "synthetic"), default="synthetic")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--samples", type=int, default=32,
                        help="Number of samples for synthetic benchmark only")
    parser.add_argument("--radius", type=float, default=16.0)
    parser.add_argument("--repeats", type=int, default=5,
                        help="Measured passes over the entire split")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Unmeasured SDF calls before timing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args(argv)
    if (args.size < 1 or args.batch_size < 1 or args.samples < 1
            or args.repeats < 1 or args.warmup < 0):
        parser.error("size, batch-size, samples, repeats must be positive; warmup >= 0")
    if not math.isfinite(args.radius) or args.radius <= 0:
        parser.error("radius must be positive and finite")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else "cpu" if args.device == "auto" else args.device
    )
    batches, sample_count = prepare_batches(args, device)
    result = benchmark(batches, radius=args.radius, repeats=args.repeats,
                       warmup=args.warmup)
    result.update({
        "dataset": args.dataset, "split": args.split, "size": args.size,
        "batch_size": args.batch_size, "radius": args.radius,
        "sample_count": sample_count, "torch_version": torch.__version__
    })
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output_json:
        target = Path(args.output_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8"
        )
    return result


if __name__ == "__main__":
    main()
