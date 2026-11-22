"""Opt-in cached-data entry point; original training scripts remain unchanged.

python train_cached_2d.py --trainer unet --dataset glas --data-path ../glas \
    --mode train --epochs 100 --size 256 --batch-size 16 --output-dir runs/glas_unet_cached

Use --trainer gcf-residual with the existing GCF script's arguments. Initialization
preloads train/val deterministic preprocessing into RAM; include that startup
time when measuring whole-run cost. No hyperparameters are overridden here.
"""

import argparse
import importlib
import json

from datasets.rgb_binary_cached import CachedRGBBinaryDataset


def main(argv=None):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--trainer', required=True, choices=('unet', 'gcf-residual'))
    selected, remaining = selector.parse_known_args(argv)
    module_name = {'unet': 'train_unet_2d', 'gcf-residual': 'train_gcf_residual_2d'}[selected.trainer]
    trainer = importlib.import_module(module_name)
    args = trainer.make_parser().parse_args(remaining)
    original = trainer.RGBBinaryDataset
    # Only this invocation's dataset factory changes. The original entry points
    # retain their behavior, and the binding is restored even on interruption.
    trainer.RGBBinaryDataset = CachedRGBBinaryDataset
    try:
        print('Dataset: eager RAM cache; deterministic preprocessing cached, per-access augmentation retained', flush=True)
        result = trainer.run(args)
    finally:
        trainer.RGBBinaryDataset = original
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == '__main__':
    main()
