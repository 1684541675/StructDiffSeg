"""GlaS/CVC RGB binary PNG dataset, adapted from feature/glas-cvc.

Expected layout:
    <root>/images/<case_id>.png
    <root>/masks/<case_id>.png
    <list_dir>/<train|val|test>.txt

GlaS instance IDs are merged to binary foreground for this initial 2D task.
This intentionally does NOT retain instance-level gland separation labels.
Only train samples receive synchronized random flips and 90-degree rotations.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class RGBBinaryDataset(Dataset):
    def __init__(self, root, split="train", *, list_dir=None, output_size=(256, 256),
                 augment=None):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be 'train', 'val', or 'test'")
        if (len(output_size) != 2 or any(
                not isinstance(v, int) or v <= 0 for v in output_size)):
            raise ValueError("output_size must be two positive integers (H, W)")
        if augment is None:
            augment = split == "train"
        if augment and output_size[0] != output_size[1]:
            raise ValueError("90-degree rotations require a square output_size")

        self.root = Path(root)
        self.split = split
        self.output_size = tuple(output_size)
        self.augment = bool(augment)
        split_file = Path(list_dir) / (split + ".txt") if list_dir is not None else self.root / (split + ".txt")
        with split_file.open(encoding="utf-8-sig") as handle:
            self.sample_list = [line.strip() for line in handle if line.strip()]
        if not self.sample_list:
            raise ValueError(f"Empty split file: {split_file}")
        if len(self.sample_list) != len(set(self.sample_list)):
            raise ValueError(f"Duplicate sample IDs in {split_file}")

        self.data = []
        for case_id in self.sample_list:
            if case_id in (".", "..") or any(char in case_id for char in "/\\:"):
                raise ValueError(f"Expected plain sample ID: {case_id!r}")
            image_path = self.root / "images" / (case_id + ".png")
            mask_path = self.root / "masks" / (case_id + ".png")
            if not image_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(f"Missing image/mask pair for {case_id}: {image_path}, {mask_path}")
            self.data.append((image_path, mask_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, mask_path = self.data[index]
        with Image.open(image_path) as image_file:
            image = np.array(image_file.convert("RGB"))
        with Image.open(mask_path) as mask_file:
            raw_mask = np.array(mask_file)

        # The existing CVC loader supports identical-channel 3-channel PNGs.
        if raw_mask.ndim == 3 and raw_mask.shape[2] == 3:
            if not np.all(raw_mask == raw_mask[..., :1]):
                raise ValueError(f"RGB mask channels must be identical: {mask_path}")
            raw_mask = raw_mask[..., 0]
        if raw_mask.ndim != 2:
            raise ValueError(f"Expected a 2D grayscale or instance-ID mask: {mask_path}")
        if image.shape[:2] != raw_mask.shape:
            raise ValueError(f"Image/mask sizes differ: {image_path}, {mask_path}")

        # Convert nonzero gland instance IDs or white polyp pixels to foreground
        # BEFORE resizing, avoiding mixed class labels after interpolation.
        binary_mask = (raw_mask > 0).astype(np.uint8)
        width, height = self.output_size[1], self.output_size[0]
        image = np.array(Image.fromarray(image).resize(
            (width, height), Image.Resampling.BILINEAR))
        binary_mask = np.array(Image.fromarray(binary_mask).resize(
            (width, height), Image.Resampling.NEAREST))

        if self.augment:
            k = int(np.random.randint(0, 4))
            image = np.rot90(image, k)
            binary_mask = np.rot90(binary_mask, k)
            axis = int(np.random.randint(0, 2))
            image = np.flip(image, axis=axis)
            binary_mask = np.flip(binary_mask, axis=axis)

        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)) / 255.0
        mask_tensor = torch.from_numpy(
            np.ascontiguousarray(binary_mask, dtype=np.int64))
        return {
            "image": image_tensor,
            "label": mask_tensor,
            "case_name": self.sample_list[index],
        }
