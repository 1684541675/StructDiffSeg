"""Opt-in eager RAM cache for RGBBinaryDataset; original loader is unchanged.

Cache lifetime is one Dataset instance. Recreate it after changing source PNGs.
Only deterministic preprocessing is cached; augmentation still runs on every
training access. Cache is built before DataLoader workers start, avoiding a
lazy cache being rebuilt every epoch when persistent_workers=False. Spawned
workers may hold their own copy, so include worker count in RAM planning.
"""

import numpy as np
import torch

from datasets.rgb_binary import RGBBinaryDataset


class CachedRGBBinaryDataset(RGBBinaryDataset):
    """Drop-in constructor/interface, with compact uint8 RGB and binary cache."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        augment = self.augment
        self.augment = False
        cache = []
        try:
            for index in range(len(self)):
                # Reuse the original decode, mask validation, interpolation and
                # normalization exactly. Recover byte RGB values for compact RAM
                # storage; round reverses float32 division by 255 without truncation.
                sample = super().__getitem__(index)
                image = np.rint(sample['image'].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                label = sample['label'].numpy().astype(np.uint8)
                image.setflags(write=False)
                label.setflags(write=False)
                cache.append((image, label))
        finally:
            self.augment = augment
        self._cache = tuple(cache)
        self.cache_nbytes = sum(image.nbytes + label.nbytes for image, label in self._cache)

    def __getitem__(self, index):
        image, label = self._cache[index]
        if self.augment:
            k = int(np.random.randint(0, 4))
            image, label = np.rot90(image, k), np.rot90(label, k)
            axis = int(np.random.randint(0, 2))
            image, label = np.flip(image, axis=axis), np.flip(label, axis=axis)
        # Always return fresh tensors: caller mutation cannot corrupt the cache.
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)) / 255.0
        label_tensor = torch.from_numpy(np.ascontiguousarray(label, dtype=np.int64))
        return {'image': image_tensor, 'label': label_tensor,
                'case_name': self.sample_list[index]}
