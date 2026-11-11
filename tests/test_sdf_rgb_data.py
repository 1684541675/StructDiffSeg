"""GlaS/CVC PNG-to-DiffUNet integration checks with temporary synthetic files.

Run:
    python -m unittest discover -s tests -p "test_sdf_rgb_data.py" -v

No original dataset, network checkpoint, or full training is required.
"""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from datasets.rgb_binary import RGBBinaryDataset
from smoke_sdf_rgb_2d import run_smoke


class TestRGBBinarySDFData(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "images").mkdir()
        (self.root / "masks").mkdir()

        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[8:28, 8:28] = 5   # Simulated gland instance ID 5
        mask[34:54, 34:54] = 9  # Simulated gland instance ID 9
        image = np.repeat(((mask > 0).astype(np.uint8) * 255)[..., None], 3, axis=2)
        for name, split in (("sample_a", "train"), ("sample_b", "val"), ("sample_c", "test")):
            Image.fromarray(image).save(self.root / "images" / f"{name}.png")
            Image.fromarray(mask).save(self.root / "masks" / f"{name}.png")
            (self.root / f"{split}.txt").write_text(f"{name}\n", encoding="utf-8")
        self.foreground_pixels = int((mask > 0).sum())

    def test_binary_instance_merge_and_augmentation_alignment(self):
        np.random.seed(123)
        dataset = RGBBinaryDataset(self.root, split="train", output_size=(64, 64))
        sample = dataset[0]
        self.assertEqual(sample["case_name"], "sample_a")
        self.assertEqual(tuple(sample["image"].shape), (3, 64, 64))
        self.assertEqual(tuple(sample["label"].shape), (64, 64))
        self.assertEqual(sample["image"].dtype, torch.float32)
        self.assertEqual(sample["label"].dtype, torch.int64)
        self.assertEqual(set(sample["label"].unique().tolist()), {0, 1})
        self.assertEqual(int(sample["label"].sum()), self.foreground_pixels)
        # With identical input colors and no resize, synchronized rotations/flips
        # must preserve the exact image-to-mask spatial alignment.
        torch.testing.assert_close(sample["image"][0], sample["label"].float(), rtol=0, atol=0)

    def test_deterministic_val_and_test_dataloader(self):
        for split in ("val", "test"):
            ds = RGBBinaryDataset(self.root, split=split, output_size=(64, 64))
            batch = next(iter(DataLoader(ds, batch_size=1, shuffle=False)))
            self.assertEqual(tuple(batch["image"].shape), (1, 3, 64, 64))
            self.assertEqual(tuple(batch["label"].shape), (1, 64, 64))
            self.assertEqual(int(batch["label"].sum()), self.foreground_pixels)
            first = ds[0]["label"]
            torch.testing.assert_close(first, ds[0]["label"])

    def test_read_only_smoke_data_mode(self):
        result = run_smoke("glas", self.root, split="val", size=64, mode="data")
        self.assertEqual(result["samples"], 1)
        self.assertEqual(result["case_name"], "sample_b")
        self.assertEqual(result["foreground_pixels"], self.foreground_pixels)
        self.assertEqual(result["image_shape"], [1, 3, 64, 64])

    def test_full_model_forward_on_png_batch(self):
        result = run_smoke("glas", self.root, split="train",
                           size=64, mode="forward", device="cpu")
        self.assertEqual(result["diffusion_shape"], [1, 2, 64, 64])
        self.assertEqual(result["geometry_shape"], [1, 2, 64, 64])
        self.assertEqual(result["sdf_shape"], [1, 1, 64, 64])

    def test_one_optimizer_update_on_png_batch(self):
        result = run_smoke("cvc", self.root, split="train",
                           size=64, mode="step", device="cpu")
        self.assertGreater(result["loss_sdf"], 0.0)
        self.assertGreater(result["grad_sdf_head"], 0.0)
        self.assertGreater(result["grad_denoiser"], 0.0)

    def test_gcf_forward_on_png_batch(self):
        result = run_smoke("glas", self.root, split="train",
                           size=64, mode="forward", device="cpu",
                           enable_gcf=True)
        self.assertTrue(result["enable_gcf"])
        self.assertEqual(result["diffusion_shape"], [1, 2, 64, 64])
        self.assertEqual(result["geometry_shape"], [1, 2, 64, 64])
        self.assertEqual(result["sdf_shape"], [1, 1, 64, 64])

    def test_gcf_one_optimizer_update_on_png_batch(self):
        result = run_smoke("glas", self.root, split="train",
                           size=64, mode="step", device="cpu",
                           enable_gcf=True)
        self.assertTrue(result["enable_gcf"])
        self.assertGreater(result["loss_sdf"], 0.0)
        self.assertGreater(result["grad_sdf_head"], 0.0)
        self.assertGreater(result["grad_denoiser"], 0.0)

    def test_bad_rgb_mask_and_split_gate(self):
        mask = np.zeros((64, 64, 3), dtype=np.uint8)
        mask[1, 1, 0] = 255
        Image.fromarray(mask).save(self.root / "masks" / "sample_b.png")
        with self.assertRaisesRegex(ValueError, "channels must be identical"):
            RGBBinaryDataset(self.root, split="val", output_size=(64, 64))[0]
        with self.assertRaisesRegex(ValueError, "only allowed for split=train"):
            run_smoke("glas", self.root, split="val", size=64, mode="step")

    def test_rejects_invalid_sample_ids(self):
        (self.root / "train.txt").write_text("../outside\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "plain sample ID"):
            RGBBinaryDataset(self.root, split="train", output_size=(64, 64))


if __name__ == "__main__":
    unittest.main()
