"""PNG visualization / checkpoint preview regression tests.

Run: python -m unittest discover -s tests -p "test_preview_sdf_2d.py" -v
Temporary files only; no real GlaS/CVC or training.
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from preview_sdf_2d import run_preview, save_case_images
from train_sdf_2d import build_model, split_hashes


class TestCheckpointPreview(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "images").mkdir()
        (self.root / "masks").mkdir()
        for i, case in enumerate(("tr0", "tr1", "v0", "te0")):
            img = np.full((64, 64, 3), 50 + 20*i, dtype=np.uint8)
            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[11:33, 17+i:40+i] = 8+i
            Image.fromarray(img).save(self.root / "images" / (case + ".png"))
            Image.fromarray(mask).save(self.root / "masks" / (case + ".png"))
        for split, cases in (("train", "tr0\ntr1\n"), ("val", "v0\n"),
                             ("test", "te0\n")):
            (self.root / (split + ".txt")).write_text(cases, encoding="utf-8")

    def make_checkpoint(self, variant="gcf", mode="overfit"):
        cfg = {
            "variant": variant, "mode": mode, "dataset": "glas",
            "data_path": str(self.root), "list_dir": None,
            "split_sha256": split_hashes(self.root, None),
            "size": 64, "ddim_steps": 3, "seed": 42,
            "overfit_samples": 2,
        }
        model = build_model(variant, ddim_steps=3, shared_seed=42)
        checkpoint = self.root / (variant + "_best.pt")
        torch.save({
            "epoch": 30, "best_dice": 0.8, "config": cfg,
            "model": model.state_dict(),
        }, checkpoint)
        return checkpoint

    def test_preview_two_train_images_saves_separate_pngs_and_metrics(self):
        checkpoint = self.make_checkpoint()
        result = run_preview(checkpoint, data_path=self.root, device="cpu", limit=2)
        directory = Path(result["output_dir"])
        self.assertEqual(result["cases_evaluated"], 2)
        self.assertEqual(result["checkpoint_epoch"], 30)
        self.assertEqual(result["split"], "train")
        self.assertEqual(len(list(directory.glob("*.png"))), 6)
        for case in ("000_tr0", "001_tr1"):
            for suffix, mode in (("image", "RGB"), ("gt", "L"), ("pred", "L")):
                png = directory / f"{case}_{suffix}.png"
                self.assertTrue(png.is_file())
                with Image.open(png) as im:
                    self.assertEqual(im.size, (64, 64))
                    self.assertEqual(im.mode, mode)
                    if mode == "L":
                        self.assertTrue(set(np.unique(np.array(im))).issubset({0, 255}))
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["checkpoint_training_mode"], "overfit")
        self.assertIn("NOT a test-set", summary["warning"])
        self.assertEqual(len(summary["cases"]), 2)
        with (directory / "metrics.csv").open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual({row["case_name"] for row in rows}, {"tr0", "tr1"})
        for row in rows:
            for key in ("image_png", "gt_png", "prediction_png"):
                self.assertTrue((directory / row[key]).is_file())
        with self.assertRaisesRegex(FileExistsError, "not empty"):
            run_preview(checkpoint, data_path=self.root, device="cpu", limit=2)

    def test_preview_validation_and_test_gate(self):
        checkpoint = self.make_checkpoint("baseline")
        result = run_preview(checkpoint, data_path=self.root, device="cpu",
                             split="val", limit=9)
        self.assertEqual(result["cases_evaluated"], 1)
        with self.assertRaisesRegex(ValueError, "never test"):
            run_preview(checkpoint, data_path=self.root, device="cpu",
                        split="test", limit=1)

    def test_reject_changed_split_lists(self):
        checkpoint = self.make_checkpoint()
        # Change only the list file's bytes; keep IDs unique and unchanged.
        # This isolates the SHA-256 mismatch guard from the overlap guard.
        (self.root / "val.txt").write_text("v0\n\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differ"):
            run_preview(checkpoint, data_path=self.root, device="cpu", limit=1)

    def test_reject_overlapping_split_ids(self):
        checkpoint = self.make_checkpoint()
        (self.root / "val.txt").write_text("te0\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Overlapping val/test"):
            run_preview(checkpoint, data_path=self.root, device="cpu", limit=1)

    def test_independent_image_and_binary_mask_png_content(self):
        rgb = np.array([[[10, 30, 50], [20, 40, 60]],
                        [[70, 90, 110], [120, 140, 160]]], dtype=np.uint8)
        gt = np.array([[0, 1], [1, 0]], dtype=bool)
        pred = np.array([[0, 1], [0, 1]], dtype=bool)
        names = save_case_images(self.root, "sample", rgb, gt, pred)
        self.assertEqual(set(names), {"image_png", "gt_png", "prediction_png"})
        self.assertEqual(len(list(self.root.glob("sample*.png"))), 3)
        with Image.open(self.root / names["image_png"]) as image:
            self.assertEqual(image.mode, "RGB")
            np.testing.assert_array_equal(np.array(image), rgb)
        with Image.open(self.root / names["gt_png"]) as image:
            self.assertEqual(image.mode, "L")
            np.testing.assert_array_equal(np.array(image), gt.astype(np.uint8) * 255)
        with Image.open(self.root / names["prediction_png"]) as image:
            self.assertEqual(image.mode, "L")
            np.testing.assert_array_equal(np.array(image), pred.astype(np.uint8) * 255)


if __name__ == "__main__":
    unittest.main()
