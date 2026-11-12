"""Regression tests for held-out test-only DiffUNet evaluation.

Run:
  python -m unittest discover -s tests -p "test_test_sdf_2d.py" -v

All datasets and checkpoints here are synthetic / temporary (no real test data).
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from test_sdf_2d import run_test, symmetric_hd95_px
from train_sdf_2d import build_model, split_hashes


class TestHeldOutEvaluation(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "images").mkdir()
        (self.root / "masks").mkdir()

        for i, name in enumerate(("tr0", "val0", "te0", "te1")):
            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[12:42, 15+i:45+i] = 10+i
            image = np.full((64, 64, 3), 45+i*12, dtype=np.uint8)
            image[mask > 0, :] = 200
            Image.fromarray(image).save(self.root / "images" / (name + ".png"))
            Image.fromarray(mask).save(self.root / "masks" / (name + ".png"))
        for split, ids in (
            ("train", "tr0\n"), ("val", "val0\n"), ("test", "te0\nte1\n")
        ):
            (self.root / (split + ".txt")).write_text(ids, encoding="utf-8")

    def checkpoint(self, variant="baseline", mode="train"):
        path = self.root / (variant + "_" + mode) / "best.pt"
        path.parent.mkdir(parents=True)
        config = {
            "variant": variant,
            "mode": mode,
            "dataset": "glas",
            "data_path": str(self.root),
            "list_dir": None,
            "split_sha256": split_hashes(self.root, None),
            "size": 64, "ddim_steps": 3, "seed": 42,
        }
        model = build_model(variant, ddim_steps=3, shared_seed=42)
        torch.save({
            "config": config,
            "model": model.state_dict(),
            "epoch": 6,
            "best_dice": 0.72
        }, path)
        return path

    def test_full_held_out_split_exports_three_separate_pngs_and_metrics(self):
        checkpoint = self.checkpoint()
        result = run_test(checkpoint, data_path=self.root, device="cpu")
        self.assertEqual(result["cases_evaluated"], 2)
        directory = Path(result["output_dir"])
        images = sorted(directory.glob("*.png"))
        self.assertEqual(len(images), 6)
        with (directory / "metrics.csv").open(
            encoding="utf-8-sig", newline=""
        ) as file:
            rows = list(csv.DictReader(file))
        self.assertEqual([r["case_name"] for r in rows], ["te0", "te1"])
        self.assertNotIn("tr0", [r["case_name"] for r in rows])
        self.assertNotIn("val0", [r["case_name"] for r in rows])
        for row in rows:
            self.assertGreaterEqual(float(row["dice"]), 0.0)
            self.assertLessEqual(float(row["dice"]), 1.0)
            self.assertGreaterEqual(float(row["iou"]), 0.0)
            for key, mode in (
                ("image_png", "RGB"),
                ("gt_png", "L"),
                ("prediction_png", "L"),
            ):
                with Image.open(directory / row[key]) as image:
                    self.assertEqual(image.size, (64, 64))
                    self.assertEqual(image.mode, mode)
                    if mode == "L":
                        self.assertTrue(
                            set(np.unique(np.array(image))).issubset({0, 255})
                        )
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["split"], "test")
        self.assertEqual(summary["checkpoint_epoch"], 6)
        self.assertEqual(summary["checkpoint_validation_best_dice"], 0.72)
        self.assertEqual(summary["cases_evaluated"], 2)
        self.assertIn("not instance challenge scores", summary["protocol_note"])
        self.assertIn("mean_hd95_px", summary)
        self.assertEqual(summary["cases"], [
            {k: (float(v) if k in ("dice", "iou") else
                 None if k == "hd95_px" and v is None else v)
             for k, v in rec.items()} for rec in summary["cases"]
        ])
        with self.assertRaisesRegex(FileExistsError, "not empty"):
            run_test(checkpoint, data_path=self.root, device="cpu")

    def test_reject_overfit_checkpoint(self):
        checkpoint = self.checkpoint(mode="overfit")
        with self.assertRaisesRegex(ValueError, "not --mode overfit"):
            run_test(checkpoint, data_path=self.root, device="cpu")
        self.assertFalse((checkpoint.parent / "test_results").exists())

    def test_reject_changed_split_hash(self):
        checkpoint = self.checkpoint()
        # Whitespace-only change keeps the test IDs disjoint; changes SHA256.
        (self.root / "test.txt").write_text("te0\nte1\n\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differ"):
            run_test(checkpoint, data_path=self.root, device="cpu")

    def test_reject_overlapping_split_ids(self):
        checkpoint = self.checkpoint()
        (self.root / "test.txt").write_text("tr0\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Overlapping train/test"):
            run_test(checkpoint, data_path=self.root, device="cpu")

    def test_reject_nonbest_file(self):
        checkpoint = self.checkpoint()
        renamed = checkpoint.parent / "last.pt"
        checkpoint.rename(renamed)
        with self.assertRaisesRegex(ValueError, "best.pt"):
            run_test(renamed, data_path=self.root, device="cpu")

    def test_hd95_convention_and_empty_masks(self):
        gt = np.zeros((32, 32), dtype=bool)
        pr = np.zeros_like(gt)
        self.assertEqual(symmetric_hd95_px(gt, pr), 0.0)
        gt[10:20, 10:20] = True
        self.assertIsNone(symmetric_hd95_px(gt, pr))
        self.assertIsNone(symmetric_hd95_px(pr, gt))
        np.testing.assert_allclose(symmetric_hd95_px(gt, gt), 0.0)
        pr[10:20, 11:21] = True
        self.assertGreater(symmetric_hd95_px(pr, gt), 0.0)
        self.assertAlmostEqual(symmetric_hd95_px(pr, gt),
                               symmetric_hd95_px(gt, pr), places=12)
        with self.assertRaisesRegex(ValueError, "same-shaped"):
            symmetric_hd95_px(pr, np.zeros((16, 16), dtype=bool))


if __name__ == "__main__":
    unittest.main()
