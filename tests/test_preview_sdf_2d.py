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

from preview_sdf_2d import error_image, run_preview, save_contact_sheet
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

    def test_preview_two_train_images_saves_contact_sheets_and_metrics(self):
        checkpoint = self.make_checkpoint()
        result = run_preview(checkpoint, data_path=self.root, device="cpu", limit=2)
        directory = Path(result["output_dir"])
        self.assertEqual(result["cases_evaluated"], 2)
        self.assertEqual(result["checkpoint_epoch"], 30)
        self.assertEqual(result["split"], "train")
        self.assertEqual(len(list(directory.glob("*.png"))), 2)
        for png in directory.glob("*.png"):
            with Image.open(png) as im:
                self.assertEqual(im.size, (192, 180))
                self.assertEqual(im.mode, "RGB")
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["checkpoint_training_mode"], "overfit")
        self.assertIn("NOT a test-set", summary["warning"])
        self.assertEqual(len(summary["cases"]), 2)
        with (directory / "metrics.csv").open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual({row["case_name"] for row in rows}, {"tr0", "tr1"})
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

    def test_mask_error_color_semantics(self):
        gt = np.array([[0, 1], [1, 0]], dtype=bool)
        pred = np.array([[0, 1], [0, 1]], dtype=bool)
        colors = error_image(gt, pred)
        np.testing.assert_array_equal(colors[0, 0], [0, 0, 0])
        np.testing.assert_array_equal(colors[0, 1], [30, 200, 50])
        np.testing.assert_array_equal(colors[1, 0], [35, 105, 245])
        np.testing.assert_array_equal(colors[1, 1], [250, 50, 50])


if __name__ == "__main__":
    unittest.main()
