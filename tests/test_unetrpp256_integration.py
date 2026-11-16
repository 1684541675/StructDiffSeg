"""UNETRPP256 integration tests on temporary synthetic binary data.

Run:
  python -m unittest discover -s tests -p "test_unetrpp256_integration.py" -v

The repository's UNETRPP256 implementation is a 2D model with total
downsampling factor 32. This test uses 64x64 inputs and the original model
configuration (feature_size=16, hidden_size=256, four EPA encoder stages).
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from models.UNETRPP256.adapter import build_unetrpp256
from test_unetrpp256_2d import run_test
from train_unetrpp256_2d import make_parser, run
from train_sdf_2d import split_hashes


class TestUNETRPP256Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "images").mkdir()
        (self.root / "masks").mkdir()
        for i, case in enumerate(("tr0", "tr1", "val0", "te0", "te1")):
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            image[:] = 40 + 12 * i
            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[13:39, 17+i:46+i] = 255
            image[mask > 0] = 220
            Image.fromarray(image).save(self.root / "images" / (case + ".png"))
            Image.fromarray(mask).save(self.root / "masks" / (case + ".png"))
        for split, names in (
            ("train", "tr0\ntr1\n"),
            ("val", "val0\n"),
            ("test", "te0\nte1\n"),
        ):
            (self.root / (split + ".txt")).write_text(names, encoding="utf-8")

    def args(self, *more):
        return make_parser().parse_args([
            "--dataset", "cvc", "--data-path", str(self.root),
            "--mode", "train", "--epochs", "1", "--size", "64",
            "--batch-size", "1", "--device", "cpu",
            "--output-dir", str(self.root / "run"),
            *more
        ])

    def fake_checkpoint(self, *, mode="train", filename="best.pt"):
        folder = self.root / "fake"
        folder.mkdir(exist_ok=True)
        config = {
            "architecture": "UNETRPP256", "dataset": "cvc", "mode": mode,
            "data_path": str(self.root), "list_dir": None,
            "split_sha256": split_hashes(self.root, None),
            "size": 64,
            "model_config": {
                "feature_size": 16,
                "hidden_size": 256,
                "num_heads": 4,
                "depths": [3, 3, 3, 3],
                "dims": [32, 64, 128, 256],
                "do_ds": False,
            },
        }
        path = folder / filename
        model = build_unetrpp256(64)
        torch.save({
            "epoch": 1, "best_dice": 0.7, "config": config,
            "model": model.state_dict()
        }, path)
        return path

    def test_model_forward_and_backward_with_64_pixels(self):
        model = build_unetrpp256(64)
        x = torch.randn(1, 3, 64, 64)
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (1, 2, 64, 64))
        logits.square().mean().backward()
        self.assertIsNotNone(next(model.parameters()).grad)

    def test_training_validation_and_heldout_testing_with_checkpoint(self):
        training = run(self.args())
        self.assertEqual(training["last_epoch"], 1)
        folder = self.root / "run"
        for filename in ("config.json", "history.jsonl", "best.pt", "last.pt"):
            self.assertTrue((folder / filename).is_file())
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["architecture"], "UNETRPP256")
        self.assertEqual(config["model_config"]["feature_size"], 16)
        self.assertEqual(config["model_config"]["num_heads"], 4)
        log = json.loads((folder / "history.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(log["cases_evaluated"], 1)
        self.assertTrue(log["did_validate"])

        output = run_test(folder / "best.pt", data_path=self.root, device="cpu")
        self.assertEqual(output["cases_evaluated"], 2)
        directory = Path(output["output_dir"])
        self.assertEqual(len(list(directory.glob("*.png"))), 6)
        with (directory / "metrics.csv").open(
            encoding="utf-8-sig", newline=""
        ) as file:
            metrics = list(csv.DictReader(file))
        self.assertEqual([row["case_name"] for row in metrics], ["te0", "te1"])
        for row in metrics:
            for field, mode in (
                ("image_png", "RGB"), ("gt_png", "L"), ("prediction_png", "L")
            ):
                with Image.open(directory / row[field]) as picture:
                    self.assertEqual(picture.mode, mode)
                    self.assertEqual(picture.size, (64, 64))
        summary = json.loads(
            (directory / "summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["architecture"], "UNETRPP256")
        self.assertEqual(summary["cases_evaluated"], 2)
        self.assertEqual(summary["split"], "test")
        with self.assertRaisesRegex(FileExistsError, "not empty"):
            run_test(folder / "best.pt", data_path=self.root, device="cpu")

    def test_reject_unsupported_image_sizes(self):
        with self.assertRaisesRegex(ValueError, "at least 64"):
            build_unetrpp256(32)
        with self.assertRaisesRegex(ValueError, "divisible by 32"):
            build_unetrpp256(80)
        for size in (64, 96, 128, 256):
            self.assertIsNotNone(size % 32 == 0)

    def test_reject_overfit_checkpoint(self):
        checkpoint = self.fake_checkpoint(mode="overfit")
        with self.assertRaisesRegex(ValueError, "not --mode overfit"):
            run_test(checkpoint, data_path=self.root, device="cpu")

    def test_reject_modified_split_hash(self):
        checkpoint = self.fake_checkpoint()
        (self.root / "test.txt").write_text("te0\nte1\n\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differ"):
            run_test(checkpoint, data_path=self.root, device="cpu")

    def test_reject_overlapping_train_and_test(self):
        checkpoint = self.fake_checkpoint()
        (self.root / "test.txt").write_text("tr0\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Overlapping train/test"):
            run_test(checkpoint, data_path=self.root, device="cpu")

    def test_reject_last_checkpoint(self):
        checkpoint = self.fake_checkpoint(filename="last.pt")
        with self.assertRaisesRegex(ValueError, "best.pt"):
            run_test(checkpoint, data_path=self.root, device="cpu")


if __name__ == "__main__":
    unittest.main()
