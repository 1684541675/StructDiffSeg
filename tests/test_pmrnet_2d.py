"""Test PMRNet integration without accessing real GlaS/CVC test images.

  python -m unittest discover -s tests -p "test_pmrnet_2d.py" -v
"""
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from models.PMRNet.pmrnet import PMRNet
from train_pmrnet_2d import make_parser, run
from train_sdf_2d import split_hashes


# unittest discovers this file as module 'test_pmrnet_2d'. The CLI script in
# the repository root has the same basename, so a regular import would import
# this partially initialized test module instead. Load the CLI under a distinct
# module name to exercise its real run_test function without changing the
# public command 'python test_pmrnet_2d.py'.
_eval_file = Path(__file__).resolve().parents[1] / "test_pmrnet_2d.py"
_spec = importlib.util.spec_from_file_location("_pmrnet_eval_entrypoint", _eval_file)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load PMRNet evaluation script: {_eval_file}")
_eval_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_module)
run_test = _eval_module.run_test


class TestPMRNetIntegration(unittest.TestCase):
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

    def fake_checkpoint(self, *, mode="train"):
        folder = self.root / "fake"
        folder.mkdir(exist_ok=True)
        config = {
            "architecture": "PMRNet", "dataset": "cvc", "mode": mode,
            "data_path": str(self.root), "list_dir": None,
            "split_sha256": split_hashes(self.root, None),
            "size": 64,
        }
        path = folder / "best.pt"
        torch.save({
            "epoch": 1, "best_dice": 0.7, "config": config,
            "model": PMRNet(3, 2).state_dict()
        }, path)
        return path

    def test_model_forward_and_backward_with_64_pixels(self):
        model = PMRNet(3, 2)
        x = torch.randn(1, 3, 64, 64)
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (1, 2, 64, 64))
        logits.square().mean().backward()
        self.assertIsNotNone(next(model.parameters()).grad)

    def test_training_validation_and_heldout_testing_with_real_checkpoint(self):
        training = run(self.args())
        self.assertEqual(training["last_epoch"], 1)
        folder = self.root / "run"
        for filename in ("config.json", "history.jsonl", "best.pt", "last.pt"):
            self.assertTrue((folder / filename).is_file())
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["architecture"], "PMRNet")
        log = json.loads((folder / "history.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(log["cases_evaluated"], 1)
        self.assertTrue(log["did_validate"])

        output = run_test(folder / "best.pt", data_path=self.root, device="cpu")
        self.assertEqual(output["cases_evaluated"], 2)
        directory = Path(output["output_dir"])
        self.assertEqual(len(list(directory.glob("*.png"))), 6)
        with (directory / "metrics.csv").open(encoding="utf-8-sig", newline="") as file:
            metrics = list(csv.DictReader(file))
        self.assertEqual([m["case_name"] for m in metrics], ["te0", "te1"])
        for m in metrics:
            for field, mode in (
                ("image_png", "RGB"), ("gt_png", "L"), ("prediction_png", "L")
            ):
                with Image.open(directory / m[field]) as picture:
                    self.assertEqual(picture.mode, mode)
                    self.assertEqual(picture.size, (64, 64))
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["architecture"], "PMRNet")
        self.assertEqual(summary["cases_evaluated"], 2)
        self.assertEqual(summary["split"], "test")
        with self.assertRaisesRegex(FileExistsError, "not empty"):
            run_test(folder / "best.pt", data_path=self.root, device="cpu")

    def test_reject_overfit_model_at_test_time(self):
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

    def test_reject_last_checkpoint_file(self):
        checkpoint = self.fake_checkpoint()
        renamed = checkpoint.parent / "last.pt"
        checkpoint.rename(renamed)
        with self.assertRaisesRegex(ValueError, "best.pt"):
            run_test(renamed, data_path=self.root, device="cpu")

    def test_reject_other_architecture_checkpoint(self):
        checkpoint = self.fake_checkpoint()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        saved["config"]["architecture"] = "DiffUNet"
        torch.save(saved, checkpoint)
        with self.assertRaisesRegex(ValueError, "not from train_pmrnet"):
            run_test(checkpoint, data_path=self.root, device="cpu")

    def test_validation_loss_and_size_arguments(self):
        args = self.args("--size", "65")
        with self.assertRaisesRegex(ValueError, "multiple of 8"):
            run(args)
        args = self.args("--val-interval", "0")
        with self.assertRaisesRegex(ValueError, "positive"):
            run(args)


if __name__ == "__main__":
    unittest.main()
