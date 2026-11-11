"""Tests for the first reproducible 2D trainer iteration.

Run: python -m unittest discover -s tests -p "test_train_sdf_2d.py" -v

Uses only temporary synthetic PNGs. No GlaS/CVC data files are required.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from train_sdf_2d import (
    baseline_update, binary_case_metrics, build_model, make_loaders,
    make_parser, run, split_hashes,
)
from utils import CeDiceLoss


class TestTrainSDF2D(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "images").mkdir()
        (self.root / "masks").mkdir()
        for i, identifier in enumerate(("a", "b", "c", "d")):
            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[12+i:42+i, 15:47] = 5 + i
            rgb = np.full((64, 64, 3), 40 + i * 20, dtype=np.uint8)
            rgb[mask > 0, :] = 200
            Image.fromarray(rgb).save(self.root / "images" / (identifier + ".png"))
            Image.fromarray(mask).save(self.root / "masks" / (identifier + ".png"))
        (self.root / "train.txt").write_text("a\nb\n", encoding="utf-8")
        (self.root / "val.txt").write_text("c\n", encoding="utf-8")
        (self.root / "test.txt").write_text("d\n", encoding="utf-8")

    def args(self, *extra):
        return make_parser().parse_args([
            "--dataset", "glas", "--data-path", str(self.root),
            "--variant", "gcf", "--mode", "overfit",
            "--epochs", "1", "--overfit-samples", "1",
            "--size", "64", "--device", "cpu",
            "--output-dir", str(self.root / "run"),
            *extra,
        ])

    def test_shared_weights_match_exactly_for_ablation(self):
        baseline = build_model("baseline", shared_seed=41)
        baseline_state = baseline.state_dict()
        self.assertFalse(any("sdf_head." in k or "gcf." in k for k in baseline_state))
        for variant in ("sdf", "gcf"):
            other = build_model(variant, shared_seed=41)
            other_state = other.state_dict()
            for key, value in baseline_state.items():
                torch.testing.assert_close(value, other_state[key], rtol=0, atol=0)
            self.assertTrue(any("sdf_head." in k for k in other_state))
            self.assertEqual(any("gcf." in k for k in other_state), variant == "gcf")
            del other

    def test_gcf_decoder_encoder_state_keys_are_shared_aliases(self):
        model = build_model("gcf", shared_seed=41)
        self.assertIs(model.denoise_model.encoder,
                      model.denoise_model.decoder.encoder)
        state = model.state_dict()
        for suffix in ("gamma", "gate_conv.weight", "time_projection.bias"):
            encoder_key = "denoise_model.encoder.gcf." + suffix
            decoder_alias_key = "denoise_model.decoder.encoder.gcf." + suffix
            self.assertIn(encoder_key, state)
            self.assertIn(decoder_alias_key, state)
            torch.testing.assert_close(
                state[encoder_key], state[decoder_alias_key], rtol=0, atol=0)

    def test_split_hashes_and_overlap_guard(self):
        hashes = split_hashes(self.root, None)
        self.assertEqual(set(hashes), {"train", "val", "test"})
        self.assertEqual(len(hashes["train"]), 64)
        (self.root / "val.txt").write_text("a\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Overlapping train/val"):
            split_hashes(self.root, None)

    def test_metrics_per_case_and_empty_policy(self):
        prediction = torch.zeros(3, 8, 8, dtype=torch.long)
        target = torch.zeros_like(prediction)
        prediction[1, :4, :4] = 1
        target[1, :4, :4] = 1
        prediction[2, :4, :4] = 1
        target[2, 4:8, 4:8] = 1
        dice, iou = binary_case_metrics(prediction, target)
        self.assertEqual(dice, [1.0, 1.0, 0.0])
        self.assertEqual(iou, [1.0, 1.0, 0.0])

    def test_loaders_use_validation_only_in_train_mode(self):
        train_args = self.args("--mode", "train")
        train_loader, val_loader = make_loaders(train_args)
        self.assertEqual(len(train_loader.dataset), 2)
        self.assertEqual(len(val_loader.dataset), 1)
        self.assertEqual(val_loader.dataset.sample_list, ["c"])
        overfit_loader, fit_eval_loader = make_loaders(self.args())
        self.assertEqual(len(overfit_loader.dataset), 1)
        self.assertEqual(len(fit_eval_loader.dataset), 1)
        self.assertIs(overfit_loader.dataset, fit_eval_loader.dataset)

    def test_baseline_one_update_without_sdf_head(self):
        model = build_model("baseline", shared_seed=17)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        report = baseline_update(
            {"image": torch.randn(1, 3, 64, 64),
             "label": torch.zeros(1, 64, 64, dtype=torch.long)},
            model, CeDiceLoss(2, [1, 1]), optimizer, epoch=1, epochs=2,
        )
        self.assertTrue(np.isfinite(report["loss_total"]))
        self.assertEqual(report["loss_sdf"], 0.0)
        self.assertAlmostEqual(report["loss_total"],
                               report["loss_diffusion"] + report["loss_geometry"],
                               places=5)

    def test_train_mode_uses_held_out_validation_and_saves_best(self):
        args = self.args("--mode", "train", "--variant", "baseline",
                         "--output-dir", str(self.root / "formal_baseline"))
        result = run(args)
        self.assertEqual(result["mode"], "train")
        self.assertTrue((self.root / "formal_baseline" / "best.pt").is_file())
        rows = (self.root / "formal_baseline" / "history.jsonl").read_text().splitlines()
        self.assertEqual(len(rows), 1)
        self.assertEqual(json.loads(rows[0])["cases_evaluated"], 1)

    def test_overfit_run_saves_artifacts_and_resumes(self):
        args = self.args()
        report = run(args)
        self.assertEqual(report["mode"], "overfit")
        self.assertTrue((self.root / "run" / "best.pt").is_file())
        self.assertTrue((self.root / "run" / "last.pt").is_file())
        history = (self.root / "run" / "history.jsonl").read_text().splitlines()
        self.assertEqual(len(history), 1)
        row = json.loads(history[0])
        self.assertEqual(row["cases_evaluated"], 1)
        self.assertTrue(0 <= row["mean_dice"] <= 1)
        with self.assertRaisesRegex(FileExistsError, "not empty"):
            run(args)
        # Completed run can be reopened using its trusted local checkpoint.
        resumed = self.args("--resume", str(self.root / "run" / "last.pt"))
        self.assertEqual(run(resumed)["last_epoch"], 1)
        self.assertEqual(len(
            (self.root / "run" / "history.jsonl").read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
