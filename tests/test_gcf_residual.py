"""Behavioral checks for the additive fusion and its independent entry points."""
import gc
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from models.DiffUNet3201.diffunet.gcf_residual import (
    ResidualGeometryConditioning, build_model,
)
from train_sdf_2d import build_model as build_legacy
from train_gcf_residual_2d import make_parser, run
from test_gcf_residual_2d import run_test


class TestResidualGCF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def tearDown(self):
        gc.collect()

    def test_initial_ddim_equals_sdf_baseline_and_encoder_alias(self):
        image = torch.randn(1, 3, 64, 64)
        baseline = build_legacy("sdf", shared_seed=42).eval()
        with torch.no_grad():
            torch.manual_seed(123)
            expected = baseline(image, ddim=True)
        del baseline
        model = build_model(shared_seed=42).eval()
        self.assertIs(model.denoise_model.encoder, model.denoise_model.decoder.encoder)
        with torch.no_grad():
            torch.manual_seed(123)
            actual = model(image, ddim=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_gate_opens_and_diffusion_loss_reaches_sdf_head(self):
        model = build_model(shared_seed=42).train()
        model.denoise_model.uncer_step = 2  # Exercise MC batch repetition too.
        image = torch.randn(2, 3, 64, 64)
        label = torch.zeros(2, 64, 64, dtype=torch.long)
        label[:, 12:40, 20:45] = 1
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        first = model(image, gt=label, return_sdf=True)[0]
        first.square().mean().backward()
        gamma = model.denoise_model.encoder.residual.gamma
        self.assertTrue(torch.isfinite(gamma.grad).all())
        self.assertGreater(gamma.grad.abs().item(), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        # No SDF supervision here: proves the conditioning path carries gradients.
        second = model(image, gt=label, return_sdf=True)[0]
        second.square().mean().backward()
        for grad in (model.edge_model.sdf_head[0].weight.grad,
                     model.denoise_model.encoder.residual.gate_conv.weight.grad):
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0)

    def test_input_ablations_and_repetition(self):
        fused = torch.randn(4, 8, 8, 8)
        geometry = torch.randn(2, 8, 8, 8)
        sdf = torch.randn(2, 1, 64, 64)
        time = torch.randn(4, 32)
        for use_sdf, use_time in ((True, True), (False, True), (True, False)):
            block = ResidualGeometryConditioning(8, use_sdf=use_sdf, use_time=use_time)
            with torch.no_grad():
                block.gamma.fill_(0.1)
                output = block(fused, geometry, sdf, time)
                delta_sdf = (output - block(fused, geometry, -sdf, time)).abs().max().item()
                delta_time = (output - block(fused, geometry, sdf, time + 1)).abs().max().item()
            self.assertEqual(delta_sdf > 0, use_sdf)
            self.assertEqual(delta_time > 0, use_time)

    def test_train_resume_and_held_out_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "images").mkdir()
            (root / "masks").mkdir()
            for index, case in enumerate(("train", "val", "test")):
                mask = np.zeros((64, 64), dtype=np.uint8)
                mask[12:44, 16 + index:48 + index] = 255
                image = np.repeat(mask[..., None], 3, axis=2)
                Image.fromarray(mask).save(root / "masks" / (case + ".png"))
                Image.fromarray(image).save(root / "images" / (case + ".png"))
                (root / (case + ".txt")).write_text(case + "\n")
            args = make_parser().parse_args([
                "--dataset", "glas", "--data-path", str(root), "--mode", "train",
                "--device", "cpu", "--size", "64", "--epochs", "1",
                "--uncer-samples", "1", "--output-dir", str(root / "run"),
            ])
            run(args)
            history = json.loads((root / "run/history.jsonl").read_text())
            for key in ("loss_sdf", "loss_geometry", "loss_diffusion", "residual_gamma"):
                self.assertTrue(np.isfinite(history[key]))
            args.resume = str(root / "run/last.pt")
            run(args)  # Completed-run resume must load the state and config cleanly.
            result = run_test(root / "run/best.pt", device="cpu")
            self.assertEqual(result["cases_evaluated"], 1)
            self.assertTrue((root / "run/test_results/metrics.csv").is_file())


if __name__ == "__main__":
    unittest.main()
