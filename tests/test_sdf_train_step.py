"""A single optimizer-update smoke test: no dataset, checkpoints or epochs.

From the repository root:
    python -m unittest discover -s tests -p "test_sdf_train_step.py" -v
"""
import unittest

import torch

from models.DiffUNet3201.diffunet.diffunet_model import DiffUNet
from train_diffunet_sdf_step import train_step_sdf
from utils import CeDiceLoss


class TestSDFTrainStep(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.batch = {
            "image": torch.randn(1, 3, 64, 64),
            "label": torch.zeros(1, 64, 64, dtype=torch.long),
        }
        self.batch["label"][:, 16:48, 16:48] = 1

    def test_one_step_updates_geometry_and_diffusion(self):
        model = DiffUNet(3, 2, ddim_steps=3, enable_sdf=True)
        criterion = CeDiceLoss(2, [1, 1])
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        head_before = model.edge_model.sdf_head[0].weight.detach().clone()
        denoiser_param = next(model.denoise_model.parameters())
        denoiser_before = denoiser_param.detach().clone()

        report = train_step_sdf(self.batch, model, criterion, optimizer,
                                epoch=1, epochs=10, lambda_sdf=0.25)
        self.assertEqual(set(report), {
            "loss_total", "loss_diffusion", "loss_geometry", "loss_sdf",
            "lambda_sdf", "grad_sdf_head", "grad_denoiser",
        })
        for key, val in report.items():
            self.assertTrue(float("-inf") < val < float("inf"), key)
        self.assertGreater(report["loss_sdf"], 0.0)
        self.assertGreater(report["grad_sdf_head"], 0.0)
        self.assertGreater(report["grad_denoiser"], 0.0)
        self.assertAlmostEqual(
            report["loss_total"],
            report["loss_diffusion"] + report["loss_geometry"] +
            report["lambda_sdf"] * report["loss_sdf"], places=4
        )
        self.assertFalse(torch.equal(head_before, model.edge_model.sdf_head[0].weight))
        self.assertFalse(torch.equal(denoiser_before, denoiser_param))

    def test_input_validation(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        with self.assertRaisesRegex(ValueError, "lambda_sdf"):
            train_step_sdf(self.batch, model, None, optimizer, lambda_sdf=-1)
        with self.assertRaisesRegex(ValueError, "epoch"):
            train_step_sdf(self.batch, model, None, optimizer, epoch=0)
        bad = dict(self.batch)
        bad["label"] = torch.full((1, 64, 64), 0.5)
        with self.assertRaisesRegex(ValueError, "binary"):
            train_step_sdf(bad, model, None, optimizer)
        bad = dict(self.batch)
        bad["image"] = torch.randn(1, 1, 64, 64)
        with self.assertRaisesRegex(ValueError, "image"):
            train_step_sdf(bad, model, None, optimizer)

    def test_one_step_cuda_when_available(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        model = DiffUNet(3, 2, ddim_steps=3, enable_sdf=True).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        report = train_step_sdf(self.batch, model, CeDiceLoss(2, [1, 1]).cuda(),
                                optimizer, epoch=1, epochs=10, lambda_sdf=0.25)
        self.assertGreater(report["grad_sdf_head"], 0)
        self.assertGreater(report["grad_denoiser"], 0)


if __name__ == "__main__":
    unittest.main()
