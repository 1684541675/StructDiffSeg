"""Integration tests for the optional SDF branch and Smooth L1 supervision.

Run from the repository root:
    python -m unittest discover -s tests -p "test_sdf_supervision.py" -v

These use synthetic images/masks, not a real dataset or training run.
"""
import unittest

import torch
import torch.nn.functional as F

from models.DiffUNet3201.diffunet.diffunet_model import DiffUNet
from models.DiffUNet3201.diffunet.sdf_loss import sdf_smooth_l1_loss
from models.DiffUNet3201.diffunet.sdf_target import binary_sdf_target


class TestSDFSupervision(unittest.TestCase):
    def test_loss_matches_torch_and_has_gradient(self):
        prediction = torch.tensor([[[[0.0, 0.5], [-0.4, 0.8]]]], requires_grad=True)
        target = torch.tensor([[[[0.2, 0.2], [-0.2, 1.0]]]])
        loss = sdf_smooth_l1_loss(prediction, target)
        expected = F.smooth_l1_loss(prediction, target, beta=1.0)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertGreater(prediction.grad.abs().sum().item(), 0)

    def test_loss_rejects_shape_and_dtype_mismatch(self):
        prediction = torch.zeros((1, 1, 8, 8))
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            sdf_smooth_l1_loss(prediction, torch.zeros((1, 8, 8)))
        with self.assertRaisesRegex(ValueError, "shape"):
            sdf_smooth_l1_loss(torch.zeros((1, 2, 8, 8)), torch.zeros((1, 2, 8, 8)))
        with self.assertRaisesRegex(TypeError, "floating point"):
            sdf_smooth_l1_loss(prediction, torch.zeros((1, 1, 8, 8), dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "beta"):
            sdf_smooth_l1_loss(prediction, prediction, beta=0.0)

    def test_full_diffunet_training_interface_and_sdf_gradient(self):
        torch.manual_seed(21)
        model = DiffUNet(3, 2, ddim_steps=3, enable_sdf=True).eval()
        image = torch.randn(1, 3, 64, 64)
        mask = torch.zeros((1, 64, 64), dtype=torch.long)
        mask[:, 16:48, 16:48] = 1

        diffusion_logits, geometry_logits, uncertainty, predicted_sdf = model(
            image, gt=mask, return_sdf=True
        )
        self.assertEqual(tuple(diffusion_logits.shape), (1, 2, 64, 64))
        self.assertEqual(tuple(geometry_logits.shape), (1, 2, 64, 64))
        self.assertEqual(tuple(uncertainty.shape), (1, 64, 64))
        self.assertEqual(tuple(predicted_sdf.shape), (1, 1, 64, 64))
        target_sdf = binary_sdf_target(mask, radius=16, dtype=predicted_sdf.dtype)
        loss = sdf_smooth_l1_loss(predicted_sdf, target_sdf)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        grad = model.edge_model.sdf_head[0].weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(grad.abs().sum().item(), 0)
        # The standard three-output training contract must remain unchanged.
        with torch.no_grad():
            baseline_outputs = model(image, gt=mask)
        self.assertEqual(len(baseline_outputs), 3)

    def test_full_diffunet_inference_does_not_need_target(self):
        torch.manual_seed(22)
        model = DiffUNet(3, 2, ddim_steps=3, enable_sdf=True).eval()
        image = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            logits, sdf = model(image, ddim=True, return_sdf=True)
        self.assertEqual(tuple(logits.shape), (1, 2, 64, 64))
        self.assertEqual(tuple(sdf.shape), (1, 1, 64, 64))
        self.assertTrue(torch.isfinite(logits).all().item())
        self.assertTrue(torch.isfinite(sdf).all().item())

    def test_opt_in_is_binary_only(self):
        with self.assertRaisesRegex(ValueError, "binary segmentation"):
            DiffUNet(1, 4, enable_sdf=True)
        self.assertIsNone(DiffUNet(1, 4).edge_model.sdf_head)


if __name__ == "__main__":
    unittest.main()
