"""Optional GCF integration checks; synthetic tensors, no full training.

Run from repo root:
    python -m unittest discover -s tests -p "test_gcf.py" -v
"""
import unittest

import torch

from models.DiffUNet3201.diffunet.gcf import GeometryConditionedFusion
from models.DiffUNet3201.diffunet.diffunet_model import DiffUNet


class TestGCF(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_standalone_shape_time_and_sdf_dependence(self):
        block = GeometryConditionedFusion(16, time_dim=32).eval()
        diff = torch.randn(2, 16, 8, 8)
        geo = torch.randn(2, 16, 8, 8)
        sdf = torch.randn(2, 1, 32, 32).tanh()
        time_a = torch.zeros(2, 32)
        time_b = torch.ones(2, 32)
        with torch.no_grad():
            result = block(diff, geo, sdf, time_a)
            changed_time = block(diff, geo, sdf, time_b)
            changed_sdf = block(diff, geo, -sdf, time_a)
        self.assertEqual(tuple(result.shape), tuple(diff.shape))
        self.assertTrue(torch.isfinite(result).all().item())
        self.assertGreater((result - changed_time).abs().max().item(), 1e-7)
        self.assertGreater((result - changed_sdf).abs().max().item(), 1e-7)

    def test_sdf_time_and_geometry_receive_gradients(self):
        block = GeometryConditionedFusion(8, time_dim=32).eval()
        diff = torch.randn(1, 8, 8, 8, requires_grad=True)
        geo = torch.randn(1, 8, 8, 8, requires_grad=True)
        sdf = torch.randn(1, 1, 32, 32, requires_grad=True)
        time = torch.randn(1, 32, requires_grad=True)
        output = block(diff, geo, sdf, time)
        output.square().mean().backward()
        for tensor in (diff, geo, sdf, time):
            self.assertIsNotNone(tensor.grad)
            self.assertGreater(tensor.grad.abs().sum().item(), 0)
        self.assertIsNotNone(block.gamma.grad)
        self.assertGreater(block.gamma.grad.abs().sum().item(), 0)

    def test_zero_gamma_is_identity(self):
        block = GeometryConditionedFusion(8).eval()
        block.gamma.data.zero_()
        diff = torch.randn(1, 8, 8, 8)
        geo = torch.randn_like(diff)
        sdf = torch.randn(1, 1, 16, 16)
        time = torch.randn(1, 32)
        with torch.no_grad():
            output = block(diff, geo, sdf, time)
        torch.testing.assert_close(output, diff, rtol=0, atol=0)

    def test_batch_repetition_and_shape_checks(self):
        block = GeometryConditionedFusion(8).eval()
        output = block(torch.randn(4, 8, 8, 8),
                       torch.randn(2, 8, 8, 8),
                       torch.randn(2, 1, 32, 32),
                       torch.randn(4, 32))
        self.assertEqual(tuple(output.shape), (4, 8, 8, 8))
        with self.assertRaisesRegex(ValueError, "batch must divide"):
            block(torch.randn(3, 8, 8, 8),
                  torch.randn(2, 8, 8, 8),
                  torch.randn(2, 1, 16, 16),
                  torch.randn(3, 32))
        with self.assertRaisesRegex(ValueError, "predicted_sdf"):
            block(torch.randn(1, 8, 8, 8),
                  torch.randn(1, 8, 8, 8),
                  torch.randn(1, 2, 8, 8),
                  torch.randn(1, 32))

    def test_full_model_training_gradient_from_diffusion_to_sdf_head(self):
        model = DiffUNet(3, 2, ddim_steps=3, enable_sdf=True,
                         enable_gcf=True, gcf_stage=2).eval()
        self.assertIsNotNone(model.denoise_model.encoder.gcf)
        self.assertEqual(model.denoise_model.encoder.gcf_stage, 2)
        image = torch.randn(1, 3, 64, 64)
        mask = torch.zeros(1, 64, 64, dtype=torch.long)
        mask[:, 14:48, 18:43] = 1

        pred_diff, pred_geo, uncertainty, pred_sdf = model(
            image, gt=mask, return_sdf=True
        )
        self.assertEqual(tuple(pred_diff.shape), (1, 2, 64, 64))
        self.assertEqual(tuple(pred_geo.shape), (1, 2, 64, 64))
        self.assertEqual(tuple(uncertainty.shape), (1, 64, 64))
        self.assertEqual(tuple(pred_sdf.shape), (1, 1, 64, 64))

        # Only diffusion loss: SDF-head gradients prove the new conditioning path.
        pred_diff.square().mean().backward()
        sdf_grad = model.edge_model.sdf_head[0].weight.grad
        gate_grad = model.denoise_model.encoder.gcf.gate_conv.weight.grad
        self.assertIsNotNone(sdf_grad)
        self.assertGreater(sdf_grad.abs().sum().item(), 0)
        self.assertIsNotNone(gate_grad)
        self.assertGreater(gate_grad.abs().sum().item(), 0)

    def test_full_ddim_without_gt_and_opt_in_compatibility(self):
        model = DiffUNet(3, 2, ddim_steps=3, enable_sdf=True,
                         enable_gcf=True).eval()
        image = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            pred, sdf = model(image, ddim=True, return_sdf=True)
        self.assertEqual(tuple(pred.shape), (1, 2, 64, 64))
        self.assertEqual(tuple(sdf.shape), (1, 1, 64, 64))
        self.assertTrue(torch.isfinite(pred).all().item())

        legacy = DiffUNet(3, 2, ddim_steps=3, enable_sdf=True).eval()
        self.assertIsNone(legacy.denoise_model.encoder.gcf)
        self.assertFalse(any("gcf." in k for k in legacy.state_dict()))
        with self.assertRaisesRegex(ValueError, "enable_sdf"):
            DiffUNet(3, 2, enable_gcf=True)
        with self.assertRaisesRegex(ValueError, "binary"):
            DiffUNet(1, 4, enable_gcf=True)


if __name__ == "__main__":
    unittest.main()
