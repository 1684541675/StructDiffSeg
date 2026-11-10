"""Optional binary SDF-head interface checks.

Run in the project's PyTorch environment:
    python -m unittest discover -s tests -p 'test_sdf_prediction_head.py' -v
"""
import unittest

import torch

from models.DiffUNet3201.diffunet.nnunet2d import get_nnunet2d


class TestSDFPredictionHead(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.image = torch.randn(1, 3, 64, 64)

    def test_legacy_interface_is_unchanged(self):
        model = get_nnunet2d(3, 2).eval()
        self.assertIsNone(model.sdf_head)
        self.assertFalse(any(key.startswith("sdf_head.") for key in model.state_dict()))
        with torch.no_grad():
            logits, features = model(self.image)
        self.assertEqual(tuple(logits.shape), (1, 2, 64, 64))
        self.assertEqual(len(features), 6)
        with self.assertRaisesRegex(RuntimeError, "SDF head is disabled"):
            model(self.image, return_sdf=True)

    def test_sdf_output_and_gradient(self):
        model = get_nnunet2d(3, 2, enable_sdf=True).eval()
        logits, features, sdf = model(self.image, return_sdf=True)
        self.assertEqual(tuple(logits.shape), (1, 2, 64, 64))
        self.assertEqual(tuple(sdf.shape), (1, 1, 64, 64))
        self.assertEqual(len(features), 6)
        self.assertEqual(model.sdf_head[0].in_channels, model.encoder.output_channels[0])
        self.assertTrue(torch.isfinite(sdf).all().item())
        self.assertTrue(((sdf >= -1) & (sdf <= 1)).all().item())
        (logits.square().mean() + sdf.square().mean()).backward()
        self.assertIsNotNone(model.sdf_head[0].weight.grad)
        self.assertGreater(model.sdf_head[0].weight.grad.abs().sum().item(), 0)

    def test_segmentation_prediction_identical_with_sdf_return(self):
        model = get_nnunet2d(3, 2, enable_sdf=True).eval()
        with torch.no_grad():
            old_logits, old_features = model(self.image)
            new_logits, new_features, _ = model(self.image, return_sdf=True)
        torch.testing.assert_close(old_logits, new_logits, rtol=0, atol=0)
        self.assertEqual(len(old_features), len(new_features))

    def test_single_channel_sdf_rejects_multiclass(self):
        with self.assertRaisesRegex(ValueError, "binary segmentation"):
            get_nnunet2d(1, 4, enable_sdf=True)


if __name__ == "__main__":
    unittest.main()
