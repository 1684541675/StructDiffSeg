"""Unit checks for supervised binary SDF ground truth.

Run: python -m unittest discover -s tests -p 'test_sdf_target.py' -v
"""
import unittest

import torch

from models.DiffUNet3201.diffunet.sdf_target import binary_sdf_target


class TestBinarySDFTarget(unittest.TestCase):
    def test_sign_distance_truncation_and_shape(self):
        mask = torch.zeros((1, 9, 9), dtype=torch.long)
        mask[:, 3:6, 3:6] = 1
        sdf = binary_sdf_target(mask, radius=4)
        self.assertEqual(tuple(sdf.shape), (1, 1, 9, 9))
        self.assertEqual(sdf.dtype, torch.float32)
        self.assertAlmostEqual(sdf[0, 0, 4, 4].item(), 0.5)
        self.assertAlmostEqual(sdf[0, 0, 3, 3].item(), 0.25)
        self.assertAlmostEqual(sdf[0, 0, 2, 3].item(), -0.25)
        self.assertAlmostEqual(sdf[0, 0, 0, 0].item(), -1.0)
        self.assertTrue(((sdf >= -1) & (sdf <= 1)).all().item())

    def test_empty_full_and_mixed_batches(self):
        mask = torch.stack([
            torch.zeros((8, 8)), torch.ones((8, 8)),
            torch.eye(8),
        ])
        sdf = binary_sdf_target(mask, radius=16)
        self.assertEqual(tuple(sdf.shape), (3, 1, 8, 8))
        self.assertTrue((sdf[0] == -1).all().item())
        self.assertTrue((sdf[1] == 1).all().item())
        self.assertTrue(torch.isfinite(sdf).all().item())

    def test_channel_dimension_and_dtype(self):
        mask = torch.zeros((2, 1, 12, 10))
        mask[:, :, 2:9, 3:8] = 1
        sdf = binary_sdf_target(mask, radius=3, dtype=torch.float64)
        self.assertEqual(tuple(sdf.shape), (2, 1, 12, 10))
        self.assertEqual(sdf.dtype, torch.float64)
        self.assertEqual(sdf.device, mask.device)

    def test_invalid_inputs(self):
        with self.assertRaisesRegex(TypeError, "torch.Tensor"):
            binary_sdf_target([[0, 1]])
        for invalid in (
            torch.empty((1, 0, 2)), torch.ones((1, 2, 2, 2)),
            torch.tensor([[[0, 2]]]), torch.tensor([[[0.0, float("nan")]]]),
        ):
            with self.assertRaises(ValueError):
                binary_sdf_target(invalid)
        with self.assertRaises(ValueError):
            binary_sdf_target(torch.zeros((1, 2, 2)), radius=0)
        with self.assertRaises(ValueError):
            binary_sdf_target(torch.zeros((1, 2, 2)), dtype=torch.long)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_input_device(self):
        mask = torch.zeros((1, 12, 12), device="cuda")
        mask[:, 3:9, 3:9] = 1
        sdf = binary_sdf_target(mask)
        self.assertEqual(sdf.device.type, "cuda")
        self.assertTrue(torch.isfinite(sdf).all().item())


if __name__ == "__main__":
    unittest.main()
