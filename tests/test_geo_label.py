"""Lightweight real-network smoke tests; no datasets or full training run.

Run: python -m unittest discover -s tests -p test_geo_label.py -v
"""

import os
import unittest
from unittest.mock import patch

import torch
from torch import nn

from models.DiffUNet3201.diffunet.nnunet2d_denoise import (
    PlainConvUNet, get_nnunet2d_denoise,
)
from models.DiffUNet3201.diffunet.sdf import multiclass_sdf
from utils import CeDiceLoss


def small_model(lambda_geo=0.25, device="cpu", dtype=torch.float32):
    # Same encoder/decoder and diffusion methods; only stage widths/count reduced.
    return PlainConvUNet(
        input_channels=1, n_stages=3, features_per_stage=[4, 8, 16],
        conv_op=nn.Conv2d, kernel_sizes=[3, 3, 3], strides=[1, 2, 2],
        n_conv_per_stage=1, num_classes=4, n_conv_per_stage_decoder=1,
        norm_op=nn.InstanceNorm2d, norm_op_kwargs={"affine": True},
        nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace": True},
        ddim_step=3, lambda_geo=lambda_geo,
    ).to(device=device, dtype=dtype)


def labels(batch=2, size=32, device="cpu"):
    mask = torch.zeros(batch, size, size, dtype=torch.long, device=device)
    mask[:, 4:12, 4:12] = 1
    mask[:, 12:20, 12:20] = 2
    mask[:, 20:28, 20:28] = 3
    return mask


class GeoLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_sdf_shape_range_direction_and_truncation(self):
        mask = labels(size=64)
        sdf = multiclass_sdf(mask)
        self.assertEqual(sdf.shape, (2, 4, 64, 64))
        self.assertTrue(torch.isfinite(sdf).all())
        self.assertGreaterEqual(sdf.min().item(), -1)
        self.assertLessEqual(sdf.max().item(), 1)
        for c in range(4):
            self.assertTrue((sdf[:, c][mask == c] < 0).all())
            self.assertTrue((sdf[:, c][mask != c] > 0).all())
        self.assertEqual(sdf[0, 1, 4, 4].item(), -1 / 16)
        self.assertEqual(sdf[0, 1, 3, 4].item(), 1 / 16)
        self.assertEqual(sdf[0, 1, 63, 63].item(), 1)
        self.assertEqual(sdf[0, 0, 63, 63].item(), -1)

    def test_missing_full_and_single_pixel_classes(self):
        for c in range(4):
            mask = torch.full((1, 16, 16), c)
            sdf = multiclass_sdf(mask)
            self.assertTrue(torch.isfinite(sdf).all())
            for k in range(4):
                self.assertTrue((sdf[:, k] == (-1 if k == c else 1)).all())
        mask[0, 0, 0] = 0
        self.assertTrue(torch.isfinite(multiclass_sdf(mask)).all())
        self.assertTrue((multiclass_sdf(mask)[:, 1:3] == 1).all())

    def test_invalid_sdf_inputs(self):
        for mask in (torch.ones(1, 2), torch.full((1, 2, 2), 4),
                     torch.full((1, 2, 2), 0.5), torch.full((1, 2, 2), float("nan"))):
            with self.assertRaises(ValueError):
                multiclass_sdf(mask)
        with self.assertRaises(ValueError):
            multiclass_sdf(labels(), radius=0)

    def test_zero_lambda_exact_original_and_skips_sdf(self):
        model = small_model(0)
        mask = labels()
        for target in (mask, mask.unsqueeze(1), mask.float()):
            old = model.embedding_table(target.long())
            old = old.squeeze(1).permute(0, 3, 1, 2)
            old = (torch.sigmoid(old) * 2 - 1) * model.bit_scale
            with patch("models.DiffUNet3201.diffunet.nnunet2d_denoise.multiclass_sdf",
                       side_effect=AssertionError("SDF must be skipped")):
                self.assertTrue(torch.equal(model.encode_mask(target), old))

    def test_fusion_formula_and_gradient(self):
        model = small_model()
        mask = labels()
        classes = 2 * model.embedding_table(mask).permute(0, 3, 1, 2).sigmoid() - 1
        expected = 0.001 * (0.75 * classes - 0.25 * multiclass_sdf(mask))
        actual = model.encode_mask(mask)
        self.assertTrue(torch.equal(actual, expected))
        actual.sum().backward()
        self.assertTrue(torch.isfinite(model.embedding_table.weight.grad).all())
        self.assertGreater(model.embedding_table.weight.grad.abs().sum().item(), 0)

    def check_forward_backward(self, device):
        model = small_model(device=device)
        mask = labels(device=device)
        image = torch.randn(2, 1, 32, 32, device=device)
        with patch.object(model, "encode_mask", wraps=model.encode_mask) as encode:
            pred, uncertainty = model(image, mask)
        self.assertEqual(encode.call_count, 2)
        for call in encode.call_args_list:
            self.assertTrue(torch.equal(call.args[0], mask))
        self.assertEqual(pred.shape, (2, 4, 32, 32))
        self.assertEqual(uncertainty.shape, (2, 32, 32))
        self.assertFalse(uncertainty.requires_grad)
        loss = CeDiceLoss(4, [1, 1])(pred, mask, uncertainty, scale=0.1)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(model.embedding_table.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.decoder.seg_layers[-1].weight.grad.abs().sum().item(), 0)

    def test_train_cpu(self):
        self.check_forward_backward("cpu")

    def test_production_wrapper_cpu_smoke(self):
        from models.DiffUNet3201.diffunet.diffunet_model import DiffUNet
        with patch.dict(os.environ, {"DIFFUNET_LAMBDA_GEO": "0.25"}):
            model = DiffUNet(1, 4, ddim_steps=3)
        mask = labels(batch=1, size=64)
        image = torch.randn(1, 1, 64, 64)
        pred, pred_edge, uncertainty = model(image, mask)
        criterion = CeDiceLoss(4, [1, 1])
        loss = criterion(pred, mask, uncertainty, scale=0.1) + criterion(pred_edge, mask)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.denoise_model.embedding_table.weight.grad.abs().sum().item(), 0)
        model.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            output = model(image, ddim=True)
        self.assertEqual(output.shape, (1, 4, 64, 64))
        self.assertTrue(torch.isfinite(output).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_train_cuda(self):
        self.check_forward_backward("cuda")

    def test_ddim_three_steps_uses_only_argmax(self):
        model = small_model().eval()
        image = torch.randn(2, 1, 32, 32)
        predictions = []
        handle = model.decoder.register_forward_hook(
            lambda module, args, output: predictions.append(output.detach().argmax(1)))
        try:
            with patch.object(model, "encode_mask", wraps=model.encode_mask) as encode:
                torch.manual_seed(7)
                pred = model(image, ddim=True)  # no GT supplied
            self.assertEqual(encode.call_count, 3)
            for call, expected in zip(encode.call_args_list, predictions):
                self.assertTrue(torch.equal(call.args[0], expected))
        finally:
            handle.remove()
        self.assertEqual(pred.shape, (2, 4, 32, 32))
        self.assertTrue(torch.isfinite(pred).all())
        torch.manual_seed(7)
        # Invalid GT sentinel must never be read on the inference path.
        other = model(image, gt=object(), ddim=True)
        self.assertTrue(torch.equal(pred, other))

    def test_dtype_device_and_conversion(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                model = small_model(device=device, dtype=dtype)
                mask = labels(device=device)
                sdf = multiclass_sdf(mask.unsqueeze(1), dtype=dtype)
                encoded = model.encode_mask(mask)
                self.assertEqual(sdf.device, mask.device)
                self.assertEqual(encoded.device, model.embedding_table.weight.device)
                self.assertEqual(sdf.dtype, dtype)
                self.assertEqual(encoded.dtype, dtype)
                self.assertTrue(torch.isfinite(encoded).all())
                # Labels on CPU are accepted by a model moved to CUDA.
                self.assertTrue(torch.equal(encoded, model.encode_mask(mask.cpu())))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_amp(self):
        model = small_model(device="cuda")
        mask = labels(device="cuda")
        image = torch.randn(2, 1, 32, 32, device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            pred, uncertainty = model(image, mask)
            loss = CeDiceLoss(4, [1, 1])(pred, mask, uncertainty)
        loss.backward()
        self.assertTrue(torch.isfinite(model.embedding_table.weight.grad).all())
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(image, ddim=True)
        self.assertEqual(output.shape, (2, 4, 32, 32))
        self.assertTrue(torch.isfinite(output).all())

    def test_factory_override_and_state_dict_compatibility(self):
        with patch.dict(os.environ, {"DIFFUNET_LAMBDA_GEO": "0"}):
            model = get_nnunet2d_denoise(1, 4)
        self.assertEqual(model.lambda_geo, 0)
        self.assertEqual(model.timesteps, 3)
        self.assertEqual(model.bit_scale, 0.001)
        # Geometry adds no keys/parameters, permitting strict baseline loading.
        state = model.state_dict()
        model.lambda_geo = 0.25
        model.load_state_dict(state, strict=True)
        for invalid in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                small_model(invalid)


if __name__ == "__main__":
    unittest.main()
