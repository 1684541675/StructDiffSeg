"""Test SDF timing parity and accounting without requiring a GPU."""

from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image
import torch

from benchmark_sdf_target import benchmark, instrumented_sdf_target, main
from models.DiffUNet3201.diffunet.sdf_target import binary_sdf_target


class SdfBenchmarkChecks(unittest.TestCase):
    def test_exact_parity_including_empty_and_full_masks(self):
        masks = torch.zeros((4, 32, 32), dtype=torch.long)
        masks[0, 5:18, 9:25] = 1
        masks[1].fill_(1)
        masks[3, 0, 0] = 1
        actual, stages = instrumented_sdf_target(masks, radius=8.0)
        self.assertTrue(torch.equal(actual, binary_sdf_target(masks, radius=8.0)))
        self.assertEqual(actual.shape, (4, 1, 32, 32))
        self.assertTrue(all(v >= 0 for v in stages.values()))
        self.assertAlmostEqual(
            stages["total"], sum(v for k, v in stages.items() if k != "total"),
            places=7
        )
        actual_4d, _ = instrumented_sdf_target(masks[:, None], radius=8.0)
        self.assertTrue(torch.equal(actual, actual_4d))

    def test_cpu_benchmark_counts(self):
        masks = torch.zeros((3, 20, 20), dtype=torch.long)
        masks[0, 2:6, 2:6] = 1
        masks[1].fill_(1)
        result = benchmark([masks[:2], masks[2:]], repeats=2, warmup=1)
        self.assertEqual(result["samples_per_epoch"], 3)
        self.assertEqual(result["batches_per_epoch"], 2)
        self.assertTrue(result["exact_output_match"])
        self.assertGreater(result["original_sdf_ms_per_epoch"]["mean"], 0)
        for stage in result["breakdown_ms_per_epoch"].values():
            self.assertGreaterEqual(stage["mean"], 0)

    def test_real_mask_loader_uses_existing_preprocessing(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "images").mkdir()
            (root / "masks").mkdir()
            Image.fromarray(np.full((18, 19, 3), 120, dtype=np.uint8)).save(
                root / "images" / "a.png"
            )
            labels = np.zeros((18, 19), dtype=np.uint16)
            labels[3:13, 4:16] = 300
            Image.fromarray(labels).save(root / "masks" / "a.png")
            (root / "train.txt").write_text("a\n")
            with redirect_stdout(io.StringIO()):
                result = main([
                    "--dataset", "glas", "--data-path", str(root),
                    "--size", "32", "--batch-size", "2",
                    "--device", "cpu", "--repeats", "1", "--warmup", "0"
                ])
            self.assertEqual(result["sample_count"], 1)
            self.assertEqual(result["batches_per_epoch"], 1)
            self.assertTrue(result["exact_output_match"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_exact_parity(self):
        masks = torch.zeros((2, 32, 32), dtype=torch.long, device="cuda")
        masks[0, 3:19, 8:26] = 1
        result, stages = instrumented_sdf_target(masks)
        self.assertTrue(torch.equal(result, binary_sdf_target(masks)))
        self.assertGreaterEqual(stages["mask_device_to_cpu"], 0)


if __name__ == "__main__":
    unittest.main()
