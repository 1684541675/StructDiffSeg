"""Run: python -m unittest discover -s tests -p test_rgb_binary_cached.py -v"""

from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from datasets.rgb_binary import RGBBinaryDataset
from datasets.rgb_binary_cached import CachedRGBBinaryDataset
from train_cached_2d import main


class CacheChecks(unittest.TestCase):
    def test_equal_outputs_rng_no_reopening_and_mutation_safety(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'images').mkdir()
            (root / 'masks').mkdir()
            # Exercise all 256 image byte values and >255 instance IDs.
            rgb = np.arange(16 * 16 * 3, dtype=np.uint16).reshape(16, 16, 3).astype(np.uint8)
            label = np.zeros((16, 16), dtype=np.uint16)
            label[2:11, 4:14] = 300
            for name in ('a', 'b'):
                Image.fromarray(rgb).save(root / 'images' / f'{name}.png')
            Image.fromarray(label).save(root / 'masks' / 'a.png')
            cvc = np.repeat(((label > 0) * 255).astype(np.uint8)[..., None], 3, axis=2)
            Image.fromarray(cvc).save(root / 'masks' / 'b.png')
            for split in ('train', 'val', 'test'):
                (root / f'{split}.txt').write_text('a\nb\n')
            for split in ('train', 'val', 'test'):
                reference = RGBBinaryDataset(root, split, output_size=(32, 32))
                np.random.seed(123)
                before = np.random.get_state()
                cached = CachedRGBBinaryDataset(root, split, output_size=(32, 32))
                after = np.random.get_state()
                self.assertTrue(np.array_equal(before[1], after[1]))
                self.assertEqual(before[2:], after[2:])
                self.assertEqual(cached.cache_nbytes, 2 * 32 * 32 * 4)
                for seed in range(12):
                    for index in range(2):
                        np.random.seed(seed)
                        expected = reference[index]
                        np.random.seed(seed)
                        with patch('PIL.Image.open', side_effect=AssertionError('Repeated decoding')):
                            actual = cached[index]
                        self.assertEqual(actual['case_name'], expected['case_name'])
                        for key in ('image', 'label'):
                            self.assertTrue(torch.equal(expected[key], actual[key]))
                            self.assertEqual(expected[key].dtype, actual[key].dtype)
                original = cached[0]
                original['image'].zero_()
                original['label'].fill_(99)
                self.assertNotIn(99, cached[0]['label'].unique().tolist())
                self.assertGreater(cached[0]['image'].sum().item(), 0)
                batch = next(iter(DataLoader(cached, batch_size=2, num_workers=0)))
                self.assertEqual(batch['image'].shape, (2, 3, 32, 32))

    def test_new_entry_restores_original_binding_without_running_training(self):
        for selected in ('unet', 'gcf-residual'):
            sentinel = object()
            trainer = SimpleNamespace(RGBBinaryDataset=sentinel)
            trainer.make_parser = lambda: SimpleNamespace(parse_args=lambda argv: argv)
            def run(args):
                self.assertIs(trainer.RGBBinaryDataset, CachedRGBBinaryDataset)
                self.assertEqual(args, ['--mode', 'train'])
                return {'status': 'stub'}
            trainer.run = run
            with patch('train_cached_2d.importlib.import_module', return_value=trainer), redirect_stdout(io.StringIO()):
                main(['--trainer', selected, '--mode', 'train'])
                self.assertIs(trainer.RGBBinaryDataset, sentinel)
                trainer.run = lambda args: (_ for _ in ()).throw(RuntimeError('interrupted'))
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    main(['--trainer', selected, '--mode', 'train'])
                self.assertIs(trainer.RGBBinaryDataset, sentinel)


if __name__ == '__main__':
    unittest.main()
