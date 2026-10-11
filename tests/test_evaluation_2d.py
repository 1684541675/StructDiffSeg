"""Short CPU checks, with synthetic data and DDIM stub; no real training.

Run: python -m unittest discover -s tests -p test_evaluation_2d.py -v
"""

from pathlib import Path
from dataclasses import asdict
from tempfile import TemporaryDirectory
from unittest.mock import patch, Mock
import unittest
import json

import numpy as np
from PIL import Image
import torch

from configs.config_setting_2d import TrainingConfig2D
from evaluation_2d import binary_region_metrics, evaluate_2d, predict_single_image
from train_2d import make_loaders, save_epoch_checkpoint, test as run_test, train, train_one_epoch_2d


class DDIMStub(torch.nn.Module):
    """Diffusion alone favors background; the existing summed output is correct."""

    def forward(self, image, *, ddim=False):
        assert ddim and not self.training and not torch.is_grad_enabled()
        torch.rand(1)  # Evaluation must preserve RNG despite stochastic DDIM.
        diffusion = torch.zeros((1, 2, *image.shape[2:]), device=image.device)
        diffusion[:, 0] = 2
        spm = torch.zeros_like(diffusion)
        spm[:, 1] = image[:, 0] * 4
        return diffusion + spm


class EvaluationChecks(unittest.TestCase):
    def test_region_metrics_and_empty_masks(self):
        pred = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        gt = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        result = binary_region_metrics(pred, gt)
        self.assertAlmostEqual(result['dice'], 2 / 3)
        self.assertEqual(result['iou'], 0.5)
        zero = np.zeros_like(pred)
        self.assertEqual(binary_region_metrics(zero, zero), {'dice': 1, 'iou': 1})
        for a, b in ((pred, zero), (zero, pred)):
            self.assertEqual(binary_region_metrics(a, b), {'dice': 0, 'iou': 0})

    def test_native_size_ddim_and_rng(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'images').mkdir()
            (root / 'masks').mkdir()
            rgb = np.zeros((12, 20, 3), dtype=np.uint8)
            rgb[:, :10, 0] = 255
            mask = np.zeros((12, 20), dtype=np.uint8)
            mask[:, :10] = 7  # GlaS instance IDs.
            Image.fromarray(rgb).save(root / 'images' / 'a.png')
            Image.fromarray(mask).save(root / 'masks' / 'a.png')
            Image.fromarray(rgb).save(root / 'images' / 'b.png')
            cvc = np.repeat(((mask > 0) * 255).astype(np.uint8)[..., None], 3, axis=2)
            Image.fromarray(cvc).save(root / 'masks' / 'b.png')
            (root / 'val.txt').write_text('a\n')
            (root / 'test.txt').write_text('b\n')
            # No train.txt: test mode must be independent of training data.
            config = TrainingConfig2D('glas', data_path=str(root))
            loaders = make_loaders(config, ('val', 'test'))
            model = DDIMStub().train()
            before = torch.get_rng_state().clone()
            validation = evaluate_2d(model, loaders['val'], 'cpu')
            self.assertTrue(model.training)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertEqual(validation['mean_dice'], 1)
            self.assertEqual(validation['mean_iou'], 1)
            self.assertNotIn('mean_hd95', validation)
            report = evaluate_2d(model, loaders['test'], 'cpu')
            self.assertEqual(report['cases'][0]['original_size'], [12, 20])
            self.assertNotIn('hd95', json.dumps(report).lower())
            self.assertEqual(report, evaluate_2d(model, loaders['test'], 'cpu'))
            model.eval()
            prediction = predict_single_image(next(iter(loaders['val']))['image'], model, (12, 20), 'cpu')
            np.testing.assert_array_equal(prediction, mask > 0)
            # Exercise the independent best-checkpoint test and JSON output too.
            best_path = root / 'best.pth'
            torch.save({'checkpoint_type': 'best', 'config': asdict(config),
                        'model_state_dict': model.state_dict(), 'epoch': 1,
                        'best_dice': 1.0, 'training_completed': True,
                        'completed_epoch': config.epochs}, best_path)
            with patch('train_2d.make_model', return_value=DDIMStub()):
                saved_report = run_test(config, loaders['test'], 'cpu', best_path, root)
            self.assertEqual(saved_report['split'], 'test')
            self.assertEqual(saved_report['mean_dice'], 1)
            self.assertTrue((root / 'test_glas.json').is_file())

    def test_checkpoint_selection_resume_and_test_guard(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.AdamW(model.parameters())
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
            config = TrainingConfig2D('glas')
            def save(epoch, score, best, best_epoch):
                return save_epoch_checkpoint(model, optimizer, scheduler, config, root,
                    epoch, 2.0, {'split': 'val', 'mean_dice': score, 'mean_iou': 0.5}, best, best_epoch)
            best, epoch = save(1, 0.6, -1, 0)
            best, epoch = save(2, 0.5, best, epoch)
            checkpoint = torch.load(root / 'latest.pth', weights_only=False)
            self.assertEqual(checkpoint['epoch'], 2)
            self.assertEqual(checkpoint['best_dice'], 0.6)
            self.assertEqual(torch.load(root / 'best.pth', weights_only=False)['epoch'], 1)
            best, epoch = save(3, 0.6, checkpoint['best_dice'], checkpoint['best_epoch'])
            self.assertEqual(epoch, 1)  # Ties retain the earlier best.
            save(4, 0.7, best, epoch)
            best_checkpoint = torch.load(root / 'best.pth', weights_only=False)
            self.assertEqual(best_checkpoint['epoch'], 4)
            self.assertEqual(best_checkpoint['checkpoint_type'], 'best')
            self.assertIn('optimizer_state_dict', checkpoint)
            self.assertIn('scheduler_state_dict', checkpoint)
            with self.assertRaisesRegex(ValueError, 'Only validation'):
                save_epoch_checkpoint(model, optimizer, scheduler, config, root,
                    5, 2, {'split': 'test', 'mean_dice': 1}, 0.7, 4)
            with self.assertRaisesRegex(ValueError, 'validation-selected'):
                run_test(config, None, 'cpu', root / 'latest.pth', root)
            with self.assertRaisesRegex(ValueError, 'completed training'):
                run_test(config, None, 'cpu', root / 'best.pth', root)

    def test_final_test_order_best_reload_resume_and_interruption(self):
        # Exercise orchestration only: stub training/evaluation, no real training.
        with TemporaryDirectory() as tmp:
            config = TrainingConfig2D('glas', epochs=2)
            loaders = {split: split for split in ('train', 'val', 'test')}
            events = []
            model = torch.nn.Linear(1, 1)
            def epoch_stub(loader, model, criterion, optimizer, scheduler, epoch, logger, config):
                events.append(f'train{epoch}')
                with torch.no_grad():
                    model.weight.fill_(epoch)
                return 2.0
            def eval_stub(model, loader, device, **kwargs):
                epoch = int(model.weight.item())
                events.append(f'{loader}{epoch}')
                return {'split': loader, 'mean_dice': 0.8 if epoch == 1 else 0.6,
                        'mean_iou': 0.5, 'cases': []}
            root = Path(tmp) / 'complete'
            with patch('train_2d.make_model', return_value=model), \
                 patch('train_2d.get_logger'), \
                 patch.object(torch.nn.Module, 'cuda', lambda self: self), \
                 patch('train_2d.train_one_epoch_2d', side_effect=epoch_stub), \
                 patch('train_2d.evaluate_2d', side_effect=eval_stub):
                report = train(config, loaders, root)
                self.assertEqual(events, ['train1', 'val1', 'train2', 'val2', 'test1'])
                self.assertEqual(report['checkpoint_epoch'], 1)
                self.assertNotIn('hd95', (root / 'test_glas.json').read_text().lower())
                self.assertEqual(len((root / 'history.jsonl').read_text().splitlines()), 2)
                events.clear()
                train(config, loaders, root, resume=root / 'checkpoints' / 'latest.pth')
                self.assertEqual(events, ['test1'])  # Completed resume: only final test.
                events.clear()
                def interrupted(*args):
                    if args[5] == 2:
                        raise RuntimeError('interrupted')
                    return epoch_stub(*args)
                with patch('train_2d.train_one_epoch_2d', side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                        train(config, loaders, Path(tmp) / 'interrupted')
                self.assertEqual(events, ['train1', 'val1'])

    def test_progress_epoch_keeps_two_branch_loss_and_scheduler(self):
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
            def forward(self, images, targets):
                return self.weight, 2 * self.weight, torch.ones_like(targets)
        model = ToyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler, logger = Mock(), Mock()
        config = TrainingConfig2D('cvc')
        loader = [{'image': torch.ones(1, 3, 1, 1), 'label': torch.ones(1, 1, 1)}]
        criterion = lambda pred, targets, *args: pred.square()
        with patch.object(torch.Tensor, 'cuda', lambda self, **kwargs: self):
            loss = train_one_epoch_2d(loader, model, criterion, optimizer, scheduler, 1, logger, config)
        self.assertEqual(loss, 5.0)
        self.assertAlmostEqual(model.weight.item(), 0.0, places=6)
        scheduler.step.assert_called_once()


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
