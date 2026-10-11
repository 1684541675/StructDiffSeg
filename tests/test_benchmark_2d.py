"""CPU/mocked-CUDA benchmark checks; no GPU workload or experiment files."""

from contextlib import ExitStack, redirect_stdout
import io
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Dataset

import train_2d
from configs.config_setting_2d import TrainingConfig2D


class Samples(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, index):
        return {'image': torch.ones(3, 2, 2), 'label': torch.zeros(2, 2)}


class QuietProgress:
    def __init__(self, iterable, **kwargs):
        self.iterable = iterable
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def __iter__(self):
        return iter(self.iterable)
    def set_postfix(self, **kwargs):
        pass
    @staticmethod
    def write(message):
        pass


class BenchmarkChecks(unittest.TestCase):
    def run_mock_benchmark(self, oom=False):
        models, optimizers, calls = [], [], []
        def make_model(config):
            model = torch.nn.Linear(1, 1)
            models.append(model)
            return model
        def make_optimizer(config, model):
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            optimizers.append(optimizer)
            return optimizer
        def loader(dataset, **kwargs):
            kwargs['pin_memory'] = False
            return DataLoader(dataset, **kwargs)
        def step(batch, model, criterion, optimizer, epoch, config):
            calls.append((config.batch_size, len(batch['image']), model, optimizer, epoch))
            if oom and len(calls) == 1:
                raise torch.cuda.OutOfMemoryError('synthetic OOM')
            return torch.tensor(2.0)
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(patch('train_2d.tqdm', QuietProgress))
            stack.enter_context(patch('train_2d.make_model', side_effect=make_model))
            stack.enter_context(patch('train_2d.get_optimizer', side_effect=make_optimizer))
            stack.enter_context(patch('train_2d.NPY_datasets', return_value=Samples()))
            stack.enter_context(patch('train_2d.DataLoader', side_effect=loader))
            stack.enter_context(patch('train_2d.training_step_2d', side_effect=step))
            stack.enter_context(patch.object(TrainingConfig2D, 'make_criterion', return_value=torch.nn.MSELoss()))
            stack.enter_context(patch.object(torch.nn.Module, 'cuda', lambda self: self))
            seeded = stack.enter_context(patch('train_2d.set_seed'))
            synced = stack.enter_context(patch('torch.cuda.synchronize'))
            reset = stack.enter_context(patch('torch.cuda.reset_peak_memory_stats'))
            stack.enter_context(patch('torch.cuda.is_available', return_value=True))
            stack.enter_context(patch('torch.cuda.empty_cache'))
            stack.enter_context(patch('torch.cuda.max_memory_allocated', return_value=512 * 2**20))
            stack.enter_context(patch('torch.cuda.max_memory_reserved', return_value=1024 * 2**20))
            stack.enter_context(patch('train_2d.time.perf_counter', side_effect=(i / 8 for i in range(100))))
            stack.enter_context(patch('torch.save', side_effect=AssertionError('Must not save checkpoints')))
            stack.enter_context(patch('train_2d.evaluate_2d', side_effect=AssertionError('Must not evaluate')))
            results = train_2d.benchmark(TrainingConfig2D('glas'), steps=10, warmup_steps=2)
        self.assertEqual(len(models), 3)
        self.assertEqual(len({id(m) for m in models}), 3)
        self.assertEqual(len({id(o) for o in optimizers}), 3)
        self.assertEqual(seeded.call_count, 3)
        self.assertTrue(all(call.args == (42,) for call in seeded.call_args_list))
        self.assertGreaterEqual(synced.call_count, 42)
        self.assertEqual(reset.call_count, 5 if oom else 6)
        for call in calls:
            self.assertEqual(call[0], call[1])  # No partial batches, including wraparound.
            self.assertEqual(call[4], 1)
        return results, calls

    def test_timing_initialization_isolation_and_loader_wrap(self):
        results, calls = self.run_mock_benchmark()
        self.assertEqual(len(calls), 3 * (2 + 10))
        for row in results:
            self.assertEqual(row['status'], 'passed')
            self.assertEqual(row['completed_steps'], 10)
            self.assertEqual(row['mean_step_seconds'], 0.125)
            self.assertEqual(row['images_per_second'], row['batch_size'] * 8)
            self.assertEqual(row['peak_allocated_mib'], 512)

    def test_oom_continues_without_reporting_partial_throughput(self):
        results, _ = self.run_mock_benchmark(oom=True)
        self.assertEqual(results[0]['status'], 'cuda_out_of_memory')
        self.assertIsNone(results[0]['images_per_second'])
        self.assertEqual([r['status'] for r in results[1:]], ['passed', 'passed'])

    def test_cli_only_dispatches_benchmark(self):
        with patch('sys.argv', ['train_2d.py', '--dataset', 'cvc', '--benchmark']), \
             patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.get_device_name', return_value='mock GPU'), \
             patch('train_2d.set_seed'), patch('torch.set_num_threads'), \
             patch('train_2d.benchmark') as benchmark, \
             patch('train_2d.make_loaders') as loaders, \
             patch('pathlib.Path.mkdir', side_effect=AssertionError('No result directory')), \
             redirect_stdout(io.StringIO()):
            train_2d.main()
        loaders.assert_not_called()
        self.assertEqual(benchmark.call_args.args[1:], (15, 3))


if __name__ == '__main__':
    unittest.main()
