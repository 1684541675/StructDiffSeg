"""GlaS/CVC StructDiffSeg entry point. Default: one-batch compatibility check.

Examples:
    python train_2d.py --dataset glas --smoke-test
    python train_2d.py --dataset cvc --smoke-test --device cpu
    python train_2d.py --dataset glas --train --batch-size 1

    python train_2d.py --dataset glas --test --work-dir results/my_run

Validation selects best.pth; completed training automatically tests best.pth.
--test also supports independent final evaluation of a trusted local checkpoint.
"""

import argparse
from dataclasses import asdict, replace
import gc
import math
import json
from pathlib import Path
import time
import shutil

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config_setting_2d import TrainingConfig2D
from datasets.dataset import NPY_datasets
from evaluation_2d import evaluate_2d
from engine_synapse import func
from models.DiffUNetSPM.diffunet.diffunet_model import DiffUNet
from utils import get_logger, get_optimizer, get_scheduler, set_seed


def make_loaders(config, splits=('train', 'val', 'test')):
    """All three TXT splits; only training is shuffled/augmented."""
    return {
        split: DataLoader(
            NPY_datasets(config.data_path, split=split, list_dir=config.list_dir,
                         output_size=(config.input_size_h, config.input_size_w)),
            batch_size=config.batch_size if split == 'train' else 1,
            shuffle=split == 'train', num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(), drop_last=False)
        for split in splits
    }


def make_model(config):
    return DiffUNet(config.input_channels, config.num_classes,
                    ddim_steps=config.ddim_steps, config=config.make_spm_config())


def smoke_test(config, loaders, device):
    """One real batch, unchanged training loss, backward and optimizer update."""
    report = {'dataset': config.datasets_name, 'device': str(device),
              'batch_size': config.batch_size, 'status': 'running',
              'samples': {split: len(loader.dataset) for split, loader in loaders.items()}}
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        report['gpu'] = torch.cuda.get_device_name(device)
        report['gpu_total_mib'] = torch.cuda.get_device_properties(device).total_memory / 2**20
    start = time.perf_counter()
    stage = 'model construction'
    try:
        model = make_model(config).to(device).train()
        criterion = config.make_criterion().to(device)
        optimizer = get_optimizer(config, model)
        scheduler = get_scheduler(config, optimizer)
        stage = 'batch loading'
        batch = next(iter(loaders['train']))
        images = batch['image'].to(device, dtype=torch.float32)
        targets = batch['label'].to(device, dtype=torch.long)
        report.update(image_shape=list(images.shape), image_dtype=str(images.dtype),
                      label_shape=list(targets.shape), label_dtype=str(targets.dtype),
                      label_values=targets.unique().tolist(), case_names=batch['case_name'])
        assert tuple(images.shape[1:]) == (3, 256, 256)
        assert tuple(targets.shape[1:]) == (256, 256)
        assert set(report['label_values']) <= {0, 1}
        assert torch.isfinite(images).all()
        embedding = model.denoise_model.embedding_table
        assert (embedding.num_embeddings, embedding.embedding_dim) == (2, 2)
        assert model.edge_model.segmentation_head.out_channels == 2
        assert tuple(model.edge_model.learnable_shape_prior.shape) == (1, 2, 256)
        report['class_embedding_shape'] = list(embedding.weight.shape)
        report['shape_prior_shape'] = list(model.edge_model.learnable_shape_prior.shape)
        report['parameters'] = sum(p.numel() for p in model.parameters())
        optimizer.zero_grad()
        stage = 'forward'
        pred, pred_edge, uncertainty = model(images, targets)
        assert pred.shape == pred_edge.shape == (images.shape[0], 2, 256, 256)
        assert uncertainty.shape == targets.shape
        for tensor in (pred, pred_edge, uncertainty):
            assert torch.isfinite(tensor).all(), 'Nonfinite model output'
        report.update(diffusion_shape=list(pred.shape), spm_shape=list(pred_edge.shape),
                      uncertainty_shape=list(uncertainty.shape),
                      uncertainty_range=[uncertainty.min().item(), uncertainty.max().item()])
        stage = 'loss'
        # Exactly the non-AMP loss strategy in engine_synapse.train_one_epoch.
        uncertainty = torch.clamp(uncertainty, 0.0, 1.0)
        diffusion_loss = criterion(pred, targets, uncertainty, func(1, config.epochs))
        spm_loss = criterion(pred_edge, targets)
        loss = diffusion_loss + spm_loss
        assert torch.isfinite(loss), 'Nonfinite loss'
        report.update(diffusion_loss=diffusion_loss.item(), spm_loss=spm_loss.item(),
                      loss=loss.item())
        stage = 'backward'
        loss.backward()
        branch_norms = {'edge_model': 0.0, 'denoise_model': 0.0}
        missing = []
        for name, param in model.named_parameters():
            if param.grad is None:
                missing.append(name)
                continue
            assert torch.isfinite(param.grad).all(), f'Nonfinite gradient: {name}'
            branch = name.split('.')[0]
            branch_norms[branch] += param.grad.double().square().sum().item()
        report['gradient_norms'] = {k: v**0.5 for k, v in branch_norms.items()}
        report['parameters_without_gradient'] = missing
        assert all(v > 0 for v in branch_norms.values()), 'Missing branch gradients'
        assert embedding.weight.grad is not None
        report['class_embedding_gradient_norm'] = embedding.weight.grad.norm().item()
        assert report['class_embedding_gradient_norm'] > 0
        before = embedding.weight.detach().clone()
        stage = 'optimizer step'
        optimizer.step()
        assert all(torch.isfinite(p).all() for p in model.parameters()), 'Nonfinite updated parameter'
        report['class_embedding_update_norm'] = (embedding.weight.detach() - before).norm().item()
        assert report['class_embedding_update_norm'] > 0
        scheduler.step()
        report['next_lr'] = optimizer.param_groups[0]['lr']
        report['status'] = 'passed'
    except torch.cuda.OutOfMemoryError as exc:
        report.update(status='cuda_out_of_memory', failed_stage=stage, error=str(exc))
    finally:
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            report['peak_allocated_mib'] = torch.cuda.max_memory_allocated(device) / 2**20
            report['peak_reserved_mib'] = torch.cuda.max_memory_reserved(device) / 2**20
        else:
            report['peak_allocated_mib'] = None
            report['peak_reserved_mib'] = None
        report['elapsed_seconds'] = time.perf_counter() - start
    return report


def save_epoch_checkpoint(model, optimizer, scheduler, config, checkpoint_dir,
                          epoch, loss, validation, best_dice, best_epoch):
    """Strict improvement in validation REGION Dice determines best.pth."""
    if validation['split'] != 'val':
        raise ValueError('Only validation metrics may select checkpoints')
    score = validation['mean_dice']
    if not 0.0 <= score <= 1.0:
        raise ValueError('Invalid validation Dice')
    improved = score > best_dice
    if improved:
        best_dice, best_epoch = score, epoch
    checkpoint = {'epoch': epoch, 'loss': loss, 'config': asdict(config),
                  'best_dice': best_dice, 'best_epoch': best_epoch,
                  'validation': validation, 'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'scheduler_state_dict': scheduler.state_dict(),
                  'checkpoint_type': 'latest'}
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if improved:
        torch.save({**checkpoint, 'checkpoint_type': 'best'}, checkpoint_dir / 'best.pth')
    torch.save(checkpoint, checkpoint_dir / 'latest.pth')
    return best_dice, best_epoch


def training_step_2d(batch, model, criterion, optimizer, epoch, config):
    """Shared formal-training/benchmark update; preserve the original strategy."""
    optimizer.zero_grad()
    images = batch['image'].cuda(non_blocking=True).float()
    targets = batch['label'].cuda(non_blocking=True).float()
    pred, pred_edge, uncertainty = model(images, targets)
    uncertainty = torch.clamp(uncertainty, 0.0, 1.0)
    loss = criterion(pred, targets, uncertainty, func(epoch, config.epochs))
    loss = loss + criterion(pred_edge, targets)
    loss.backward()
    optimizer.step()
    return loss


def benchmark(config, steps=15, warmup_steps=3):
    """CUDA end-to-end throughput; no validation, scheduler steps or file writes.

    Each measurement includes loading/augmentation, H2D transfer and the full
    training update. Initialization, warmup, loss checks and progress output
    are outside measured intervals. Peaks after warmup include resident model,
    gradients and AdamW state. Full batches only; cycle the loader if needed.
    """
    if not torch.cuda.is_available():
        raise ValueError('Benchmark requires CUDA')
    if not 10 <= steps <= 20 or warmup_steps < 1:
        raise ValueError('Benchmark requires 10-20 measured steps and at least one warmup step')
    results = []
    for batch_size in (1, 2, 4):
        trial = replace(config, batch_size=batch_size)
        model = optimizer = criterion = loader = iterator = batch = loss = None
        row = {'batch_size': batch_size, 'status': 'running', 'steps': steps,
               'warmup_steps': warmup_steps, 'completed_steps': 0,
               'mean_step_seconds': None, 'images_per_second': None}
        stage = 'initialization'
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            set_seed(trial.seed)
            model = make_model(trial).cuda().train()
            criterion = trial.make_criterion().cuda()
            optimizer = get_optimizer(trial, model)
            dataset = NPY_datasets(trial.data_path, split='train', list_dir=trial.list_dir,
                                   output_size=(trial.input_size_h, trial.input_size_w))
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=trial.num_workers, pin_memory=True,
                                drop_last=True, generator=torch.Generator().manual_seed(trial.seed))
            if not len(loader):
                raise ValueError(f'Train split has fewer than {batch_size} samples')
            iterator = iter(loader)
            def next_batch():
                nonlocal iterator
                try:
                    return next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    return next(iterator)
            stage = 'warmup'
            for _ in range(warmup_steps):
                loss = training_step_2d(next_batch(), model, criterion, optimizer, 1, trial)
                if not math.isfinite(loss.item()):
                    raise FloatingPointError('Nonfinite warmup loss')
            torch.cuda.synchronize()
            loss = None
            torch.cuda.reset_peak_memory_stats()
            elapsed = 0.0
            stage = 'measurement'
            with tqdm(range(steps), desc=f'Benchmark BS={batch_size}', unit='step') as progress:
                for _ in progress:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    batch = next_batch()
                    loss = training_step_2d(batch, model, criterion, optimizer, 1, trial)
                    torch.cuda.synchronize()
                    elapsed += time.perf_counter() - start
                    row['completed_steps'] += 1
                    value = loss.item()
                    if not math.isfinite(value):
                        raise FloatingPointError('Nonfinite benchmark loss')
                    progress.set_postfix(loss=f'{value:.4f}',
                                         step_s=f"{elapsed / row['completed_steps']:.4f}")
                    loss = batch = None
            row.update(status='passed', mean_step_seconds=elapsed / steps,
                       images_per_second=batch_size * steps / elapsed,
                       measured_seconds=elapsed)
        except torch.cuda.OutOfMemoryError as exc:
            row.update(status='cuda_out_of_memory', failed_stage=stage, error=str(exc))
            tqdm.write(f'BS={batch_size}: CUDA OOM during {stage}; continuing to next batch size')
        except FloatingPointError as exc:
            row.update(status='nonfinite_loss', failed_stage=stage, error=str(exc))
        finally:
            row['peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
            row['peak_reserved_mib'] = torch.cuda.max_memory_reserved() / 2**20
            model = optimizer = criterion = loader = iterator = batch = loss = None
            gc.collect()
            torch.cuda.empty_cache()
        results.append(row)
    print('\nBenchmark: initialization/warmup excluded; includes data loading, H2D and full update.')
    print(' BS  status                   mean step (s)  images/s  peak allocated/reserved (MiB)')
    for row in results:
        duration = f"{row['mean_step_seconds']:.4f}" if row['mean_step_seconds'] is not None else '-'
        throughput = f"{row['images_per_second']:.2f}" if row['images_per_second'] is not None else '-'
        print(f" {row['batch_size']:2}  {row['status']:<24} {duration:>12}  {throughput:>8}  "
              f"{row['peak_allocated_mib']:.1f}/{row['peak_reserved_mib']:.1f}")
    return results


def train_one_epoch_2d(loader, model, criterion, optimizer, scheduler, epoch, logger, config):
    """Same non-AMP steps as engine_synapse.train_one_epoch, with 2D progress."""
    start = time.perf_counter()
    model.train()
    total_loss = 0.0
    with tqdm(loader, desc=f'Train {epoch}/{config.epochs}', unit='batch') as progress:
        for index, batch in enumerate(progress):
            loss = training_step_2d(batch, model, criterion, optimizer, epoch, config)
            total_loss += loss.item()
            mean_loss = total_loss / (index + 1)
            lr = optimizer.param_groups[0]['lr']
            progress.set_postfix(loss=f'{loss.item():.4f}', mean_loss=f'{mean_loss:.4f}', lr=f'{lr:.6g}')
            if index % config.print_interval == 0:
                logger.info('train: epoch %s, iter %s, loss %.4f, lr %s', epoch, index, loss.item(), lr)
    scheduler.step()
    logger.info('Train epoch %s: loss %.4f, time %.2fs', epoch, mean_loss, time.perf_counter() - start)
    tqdm.write(f'Train epoch {epoch}: mean Loss={mean_loss:.4f}')
    return mean_loss


def train(config, loaders, work_dir, resume=None):
    """Validate each epoch; test best only after the full epoch loop completes."""
    checkpoint_dir = work_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger('train_2d', str(work_dir / 'logs'))
    logger.info('Config: %s', asdict(config))
    model = make_model(config).cuda()
    criterion = config.make_criterion().cuda()
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    start_epoch = 1
    best_dice, best_epoch = -1.0, 0
    if resume:
        checkpoint = torch.load(resume, map_location='cpu', weights_only=False)
        saved_config = checkpoint['config']
        for key in ('datasets_name', 'data_path', 'list_dir', 'batch_size', 'epochs',
                    'seed', 'input_channels', 'num_classes', 'lr', 'weight_decay'):
            if saved_config[key] != asdict(config)[key]:
                raise ValueError(f'Resume config mismatch: {key}')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', -1.0)
        best_epoch = checkpoint.get('best_epoch', 0)
        # Older stage-2 latest checkpoints have no validation/best metadata.
        source_best = Path(resume).parent / 'best.pth'
        destination_best = checkpoint_dir / 'best.pth'
        if best_epoch and source_best.is_file():
            if source_best.resolve() != destination_best.resolve():
                shutil.copy2(source_best, destination_best)
        elif best_epoch:
            logger.warning('Historical best.pth missing; selecting best from future validation')
            best_dice, best_epoch = -1.0, 0
    completed_epoch = start_epoch - 1
    for epoch in range(start_epoch, config.epochs + 1):
        loss = train_one_epoch_2d(loaders['train'], model, criterion, optimizer,
                               scheduler, epoch, logger, config)
        validation = evaluate_2d(model, loaders['val'], 'cuda', seed=config.seed,
                                 description=f'Val {epoch}/{config.epochs}')
        logger.info('Val epoch %s: region Dice=%.6f, IoU=%.6f', epoch,
                    validation['mean_dice'], validation['mean_iou'])
        print(f"Val epoch {epoch}: region Dice={validation['mean_dice']:.6f}, "
              f"IoU={validation['mean_iou']:.6f}", flush=True)
        best_dice, best_epoch = save_epoch_checkpoint(
            model, optimizer, scheduler, config, checkpoint_dir, epoch,
            loss, validation, best_dice, best_epoch)
        completed_epoch = epoch
        with (work_dir / 'history.jsonl').open('a', encoding='utf-8') as history:
            history.write(json.dumps({'epoch': epoch, 'train_loss': loss,
                                      'val_dice': validation['mean_dice'],
                                      'val_iou': validation['mean_iou'],
                                      'best_dice': best_dice, 'best_epoch': best_epoch},
                                     allow_nan=False) + '\n')
    if completed_epoch != config.epochs:
        raise ValueError('Training is not complete; final test was not run')
    best_path = checkpoint_dir / 'best.pth'
    if not best_path.is_file():
        raise FileNotFoundError('Completed run has no validation-selected best.pth')
    # A copied best.pth retains proof of completion for independent --test.
    best_checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
    best_checkpoint.update(training_completed=True, completed_epoch=completed_epoch)
    torch.save(best_checkpoint, best_path)
    del best_checkpoint
    # Reuse the trained model to avoid holding two full models on the GPU.
    report = test(config, loaders['test'], 'cuda', best_path, work_dir, model=model)
    logger.info('Final Test Dice=%.6f, IoU=%.6f (best epoch %s)',
                report['mean_dice'], report['mean_iou'], report['checkpoint_epoch'])
    return report


def test(config, loader, device, checkpoint_path, work_dir, model=None):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint.get('checkpoint_type') != 'best':
        raise ValueError('Test requires a validation-selected best.pth checkpoint')
    if (not checkpoint.get('training_completed', False) or
            checkpoint.get('completed_epoch') != checkpoint['config']['epochs']):
        raise ValueError('Final test requires a best.pth from completed training')
    for key in ('datasets_name', 'input_channels', 'num_classes',
                'input_size_h', 'input_size_w', 'ddim_steps'):
        if checkpoint['config'][key] != asdict(config)[key]:
            raise ValueError(f'Test checkpoint config mismatch: {key}')
    if model is None:
        model = make_model(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    report = evaluate_2d(model, loader, device, seed=config.seed, description='Test best.pth')
    report.update(dataset=config.datasets_name, checkpoint=str(Path(checkpoint_path).resolve()),
                  checkpoint_epoch=checkpoint['epoch'], best_val_dice=checkpoint['best_dice'])
    work_dir.mkdir(parents=True, exist_ok=True)
    report_path = work_dir / f'test_{config.datasets_name}.json'
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(f"Test Dice={report['mean_dice']:.6f}, Test IoU={report['mean_iou']:.6f}", flush=True)
    print(f'Report: {report_path}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=('glas', 'cvc'))
    parser.add_argument('--data-path')
    parser.add_argument('--list-dir')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--work-dir')
    parser.add_argument('--resume', help='Explicit latest.pth checkpoint for training')
    parser.add_argument('--checkpoint', help='best.pth for --test; defaults to work-dir/checkpoints/best.pth')
    parser.add_argument('--benchmark-steps', type=int, default=15, choices=range(10, 21))
    parser.add_argument('--benchmark-warmup', type=int, default=3)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke-test', action='store_true', help='Default: check one batch only')
    mode.add_argument('--train', action='store_true', help='Explicitly run full training')
    mode.add_argument('--test', action='store_true', help='Evaluate best.pth on test only')
    mode.add_argument('--benchmark', action='store_true', help='Compare BS=1,2,4 without saving files')
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use --device cpu for a smoke test')
    if args.train and args.device != 'cuda':
        parser.error('The existing training engine requires CUDA')
    if args.benchmark and args.device != 'cuda':
        parser.error('Benchmark requires --device cuda')
    if args.benchmark and args.benchmark_warmup < 1:
        parser.error('--benchmark-warmup must be positive')
    if args.resume and not args.train:
        parser.error('--resume is only supported with --train')
    if args.checkpoint and not args.test:
        parser.error('--checkpoint is only supported with --test')
    if args.test and not (args.work_dir or args.checkpoint):
        parser.error('--test requires --work-dir or --checkpoint')
    config = TrainingConfig2D(args.dataset, args.data_path, args.list_dir,
                              args.batch_size, args.epochs, args.num_workers, args.seed)
    set_seed(config.seed)
    torch.set_num_threads(args.cpu_threads)
    if args.benchmark:
        print(f'GPU: {torch.cuda.get_device_name()} | dataset: {args.dataset} | '
              f'BS=1,2,4 | warmup={args.benchmark_warmup} | measured steps={args.benchmark_steps}', flush=True)
        benchmark(config, args.benchmark_steps, args.benchmark_warmup)
        return
    splits = ('test',) if args.test else ('train', 'val', 'test')
    loaders = make_loaders(config, splits)
    print('Split sizes:', {k: len(v.dataset) for k, v in loaders.items()}, flush=True)
    work_dir = Path(args.work_dir) if args.work_dir else Path(__file__).resolve().parent / 'results' / (
        f'structdiffseg_{args.dataset}_' + time.strftime('%Y%m%d_%H%M%S'))
    if args.train:
        train(config, loaders, work_dir, args.resume)
    elif args.test:
        checkpoint_path = Path(args.checkpoint) if args.checkpoint else work_dir / 'checkpoints' / 'best.pth'
        if not checkpoint_path.is_file():
            parser.error(f'Best checkpoint not found: {checkpoint_path}')
        test(config, loaders['test'], torch.device(args.device), checkpoint_path, work_dir)
    else:
        report = smoke_test(config, loaders, torch.device(args.device))
        work_dir.mkdir(parents=True, exist_ok=True)
        report_path = work_dir / f'smoke_{args.dataset}_{args.device}.json'
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
        print(f'Report: {report_path}', flush=True)
        if report['status'] != 'passed':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
