"""GlaS/CVC StructDiffSeg entry point. Default: one-batch compatibility check.

Examples:
    python train_2d.py --dataset glas --smoke-test
    python train_2d.py --dataset cvc --smoke-test --device cpu
    python train_2d.py --dataset glas --train --batch-size 1

Formal validation/test and best-by-validation checkpoints are deferred.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from configs.config_setting_2d import TrainingConfig2D
from datasets.dataset import NPY_datasets
from engine_synapse import func, train_one_epoch
from models.DiffUNetSPM.diffunet.diffunet_model import DiffUNet
from utils import get_logger, get_optimizer, get_scheduler, set_seed


def make_loaders(config):
    """All three TXT splits; only training is shuffled/augmented."""
    return {
        split: DataLoader(
            NPY_datasets(config.data_path, split=split, list_dir=config.list_dir,
                         output_size=(config.input_size_h, config.input_size_w)),
            batch_size=config.batch_size if split == 'train' else 1,
            shuffle=split == 'train', num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(), drop_last=False)
        for split in ('train', 'val', 'test')
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


def train(config, loaders, work_dir, resume=None):
    """Reuse the CUDA engine; save latest state without any evaluation."""
    checkpoint_dir = work_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger('train_2d', str(work_dir / 'logs'))
    logger.info('Config: %s', asdict(config))
    model = make_model(config).cuda()
    criterion = config.make_criterion().cuda()
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    start_epoch = 1
    if resume:
        checkpoint = torch.load(resume, map_location='cpu', weights_only=True)
        saved_config = checkpoint['config']
        for key in ('datasets_name', 'data_path', 'list_dir', 'batch_size', 'epochs',
                    'seed', 'input_channels', 'num_classes', 'lr', 'weight_decay'):
            if saved_config[key] != asdict(config)[key]:
                raise ValueError(f'Resume config mismatch: {key}')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    for epoch in range(start_epoch, config.epochs + 1):
        loss = train_one_epoch(loaders['train'], model, criterion, optimizer,
                               scheduler, epoch, logger, config)
        torch.save({'epoch': epoch, 'loss': loss, 'config': asdict(config),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()},
                   checkpoint_dir / 'latest.pth')


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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke-test', action='store_true', help='Default: check one batch only')
    mode.add_argument('--train', action='store_true', help='Explicitly run full training')
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use --device cpu for a smoke test')
    if args.train and args.device != 'cuda':
        parser.error('The existing training engine requires CUDA')
    if args.resume and not args.train:
        parser.error('--resume is only supported with --train')
    config = TrainingConfig2D(args.dataset, args.data_path, args.list_dir,
                              args.batch_size, args.epochs, args.num_workers, args.seed)
    set_seed(config.seed)
    torch.set_num_threads(args.cpu_threads)
    loaders = make_loaders(config)
    print('Split sizes:', {k: len(v.dataset) for k, v in loaders.items()}, flush=True)
    work_dir = Path(args.work_dir) if args.work_dir else Path(__file__).resolve().parent / 'results' / (
        f'structdiffseg_{args.dataset}_' + time.strftime('%Y%m%d_%H%M%S'))
    if args.train:
        train(config, loaders, work_dir, args.resume)
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
