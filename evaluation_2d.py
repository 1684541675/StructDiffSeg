"""Foreground REGION metrics for RGB binary segmentation (not Object Dice)."""

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm


def binary_region_metrics(prediction, target):
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.ndim != 2 or prediction.shape != target.shape:
        raise ValueError('Expected equally sized 2D prediction and target')
    if not set(np.unique(prediction)) <= {0, 1} or not set(np.unique(target)) <= {0, 1}:
        raise ValueError('Region metrics require binary class labels')
    pred, gt = prediction.astype(bool), target.astype(bool)
    p, g = int(pred.sum()), int(gt.sum())
    intersection = int(np.count_nonzero(pred & gt))
    union = p + g - intersection
    result = {'dice': 2 * intersection / (p + g) if p + g else 1.0,
              'iou': intersection / union if union else 1.0}
    return result


@torch.no_grad()
def predict_single_image(image, model, original_size, device):
    """Prepared [1,3,H,W] RGB input; restore argmax mask with nearest resize.

    DiffUNet's DDIM return value already sums diffusion and SPM logits, exactly
    as in test_single_volume. No target is passed into inference.
    """
    if image.ndim != 4 or image.shape[0:2] != (1, 3):
        raise ValueError('Expected one RGB image [1,3,H,W]')
    logits = model(image.to(device, dtype=torch.float32), ddim=True)
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 4 or
            logits.shape[:2] != (1, 2) or logits.shape[2:] != image.shape[2:]):
        raise ValueError('Expected binary DDIM logits [1,2,H,W]')
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite DDIM logits')
    prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    height, width = original_size
    return np.array(Image.fromarray(prediction).resize(
        (width, height), Image.Resampling.NEAREST))


def evaluate_2d(model, loader, device, *, seed=42, description=None):
    """Per-image macro mean on native masks; deterministic DDIM random seed.

    Original masks are read using the existing dataset's ID-paired paths;
    resized loader labels are deliberately not used for native-size metrics.
    Restore model mode and RNG state so validation doesn't perturb training.
    """
    dataset = loader.dataset
    dataset = getattr(dataset, 'binary_dataset', None) or dataset
    if dataset.split not in ('val', 'test') or loader.batch_size != 1:
        raise ValueError('Evaluation requires a val/test loader with batch_size=1')
    if not len(dataset):
        raise ValueError('Cannot evaluate an empty split')
    paths = dict(zip(dataset.sample_list, dataset.data))
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    was_training = model.training
    rows, seen = [], set()
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.random.default_generator.manual_seed(seed)
            if device.type == 'cuda':
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            with tqdm(loader, desc=description or dataset.split.title(), unit='image') as progress:
                dice_sum = iou_sum = 0.0
                for batch in progress:
                    case = batch['case_name'][0]
                    if case not in paths or case in seen:
                        raise ValueError(f'Unexpected/duplicate evaluation case: {case}')
                    seen.add(case)
                    with Image.open(paths[case][1]) as mask_file:
                        raw = np.array(mask_file)
                    if raw.ndim == 3 and raw.shape[2] == 3 and np.all(raw == raw[..., :1]):
                        raw = raw[..., 0]
                    if raw.ndim != 2:
                        raise ValueError(f'Expected a grayscale or identical-channel mask: {case}')
                    target = (raw > 0).astype(np.uint8)
                    prediction = predict_single_image(batch['image'], model, target.shape, device)
                    scores = binary_region_metrics(prediction, target)
                    rows.append({'case_name': case, 'original_size': list(target.shape), **scores})
                    dice_sum += scores['dice']
                    iou_sum += scores['iou']
                    progress.set_postfix(Dice=f'{dice_sum / len(rows):.4f}',
                                         IoU=f'{iou_sum / len(rows):.4f}')
        if seen != set(paths):
            raise ValueError('Evaluation did not visit every split case')
    finally:
        model.train(was_training)
    report = {'split': dataset.split, 'samples': len(rows),
              'prediction': 'argmax(diffusion_ddim_logits + spm_logits)',
              'metric_type': 'binary_foreground_region',
              'aggregation': 'per_image_macro_mean_at_original_resolution',
              'seed': seed, 'mean_dice': float(np.mean([r['dice'] for r in rows])),
              'mean_iou': float(np.mean([r['iou'] for r in rows])), 'cases': rows}
    report['empty_policy'] = 'both_empty: Dice=IoU=1; one_empty: Dice=IoU=0'
    return report
