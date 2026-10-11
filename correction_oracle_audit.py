"""Normal-inference correction oracle statistics; imported only by its audit flag."""

import json
import os
import time

import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion, zoom
from sklearn.metrics import average_precision_score, roc_auc_score


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else float('nan')


def _auc_ap(target, scores):
    """Use raw score direction; undefined ROC-AUC/empty scopes return NaN."""
    if target.size == 0 or not np.isfinite(scores).all():
        return float('nan'), float('nan')
    positives = int(np.count_nonzero(target))
    if positives == 0:
        return float('nan'), 0.0
    if positives == target.size:
        return float('nan'), 1.0
    return float(roc_auc_score(target, scores)), float(average_precision_score(target, scores))


def _gt_boundary_band(label, radius=3):
    """Union of class-specific 2D bands, on the original GT grid, including background side."""
    offsets = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(offsets, offsets, indexing='ij')
    disk = xx ** 2 + yy ** 2 <= radius ** 2
    band = np.zeros(label.shape, dtype=bool)
    for index, gt_slice in enumerate(label):
        for cls in (1, 2, 3):
            mask = gt_slice == cls
            dilated = binary_dilation(mask, structure=disk, border_value=0)
            eroded = binary_erosion(mask, structure=disk, border_value=0)
            band[index] |= dilated ^ eroded
    return band


class CorrectionOracleCase:
    """Keep only this case's discrete predictions and seven float32 CPU signal maps."""

    def __init__(self, label, case):
        self.label = label
        self.case = case
        self.predictions = {name: np.empty(label.shape, dtype=np.uint8)
                            for name in ('spm', 'diffusion')}
        self.signals = {name: np.empty(label.shape, dtype=np.float32) for name in (
            'spm_confidence', 'diff_confidence', 'confidence_diff',
            'spm_entropy', 'diff_entropy', 'entropy_diff', 'probability_l1_disagreement',
        )}

    def add_slice(self, spm_logits, diffusion_logits, index):
        spm_prob = torch.softmax(spm_logits, dim=1)
        diff_prob = torch.softmax(diffusion_logits, dim=1)
        spm_confidence = spm_prob.max(dim=1).values
        diff_confidence = diff_prob.max(dim=1).values
        spm_pred = spm_prob.argmax(dim=1)
        diff_pred = diff_prob.argmax(dim=1)
        spm_entropy = -(spm_prob * torch.log(spm_prob + 1e-8)).sum(dim=1)
        diff_entropy = -(diff_prob * torch.log(diff_prob + 1e-8)).sum(dim=1)
        maps = {
            'spm_confidence': spm_confidence,
            'diff_confidence': diff_confidence,
            'confidence_diff': diff_confidence - spm_confidence,
            'spm_entropy': spm_entropy,
            'diff_entropy': diff_entropy,
            'entropy_diff': spm_entropy - diff_entropy,
            'probability_l1_disagreement': (spm_prob - diff_prob).abs().mean(dim=1),
        }

        def restore(tensor):
            array = tensor.squeeze(0).detach().cpu().numpy()
            height, width = self.label.shape[1:]
            if array.shape != (height, width):
                array = zoom(array, (height / array.shape[0], width / array.shape[1]), order=0)
            return array

        self.predictions['spm'][index] = restore(spm_pred)
        self.predictions['diffusion'][index] = restore(diff_pred)
        for name, tensor in maps.items():
            self.signals[name][index] = restore(tensor)

    def finalize(self):
        from utils import calculate_metric_percase

        gt = self.label
        spm_pred = self.predictions['spm']
        diff_pred = self.predictions['diffusion']
        spm_correct = spm_pred == gt
        diff_correct = diff_pred == gt
        categories = {
            'both_correct': spm_correct & diff_correct,
            'spm_only_correct': spm_correct & ~diff_correct,
            'diffusion_only_correct': ~spm_correct & diff_correct,
            'both_wrong': ~spm_correct & ~diff_correct,
        }
        benefit = categories['diffusion_only_correct']
        harm = categories['spm_only_correct']
        oracle_pred = spm_pred.copy()
        oracle_pred[benefit] = diff_pred[benefit]
        row = {'case': self.case, 'condition_mode': 'normal'}
        for name, prediction in (('spm', spm_pred), ('diffusion', diff_pred), ('oracle', oracle_pred)):
            metrics = np.asarray([calculate_metric_percase(prediction == cls, gt == cls)
                                  for cls in (1, 2, 3)], dtype=float)
            for index, cls_name in enumerate(('rv', 'myo', 'lv')):
                row[f'{name}_{cls_name}_dice'] = float(metrics[index, 0])
                row[f'{name}_{cls_name}_hd95'] = float(metrics[index, 1])
            row[f'{name}_mean_dice'] = float(metrics[:, 0].mean())
            row[f'{name}_mean_hd95'] = float(metrics[:, 1].mean())
        row['oracle_gain_vs_spm'] = row['oracle_mean_dice'] - row['spm_mean_dice']
        row['oracle_gain_vs_diff'] = row['oracle_mean_dice'] - row['diffusion_mean_dice']

        foreground = gt != 0
        scopes = {'all': np.ones(gt.shape, dtype=bool), 'foreground': foreground,
                  'rv': gt == 1, 'myo': gt == 2, 'lv': gt == 3}
        for scope, selected in scopes.items():
            count = int(np.count_nonzero(selected))
            row[f'{scope}_pixel_count'] = count
            for category, mask in categories.items():
                category_count = int(np.count_nonzero(mask & selected))
                row[f'{scope}_{category}_count'] = category_count
                row[f'{scope}_{category}_ratio'] = _ratio(category_count, count)

        boundary = _gt_boundary_band(gt, radius=3)
        for name, mask in (('benefit', benefit), ('harm', harm)):
            count = int(np.count_nonzero(mask))
            boundary_count = int(np.count_nonzero(mask & boundary))
            row[f'{name}_boundary_count'] = boundary_count
            row[f'{name}_non_boundary_count'] = count - boundary_count
            row[f'{name}_boundary_ratio'] = _ratio(boundary_count, count)
            row[f'{name}_non_boundary_ratio'] = _ratio(count - boundary_count, count)
            foreground_count = int(np.count_nonzero(mask & foreground))
            foreground_boundary_count = int(np.count_nonzero(mask & foreground & boundary))
            row[f'foreground_{name}_boundary_ratio'] = _ratio(foreground_boundary_count,
                                                            foreground_count)

        disagreement = spm_pred != diff_pred
        for scope in ('all', 'foreground'):
            selected = scopes[scope]
            target = benefit[selected]
            row[f'{scope}_benefit_prevalence'] = _ratio(int(target.sum()), target.size)
            for signal, values in self.signals.items():
                scores = values[selected]
                auc, ap = _auc_ap(target, scores)
                row[f'{scope}_{signal}_auc'] = auc
                row[f'{scope}_{signal}_ap'] = ap
                row[f'{scope}_{signal}_nonfinite_count'] = int(np.count_nonzero(~np.isfinite(scores)))
            # Aliases match the shorter field names used in the audit request.
            row[f'{scope}_probability_disagreement_auc'] = row[f'{scope}_probability_l1_disagreement_auc']
            row[f'{scope}_probability_disagreement_ap'] = row[f'{scope}_probability_l1_disagreement_ap']
            predicted = disagreement[selected]
            tp = int(np.count_nonzero(predicted & target))
            fp = int(np.count_nonzero(predicted & ~target))
            fn = int(np.count_nonzero(~predicted & target))
            row[f'{scope}_prediction_disagreement_precision'] = (
                float(tp / (tp + fp)) if tp + fp else 0.0)
            row[f'{scope}_prediction_disagreement_recall'] = (
                float(tp / (tp + fn)) if tp + fn else 0.0)
            row[f'{scope}_prediction_disagreement_f1'] = (
                float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0)
        return row


def summarize_correction_oracle(rows):
    """Unweighted per-case mean and population std, excluding nonfinite values."""
    summary = {
        'case_count': len(rows),
        'condition_mode': 'normal',
        'spm_scale': 1.0,
        'boundary_radius_pixels': 3,
        'boundary_definition': 'union of per-class 2D dilation XOR erosion, Euclidean disk, border_value=0',
        'boundary_grid': 'original GT pixels; no morphology along the slice axis',
        'boundary_ratio_denominator': 'all benefit/harm pixels; foreground variants use GT foreground only',
        'non_boundary_definition': 'complement of union band, including distant background',
        'signal_direction': 'raw signals; higher score predicts benefit; no sign inversion or fitting',
        'entropy_diff_definition': 'spm_entropy - diff_entropy',
        'probability_l1_disagreement_definition': 'mean(abs(spm_prob - diff_prob), class_dim)',
        'pr_metric': 'average_precision_score, not trapezoidal PR area',
        'undefined_roc_auc': 'NaN for single target class or empty scope',
        'ap_single_class': '0 if no positives, 1 if all positive; NaN for empty scope',
        'undefined_ratios': 'NaN for empty denominator',
        'binary_metric_zero_division': 0.0,
        'nonfinite_signal_policy': 'AUC/AP are NaN for the affected case/scope/signal',
        'summary_weighting': 'equal case weights; finite values only',
        'std_ddof': 0,
        'case_names': [row['case'] for row in rows],
        'metrics': {},
    }
    if rows:
        for key, value in rows[0].items():
            if not isinstance(value, (int, float)):
                continue
            values = np.asarray([row[key] for row in rows], dtype=float)
            finite = values[np.isfinite(values)]
            summary['metrics'][key] = {
                'mean': float(finite.mean()) if finite.size else float('nan'),
                'std': float(finite.std(ddof=0)) if finite.size else float('nan'),
                'valid_cases': int(finite.size),
                'nonfinite_cases': int(values.size - finite.size),
            }
    return summary


def run_correction_oracle_audit(val_loader, model, config, outputs, logger):
    from utils import set_seed, test_single_volume

    cases_path = os.path.join(outputs, 'correction_oracle_cases.jsonl')
    summary_path = os.path.join(outputs, 'correction_oracle_summary.json')
    rows = []
    model.eval()
    set_seed(config.seed)
    with open(cases_path, 'w', encoding='utf-8') as record, torch.no_grad():
        for case_index, data in enumerate(val_loader):
            case_name = data['case_name'][0]
            start = time.perf_counter()
            row = test_single_volume(
                data['image'], data['label'], model,
                classes=config.num_classes,
                patch_size=[config.input_size_h, config.input_size_w],
                z_spacing=config.z_spacing, val_or_test=False, case=case_name,
                condition_mode='normal', audit_seed=config.seed + case_index * 100000,
                correction_oracle=True,
            )
            row['case_audit_seconds'] = time.perf_counter() - start
            record.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + '\n')
            record.flush()
            rows.append(row)
            message = f'ACDC correction oracle audit: case={case_name}, completed={len(rows)}'
            print(message)
            logger.info(message)
    summary = summarize_correction_oracle(rows)
    with open(summary_path, 'w', encoding='utf-8') as record:
        json.dump(summary, record, ensure_ascii=False, indent=2, allow_nan=True)
    print(f'ACDC correction oracle cases: {cases_path}')
    print(f'ACDC correction oracle summary: {summary_path}')
    logger.info('ACDC correction oracle output: %s; %s', cases_path, summary_path)
    return cases_path, summary_path
