"""Per-case correction utility conditioned on normal branch disagreement."""

import json
import os
import time

import numpy as np
import torch

from correction_oracle_audit import CorrectionOracleCase, _auc_ap, _gt_boundary_band, _ratio


def _volume_metrics(prediction, gt):
    from utils import calculate_metric_percase

    return np.asarray([calculate_metric_percase(prediction == cls, gt == cls)
                       for cls in (1, 2, 3)], dtype=float)


class DisagreementUtilityCase(CorrectionOracleCase):
    """Reuse probability extraction; evaluate only disagreement-conditioned targets."""

    def finalize(self):
        gt = self.label
        spm_pred = self.predictions['spm']
        diff_pred = self.predictions['diffusion']
        spm_correct = spm_pred == gt
        diff_correct = diff_pred == gt
        disagreement = spm_pred != diff_pred
        benefit = ~spm_correct & diff_correct
        harm = spm_correct & ~diff_correct
        neither = disagreement & ~spm_correct & ~diff_correct
        if not np.array_equal(disagreement, benefit | harm | neither):
            raise RuntimeError(f'Disagreement partition mismatch in case {self.case}')

        # Reconstruct both definitions independently; do not run the old all-pixel AUC audit.
        oracle_pred = spm_pred.copy()
        oracle_pred[benefit] = diff_pred[benefit]
        disagreement_oracle_pred = spm_pred.copy()
        switch = disagreement & benefit
        disagreement_oracle_pred[switch] = diff_pred[switch]
        if not np.array_equal(oracle_pred, disagreement_oracle_pred):
            raise RuntimeError(f'Oracle prediction mismatch in case {self.case}')

        row = {'case': self.case, 'condition_mode': 'normal'}
        for name, prediction in (('spm', spm_pred), ('diffusion', diff_pred),
                                 ('oracle', oracle_pred),
                                 ('disagreement_oracle', disagreement_oracle_pred)):
            metrics = _volume_metrics(prediction, gt)
            for index, cls_name in enumerate(('rv', 'myo', 'lv')):
                row[f'{name}_{cls_name}_dice'] = float(metrics[index, 0])
                row[f'{name}_{cls_name}_hd95'] = float(metrics[index, 1])
            row[f'{name}_mean_dice'] = float(metrics[:, 0].mean())
            row[f'{name}_mean_hd95'] = float(metrics[:, 1].mean())
        if not np.isclose(row['oracle_mean_dice'], row['disagreement_oracle_mean_dice'],
                          rtol=1e-7, atol=1e-9):
            raise RuntimeError(f'Oracle mean Dice mismatch in case {self.case}')
        row['oracle_consistency_passed'] = True
        row['oracle_consistency_abs_dice_difference'] = abs(
            row['oracle_mean_dice'] - row['disagreement_oracle_mean_dice'])
        row['oracle_gain_vs_spm'] = row['oracle_mean_dice'] - row['spm_mean_dice']
        row['oracle_gain_vs_diff'] = row['oracle_mean_dice'] - row['diffusion_mean_dice']

        foreground = gt != 0
        boundary = _gt_boundary_band(gt, radius=3)
        scopes = {
            'all': np.ones(gt.shape, dtype=bool),
            'foreground': foreground,
            'boundary': boundary,
            'foreground_boundary': foreground & boundary,
            'rv': gt == 1, 'myo': gt == 2, 'lv': gt == 3,
        }
        winner_mask = benefit | harm
        signal_specs = [(name, name, False) for name in self.signals]
        signal_specs += [('low_spm_confidence', 'spm_confidence', True),
                         ('low_diff_confidence', 'diff_confidence', True)]
        for scope, region in scopes.items():
            selected_disagreement = region & disagreement
            scope_count = int(np.count_nonzero(region))
            disagreement_count = int(np.count_nonzero(selected_disagreement))
            counts = {name: int(np.count_nonzero(mask & selected_disagreement))
                      for name, mask in (('benefit', benefit), ('harm', harm), ('neither', neither))}
            if disagreement_count != sum(counts.values()):
                raise RuntimeError(f'Disagreement count mismatch: case={self.case}, scope={scope}')
            prefix = '' if scope == 'all' else f'{scope}_'
            row[f'{scope}_pixel_count'] = scope_count
            row[f'{prefix}disagreement_pixel_count'] = disagreement_count
            row[f'{prefix}disagreement_ratio'] = _ratio(disagreement_count, scope_count)
            for category, count in counts.items():
                row[f'{prefix}disagreement_{category}_count'] = count
                row[f'{prefix}disagreement_{category}_ratio'] = _ratio(count, disagreement_count)

            for task in ('taskA', 'taskB'):
                selected = (selected_disagreement if task == 'taskA'
                            else selected_disagreement & winner_mask)
                target = benefit[selected]
                positives = int(np.count_nonzero(target))
                row[f'{task}_{scope}_pixel_count'] = int(target.size)
                row[f'{task}_{scope}_positive_count'] = positives
                row[f'{task}_{scope}_negative_count'] = int(target.size - positives)
                row[f'{task}_{scope}_positive_ratio'] = _ratio(positives, target.size)
                for signal, source, invert in signal_specs:
                    scores = self.signals[source][selected]
                    if invert:
                        scores = 1.0 - scores
                    auc, ap = _auc_ap(target, scores)
                    key = f'{task}_{scope}_{signal}'
                    row[f'{key}_auc'] = auc
                    row[f'{key}_ap'] = ap
                    row[f'{key}_direction_free_auc'] = (
                        float(max(auc, 1.0 - auc)) if np.isfinite(auc) else float('nan'))
                    row[f'{key}_nonfinite_count'] = int(np.count_nonzero(~np.isfinite(scores)))
        return row


def summarize_disagreement_utility(rows):
    summary = {
        'case_count': len(rows),
        'case_names': [row['case'] for row in rows],
        'condition_mode': 'normal',
        'spm_scale': 1.0,
        'task_definitions': {
            'taskA': 'scope AND disagreement; positive=benefit; negative=harm OR neither',
            'taskB': 'scope AND disagreement AND (benefit OR harm); positive=benefit; negative=harm',
        },
        'scope_definitions': {
            'all': 'all pixels before conditioning on disagreement',
            'foreground': 'GT != 0',
            'boundary': 'union of per-class GT boundary bands',
            'foreground_boundary': '(GT != 0) AND boundary',
            'rv': 'GT == 1', 'myo': 'GT == 2', 'lv': 'GT == 3',
        },
        'disagreement_ratio_denominator': 'all pixels in the named scope',
        'composition_ratio_denominator': 'disagreement pixels in the named scope',
        'boundary_radius_pixels': 3,
        'boundary_definition': 'reuse _gt_boundary_band: per-class 2D dilation XOR erosion, Euclidean disk, border_value=0',
        'boundary_grid': 'original GT pixels; no morphology along the slice axis',
        'signal_definitions': {
            'low_spm_confidence': '1 - spm_confidence',
            'low_diff_confidence': '1 - diff_confidence',
            'confidence_diff': 'diff_confidence - spm_confidence',
            'entropy_diff': 'spm_entropy - diff_entropy',
            'probability_l1_disagreement': 'mean(abs(spm_prob - diff_prob), class_dim)',
        },
        'signal_aliases': {'confidence_advantage': 'confidence_diff',
                           'entropy_advantage': 'entropy_diff'},
        'direction_free_auc': 'per-case max(auc, 1 - auc); information diagnostic only',
        'pr_metric': 'average_precision_score',
        'undefined_roc_auc': 'NaN for single target class or empty scope',
        'ap_single_class': '0 if no positives, 1 if all positive; NaN for empty scope',
        'nonfinite_signal_policy': 'AUC/AP/direction-free AUC are NaN for the affected task/scope/signal',
        'undefined_ratios': 'NaN for empty denominator',
        'summary_weighting': 'equal case weights; finite values only; no pooled pixel AUC',
        'std_ddof': 0,
        'oracle_consistency_rtol': 1e-7,
        'oracle_consistency_atol': 1e-9,
        'oracle_consistency_all_passed': bool(rows) and all(row['oracle_consistency_passed'] for row in rows),
        'metrics': {},
        'winner_identification_summary': {},
    }
    if rows:
        for key, value in rows[0].items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            values = np.asarray([row[key] for row in rows], dtype=float)
            finite = values[np.isfinite(values)]
            summary['metrics'][key] = {
                'mean': float(finite.mean()) if finite.size else float('nan'),
                'std': float(finite.std(ddof=0)) if finite.size else float('nan'),
                'valid_cases': int(finite.size),
                'nonfinite_cases': int(values.size - finite.size),
            }
    for scope in ('foreground', 'foreground_boundary'):
        focused = {}
        for signal in ('confidence_diff', 'entropy_diff', 'probability_l1_disagreement'):
            for metric in ('auc', 'ap'):
                key = f'{signal}_{metric}'
                focused[key] = summary['metrics'].get(f'taskB_{scope}_{key}', {
                    'mean': float('nan'), 'std': float('nan'),
                    'valid_cases': 0, 'nonfinite_cases': len(rows),
                })
        summary['winner_identification_summary'][scope] = focused
    return summary


def run_disagreement_utility_audit(val_loader, model, config, outputs, logger):
    from utils import set_seed, test_single_volume

    cases_path = os.path.join(outputs, 'disagreement_utility_cases.jsonl')
    summary_path = os.path.join(outputs, 'disagreement_utility_summary.json')
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
                disagreement_utility=True,
            )
            row['case_audit_seconds'] = time.perf_counter() - start
            record.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + '\n')
            record.flush()
            rows.append(row)
            message = f'ACDC disagreement utility audit: case={case_name}, completed={len(rows)}'
            print(message)
            logger.info(message)
    summary = summarize_disagreement_utility(rows)
    with open(summary_path, 'w', encoding='utf-8') as record:
        json.dump(summary, record, ensure_ascii=False, indent=2, allow_nan=True)
    print(f'ACDC disagreement utility cases: {cases_path}')
    print(f'ACDC disagreement utility summary: {summary_path}')
    logger.info('ACDC disagreement utility output: %s; %s', cases_path, summary_path)
    return cases_path, summary_path
