"""Official handcrafted K/G/CD features, indexed by samples.json rather than glob."""
import json
from pathlib import Path
import numpy as np
from scipy import linalg
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
from utils.features.kinetic import extract_kinetic_features
from utils.features.manual_new import extract_manual_features
from .data import test_references
from .paths import project_path, write_json

POINTS = [0, 7, 8, 10, 11, 15, 16, 17, 20, 21]


def cd_feature(follower, leader):
    a, b = follower[:, POINTS], leader[:, POINTS]
    return np.linalg.norm(a[:, None] - b[:, :, None], axis=-1).reshape(len(a), -1).mean(axis=0)


def distribution(predictions, references, kind='k'):
    if len(predictions) < 2:
        return {'fid': None, 'diversity': None}
    pred, ref = np.stack(predictions), np.stack(references)
    # Historical solo and duet evaluators use different normalization rules.
    base = ref if kind == 'cd' else pred
    pred = (pred - base.mean(axis=0)) / (base.std(axis=0) + 1e-10)
    ref = (ref - ref.mean(axis=0)) / (ref.std(axis=0) + 1e-10)
    cp, cr = np.atleast_2d(np.cov(pred, rowvar=False)), np.atleast_2d(np.cov(ref, rowvar=False))
    offset = np.eye(cp.shape[0]) * 1e-5
    if kind != 'cd':
        cp, cr = cp + offset, cr + offset
    product = cp.dot(cr)
    covmean = linalg.sqrtm(product) if np.any(product) else np.zeros_like(product)
    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm((cp + offset).dot(cr + offset))
    if np.iscomplexobj(covmean) and not np.allclose(np.diag(covmean).imag, 0, atol=1e-3):
        covmean = covmean.real
    delta = pred.mean(axis=0) - ref.mean(axis=0)
    fid = delta.dot(delta) + np.trace(cp) + np.trace(cr) - 2 * np.trace(covmean)
    # The historical checkpoint evaluator also checked the final scalar in _real().
    if abs(np.imag(fid)) >= 1e-3:
        raise ValueError('Non-real FID: %s' % fid)
    distances = [np.linalg.norm(pred[i] - pred[j]) for i in range(len(pred)) for j in range(i + 1, len(pred))]
    return {'fid': float(np.real(fid)), 'diversity': float(sum(distances) / len(distances))}


def beats(motion):
    speed = np.linalg.norm(np.diff(motion, axis=0), axis=-1).mean(axis=-1)
    return argrelextrema(gaussian_filter1d(speed, 5), np.less)[0]


def beat_alignment(reference, predicted):
    if not len(reference) or not len(predicted):
        return None
    return float(np.exp(-((np.asarray(reference)[:, None] - predicted[None, :]) ** 2).min(axis=1) / 18).mean())


def evaluate(destination, data_root=None):
    destination = Path(destination)
    manifest = json.loads((destination / 'samples.json').read_text())
    config = manifest['config']
    config['data']['root'] = data_root or manifest['data_root']
    source, by_name = test_references(config, manifest['samples'])
    pred_features = {key: [] for key in ('k', 'g', 'cd')}
    gt_features = {key: [] for key in pred_features}
    rows = []
    root = project_path(config['data']['root']) / 'motion/pos3d' / config['data']['test']['split']
    for record in manifest['samples']:
        name = record['sample']
        reference = source[by_name[name]]
        pred = np.load(destination / 'pos3d_npy' / (name + '_00.npy')).reshape(-1, 22, 3)
        leader = np.load(destination / 'pos3d_npy' / (name + '_01.npy')).reshape(-1, 22, 3)
        target = reference['pos3df'].reshape(-1, 22, 3)
        length = min(len(pred), len(target), len(leader))
        pred, target, leader = pred[:length], target[:length], leader[:length]
        centered_pred, centered_target = pred - pred[:, :1], target - target[:, :1]
        row = {'sample': name, 'frames': length,
            'mpjpe_mm': float(np.linalg.norm(centered_pred - centered_target, axis=-1).mean() * 1000),
            'mpjve_mm_per_frame': float(np.linalg.norm(np.diff(centered_pred, axis=0) - np.diff(centered_target, axis=0), axis=-1).mean() * 1000),
            'root_error_cm': float(np.linalg.norm(pred[:, 0] - target[:, 0], axis=-1).mean() * 100),
            'bas': beat_alignment(np.flatnonzero(reference['music'][:length-1, 53].astype(bool)), beats(pred)),
            'bed': beat_alignment(beats(leader), beats(pred))}
        rows.append(row)
        if len(manifest['samples']) < 2:
            continue
        relative_pred = pred - pred[0, :1]
        # Distribution GT uses the full raw 22-joint sequence, including for prefix predictions.
        raw_target = np.load(root / (name + '_00.npy'))[:, :66].reshape(-1, 22, 3)
        relative_target = raw_target - raw_target[0, :1]
        functions = {'k': extract_kinetic_features, 'g': extract_manual_features}
        caches = {'k': 'kinetic_features', 'g': 'manual_features_new', 'cd': 'duet_features'}
        for key, function in functions.items():
            pred_features[key].append(function(relative_pred))
            path = root / caches[key] / (name + '_00.npy')
            gt_features[key].append(np.load(path) if path.is_file() else function(relative_target))
        pred_features['cd'].append(cd_feature(pred, leader))
        path = root / caches['cd'] / (name + '_00.npy')
        if path.is_file():
            ref_cd = np.load(path)
        else:
            raw_leader = np.load(root / (name + '_01.npy'))[:, :66].reshape(-1, 22, 3)
            common = min(len(raw_target), len(raw_leader))
            origin = raw_leader[0, :1]
            ref_cd = cd_feature(raw_target[:common] - origin, raw_leader[:common] - origin)
        gt_features['cd'].append(ref_cd)
    result = {'samples': len(rows), 'frames': sum(r['frames'] for r in rows),
              'protocol': 'K/G standardized separately, CD standardized against GT, full raw GT features; pose/beat: common prefix; FID needs >=2 samples',
              'sample_metrics': rows}
    for key in ('mpjpe_mm', 'mpjve_mm_per_frame', 'root_error_cm', 'bas', 'bed'):
        values = [r[key] for r in rows if r[key] is not None]
        result[key] = float(np.mean(values)) if values else None
    result['official_jitter_mm_per_frame'] = result['mpjve_mm_per_frame']
    result['aits_seconds'] = float(np.mean([record['seconds'] for record in manifest['samples']]))
    result['aits_scope'] = 'condition preparation + model sampling + decode + CPU output; excludes weight load, data load, file writes and metrics'
    for key in pred_features:
        result[key] = distribution(pred_features[key], gt_features[key], key)
    write_json(destination / 'metrics.json', result)
    return result


if __name__ == '__main__':
    import sys
    from torch.utils.tensorboard import SummaryWriter
    folder, log_dir, epoch = sys.argv[1:]
    result = evaluate(folder)
    with SummaryWriter(log_dir) as writer:
        for key in ('mpjpe_mm', 'mpjve_mm_per_frame', 'root_error_cm', 'bas', 'bed'):
            if result.get(key) is not None:
                writer.add_scalar('test/' + key, result[key], int(epoch))
        for key in ('k', 'g', 'cd'):
            for metric, value in result[key].items():
                if value is not None:
                    writer.add_scalar('test/' + metric + '_' + key, value, int(epoch))
