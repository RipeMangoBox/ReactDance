import contextlib
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
import torch.distributed as dist
from .models import load_components, read_checkpoint
from .paths import relative_path, write_json


def load_engine(path, model, device):
    checkpoint = read_checkpoint(path)
    primary, hfsq, normalizer, meta = load_components(checkpoint, model)
    primary.to(device).eval()
    normalizer.to(device)
    if hfsq is not None:
        hfsq.to(device).eval()
    return SimpleNamespace(model=primary, hfsq=hfsq, normalizer=normalizer,
                           model_name=model, device=device), meta


def synthesize(decoded, leader):
    b, t, _ = decoded['pos3df'].shape
    body = decoded['pos3df'].clone()
    body[:, :, :3] = 0
    follower = (body.reshape(b, t, 22, 3) +
                (leader[:, :, :3] + decoded['translf']).unsqueeze(-2)).reshape(b, t, 66)
    return follower


def predict(engine, config, loader, destination, checkpoint, precision='fp32', rank=0, world=1,
            source='standalone'):
    destination = Path(destination)
    error = None
    records = []
    try:
        records = _predict_rank(engine, config, loader, destination, precision, rank, world)
    except Exception as exc:
        if world == 1:
            raise
        error = 'rank %d: %r' % (rank, exc)
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, (error, records))
        failures = [message for message, _ in gathered if message]
        if failures:
            raise RuntimeError('Prediction failed: ' + ', '.join(failures))
        records = [entry for _, group in gathered for entry in group]
    records.sort(key=lambda r: r['sample_index'])
    if rank == 0:
        attention = None
        if engine.model_name == 'ReactDance':
            fused = {layer.fused_attn for layer in engine.model.modules() if hasattr(layer, 'fused_attn')}
            attention = {frozenset({True}): 'sdpa', frozenset({False}): 'math'}.get(frozenset(fused), 'mixed')
        write_json(destination / 'samples.json', {
            'model': engine.model_name, 'checkpoint': relative_path(checkpoint),
            'data_root': relative_path(config['data']['root']), 'config': config,
            'source': source, 'attention': attention, 'precision': precision, 'guidance': 1, 'inference_phase': 0,
            'fps': 30, 'world_size': world, 'samples': records})
    return records


def _predict_rank(engine, config, loader, destination, precision, rank, world):
    poses = destination / 'pos3d_npy'
    poses.mkdir(parents=True, exist_ok=True)
    records = []
    device = engine.device
    engine.model.eval()
    if engine.hfsq is not None:
        engine.hfsq.eval()
    for number, raw in enumerate(loader):
        if number % world != rank:
            continue
        name = raw['fname'][0]
        seed = int(config['seed'] + raw['sample_index'][0])
        torch.random.default_generator.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.synchronize(device)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in raw.items()}
        start = time.perf_counter()
        amp_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16}.get(precision)
        amp = torch.autocast('cuda', dtype=amp_dtype) if amp_dtype else contextlib.nullcontext()
        with torch.no_grad(), amp:
            if engine.model_name == 'HFSQ':
                latent = engine.model.encode(batch, encode_mode='all_quantizeds')
                decoded = engine.model.decode(latent, decode_mode='all_quantizeds')
            else:
                leader, _ = engine.hfsq.rootl_full_forward(batch['pos3dl'])
                latent = engine.model.sample(use_EMA=False, cond_tuple=(batch['music'], leader))
                decoded = engine.hfsq.decode(latent, decode_mode=engine.model.prior_mode)
            follower = synthesize(decoded, batch['pos3dl'])[0].float().cpu().numpy()
        if not np.isfinite(follower).all():
            raise RuntimeError('Nonfinite prediction: ' + name)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
        np.save(poses / (name + '_00.npy'), follower)
        np.save(poses / (name + '_01.npy'), batch['pos3dl'][0].float().cpu().numpy())
        records.append({'sample': name, 'frames': len(follower), 'seed': seed,
                        'sample_index': int(raw['sample_index'][0]), 'seconds': seconds, 'rank': rank})
    return records
