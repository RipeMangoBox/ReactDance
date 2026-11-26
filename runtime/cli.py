from datetime import timedelta
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml
import torch
import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy
from .paths import ROOT, Run, atomic_write, checkpoint_number, project_path, relative_path, sample_transaction
from .models import metadata, prefixed_state, read_checkpoint


PRESETS = {'HFSQ': 'configs/hfsq.yaml', 'ReactDance': 'configs/reactdance.yaml'}


def batch_limit(value):
    return float(value) if '.' in value else int(value)


def parse(model):
    parser = argparse.ArgumentParser(description=model + ' train / sample / evaluate / render')
    parser.add_argument('--mode', choices=['train', 'sample', 'evaluate', 'render'], default='train')
    parser.add_argument('--config', help='Training YAML for a new run (default: %s)' % PRESETS[model])
    parser.add_argument('--checkpoint')
    if model == 'ReactDance':
        parser.add_argument('--hfsq', help='HFSQ checkpoint for ReactDance training')
    parser.add_argument('--resume', help='Checkpoint to resume with its matching training state; keeps its version directory')
    parser.add_argument('--init-checkpoint', help='Initialize weights, start a new version and optimizer')
    parser.add_argument('--results-root')
    parser.add_argument('--output-dir', help='Explicit directory for sample, evaluate or render')
    parser.add_argument('--version', type=int)
    parser.add_argument('--data-root')
    parser.add_argument('--device', default=None, help='cpu, cuda, cuda:0; default from config')
    parser.add_argument('--devices', help='CUDA IDs, e.g. 0,1,2,3; on CPU an integer process count')
    parser.add_argument('--precision', choices=['auto', 'fp32', 'fp16', 'bf16'])
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--accumulate', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--limit-train-batches', type=batch_limit)
    parser.add_argument('--limit-val-batches', type=batch_limit)
    parser.add_argument('--checkpoint-every', type=int)
    parser.add_argument('--evaluate-every', type=int)
    parser.add_argument('--no-evaluation', action='store_true')
    parser.add_argument('--render', action='store_true', help='Render the full test set after sampling')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.resume and args.init_checkpoint:
        parser.error('--resume and --init-checkpoint are mutually exclusive')
    if args.version is not None and args.version < 0:
        parser.error('--version must be nonnegative')
    if args.config and args.resume:
        parser.error('--resume uses the saved config; use a new run to train with --config')
    for option in ('config', 'checkpoint', 'hfsq', 'resume', 'init_checkpoint', 'results_root', 'output_dir', 'data_root'):
        value = getattr(args, option, None)
        if value and Path(value).expanduser().is_absolute():
            parser.error('--%s must be relative to the repository root' % option.replace('_', '-'))
    return args


def effective_config(model, args):
    if args.resume:
        checkpoint = read_checkpoint(project_path(args.resume))
        config = copy.deepcopy(metadata(checkpoint, model)['config'])
        del checkpoint
    else:
        config = yaml.safe_load(project_path(args.config or PRESETS[model]).read_text(encoding='utf-8'))
    if args.results_root:
        config['results_root'] = args.results_root
    if args.data_root:
        config['data']['root'] = args.data_root
    if args.workers is not None:
        config['data']['num_workers'] = args.workers
    if args.batch_size is not None:
        config['data']['train']['batch_size'] = args.batch_size
        config['data']['val']['batch_size'] = min(config['data']['val']['batch_size'], args.batch_size)
    if args.epochs is not None:
        config['train']['epochs'] = args.epochs
    if args.accumulate is not None:
        if args.accumulate < 1:
            raise ValueError('--accumulate must be positive')
        config['train']['accumulate_grad_batches'] = args.accumulate
    if args.checkpoint_every is not None:
        config['train']['checkpoint_every'] = args.checkpoint_every
    if args.evaluate_every is not None:
        config['train']['evaluate_every'] = args.evaluate_every
    if args.no_evaluation:
        config['train']['evaluate_every'] = 0
    if getattr(args, 'hfsq', None):
        config['hfsq_checkpoint'] = args.hfsq
    if args.device == 'cpu':
        config['trainer'].update(accelerator='cpu', devices=1, precision=32)
    elif args.device and args.device.startswith('cuda'):
        config['trainer'].update(accelerator='gpu', devices=[int(args.device.split(':')[1]) if ':' in args.device else 0])
    if args.devices:
        config['trainer']['devices'] = int(args.devices) if config['trainer']['accelerator']=='cpu' else [int(v) for v in args.devices.split(',')]
    if args.precision and args.precision != 'auto':
        config['trainer']['precision'] = {'fp32': 32, 'fp16': 16, 'bf16': 'bf16'}[args.precision]
        if args.precision == 'fp16' and config['trainer']['accelerator'] == 'cpu':
            raise ValueError('FP16 requires CUDA.')
    if args.limit_train_batches is not None:
        config['trainer']['limit_train_batches'] = args.limit_train_batches
    if args.limit_val_batches is not None:
        config['trainer']['limit_val_batches'] = args.limit_val_batches
    every, evaluate = config['train']['checkpoint_every'], config['train']['evaluate_every']
    if args.mode == 'train' and (every < 1 or evaluate < 0 or evaluate % every):
        raise ValueError('evaluate_every must be zero or a multiple of checkpoint_every')
    return portable_config(config)


def portable_config(config):
    config['results_root'] = relative_path(config['results_root'])
    config['data']['root'] = relative_path(config['data']['root'])
    for key in ('hfsq_checkpoint', 'init_checkpoint'):
        if config.get(key):
            config[key] = relative_path(config[key])
    return config


def selected_checkpoint(config, model, args):
    if args.checkpoint:
        return project_path(args.checkpoint).resolve(strict=True)
    stage = project_path(config['results_root']) / model
    name = ('version_%d' % args.version) if args.version is not None else 'latest'
    return (stage / name / 'last.ckpt').resolve(strict=True)


def concrete_hfsq(config):
    path = project_path(config['hfsq_checkpoint']).resolve(strict=True)
    config['hfsq_checkpoint'] = relative_path(path)


def training_run(model, config, args):
    if args.resume:
        selected = project_path(args.resume).resolve(strict=True)
        run = Run.from_checkpoint(selected, model)
        last = run.directory / 'last.ckpt'
        if last.exists() and last.resolve() != selected:
            raise ValueError('--resume must use this version\'s last checkpoint; use --init-checkpoint to branch from an older checkpoint.')
        selected_number = checkpoint_number(selected)
        if selected_number is None:
            raise ValueError('Expected a numbered training checkpoint for --resume.')
        for path in run.directory.iterdir():
            if path.is_symlink() or path.suffix != '.ckpt' or path == selected:
                continue
            number = checkpoint_number(path)
            if number is not None and number >= selected_number:
                raise ValueError('Checkpoint collision on resume: %s. Inspect the version before resuming.' % path)
        return run
    inherited = os.environ.get('REACTDANCE_V2_RUN_DIR')
    if inherited:
        path = project_path(inherited)
        return Run(path.parents[1], model, int(path.name.split('_')[1]))
    external_world = int(os.environ.get('WORLD_SIZE', '1'))
    external_rank = int(os.environ.get('RANK', '0'))
    if external_world > 1 and args.version is None:
        raise ValueError('With external torchrun, pass --version N; python entrypoints allocate versions automatically.')
    if external_rank > 0:
        run = Run(project_path(config['results_root']).resolve(), model, args.version)
        for _ in range(3000):
            if (run.directory / 'config.yaml').is_file():
                return run
            time.sleep(.1)
        raise RuntimeError('Rank0 did not create the run directory')
    run = Run.allocate(project_path(config['results_root']), model, args.version)
    os.environ['REACTDANCE_V2_RUN_DIR'] = relative_path(run.directory)
    return run


def train(model, config, args):
    from .checkpoints import SplitCheckpointIO
    from .data import loader
    from .training import DanceModule, TensorBoard, PeriodicCheckpoints, new_module, resume_module
    from .inference import predict
    rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))
    if max(int(config['trainer'].get('num_nodes', 1)), int(os.environ.get('GROUP_WORLD_SIZE', 1)),
           int(os.environ.get('SLURM_NNODES', 1))) > 1:
        raise ValueError('This entry point supports single-node training.')
    run = None
    if model == 'ReactDance' and not args.resume and not args.init_checkpoint:
        if rank == 0:
            concrete_hfsq(config)
        else:
            run = training_run(model, config, args)
            config['hfsq_checkpoint'] = yaml.safe_load((run.directory / 'config.yaml').read_text())['hfsq_checkpoint']
    pl.seed_everything(config['seed'], workers=True)
    if args.resume:
        module = resume_module(model, project_path(args.resume), config)
    elif args.init_checkpoint:
        initial = read_checkpoint(project_path(args.init_checkpoint))
        meta = metadata(initial, model)
        config['init_checkpoint'] = relative_path(project_path(args.init_checkpoint).resolve())
        if model == 'ReactDance':
            config['hfsq_checkpoint'] = meta['config'].get('hfsq_checkpoint')
            portable_config(config)
        hstate = prefixed_state(initial['state_dict'], 'hfsq.') if model=='ReactDance' else None
        module = DanceModule(model, config, meta['normalizers'], meta.get('hfsq_config'), hstate)
        module.load_state_dict(initial['state_dict'], strict=True)
        del initial
    else:
        module = new_module(model, config)
    devices = config['trainer']['devices']
    device_count = len(devices) if isinstance(devices, list) else int(devices)
    world = int(os.environ.get('WORLD_SIZE', device_count))
    train_loader = loader(config, 'train', world=world, rank=rank)
    val_loader = loader(config, 'val')
    test_loader = None
    if config['train'].get('evaluate_every', 0):
        test_loader = loader(config, 'test')

    def checkpoint_evaluation(module, run, checkpoint, trainer):
        destination = run.evaluation(checkpoint)
        precision = {'16': 'fp16', 'bf16': 'bf16'}.get(str(trainer.precision), 'fp32')
        predict(module, config, test_loader, destination, checkpoint, precision,
                rank=trainer.global_rank, world=trainer.world_size, source='training')
        if trainer.is_global_zero:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            with (destination / 'evaluation.log').open('w') as log:
                subprocess.Popen([sys.executable, '-m', 'runtime.metrics', str(destination),
                    str(run.directory), str(trainer.current_epoch + 1)], cwd=str(ROOT),
                    env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    trainer_options = copy.deepcopy(config['trainer'])
    if trainer_options['accelerator'] == 'gpu' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Set an available device in the training config.')
    run = run or training_run(model, config, args)
    if rank == 0:
        atomic_write(run.directory / 'config.yaml', yaml.safe_dump(config, sort_keys=False))
    logger = TensorBoard(str(run.stage), name='', version=run.directory.name, default_hp_metric=False)
    callback = PeriodicCheckpoints(run, config['train']['checkpoint_every'],
        config['train'].get('evaluate_every', 0), checkpoint_evaluation if test_loader is not None else None)
    trainer = pl.Trainer(default_root_dir=str(run.directory), logger=logger,
        callbacks=[callback], plugins=[SplitCheckpointIO()], enable_checkpointing=True, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=0,
        strategy=DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True,
            broadcast_buffers=False, timeout=timedelta(seconds=3600 if model == 'ReactDance' else 1800)) if world > 1 else None,
        replace_sampler_ddp=False,
        max_epochs=config['train']['epochs'], accumulate_grad_batches=config['train']['accumulate_grad_batches'],
        gradient_clip_val=config['train']['gradient_clip'], **trainer_options)
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=str(project_path(args.resume)) if args.resume else None)
    if trainer.is_global_zero:
        print('Completed: %s (global_step=%d)' % (run.directory, trainer.global_step), flush=True)


def inference(model, config, args):
    from .data import loader
    from .inference import load_engine, predict
    from .metrics import evaluate
    from .render import render_evaluation
    checkpoint = selected_checkpoint(config, model, args)
    if args.output_dir:
        destination = project_path(args.output_dir)
    else:
        destination = Run.from_checkpoint(checkpoint, model).evaluation(checkpoint)
    if args.mode in ('evaluate', 'render'):
        if args.mode == 'evaluate':
            print(json.dumps(evaluate(destination, data_root=args.data_root), ensure_ascii=False, indent=2))
        else:
            render_evaluation(destination, data_root=args.data_root, workers=args.workers)
        return
    device = torch.device(args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    if device.type=='cuda':
        torch.cuda.set_device(device)
    precision = args.precision or 'auto'
    if precision=='auto':
        precision = 'bf16' if model == 'ReactDance' and device.type=='cuda' and torch.cuda.is_bf16_supported() else 'fp32'
    if precision=='fp16' and device.type != 'cuda':
        raise ValueError('FP16 requires CUDA.')
    if precision=='bf16' and (device.type!='cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('BF16 requires a supported CUDA GPU; use --precision fp32.')
    engine, meta = load_engine(checkpoint, model, device)
    saved = copy.deepcopy(meta['config'])
    if args.data_root:
        saved['data']['root'] = args.data_root
    if args.workers is not None:
        saved['data']['num_workers'] = args.workers
    portable_config(saved)
    cases = loader(saved, 'test')
    with sample_transaction(destination, args.overwrite) as stage:
        predict(engine, saved, cases, stage, checkpoint, precision)
        result = evaluate(stage)
        if args.render:
            render_evaluation(stage, workers=args.workers)
    print('Evaluation: %s (%d samples)' % (destination, result['samples']), flush=True)


def main(model):
    args = parse(model)
    torch.set_num_threads(args.threads)
    from models.nets.config import set_fused_attn
    set_fused_attn(model == 'ReactDance' and args.mode == 'train')
    torch.set_float32_matmul_precision('medium' if args.mode == 'train' else 'high')
    try:
        config = effective_config(model, args)
        if args.mode == 'train':
            train(model, config, args)
        else:
            inference(model, config, args)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
