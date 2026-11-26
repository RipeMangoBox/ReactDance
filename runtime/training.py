import copy
import gc
from pathlib import Path
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from .checkpoints import resume_sidecar
from .models import components, metadata, prefixed_state, read_checkpoint
from .normalization import dataset_normalizers
from .paths import CHECKPOINT_FILENAME, project_path, relative_path


class TensorBoard(TensorBoardLogger):
    """One event stream in version_x; config.yaml is the sole human config snapshot."""
    def save(self):
        self.experiment.flush()


class DanceModule(pl.LightningModule):
    def __init__(self, model_name, config, normalizers, hfsq_config=None, hfsq_state=None):
        super().__init__()
        self.model_name = model_name
        self.run_config = copy.deepcopy(config)
        self.hfsq_config = copy.deepcopy(hfsq_config)
        self.model, self.hfsq, self.normalizer = components(
            model_name, config, normalizers, hfsq_config, hfsq_state)
        self.last_train_loss = None
        self.completed_epochs = 0

    def train(self, mode=True):
        super().train(mode)
        if self.hfsq is not None:
            self.hfsq.eval()
        return self

    def forward(self, batch):
        if self.model_name == 'HFSQ':
            _, losses, _ = self.model(batch)
            return losses['total']
        with torch.no_grad():
            leader, _ = self.hfsq.rootl_full_forward(batch['pos3dl'])
            target_latent = self.hfsq.encode(batch, encode_mode=self.model.prior_mode)
            teacher = self.hfsq.decode(target_latent, decode_mode=self.model.prior_mode)
        target = {'pos3df': teacher['pos3df'], 'translf': teacher['translf'], 'pos3dl': batch['pos3dl']}
        _, losses, _ = self.model(target_latent, (batch['music'], leader), motion_target=target)
        return losses['total']

    def check_loss(self, loss, stage):
        if torch.isfinite(loss):
            return
        if self.model_name == 'HFSQ' and str(self.trainer.precision) == '16':
            print('Nonfinite HFSQ FP16 %s loss on rank %d' % (stage, self.global_rank), flush=True)
            return
        raise RuntimeError('Nonfinite %s loss' % stage)

    def training_step(self, batch, batch_idx):
        loss = self(batch)
        self.check_loss(loss, 'training')
        self.last_train_loss = float(loss.detach())
        self.log('loss/train', loss.detach(), on_step=True, on_epoch=True,
                 batch_size=len(batch['music']), sync_dist=True)
        optimizer = self.optimizers(use_pl_optimizer=False)
        self.log('lr', optimizer.param_groups[0]['lr'], on_step=True, sync_dist=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        self.check_loss(loss, 'validation')
        self.log('loss/val', loss, on_step=False, on_epoch=True,
                 batch_size=len(batch['music']), sync_dist=True)
        return loss

    def configure_optimizers(self):
        options = copy.deepcopy(self.run_config['train']['optimizer'])
        name = options.pop('name')
        if name == 'Lion':
            from models.model_utils.optimizer import Lion
            factory = Lion
        elif name == 'AdamW':
            options.setdefault('eps', 1e-6)
            factory = torch.optim.AdamW
        else:
            raise ValueError('Supported optimizers: Lion, AdamW')
        optimizer = factory((p for p in self.model.parameters() if p.requires_grad), **options)
        schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.run_config['train']['lr_decay'])
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': schedule, 'interval': 'epoch'}}

    def on_save_checkpoint(self, checkpoint):
        checkpoint['reactdance_v2'] = {
            'format': 2, 'model': self.model_name, 'config': copy.deepcopy(self.run_config),
            'hfsq_config': copy.deepcopy(self.hfsq_config),
            'normalizers': self.normalizer.export(), 'completed_epochs': self.current_epoch + 1,
            'latent_normalization': 'identity, identical to upstream effective normalization',
            'last_train_loss': self.last_train_loss,
            'world_size': self.trainer.world_size,
            'effective_batch_size': self.run_config['data']['train']['batch_size'] * self.trainer.world_size * self.run_config['train']['accumulate_grad_batches'],
        }

    def on_train_start(self):
        self.print('world_size=%d, batch_per_rank=%d, accumulation=%d, effective_batch=%d' % (
            self.trainer.world_size, self.run_config['data']['train']['batch_size'],
            self.run_config['train']['accumulate_grad_batches'],
            self.run_config['data']['train']['batch_size'] * self.trainer.world_size * self.run_config['train']['accumulate_grad_batches']))


class PeriodicCheckpoints(ModelCheckpoint):
    """Save numbered training snapshots, update links and run scheduled evaluations."""
    def __init__(self, run, every, evaluate_every=0, evaluation=None):
        super().__init__(dirpath=str(run.directory), filename=CHECKPOINT_FILENAME,
            auto_insert_metric_name=False, monitor=None, save_top_k=-1,
            every_n_epochs=every, save_last=False, save_on_train_epoch_end=True)
        self.run = run
        self.evaluate_every = evaluate_every
        self.evaluation = evaluation

    def format_checkpoint_name(self, metrics, filename=None, ver=None):
        metrics = dict(metrics)
        if 'epoch' in metrics:
            metrics['epoch'] = metrics['epoch'] + 1
        return super().format_checkpoint_name(metrics, filename=filename, ver=ver)

    @staticmethod
    def checkpoint_paths(state, convert):
        state = dict(state)
        for key in ('dirpath', 'best_model_path', 'kth_best_model_path', 'last_model_path'):
            if state.get(key):
                state[key] = convert(state[key])
        state['best_k_models'] = {convert(path): score for path, score in state.get('best_k_models', {}).items()}
        return state

    def state_dict(self):
        return self.checkpoint_paths(super().state_dict(), relative_path)

    def load_state_dict(self, state_dict):
        super().load_state_dict(self.checkpoint_paths(state_dict, lambda path: str(project_path(path))))

    def on_train_epoch_end(self, trainer, module):
        previous_step = self._last_global_step_saved
        super().on_train_epoch_end(trainer, module)
        if self._last_global_step_saved == previous_step:
            return
        completed = trainer.current_epoch + 1
        checkpoint = Path(self.best_model_path)
        if trainer.is_global_zero:
            self.run.update_links(checkpoint)
            print('Checkpoint: %s' % checkpoint, flush=True)
        trainer.strategy.barrier('checkpoint_saved')
        if self.evaluation and self.evaluate_every and completed % self.evaluate_every == 0:
            # Preserve training RNG, model mode, optimizer and distributed process state.
            state = module.training
            devices = [module.device.index] if module.device.type == 'cuda' else []
            import numpy as np
            import random
            numpy_state, python_state = np.random.get_state(), random.getstate()
            try:
                with torch.random.fork_rng(devices=devices):
                    self.evaluation(module, self.run, checkpoint, trainer)
            except Exception as exc:
                # Rank-local prediction errors have already been collected in predict().
                print('Periodic evaluation failed at epoch %d: %r' % (completed, exc), flush=True)
            finally:
                np.random.set_state(numpy_state)
                random.setstate(python_state)
                module.train(state)
            trainer.strategy.barrier('checkpoint_evaluation')


def new_module(model_name, config, hfsq_path=None):
    if model_name == 'HFSQ':
        norms = dataset_normalizers(project_path(config['data']['root']), config['model'])
        return DanceModule(model_name, config, norms)
    path = hfsq_path or project_path(config['hfsq_checkpoint'])
    checkpoint = read_checkpoint(path)
    meta = metadata(checkpoint, 'HFSQ')
    state = prefixed_state(checkpoint['state_dict'], 'model.')
    result = DanceModule(model_name, config, meta['normalizers'], meta['config']['model'], state)
    del checkpoint, state
    gc.collect()
    return result


def resume_module(model_name, checkpoint_path, config):
    checkpoint = read_checkpoint(checkpoint_path)
    meta = metadata(checkpoint, model_name)
    resume_sidecar(checkpoint_path, checkpoint)
    hfsq_state = prefixed_state(checkpoint['state_dict'], 'hfsq.') if model_name == 'ReactDance' else None
    result = DanceModule(model_name, config, meta['normalizers'], meta.get('hfsq_config'), hfsq_state)
    # Trainer restores full state, optimizer and scheduler after DDP setup.
    return result
