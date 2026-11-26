"""Store inference weights and Lightning resume state in separate local files."""
import fcntl
import os
from pathlib import Path
from uuid import uuid4

from lightning.pytorch.plugins.io import TorchCheckpointIO


WEIGHT_KEYS = ('state_dict', 'reactdance_v2', 'epoch', 'global_step', 'pytorch-lightning_version')
PAIR_KEY = 'reactdance_checkpoint_id'


def resume_sidecar(path, checkpoint):
    if checkpoint.get('optimizer_states'):
        return None  # Legacy mixed checkpoint.
    sidecar = Path(path).resolve(strict=True).with_suffix('.state.pt')
    if checkpoint.get(PAIR_KEY):
        if not sidecar.is_file():
            raise ValueError('Resume needs the matching training state: %s. Use --init-checkpoint to start a new run.' % sidecar)
        return sidecar
    raise ValueError('Imported inference weights cannot resume optimizer state; use --init-checkpoint instead.')


class SplitCheckpointIO(TorchCheckpointIO):
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        path = Path(path)
        weights = {key: checkpoint[key] for key in WEIGHT_KEYS if key in checkpoint}
        if not checkpoint.get('optimizer_states'):
            super().save_checkpoint(weights, str(path), storage_options=storage_options)
            return
        state = {key: value for key, value in checkpoint.items() if key not in weights}
        state[PAIR_KEY] = weights[PAIR_KEY] = uuid4().hex
        sidecar = path.with_suffix('.state.pt')
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            if path.exists() or path.is_symlink():
                raise FileExistsError('Checkpoint already exists: %s' % path)
            # Publish weights last. An interrupted save can leave a sidecar but cannot expose an incomplete pair.
            super().save_checkpoint(state, str(sidecar), storage_options=storage_options)
            try:
                super().save_checkpoint(weights, str(path), storage_options=storage_options)
            except Exception:
                if not path.exists():
                    sidecar.unlink(missing_ok=True)
                raise
        finally:
            os.close(descriptor)

    def load_checkpoint(self, path, map_location='cpu'):
        weights = super().load_checkpoint(path, map_location=map_location)
        sidecar = resume_sidecar(path, weights)
        if sidecar is None:
            return weights
        state = super().load_checkpoint(str(sidecar), map_location=map_location)
        if state.get(PAIR_KEY) != weights[PAIR_KEY] or not state.get('optimizer_states'):
            raise ValueError('Training state does not match checkpoint: %s' % sidecar)
        return {**state, **weights}

    def remove_checkpoint(self, path):
        super().remove_checkpoint(path)
        super().remove_checkpoint(str(Path(path).with_suffix('.state.pt')))
