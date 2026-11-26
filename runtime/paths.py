"""Deterministic run/checkpoint/evaluation paths. No checkpoint globbing."""
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile
import fcntl

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_FILENAME = 'checkpoint_{epoch:04d}'


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def relative_path(path):
    return os.path.relpath(project_path(path), ROOT)


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(text)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def symlink_replace(path, relative_target):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.link.' + str(os.getpid()))
    try:
        temporary.symlink_to(relative_target)
        os.replace(temporary, path)
    finally:
        if temporary.is_symlink():
            temporary.unlink()


@contextmanager
def sample_transaction(destination, overwrite):
    """Stage the whole sample run; publish its manifest after all referenced files."""
    destination.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        manifest = destination / 'samples.json'
        previous = json.loads(manifest.read_text()) if manifest.exists() else {}
        if previous.get('source') == 'training' or (destination / 'evaluation.log').exists():
            raise FileExistsError('Training evaluation: select a separate --output-dir.')
        if previous and not overwrite:
            raise FileExistsError('%s exists; use --overwrite to regenerate.' % destination)
        old_names = {record['sample'] for record in previous.get('samples', ())}
        if any(not name or Path(name).name != name for name in old_names):
            raise ValueError('Invalid sample name in existing manifest')
        with tempfile.TemporaryDirectory(prefix='.sampling-', dir=str(destination)) as temp:
            stage = Path(temp)
            yield stage
            new_manifest = stage / 'samples.json'
            new_names = {record['sample'] for record in json.loads(new_manifest.read_text())['samples']}
            if manifest.exists():
                os.replace(manifest, stage / 'previous.json')
            for folder, suffixes in [('pos3d_npy', ('_00.npy', '_01.npy')), ('videos', ('.mp4',))]:
                target = destination / folder
                if (stage / folder).exists():
                    target.mkdir(exist_ok=True)
                for name in old_names | new_names:
                    for suffix in suffixes:
                        filename = name + suffix
                        output = stage / folder / filename
                        if output.exists():
                            os.replace(output, target / filename)
                        elif name in old_names:
                            (target / filename).unlink(missing_ok=True)
            os.replace(stage / 'metrics.json', destination / 'metrics.json')
            os.replace(new_manifest, manifest)
    finally:
        os.close(descriptor)


def checkpoint_number(path):
    # Read the numbering used by both current and earlier training runs.
    match = re.fullmatch(r'(?:checkpoint|epoch)_(\d+)(?:-.*)?', Path(path).stem)
    return int(match[1]) if match else None


@dataclass(frozen=True)
class Run:
    root: Path
    model: str
    version: int

    @property
    def stage(self):
        return self.root / self.model

    @property
    def directory(self):
        return self.stage / ('version_%d' % self.version)

    def checkpoint(self, completed_epochs):
        return self.directory / (CHECKPOINT_FILENAME.format(epoch=completed_epochs) + '.ckpt')

    def evaluation(self, checkpoint):
        checkpoint = Path(checkpoint).resolve(strict=True)
        if checkpoint.parent != self.directory.resolve():
            raise ValueError('Checkpoint does not belong to %s' % self.directory)
        if checkpoint.suffix != '.ckpt' or not checkpoint.is_file():
            raise ValueError('Expected a .ckpt file: %s' % checkpoint)
        return self.stage / 'evaluation' / self.directory.name / checkpoint.stem

    def update_links(self, checkpoint):
        symlink_replace(self.directory / 'last.ckpt', Path(checkpoint).name)
        # Lock the directory inode itself: concurrent runs need no extra lock file.
        descriptor = os.open(self.stage, os.O_RDONLY)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            latest = self.stage / 'latest'
            current = -1
            if latest.is_symlink():
                match = re.fullmatch(r'version_(\d+)', os.readlink(latest))
                if match:
                    current = int(match.group(1))
            if self.version >= current:
                symlink_replace(latest, self.directory.name)
        finally:
            os.close(descriptor)

    @classmethod
    def allocate(cls, root, model, version=None):
        root = Path(root).resolve()
        (root / model).mkdir(parents=True, exist_ok=True)
        candidate = 0 if version is None else version
        while True:
            run = cls(root, model, candidate)
            try:
                run.directory.mkdir()
                return run
            except FileExistsError:
                if version is not None:
                    raise FileExistsError('Run already exists: %s; use --resume to continue it.' % run.directory)
                candidate += 1

    @classmethod
    def from_checkpoint(cls, checkpoint, model):
        checkpoint = Path(checkpoint).resolve(strict=True)
        match = re.fullmatch(r'version_(\d+)', checkpoint.parent.name)
        if not match or checkpoint.parent.parent.name != model:
            raise ValueError('Expected a checkpoint under %s/version_N. For inference with a standalone checkpoint, set --output-dir.' % model)
        return cls(checkpoint.parents[2], model, int(match.group(1)))
