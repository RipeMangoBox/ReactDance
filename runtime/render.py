"""ReactDance skeleton renderer adapted from utils/visualize.py.

Uses the same SMPL-X chains, red leader / green follower, Z-up, 30fps.
Frames stream to the bundled FFmpeg; both people share a single floor offset.
"""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from multiprocessing import get_context
from pathlib import Path
import os
import shutil
import subprocess
import tempfile

from .data import test_references
from .paths import project_path

CHAINS = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15],
          [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]


def render_duet(leader, follower, output, audio=None):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import imageio_ffmpeg

    leader = np.asarray(leader).reshape(-1, 22, 3).copy()
    follower = np.asarray(follower).reshape(-1, 22, 3).copy()
    count = min(len(leader), len(follower))
    if count < 1:
        raise ValueError('No frames to render')
    motions = [leader[:count], follower[:count]]
    floor = min(float(m[:, :, 2].min()) for m in motions)
    for motion in motions:
        motion[:, :, 2] -= floor
    combined = np.concatenate(motions, axis=1)
    lo = combined.min(axis=(0, 1))
    hi = combined.max(axis=(0, 1))
    center = (lo + hi) / 2
    radius = max(float((hi - lo).max()) * .62, 1.05)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = os.environ.get('FFMPEG_BINARY') or imageio_ffmpeg.get_ffmpeg_exe()
    fig = plt.figure(figsize=(6.4, 4.8), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(azim=-60, elev=25)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(-.1, center[2] - radius), max(2 * radius, center[2] + radius))
    if hasattr(ax, 'set_box_aspect'):
        ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    vertices = [[center[0]-radius, center[1]-radius, 0],
                [center[0]-radius, center[1]+radius, 0],
                [center[0]+radius, center[1]+radius, 0],
                [center[0]+radius, center[1]-radius, 0]]
    ax.add_collection3d(Poly3DCollection([vertices], facecolors='lightgrey', linewidths=0, alpha=.25))
    artists = []
    for motion, color in zip(motions, ('#d83932', '#29915c')):
        for chain in CHAINS:
            line, = ax.plot3D([], [], [], color=color, linewidth=2)
            artists.append((line, motion, chain))
    fig.tight_layout(pad=0)
    process = None
    try:
        with tempfile.TemporaryDirectory(prefix='reactdance_render_', dir=str(output.parent)) as temp:
            temp = Path(temp)
            silent = temp / 'silent.mp4'
            with (temp / 'ffmpeg.log').open('wb') as log:
                command = [ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo',
                           '-pix_fmt', 'rgb24', '-s', '640x480', '-r', '30', '-i', '-', '-an',
                           '-c:v', 'libx264', '-threads', '2', '-pix_fmt', 'yuv420p',
                           '-crf', '20', '-movflags', '+faststart', str(silent)]
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
                try:
                    for frame in range(count):
                        for line, motion, chain in artists:
                            points = motion[frame, chain]
                            line.set_data(points[:, 0], points[:, 1])
                            line.set_3d_properties(points[:, 2])
                        fig.canvas.draw()
                        process.stdin.write(fig.canvas.tostring_rgb())
                finally:
                    process.stdin.close()
                code = process.wait()
            if code:
                raise RuntimeError('FFmpeg failed: ' + (temp / 'ffmpeg.log').read_text(errors='replace')[-2000:])
            if audio:
                merged = temp / 'with_audio.mp4'
                result = subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-i', str(silent),
                    '-i', str(Path(audio).resolve()), '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy',
                    '-c:a', 'aac', '-shortest', '-movflags', '+faststart', str(merged)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if result.returncode:
                    raise RuntimeError('Audio mux failed: ' + result.stderr.decode(errors='replace')[-2000:])
                shutil.move(str(merged), str(output))
            else:
                shutil.move(str(silent), str(output))
    finally:
        plt.close(fig)
        if process and process.poll() is None:
            process.terminate()
            process.wait()
    print('[OK] Video: %s (%d frames)' % (output, count), flush=True)


def render_prediction(destination, data, name):
    import numpy as np

    poses = destination / 'pos3d_npy'
    follower = np.load(poses / (name + '_00.npy'))
    leader = np.load(poses / (name + '_01.npy'))
    audio = next((path for split in ('test', 'all', 'train')
                  if (path := data / 'music/mp3' / split / (name + '.mp3')).is_file()), None)
    render_duet(leader, follower, destination / 'videos' / (name + '.mp4'), audio=audio)


def render_evaluation(destination, data_root=None, workers=None):
    destination = Path(destination)
    manifest = json.loads((destination / 'samples.json').read_text())
    config = manifest['config']
    config['data']['root'] = data_root or manifest['data_root']
    _, names = test_references(config, manifest['samples'])
    render = partial(render_prediction, destination, project_path(config['data']['root']))
    if workers is None:
        workers = config['data']['num_workers']
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
            for _ in pool.map(render, names):
                pass
    else:
        for name in names:
            render(name)
