"""Display-free, low-cost skeleton videos for debugging simulated motion."""
from datetime import datetime
from pathlib import Path

import imageio
import numpy as np
from PIL import Image, ImageDraw

_recorders = []


def close_debug_videos():
    for recorder in _recorders:
        recorder.close()
    _recorders.clear()


class DebugVideo:
    """Record environment zero using a projected 3D view, without OpenGL."""

    def __init__(self, task):
        cfg = task.cfg['env']
        self.limit = int(cfg.get('num_rec_frames', 300))
        if task.args.rec_once:
            self.limit = int(np.ceil(float(task._motion_lib._motion_lengths.max()) / task.dt))
        self.fps = float(cfg.get('rec_fps', 1.0 / task.dt))
        if self.limit <= 0 or self.fps <= 0:
            raise ValueError('num_rec_frames and rec_fps must be positive')
        self.path = Path(cfg.get('rec_fname', 'out/recording/debug_' +
                             datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f') + '.mp4'))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(str(self.path), fps=self.fps, codec='libx264',
                                         quality=3, ffmpeg_params=['-preset', 'ultrafast'],
                                         macro_block_size=16)
        self.frames = 0
        self.closed = False
        names = {name: i for i, name in enumerate(task.body_names)}
        chains = [
            ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe'],
            ['Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe'],
            ['Pelvis', 'Torso', 'Spine', 'Chest', 'Neck', 'Head'],
            ['Chest', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand'],
            ['Chest', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand'],
        ]
        self.edges = [(names[a], names[b]) for chain in chains
                      for a, b in zip(chain, chain[1:]) if a in names and b in names]
        _recorders.append(self)
        print(f'Headless debug video (environment 0): {self.path.resolve()}')

    def append(self, task):
        if self.closed:
            return
        actual = task._rigid_body_pos[0].detach().cpu().numpy()
        target = getattr(task, '_prev_target_rb_pos', None)
        target = target[0].detach().cpu().numpy() if target is not None else None
        frame = Image.new('RGB', (512, 256), (245, 245, 245))
        draw = ImageDraw.Draw(frame)
        draw.text((8, 6), 'Blue: simulation  Orange: reference  Env: 0', fill='black')
        draw.text((8, 22), f'Frame {self.frames}  t={(self.frames + 1) * task.dt:.2f}s', fill='black')
        # Fixed oblique camera: retain all three coordinates, with perspective.
        center = actual[0].copy()
        center[2] = 0.9
        right = np.array([0.7071, -0.7071, 0.0])
        up = np.array([0.2988, 0.2988, 0.9063])
        forward = np.cross(up, right)

        def project(points):
            delta = np.asarray(points) - center
            depth = np.maximum(5.0 + delta @ forward, 0.2)
            return np.column_stack((256 + 440 * (delta @ right) / depth,
                                    155 - 440 * (delta @ up) / depth))

        # World-aligned ground grid makes depth and horizontal travel visible.
        x0, y0 = np.floor(actual[0, :2])
        for offset in range(-3, 4):
            for ends in ([(x0+offset, y0-3, 0), (x0+offset, y0+3, 0)],
                         [(x0-3, y0+offset, 0), (x0+3, y0+offset, 0)]):
                draw.line([tuple(p) for p in project(ends)], fill=(215, 215, 215))
        segments = []
        for points, color in ((target, (225, 140, 30)), (actual, (35, 100, 220))):
            if points is None or not np.isfinite(points).all():
                continue
            pixels = project(points)
            depths = (points - center) @ forward
            for a, b in self.edges:
                segments.append(((depths[a] + depths[b]) / 2,
                                 [tuple(pixels[a]), tuple(pixels[b])], color))
        for _, ends, color in sorted(segments, key=lambda s: s[0], reverse=True):
            draw.line(ends, fill=color, width=3)
            for x, y in ends:
                draw.ellipse((x-2, y-2, x+2, y+2), fill=color)
        self.writer.append_data(np.asarray(frame))
        self.frames += 1
        if self.frames >= self.limit:
            self.close()

    def close(self):
        if not self.closed:
            self.closed = True
            self.writer.close()
            print(f'Saved headless debug video: {self.path.resolve()} ({self.frames} frames)')
