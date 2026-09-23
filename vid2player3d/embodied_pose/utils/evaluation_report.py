"""Persistent playback diagnostics; errors compare simulated and reference bodies."""
import csv
import json
import sys
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from pathlib import Path

import numpy as np


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.log.write(text)
        self.log.flush()
        return self.stream.write(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()


class EvaluationReport:
    def __init__(self, task):
        self.task = task
        self.directory = Path(task.cfg['env']['evaluation_dir'])
        self.file = (self.directory / 'history.csv').open('w', newline='')
        self.writer = None
        self.rows = []
        self.finished = set()

    def append(self, game, step, reward, done, info):
        task = self.task
        actual = task._rigid_body_pos.detach().cpu().numpy()
        target = task._prev_target_rb_pos.detach().cpu().numpy()
        diff = actual - target
        mpjpe = np.linalg.norm(diff, axis=-1).mean(axis=-1)
        aligned = np.linalg.norm(diff - diff[:, :1], axis=-1).mean(axis=-1)
        root = np.linalg.norm(diff[:, 0], axis=-1)
        head_id = task._humanoid_head_id
        head_threshold = float(task._termination_heights[head_id])
        for env in range(len(actual)):
            key = (game, env)
            if key in self.finished:
                continue
            motion = int(task._reset_ref_motion_ids[env])
            time = float(task._cur_ref_motion_times[env])
            length = float(task._motion_lib._motion_lengths[motion])
            terminal = bool(done[env])
            terminated = bool(info['terminate'][env])
            reason = ''
            if terminal:
                reason = ('early_termination_height' if terminated else
                          'motion_complete' if time >= length else 'episode_limit')
                self.finished.add(key)
            row = dict(game=game, env=env, step=step, elapsed_s=step * task.dt,
                       motion_id=motion, reference_time_s=time, motion_length_s=length,
                       reward=float(reward[env]), body_position_error_m=float(mpjpe[env]),
                       root_aligned_body_error_m=float(aligned[env]), root_error_m=float(root[env]),
                       head_height_m=float(actual[env, head_id, 2]),
                       reference_head_height_m=float(target[env, head_id, 2]),
                       head_termination_height_m=head_threshold,
                       head_below_threshold=int(actual[env, head_id, 2] < head_threshold),
                       done=int(terminal), terminated=int(terminated), stop_reason=reason)
            names = info.get('sub_rewards_names', '')
            if isinstance(names, str):
                names = names.split(',')
            for i, name in enumerate(names):
                if name:
                    row[name] = float(info['sub_rewards'][env, i])
            if self.writer is None:
                self.writer = csv.DictWriter(self.file, fieldnames=list(row))
                self.writer.writeheader()
            self.writer.writerow(row)
            self.file.flush()
            self.rows.append(row)

    def close(self, status):
        self.file.close()
        groups = {}
        for row in self.rows:
            groups.setdefault((row['game'], row['env']), []).append(row)
        episodes = []
        for (game, env), rows in groups.items():
            last = rows[-1]
            episodes.append(dict(game=game, env=env, motion_id=last['motion_id'], steps=len(rows),
                duration_s=last['elapsed_s'], motion_length_s=last['motion_length_s'],
                reference_time_s=last['reference_time_s'],
                stop_reason=last['stop_reason'] or ('evaluation_stopped' if status == 'completed' else status),
                total_reward=sum(r['reward'] for r in rows),
                mean_body_position_error_m=float(np.mean([r['body_position_error_m'] for r in rows])),
                max_body_position_error_m=max(r['body_position_error_m'] for r in rows),
                mean_root_aligned_body_error_m=float(np.mean([r['root_aligned_body_error_m'] for r in rows])),
                mean_root_error_m=float(np.mean([r['root_error_m'] for r in rows]))))
        summary = dict(status=status, recorded_steps=len(self.rows), episodes=episodes,
                       metric_note='Position errors are unweighted mean body-origin distances in meters; root-aligned error removes pelvis translation only. Rewards are similarity scores, not error counts.')
        (self.directory / 'summary.json').write_text(json.dumps(summary, indent=2))
        if self.rows:
            # Explicit Agg canvas works over SSH without a display.
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            figure = Figure(figsize=(10, 11), tight_layout=True)
            FigureCanvasAgg(figure)
            axes = figure.subplots(4, 1)
            for key, rows in groups.items():
                times = [r['elapsed_s'] for r in rows]
                label = f'game {key[0]}, env {key[1]}'
                axes[0].plot(times, [r['reward'] for r in rows], label=label)
                for field, title in [('head_height_m', 'simulated'),
                                     ('reference_head_height_m', 'reference'),
                                     ('head_termination_height_m', 'threshold')]:
                    axes[3].plot(times, [r[field] for r in rows],
                                 linestyle='--' if title == 'threshold' else '-',
                                 label=f'{label}: {title}')
                for field, title in [('body_position_error_m', 'world body'),
                                     ('root_aligned_body_error_m', 'root-aligned body'),
                                     ('root_error_m', 'root')]:
                    axes[1].plot(times, [r[field] for r in rows], label=f'{label}: {title}')
                for field in ('dof_reward', 'vel_reward', 'body_pos_reward', 'body_rot_reward'):
                    if field in rows[0]:
                        axes[2].plot(times, [r[field] for r in rows], label=f'{label}: {field}')
            for ax, title, ylabel in zip(axes, ['Total reward', 'Tracking errors', 'Reward components', 'Head height (world Z)'],
                                        ['reward', 'meters', 'similarity', 'meters']):
                ax.set_title(title)
                ax.set_ylabel(ylabel)
                ax.set_xlabel('Episode elapsed time (s)')
                ax.grid(alpha=0.3)
                ax.legend(fontsize=6)
            figure.savefig(str(self.directory / 'metrics.png'), dpi=110)
        print(f'Evaluation report: {self.directory.resolve()}')
        for episode in episodes:
            print(f"Env {episode['env']}: {episode['stop_reason']}, "
                  f"{episode['duration_s']:.2f}s, mean body error "
                  f"{episode['mean_body_position_error_m']:.4f}m")
