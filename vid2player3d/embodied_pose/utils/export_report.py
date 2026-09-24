"""One summary row per exported motion, without retaining per-step history."""
import csv
from pathlib import Path

import numpy as np


class ExportMotionMetrics:
    def __init__(self, task, motion_id, env_id, planned_steps):
        self.task = task
        self.motion_id = motion_id
        self.env_id = env_id
        self.planned_steps = planned_steps
        self.steps = 0
        self.sums = {}
        self.max_body_error = 0.0
        self.min_head_height = float('inf')
        self.head_below_steps = 0
        self.terminated = False
        self.done = False

    def append(self, reward, done, info):
        task, env = self.task, self.env_id
        # Slice before copying to CPU: background environments are not exported.
        actual = task._rigid_body_pos[env].detach().cpu().numpy()
        target = task._prev_target_rb_pos[env].detach().cpu().numpy()
        diff = actual - target
        body_error = float(np.linalg.norm(diff, axis=-1).mean())
        values = dict(
            reward=float(reward[env]),
            body_position_error_m=body_error,
            root_aligned_body_error_m=float(np.linalg.norm(diff - diff[:1], axis=-1).mean()),
            root_error_m=float(np.linalg.norm(diff[0])),
        )
        names = info.get('sub_rewards_names', '')
        if isinstance(names, str):
            names = names.split(',')
        for i, name in enumerate(names):
            if name:
                values[name] = float(info['sub_rewards'][env, i])
        for name, value in values.items():
            self.sums[name] = self.sums.get(name, 0.0) + value
        self.steps += 1
        self.max_body_error = max(self.max_body_error, body_error)
        head_height = float(actual[task._humanoid_head_id, 2])
        self.min_head_height = min(self.min_head_height, head_height)
        self.head_below_steps += int(head_height < float(task._termination_heights[task._humanoid_head_id]))
        self.done = bool(done[env])
        self.terminated |= bool(info['terminate'][env])
        self.reference_time = float(task._cur_ref_motion_times[env])

    def summary(self, export_file):
        length = float(self.task._motion_lib._motion_lengths[self.motion_id])
        reached_end = self.reference_time >= length
        if self.terminated:
            reason = 'early_termination_height'
        elif reached_end:
            reason = 'motion_complete'
        elif self.done:
            reason = 'episode_limit'
        else:
            reason = 'export_step_limit'
        row = dict(
            export_file=Path(export_file).name, motion_id=self.motion_id, env_id=self.env_id,
            frames_exported=self.steps, planned_steps=self.planned_steps,
            duration_s=self.steps * self.task.dt, motion_length_s=length,
            reference_time_s=self.reference_time,
            completion_fraction=min(1.0, max(0.0, self.reference_time / length)) if length > 0 else 1.0,
            survived=int(not self.terminated),
            completed_motion=int(reached_end and not self.terminated),
            done=int(self.done), terminated=int(self.terminated), stop_reason=reason,
            total_reward=self.sums['reward'],
            max_body_position_error_m=self.max_body_error,
            min_head_height_m=self.min_head_height,
            head_termination_height_m=float(self.task._termination_heights[self.task._humanoid_head_id]),
            head_below_threshold_steps=self.head_below_steps,
        )
        for name, total in self.sums.items():
            row['mean_' + name] = total / self.steps
            if name.endswith('_reward'):
                row['total_' + name] = total
        return row


class ExportEvaluationReport:
    """CSV for this export invocation; flush every successfully saved motion."""
    def __init__(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / 'metrics.csv'
        self.file = None
        self.writer = None

    def __enter__(self):
        self.file = self.path.open('w', newline='')
        return self

    def append(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()
