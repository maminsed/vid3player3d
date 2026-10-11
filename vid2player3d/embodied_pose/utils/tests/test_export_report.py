"""CPU checks for export metrics, file persistence, and player integration."""
import csv
import io
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import joblib
import numpy as np
import torch

from embodied_pose.players.im_player import ImitatorPlayer, logger
from embodied_pose.utils.export_report import ExportEvaluationReport, ExportMotionMetrics


def make_task():
    target = torch.tensor([[[0., 0., 1.], [0., 0., 2.]]] * 2)
    actual = target.clone()
    actual[0] += 100  # Background data must not contaminate env 1's metrics.
    actual[1, :, 0] += 0.5
    return SimpleNamespace(
        dt=0.25, context_length=8, _rigid_body_pos=actual,
        _prev_target_rb_pos=target, _humanoid_head_id=1,
        _termination_heights=torch.tensor([0.1, 0.3]),
        _cur_ref_motion_times=torch.tensor([0., 0.]),
        _reset_ref_motion_ids=torch.tensor([0, 1]),
        _motion_lib=SimpleNamespace(
            _motion_lengths=torch.tensor([1., 1.]),
            _motion_num_frames=torch.tensor([4, 4]),
            _motion_seq_names=['fall.pk', 'complete.pk'],
            num_motions=lambda: 2),
        render_vis=Mock(),
    )


def info(terminated=False):
    return {'terminate': torch.tensor([1, int(terminated)]),
            'sub_rewards_names': 'dof_reward,vel_reward',
            'sub_rewards': torch.tensor([[99., 99.], [0.2, 0.4]])}


class ExportMetricsTests(unittest.TestCase):
    def test_selected_environment_totals_and_completion(self):
        task = make_task()
        metrics = ExportMotionMetrics(task, 1, 1, 4)
        for reward, time in [(0.5, 0.25), (0.75, 0.5), (1., 1.)]:
            task._cur_ref_motion_times[1] = time
            metrics.append(torch.tensor([100., reward]), torch.tensor([1, time == 1]), info())
        row = metrics.summary('/tmp/complete.pk')
        self.assertEqual(row['export_file'], 'complete.pk')
        self.assertEqual(row['frames_exported'], 3)
        self.assertEqual(row['duration_s'], 0.75)
        self.assertEqual(row['total_reward'], 2.25)
        self.assertEqual(row['mean_reward'], 0.75)
        self.assertAlmostEqual(row['total_dof_reward'], 0.6)
        self.assertAlmostEqual(row['mean_vel_reward'], 0.4)
        self.assertEqual(row['mean_body_position_error_m'], 0.5)
        self.assertEqual(row['mean_root_aligned_body_error_m'], 0.)
        self.assertEqual(row['mean_root_error_m'], 0.5)
        self.assertEqual(row['survived'], 1)
        self.assertEqual(row['completed_motion'], 1)
        self.assertEqual(row['stop_reason'], 'motion_complete')
        self.assertEqual(row['completion_fraction'], 1.)

    def test_stop_reasons_and_survival_are_separate(self):
        cases = [(True, True, 0.5, 'early_termination_height', 0, 0),
                 (True, True, 1., 'early_termination_height', 0, 0),
                 (False, True, 0.5, 'episode_limit', 1, 0),
                 (False, False, 0.5, 'export_step_limit', 1, 0),
                 (False, True, 1., 'motion_complete', 1, 1)]
        for terminated, done, time, reason, survived, completed in cases:
            with self.subTest(reason=reason, time=time):
                task = make_task()
                task._cur_ref_motion_times[1] = time
                task._rigid_body_pos[1, 1, 2] = 0.1
                metrics = ExportMotionMetrics(task, 1, 1, 4)
                metrics.append(torch.tensor([99., 0.5]), torch.tensor([1, done]), info(terminated))
                row = metrics.summary('motion.pk')
                self.assertEqual(row['stop_reason'], reason)
                self.assertEqual(row['survived'], survived)
                self.assertEqual(row['completed_motion'], completed)
                self.assertEqual(row['head_below_threshold_steps'], 1)
                self.assertAlmostEqual(row['min_head_height_m'], 0.1)

    def test_csv_is_flushed_and_closed_on_interruption(self):
        with tempfile.TemporaryDirectory() as directory:
            report = ExportEvaluationReport(directory)
            with self.assertRaises(KeyboardInterrupt):
                with report:
                    report.append({'export_file': 'one.pk', 'total_reward': 5.})
                    with report.path.open() as saved:
                        rows = list(csv.DictReader(saved))
                    self.assertEqual(len(rows), 1)
                    raise KeyboardInterrupt
            self.assertTrue(report.file.closed)
            with ExportEvaluationReport(directory) as replacement:
                replacement.append({'export_file': 'two.pk', 'total_reward': 3.})
            with report.path.open() as saved:
                self.assertEqual([r['export_file'] for r in csv.DictReader(saved)], ['two.pk'])

    def test_player_exports_two_files_and_matching_summary_rows(self):
        player = ImitatorPlayer.__new__(ImitatorPlayer)
        player.task = make_task()
        player.config = {'name': 'test_export'}
        player.args = SimpleNamespace(motion_id=-1)
        player.max_steps = 4
        player.render_env = False
        player.is_determenistic = True
        player.is_rnn = False
        player.is_tensor_obses = True
        player.value_size = 1
        player.device = 'cpu'
        player.get_batch_size = Mock(return_value=2)
        player.get_action = Mock(return_value=torch.zeros((2, 1)))
        player.env_reset = Mock(return_value={'obs': torch.zeros((2, 1))})
        active = {}

        def prepare(motion_id, env_id):
            active.update(motion_id=motion_id, env_id=env_id, step=0)
            player.task._cur_ref_motion_times.zero_()
            return {'obs': torch.zeros((2, 1))}

        def step(actions):
            active['step'] += 1
            env = active['env_id']
            player.task._cur_ref_motion_times[env] = active['step'] * player.task.dt
            done = torch.ones(2, dtype=torch.long)  # Background env always ends.
            early = active['motion_id'] == 0
            done[env] = int(active['step'] >= (2 if early else 4))
            flags = torch.zeros(2, dtype=torch.long)
            flags[env] = int(early and bool(done[env]))
            return torch.zeros((2, 1)), torch.tensor([0.25, 0.5]), done, {
                'terminate': flags, 'sub_rewards_names': 'dof_reward',
                'sub_rewards': torch.tensor([[0.1], [0.2]])}

        def collect(env_id):
            self.assertEqual(env_id, active['env_id'])
            player.motion_export_data['pose_aa'].append(np.zeros(72))
            player.motion_export_data['trans_orig'].append(np.zeros(3))
            player.motion_export_data['reference_times'].append(float(player.task._cur_ref_motion_times[env_id]))

        player._prepare_export_motion = prepare
        player._collect_motion_data = collect
        player.env = SimpleNamespace(step=step, task=player.task)
        old_cwd = Path.cwd()
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        old_level = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                player._run_export_dataset()
                output = Path('exports/test_export')
                with (output / 'metrics.csv').open() as saved:
                    rows = list(csv.DictReader(saved))
                self.assertEqual(len(rows), 2)
                self.assertEqual([r['stop_reason'] for r in rows],
                                 ['early_termination_height', 'motion_complete'])
                self.assertEqual([float(r['total_reward']) for r in rows], [0.5, 2.])
                for row in rows:
                    data = next(iter(joblib.load(output / row['export_file']).values()))
                    self.assertEqual(len(data['pose_aa']), int(row['frames_exported']))
                    self.assertEqual(set(data), {'pose_aa', 'trans_orig', 'beta', 'gender', 'fps', 'reference_times', 'export_fps'})
                    np.testing.assert_allclose(data['reference_times'],
                                               np.arange(1, len(data['pose_aa']) + 1) * player.task.dt)
                    self.assertAlmostEqual(data['export_fps'], 1. / player.task.dt)
                self.assertNotIn('Export step', stream.getvalue())
                self.assertNotIn('Environment step details', stream.getvalue())
                self.assertIn('Saved complete.pk', stream.getvalue())
                stream.truncate(0)
                stream.seek(0)
                logger.setLevel(logging.DEBUG)
                player._run_export_dataset()
                self.assertIn('Export step', stream.getvalue())
                self.assertIn('Environment step details', stream.getvalue())
        finally:
            os.chdir(old_cwd)
            logger.removeHandler(handler)
            logger.setLevel(old_level)


if __name__ == '__main__':
    unittest.main()
