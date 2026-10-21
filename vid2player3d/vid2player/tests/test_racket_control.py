"""Interception timing, experiment isolation, and arm-preset behavior on CPU."""
import math
from pathlib import Path
import unittest

import torch
import yaml

from vid2player.utils.racket_control import (
    InterceptionSettings, InterceptionWindow, resolve_racket_control, corrected_racket_rotations)
from vid2player.utils.konia_transform import angle_axis_to_rotation_matrix
from vid2player.tests.test_controller_baseline import residual_player


class RacketControlTests(unittest.TestCase):
    def fixture(self):
        settings = InterceptionSettings(enabled=True)
        window = InterceptionWindow(2, 'cpu', 1/30, settings, .1)
        trajectory = torch.zeros(2, 100, 3)
        trajectory[:, :, 0] = torch.tensor([.7, -.7])[:, None]
        trajectory[:, :, 1] = torch.linspace(8., -8., 100)
        trajectory[:, :, 2] = 1.
        window.reset(torch.arange(2), trajectory)
        root = torch.tensor([[0., -1., 1.]]).repeat(2, 1)
        return window, trajectory, root

    def step(self, window, trajectory, root, frame, contact=False, attempting=True):
        clock = torch.full((2,), frame, dtype=torch.long)
        return window.step(clock, root, torch.zeros_like(root), trajectory[:, min(frame, 99)],
                           torch.full((2,), attempting), torch.full((2,), contact))

    def test_prediction_window_lock_and_contact_fade(self):
        window, trajectory, root = self.fixture()
        self.step(window, trajectory, root, 0)
        self.assertTrue(window.valid.all())
        self.assertFalse(window.active.any())
        self.assertEqual(window.side.tolist(), [1, 2])
        self.assertTrue(((window.position[:, 1] - root[:, 1] - .6).abs() < .12).all())
        self.assertTrue((window.weight == 0).all())
        for i in range(1, 44):
            self.step(window, trajectory, root, i)
        self.assertTrue(window.active.all())
        self.assertTrue(window.locked.all())
        deadline = window.deadline.clone()
        moved = root.clone(); moved[:, 1] += .2
        self.step(window, trajectory, moved, 44)
        torch.testing.assert_close(window.deadline, deadline)
        for i in range(45, 49):
            self.step(window, trajectory, root, i, contact=True)
        self.assertFalse(window.valid.any())
        self.assertFalse(window.active.any())
        torch.testing.assert_close(window.weight, torch.zeros(2))

    def test_bad_preview_miss_and_expired_window_do_not_retrigger(self):
        for condition in ['divergence', 'miss', 'expired', 'recovery']:
            window, trajectory, root = self.fixture()
            for i in range(49):
                self.step(window, trajectory, root, i)
            if condition == 'divergence':
                trajectory = trajectory + 5
            elif condition == 'miss':
                root = root.clone(); root[:, 1] = 10
            for i in range(49, 80):
                self.step(window, trajectory, root, i, attempting=condition != 'recovery')
            self.assertFalse(window.active.any(), condition)
            torch.testing.assert_close(window.weight, torch.zeros(2))
        window, trajectory, root = self.fixture()
        window.reset(torch.tensor([0]), trajectory[:1])
        self.assertFalse(window.locked[0])
        self.assertEqual(window.weight[0], 0)

    def test_unreachable_or_nonfinite_has_no_window(self):
        for bad in [100., float('nan')]:
            window, trajectory, root = self.fixture()
            trajectory[:, :, 2] = bad
            window.reset(torch.arange(2), trajectory)
            self.step(window, trajectory, root, 0)
            self.assertFalse(window.valid.any())
            self.assertFalse(window.active.any())

    def test_none_keeps_decoded_pose_and_additive_policy_controls(self):
        pose = torch.zeros(2, 2, 3)
        pose[:, 0] = torch.tensor([.3, .4, -.2])
        pose[:, 1] = torch.tensor([.1, -.2, .3])
        residual = torch.tensor([[.4, -.4, .4]]).repeat(2, 1)
        result, _ = corrected_racket_rotations(angle_axis_to_rotation_matrix(pose), residual,
            torch.tensor([1., 3.1]), torch.tensor([-1, 1]), 'none')
        expected = pose.clone()
        expected[:, 0, 0] += .1 * math.pi
        expected[:, 1, 1] -= .1 * math.pi
        expected[:, 1, 2] += .1 * math.pi
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_window_preserves_pose_when_closed_and_smooths_preset_switches(self):
        pose = torch.zeros(1, 2, 3); pose[:, 1, 0] = .1
        decoded = angle_axis_to_rotation_matrix(pose)
        delta = torch.eye(3).expand(1, 2, 3, 3).clone()
        phase, swing = torch.tensor([2.8]), torch.tensor([1])
        result, delta = corrected_racket_rotations(decoded, torch.zeros(1, 3), phase, swing,
                                                   'interception_window', torch.zeros(1), delta, 1/3)
        torch.testing.assert_close(result, pose, atol=1e-5, rtol=1e-5)
        for _ in range(40):
            result, delta = corrected_racket_rotations(decoded, torch.zeros(1, 3), phase, swing,
                                                       'interception_window', torch.ones(1), delta, 1/3)
        self.assertAlmostEqual(result[0, 0, 0].item(), -.75 * math.pi, places=4)
        self.assertAlmostEqual(result[0, 1, 2].item(), -.25 * math.pi, places=4)
        switched, delta = corrected_racket_rotations(decoded, torch.zeros(1, 3), torch.tensor([3.11]), swing,
                                                     'interception_window', torch.ones(1), delta, 1/3)
        # The phase threshold cannot instantly switch from -45 to +45 degrees.
        self.assertLess(abs(switched[0, 1, 2] - result[0, 1, 2]), .6)
        for _ in range(45):
            result, delta = corrected_racket_rotations(decoded, torch.zeros(1, 3), phase, swing,
                                                       'interception_window', torch.zeros(1), delta, 1/3)
        torch.testing.assert_close(result, pose, atol=1e-5, rtol=1e-5)

    def test_mvae_integration_same_in_training_and_evaluation(self):
        for mode in ['none', 'interception_window']:
            output = []
            for training in [True, False]:
                player, feature, _ = residual_player(training)
                player._update_mvae_state.__func__.__globals__['corrected_racket_rotations'] = corrected_racket_rotations
                player.cfg['racket_phase_correction'] = {'mode': mode}
                player._racket_control = resolve_racket_control({'mode': mode, 'prediction': {'enabled': True}})
                player._racket_delta = torch.eye(3).expand(2, 2, 3, 3).clone()
                player._racket_window_weight = torch.ones(2)
                player._racket_blend_alpha = 1/3
                player._predict_phase = True
                player._phase_pred = torch.tensor([1., 2.8])
                player._swing_type = torch.tensor([-1, 1])
                player._update_mvae_state(feature.clone(), residual=torch.full((2, 3), .4))
                output.append(player._joint_rotmat)
            torch.testing.assert_close(*output)

    def test_yaml_experiments_and_validation(self):
        root = Path(__file__).resolve().parents[1] / 'cfg/controller/usopen_train_stage_1'
        configs = [yaml.safe_load((root / f'high_level_{i}.yaml').read_text()) for i in [4, 5, 6]]
        self.assertEqual(resolve_racket_control().mode, 'legacy')
        for i, cfg in zip([4, 5, 6], configs):
            self.assertEqual(cfg['params']['config']['name'], f'high_level_{i}')
            if i > 4:
                settings = resolve_racket_control(cfg['env']['vid2player'].pop('racket_phase_correction'))
                self.assertEqual(settings.mode, 'interception_window' if i == 5 else 'none')
                self.assertTrue(settings.prediction.enabled)
                cfg['params']['config']['name'] = 'high_level_4'
                cfg['params']['config']['description'] = configs[0]['params']['config']['description']
                self.assertEqual(cfg, configs[0])
        for bad in [{'mode':'bad'}, {'mode':'interception_window'}, {'prediction':{'update_interval':0}},
                    {'prediction':{'preparation_s':float('nan')}}, {'blend_s':0}]:
            with self.assertRaises(ValueError):
                resolve_racket_control(bad)


if __name__ == '__main__':
    unittest.main()
