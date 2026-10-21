"""Preview diagnostics follow the launch without leaking into history observations."""
import unittest
from types import SimpleNamespace as NS

import torch

from vid2player.tests.test_controller_baseline import ROOT, definitions


class ControllerPreviewLoggingTests(unittest.TestCase):
    def test_partial_ball_reset_and_step_keep_preview_in_both_observation_modes(self):
        namespace = definitions(ROOT / 'env/tasks/physics_mvae_controller.py',
                                {'PhysicsMVAEController'}, {'torch': torch})
        for history in (False, True):
            with self.subTest(history=history):
                task = object.__new__(namespace['PhysicsMVAEController'])
                task.device, task.num_envs, task.headless = 'cpu', 3, True
                task.cfg_v2p = {'use_history_ball_obs': history}
                task._reset_reaction_buf = torch.tensor([False, True, False])
                task._reset_recovery_buf = torch.zeros(3, dtype=torch.bool)
                task.reset_buf = torch.zeros(3, dtype=torch.long)
                task._tar_action = torch.tensor([0, 1, 0])
                task._tar_time = torch.zeros(3, dtype=torch.long)
                task._ball_traj = torch.full((3, 100, 3), -7.)
                preview = torch.arange(300, dtype=torch.float32).reshape(1, 100, 3)
                task._ball_pos = torch.zeros(3, 3)
                task._ball_pos[1] = preview[0, 0]
                task._obs_ball_traj_length = 2
                task._ball_obs = torch.full((3, 2, 3), -3.)
                task._racket_body_id = 0
                task._interception_window = None
                player = NS(
                    reset=lambda *args: preview,
                    _rigid_body_pos=torch.zeros(3, 1, 3),
                    _has_racket_ball_contact=torch.zeros(3, dtype=torch.bool),
                    _has_racket_ball_contact_now=torch.zeros(3, dtype=torch.bool),
                    _has_bounce_now=torch.zeros(3, dtype=torch.bool),
                    _bounce_pos=torch.zeros(3, 3))
                task._physics_player = NS(task=player)
                task._reset_reaction_tasks = lambda *args: None
                task._compute_observations = lambda *args: None
                task._est_bounce_in = torch.zeros(3, dtype=torch.bool)
                task._bounce_in = torch.zeros(3, dtype=torch.bool)
                task._est_bounce_pos = task._target_bounce_pos = torch.zeros(3, 3)
                task._est_bounce_time = torch.zeros(3)

                # Reset only the ball in slot 1, as on a normal reaction relaunch.
                task._reset_envs(torch.empty(0, dtype=torch.long))
                torch.testing.assert_close(task._ball_traj[1:2], preview)
                torch.testing.assert_close(task._ball_traj[[0, 2]], torch.full((2, 100, 3), -7.))
                stats = task._collect_task_stats()
                self.assertEqual(stats['preview_count'].item(), 1)
                self.assertEqual(stats['preview_error_sum'].item(), 0.)

                # Loading a diagnostic preview must not expose future samples to
                # an actor configured to observe its ball position history.
                obs = task._compute_task_obs(torch.tensor([1])).reshape(1, 2, 3)
                expected = torch.stack((torch.full((3,), -3.), preview[0, 0])) if history else preview[0, :2]
                torch.testing.assert_close(obs[0], expected)

                def advance_ball():
                    task._ball_pos[1] = preview[0, 1] + torch.tensor([0., .25, 0.])

                task._physics_player.run_one_step = advance_ball
                task.physics_step()
                task._tar_time += 1
                stats = task._collect_task_stats()
                self.assertAlmostEqual(stats['preview_error_sum'].item(), .25)


if __name__ == '__main__':
    unittest.main()
