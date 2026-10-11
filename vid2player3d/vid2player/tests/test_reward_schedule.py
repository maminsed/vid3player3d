"""Curriculum boundaries and controller integration without IsaacGym."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from vid2player.utils.reward_schedule import LinearRewardSchedule


ROOT = Path(__file__).resolve().parents[1]


class RewardScheduleTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(start=1., end=5., start_frame=100, end_frame=300)
        self.schedule = LinearRewardSchedule(self.config)

    def test_boundaries_and_interpolation(self):
        for frame, expected in [(0, 1), (100, 1), (150, 2), (200, 3),
                                (300, 5), (1000, 5)]:
            self.assertEqual(self.schedule.value(frame), expected)

    def test_resume_and_environment_count_independence(self):
        # No mutable epoch counter: a restored global frame gives the same scale.
        resumed = LinearRewardSchedule(self.config)
        self.assertEqual(resumed.value(200), self.schedule.value(200))
        self.assertEqual(self.schedule.value(2 * 50 * 2),
                         self.schedule.value(4 * 25 * 2))

    def test_invalid_configuration(self):
        for changes in [dict(type='cosine'), dict(start=0), dict(end=float('nan')),
                        dict(start_frame=-1), dict(end_frame=100)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                LinearRewardSchedule(dict(self.config, **changes))

    def test_controller_training_evaluation_and_opt_out(self):
        path = ROOT / 'env/tasks/physics_mvae_controller.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == 'PhysicsMVAEController')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                      and n.name == 'set_reward_schedule_progress')
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
        update = namespace['set_reward_schedule_progress']
        task = SimpleNamespace(_is_train=True, _pos_reward_schedule=self.schedule,
                               _reward_scales={'pos': 5., 'phase': 0.})
        update(task, 0)
        self.assertEqual(task._reward_scales, {'pos': 1., 'phase': 0.})
        update(task, 200)  # First resumed rollout must not restart at 1.
        self.assertEqual(task._reward_scales['pos'], 3.)
        task._is_train = False
        task._reward_scales['pos'] = 5.
        update(task, 0)
        self.assertEqual(task._reward_scales['pos'], 5.)
        task._is_train, task._pos_reward_schedule = True, None
        update(task, 0)
        self.assertEqual(task._reward_scales['pos'], 5.)

if __name__ == '__main__':
    unittest.main()
