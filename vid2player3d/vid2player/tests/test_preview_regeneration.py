"""Safety and format contracts for the standalone preview regeneration command."""
from pathlib import Path
import ast
import math
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from vid2player.tools.regenerate_ball_previews import (
    load_config, parse_args, reserve_output, sha256, validate_pool)


class TestPreviewRegeneration(unittest.TestCase):
    def test_three_shot_tail_and_live_reset_preserve_per_shot_spin(self):
        # torch.cross without dim silently crosses along the batch axis when N=3.
        # Execute the actual launch conversion expressions without importing Gym.
        root = Path(__file__).resolve().parents[1]
        cases = [('utils/tennis_ball.py', 'launch_ang_vel'),
                 ('utils/tennis_ball_out_estimator.py', 'launch_ang_vel'),
                 ('env/tasks/humanoid_smpl_im_mvae.py', 'launch_ang_vel')]
        for rel, variable in cases:
            tree = ast.parse((root / rel).read_text())
            expr = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == variable for t in node.targets))
            for count in (1, 3, 4):
                velocity = torch.tensor([[2., -25., 4.], [-3., -26., 3.], [1., -27., 2.],
                                         [4., -24., 1.]])[:count]
                spin = torch.tensor([5., -8., 9., 6.])[:count]
                gravity = torch.tensor([[0., 0., -1.]]).repeat(count, 1)
                actual = eval(compile(ast.Expression(expr), rel, 'eval'), dict(
                    torch=torch, F=F, math=math, pi=math.pi, launch_vel=velocity,
                    launch_vspin=spin, g_tensor=gravity, num_ball=count, env_ids=range(count)))
                expected = torch.stack([torch.linalg.cross(v, g) for v, g in zip(velocity, gravity)])
                expected = F.normalize(expected, dim=-1) * spin[:, None] * (2 * math.pi)
                with self.subTest(source=rel, count=count):
                    torch.testing.assert_close(actual, expected)

    def test_existing_output_is_never_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = reserve_output(Path(tmp) / 'run')
            original = output / 'ball_traj_in.npy'
            original.write_bytes(b'existing pool')
            before = sha256(original)
            with self.assertRaises(FileExistsError):
                reserve_output(output)
            self.assertEqual(sha256(original), before)

    def test_only_incoming_pool_layout_is_accepted(self):
        validate_pool(np.zeros((2, 307), dtype='float32'))
        for shape in ((0, 307), (2, 300), (2, 100, 3), (307,)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                validate_pool(np.zeros(shape))

    def test_stage_name_and_full_path_resolve_identically(self):
        path, cfg = load_config('usopen_train_stage_1')
        self.assertEqual(load_config(str(path)), (path, cfg))
        self.assertEqual(cfg['env']['physics']['substeps'], 12)

    def test_replay_defaults_do_not_resample(self):
        args = parse_args(['--cfg', 'usopen_train_stage_1'])
        self.assertFalse(args.resample)
        self.assertFalse(args.test_only)
        self.assertIsNone(args.num_shots)
        self.assertEqual(args.artifacts, 'all')

    def test_outgoing_only_and_movie_selector(self):
        args = parse_args(['--cfg', 'x', '--artifacts', 'outgoing', '--test-only'])
        self.assertEqual(args.artifacts, 'outgoing')
        self.assertTrue(args.test_only)
        with self.assertRaises(SystemExit):
            parse_args(['--cfg', 'x', '--artifacts', 'outgoing', '--num-shots', '5'])

    def test_resampling_requires_bounded_output_size(self):
        with self.assertRaises(SystemExit):
            parse_args(['--cfg', 'x', '--resample'])
        args = parse_args(['--cfg', 'x', '--resample', '--test-only'])
        self.assertTrue(args.test_only)

    def test_invalid_sizes_devices_and_conflicting_sources_fail(self):
        for extra in (['--batch-size', '0'], ['--num-shots', '-1'], ['--device', 'cuda'],
                      ['--resample', '--num-shots', '4', '--shots', 'old.npy']):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                parse_args(['--cfg', 'x'] + extra)


if __name__ == '__main__':
    unittest.main()
