"""CPU validation protocol and artifact coverage without requiring IsaacGym."""
import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from vid2player.utils.controller_validation import (
    configure_validation_args, select_validation_balls, prepare_validation,
    rollout_shot, rollout_batch, aggregate, validate_controller, _reset_slots)


def ball_data(path, count=30):
    data = np.zeros((count, 307), dtype=np.float32)
    for i in range(count):
        trajectory = np.zeros((100, 3), dtype=np.float32)
        trajectory[:, 0] = [-2, 0, 2][i % 3] + .01*(i//6)
        trajectory[:, 1] = np.linspace(13, -15, 100)
        trajectory[:, 2] = [.9, 1.4][(i//3) % 2]
        data[i, :7] = [0, 13, 1, 0, -15, 1, 5]
        data[i, 7:] = trajectory.ravel()
    np.save(path, data)
    return data


def fake_player(contact=True, landing=True, terminate=False, num_envs=1):
    n = num_envs
    physics = NS(dt=1/30, _ball_generator=NS(sample_idx=torch.zeros(n, dtype=torch.long)),
                 _has_racket_ball_contact=torch.zeros(n, dtype=torch.bool),
                 _has_bounce_now=torch.zeros(n, dtype=torch.bool), _bounce_pos=torch.zeros(n, 3),
                 _rigid_body_pos=torch.zeros(n, 24, 3))
    task = NS(device='cpu', num_envs=n, _physics_player=NS(task=physics),
              _mvae_player=NS(_init_data=torch.arange(n, dtype=torch.float).reshape(n, 1, 1)),
              cfg_v2p={'court_min': [-5, -16], 'court_max': [5, -10],
                       'reward_weights': {'pos': 2.}, 'aux_loss_specs': {'dof_res': .01}},
              _reset_reaction_buf=torch.zeros(n, dtype=torch.bool),
              _reset_recovery_buf=torch.zeros(n, dtype=torch.bool),
              _tar_time_total=torch.zeros(n), _target_bounce_pos=torch.zeros(n, 3),
              _ball_pos=torch.tensor([[0., -13., 1.]]).repeat(n, 1),
              _racket_pos=torch.tensor([[0., -13., 1.]]).repeat(n, 1),
              _root_pos=torch.tensor([[0., -13., 1.]]).repeat(n, 1),
              _est_bounce_in=torch.ones(n, dtype=torch.bool),
              obs_buf=torch.zeros(n, 1), _compute_observations=lambda: None,
              _num_mvae_action=2, _num_res_dof_action=1)
    resets, launches, seeds = [], [], []
    counters = torch.zeros(n, dtype=torch.long)
    def reset(ids):
        resets.append(ids.tolist())
        for i in ids.tolist():
            launches.append(int(physics._ball_generator.sample_idx[i]))
        counters[ids] = 0
        physics._has_racket_ball_contact[ids] = False
        return {'obs': task.obs_buf}
    def step(env, action):
        counters[:] += 1
        physics._has_racket_ball_contact[:] = contact & (counters >= 2)
        physics._has_bounce_now[:] = (counters <= 2) | landing
        physics._bounce_pos[:] = torch.tensor([0., 10., 0.])
        physics._bounce_pos[counters <= 2] = torch.tensor([0., -10., 0.])
        task._reset_recovery_buf[:] = counters == 2
        return {'obs': task.obs_buf}, torch.full((n,), .2), torch.full((n,), terminate), {
            'sub_rewards': torch.full((n, 1), .1), 'sub_rewards_names': 'pos_reward'}
    player = NS(task=task, device='cpu', model=torch.nn.Linear(1, 1), normalize_input=False,
                is_rnn=True, init_rnn=lambda: seeds.append(True), config={'bounds_loss_coef': 10.},
                env=object(), env_reset=reset, get_batch_size=lambda *a: n,
                obs_to_torch=lambda obs: {'obs': obs},
                get_action=lambda *a, **k: torch.tensor([[1.1, 0., .2]]).repeat(n, 1), env_step=step)
    return player, resets, launches, seeds


class ValidationTests(unittest.TestCase):
    def test_cpu_cli_without_pipeline_overrides(self):
        path = Path(__file__).resolve().parents[1]/'utils/config.py'
        node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'get_args')
        parsed = NS(resume_id=None, validation=True, eval_controller=False, rl_device='cpu',
                    sim_device_type='cuda', use_gpu_pipeline=True, use_gpu=True,
                    test=False, play=False, compute_device_id=0)
        namespace = {'gymutil': NS(parse_arguments=lambda **kw: parsed)}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        args = namespace['get_args']()
        self.assertEqual(args.device, 'cpu')
        self.assertEqual(args.sim_device, 'cpu')
        self.assertFalse(args.use_gpu or args.use_gpu_pipeline)

    def args(self, root):
        checkpoint = root/'model.pth'
        checkpoint.touch()
        return NS(eval_controller=False, horovod=False, validation_shots_per_preset=1, validation_num_envs=1,
                  eval_checkpoint=str(checkpoint), eval_steps=5, eval_ball='center_low',
                  eval_target='center', eval_ball_x=None, eval_ball_height=None, seed=None,
                  rl_device='cpu', use_gpu=True, use_gpu_pipeline=True, cfg='test',
                  eval_output=str(root/'validation'))

    def test_selection_and_preparation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root/'balls.npy'
            data = ball_data(source)
            rows, cases = select_validation_balls(source, 5)
            self.assertEqual(len(rows), 30)
            self.assertEqual(len({c['source_row'] for c in cases}), 30)
            np.testing.assert_array_equal(rows, data[[c['source_row'] for c in cases]])
            np.testing.assert_array_equal(rows, select_validation_balls(source, 5)[0])
            with self.assertRaises(ValueError):
                select_validation_balls(source, 6)
            args = self.args(root)
            configure_validation_args(args)
            self.assertFalse(args.use_gpu or args.use_gpu_pipeline or args.train)
            self.assertEqual(args.seed, 123)
            cfg = {'name': 'MVAEControllerVis', 'env': {'episodeLength': 2,
                   'enableEarlyTermination': True, 'vid2player': {'ball_traj_file': str(source)}}}
            train = {'params': {'config': {}}}
            output, suite, rows, temporary = prepare_validation(args, cfg, train)
            try:
                self.assertEqual(len(suite), 18)
                self.assertEqual([c['seed'] for c in suite[:3]], [123]*3)
                self.assertEqual(len({(c['ball_index'], c['target']) for c in suite}), 18)
                self.assertFalse(cfg['env']['enableEarlyTermination'])
                self.assertEqual(cfg['env']['episodeLength'], 7)
                self.assertEqual(cfg['name'], 'PhysicsMVAEController')
                self.assertTrue(Path(temporary.name, 'launches.npy').is_file())
                json.loads((output/'validation_config.json').read_text())
                with self.assertRaises(FileExistsError):
                    prepare_validation(args, cfg, train)
            finally:
                temporary.cleanup()
            args.rl_device = 'cuda:0'
            args.use_gpu = args.use_gpu_pipeline = True
            configure_validation_args(args)
            self.assertTrue(args.use_gpu and args.use_gpu_pipeline)

    def run_shot(self, **kwargs):
        player, resets, launches, rnn = fake_player(**kwargs)
        case = {'ball_index': 4, 'seed': 123, 'target_xyz_m': [0., 10., 0.]}
        with patch.dict(sys.modules, {'vid2player.utils.config': NS(set_seed=lambda *a: None)}):
            shot = rollout_shot(player, NS(eval_steps=4), case)
        self.assertEqual(launches, [4])
        self.assertEqual(resets[0], [0])
        self.assertTrue(all(ids == [] for ids in resets[1:]))
        self.assertEqual(rnn, [True])
        return shot

    def test_first_post_contact_landing_and_losses(self):
        shot = self.run_shot()
        self.assertEqual(shot['contact_step'], 2)
        self.assertEqual(shot['landing_xyz_m'], [0., 10., 0.])
        self.assertEqual(shot['landing_error_m'], 0)
        self.assertTrue(shot['return_in'])
        self.assertEqual(shot['end_reason'], 'landing_observed')
        self.assertEqual(len(shot['steps']), 3)
        row = shot['steps'][0]
        self.assertAlmostEqual(row['weighted_reward_components']['pos_reward'], .2)
        self.assertAlmostEqual(row['weighted_aux_dof_res_loss'], .0004)
        self.assertAlmostEqual(row['weighted_bounds_loss'], .1, places=6)

    def test_missing_landings_are_not_estimated_successes(self):
        missing = self.run_shot(landing=False)
        missed = self.run_shot(contact=False)
        terminated = self.run_shot(terminate=True)
        self.assertTrue(missing['unresolved'])
        self.assertFalse(missed['unresolved'])
        self.assertIsNone(missing['landing_error_m'])
        self.assertIsNone(missed['landing_xyz_m'])
        self.assertEqual(terminated['end_reason'], 'episode_termination')
        stats = aggregate([missing, missed, self.run_shot()])
        self.assertAlmostEqual(stats['return_in_rate'], 1/3)
        self.assertAlmostEqual(stats['contact_rate'], 2/3)
        self.assertEqual(stats['mean_landing_error_m'], 0)
        self.assertEqual(stats['unresolved'], 1)

    def test_complete_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            args = self.args(root)
            args.validation_shots_per_preset = 5
            args.validation_num_envs = 4
            player, _, launches, rnn = fake_player(num_envs=4)
            player.normalize_input = True
            player.running_mean_std = torch.nn.Linear(1, 1)
            normalization = player.running_mean_std.state_dict()
            torch.save({'model': player.model.state_dict(), 'running_mean_std': normalization}, args.eval_checkpoint)
            source = root/'balls.npy'
            ball_data(source)
            configure_validation_args(args)
            cfg = {'name': 'PhysicsMVAEController', 'env': {'episodeLength': 600,
                   'vid2player': {'ball_traj_file': str(source)}}}
            evaluation = prepare_validation(args, cfg, {'params': {'config': {}}})
            with patch.dict(sys.modules, {'vid2player.utils.config': NS(set_seed=lambda *a: None)}):
                validate_controller(player, args, evaluation)
            output = evaluation[0]
            self.assertEqual({p.name for p in output.iterdir()},
                             {'validation_config.json', 'metrics.png', 'values.json'})
            values = json.loads((output/'values.json').read_text())
            self.assertEqual(values['summary']['overall']['shots'], 90)
            self.assertEqual(len(launches), 92)  # last batch has two ignored slots
            self.assertEqual(launches[:6], [0, 0, 0, 1, 1, 1])
            self.assertEqual(len(rnn), 23)
            self.assertEqual(len(values['shots']), 90)
            self.assertEqual([s['case']['target'] for s in values['shots'][:3]], ['left', 'center', 'right'])
            config = json.loads((output/'validation_config.json').read_text())
            self.assertEqual(config['execution']['num_envs'], 4)
            self.assertEqual(config['execution']['batches'], 23)
            self.assertFalse(Path(evaluation[3].name).exists())
            with Image.open(output/'metrics.png') as plot:
                self.assertEqual(plot.size, (2400, 1950))

    def test_nonfinite_action_stops_before_simulation(self):
        player, resets, launches, _ = fake_player()
        player.get_action = lambda *a, **k: torch.full((1, 3), float('nan'))
        with patch.dict(sys.modules, {'vid2player.utils.config': NS(set_seed=lambda *a: None)}):
            shot = rollout_shot(player, NS(eval_steps=4),
                                {'ball_index': 0, 'seed': 123, 'target_xyz_m': [0., 10., 0.]})
        self.assertEqual(shot['end_reason'], 'nonfinite_action')
        self.assertEqual(shot['steps'], [])
        self.assertIsNone(shot['min_ball_racket_distance_m'])
        self.assertEqual(aggregate([shot])['contact_rate'], 0)
        json.dumps(shot, allow_nan=False)

    def test_parallel_slots_finish_independently(self):
        player, resets, launches, _ = fake_player(num_envs=4)
        original_step = player.env_step
        actions = []
        def step(env, action):
            actions.append(action.clone())
            obs, rewards, dones, info = original_step(env, action)
            dones[0] = True
            player.task._physics_player.task._has_racket_ball_contact[2] = False
            return obs, rewards, dones, info
        player.env_step = step
        means = torch.tensor([[1.1, 0., .2]]).repeat(4, 1)
        means[3] = float('nan')
        player.get_action = lambda *a, **k: means
        cases = [{'ball_index': i+1, 'seed': 123+i, 'target_xyz_m': [float(i), 10., 0.]} for i in range(4)]
        with patch.dict(sys.modules, {'vid2player.utils.config': NS(set_seed=lambda *a: None)}):
            shots = rollout_batch(player, NS(eval_steps=4), cases)
        self.assertEqual(len(actions), 4)  # four environments share four physics calls
        self.assertEqual([len(s['steps']) for s in shots], [1, 3, 4, 0])
        self.assertEqual([s['end_reason'] for s in shots],
                         ['episode_termination', 'landing_observed', 'step_limit', 'nonfinite_action'])
        self.assertEqual(shots[1]['landing_error_m'], 1.)
        self.assertIsNone(shots[2]['contact_step'])
        self.assertEqual(launches[:4], [1, 2, 3, 4])
        np.testing.assert_array_equal(player.task._target_bounce_pos[1:3].numpy(), [[1., 10., 0.], [2., 10., 0.]])
        self.assertTrue(all(torch.equal(a[[0, 3]], torch.zeros(2, 3)) for a in actions[1:]))
        self.assertTrue(torch.equal(player.task._mvae_player._init_data, torch.zeros(4, 1, 1)))

    def test_auto_concurrency_and_cap(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.args(Path(d))
            args.validation_num_envs = 0
            configure_validation_args(args)
            self.assertEqual(args.num_envs, 8)
            args.rl_device = 'cuda:0'
            args.validation_shots_per_preset = 5
            configure_validation_args(args)
            self.assertEqual(args.num_envs, 90)
            args.validation_num_envs = 32
            configure_validation_args(args)
            self.assertEqual(args.num_envs, 32)
            args.validation_shots_per_preset = 1
            configure_validation_args(args)
            self.assertEqual(args.num_envs, 18)
            args.validation_num_envs = -1
            with self.assertRaises(ValueError):
                configure_validation_args(args)

    def test_deterministic_metrics_match_sequential_rollouts(self):
        cases = [{'ball_index': i, 'seed': 123+i, 'target_xyz_m': [float(i), 10., 0.]} for i in range(3)]
        parallel, _, _, _ = fake_player(num_envs=3)
        sequential, _, _, _ = fake_player()
        with patch.dict(sys.modules, {'vid2player.utils.config': NS(set_seed=lambda *a: None)}):
            expected = [rollout_shot(sequential, NS(eval_steps=4), case) for case in cases]
            actual = rollout_batch(parallel, NS(eval_steps=4), cases)
        self.assertEqual(actual, expected)

    def test_parking_reset_defers_other_slots_recovery(self):
        player, _, _, _ = fake_player(num_envs=3)
        player.task._reset_recovery_buf[:] = True
        original = player.env_reset
        during_reset = []
        def reset(ids):
            during_reset.append(player.task._reset_recovery_buf.clone())
            return original(ids)
        player.env_reset = reset
        _reset_slots(player, torch.tensor([1]), [2])
        self.assertEqual(during_reset[0].tolist(), [False, False, False])
        self.assertEqual(player.task._reset_recovery_buf.tolist(), [True, False, True])


if __name__ == '__main__':
    unittest.main()
