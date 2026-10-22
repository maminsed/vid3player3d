"""Event, curriculum and controller integration checks; no IsaacGym simulation."""
import math
from types import SimpleNamespace as NS
import unittest

import torch
import yaml

from vid2player.tests.test_controller_baseline import ROOT, definitions
from vid2player.utils.return_curriculum import COMPONENTS, ReturnCurriculum
from vid2player.utils.tennis_contacts import racket_impact
from vid2player.utils.ball_physics import resolve_ball_physics
from vid2player.utils.tennis_court import NET_HEIGHT_M


class ReturnCurriculumTests(unittest.TestCase):
    def make(self, count=2, config=None, weights=None, scales=None):
        obj = ReturnCurriculum(count, 'cpu', config or {}, weights or {'pos': 1.}, scales or {},
                               ball_radius=resolve_ball_physics({'model': 'aerodynamic_v2'}).radius)
        ball = torch.tensor([[0., -10., 1.]]).repeat(count, 1)
        obj.reset(torch.arange(count), ball)
        return obj, ball

    def test_clearance_uses_active_ball_radius_and_shared_net_height(self):
        for model in ('legacy', 'aerodynamic_v2'):
            radius = resolve_ball_physics({'model': model}).radius
            obj = ReturnCurriculum(2, 'cpu', {}, {'pos': 1.}, {}, ball_radius=radius)
            self.assertEqual(obj.ball_radius, radius)
            self.assertEqual(obj.net_height, NET_HEIGHT_M)
            # Two horizontal flights straddle the clearance threshold by 1 mm.
            initial = torch.tensor([[0., -1., NET_HEIGHT_M + radius - .001],
                                    [0., -1., NET_HEIGHT_M + radius + .001]])
            obj.reset(torch.arange(2), initial)
            self.step(obj, initial, contact=True)
            outgoing = initial.clone()
            outgoing[:, 1] = 1.
            self.step(obj, outgoing)
            self.assertEqual(obj.crossed_net.tolist(), [False, True])
        for key in ('ball_radius_m', 'net_height_m'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.make(config={key: 1.})

    def step(self, obj, ball, contact=False, bounce=False, velocity=None, landing=None, valid=False):
        count = len(ball)
        velocity = torch.tensor([[0., 2., 0.]]).repeat(count, 1) if velocity is None else velocity
        return obj.step(ball, velocity, torch.full((count,), contact), torch.full((count,), bounce),
                        ball if landing is None else landing, torch.tensor([[0., 10., 0.]]).repeat(count, 1),
                        torch.full((count,), valid))

    def test_weak_contact_is_rewarded_without_estimator_and_only_once(self):
        obj, ball = self.make()
        first = self.step(obj, ball, contact=True)
        torch.testing.assert_close(first[:, 0], torch.ones(2))
        torch.testing.assert_close(first[:, 1], torch.full((2,), .2))
        self.assertFalse(obj.estimate_valid.any())
        self.assertTrue(obj.hit.all())
        torch.testing.assert_close(self.step(obj, ball, contact=True), torch.zeros(2, 5))

    def test_real_contact_detector_accepts_weak_return_rejected_by_lookup(self):
        previous = torch.zeros(1, 13)
        previous[:, :3] = torch.tensor([0., -13., 1.])
        previous[:, 8] = -25.
        current = previous.clone()
        current[:, 8] = 2.
        racket = previous[:, :3].clone()
        zeros = torch.zeros(1, 3)
        hit = racket_impact(previous, current, racket, racket, zeros, zeros, zeros,
                            .0335, .05, 1/60)
        self.assertTrue(hit.item())
        namespace = definitions(ROOT / 'utils/tennis_ball_out_estimator.py',
                                {'TennisBallOutEstimator', 'traj_out_params'}, {'torch': torch})
        estimator = object.__new__(namespace['TennisBallOutEstimator'])
        estimator.params = namespace['traj_out_params']
        valid, landing, duration, height = estimator.estimate(current)
        self.assertFalse(valid.item())
        self.assertIsNone(landing)

    def test_backward_contact_and_bounded_fast_forward_bonus(self):
        obj, ball = self.make()
        events = self.step(obj, ball, contact=True,
                           velocity=torch.tensor([[0., -5., 0.], [0., 50., 0.]]))
        torch.testing.assert_close(events[:, 0], torch.ones(2))
        torch.testing.assert_close(events[:, 1], torch.tensor([0., 1.]))

    def test_signed_direction_distinguishes_backward_stationary_and_weak_forward(self):
        speeds = torch.tensor([-100., -10., -5., -2., 0., 2., 5., 10., 100.])
        velocity = torch.zeros(len(speeds), 3)
        velocity[:, 1] = speeds
        for penalty in (0., .5, 1., 2.):
            obj, ball = self.make(count=len(speeds), config={
                'forward_reward_mode': 'signed_linear', 'forward_speed_mps': 10.,
                'backward_penalty_scale': penalty})
            events = self.step(obj, ball, contact=True, velocity=velocity)
            expected = (speeds / 10).clamp(0, 1) - penalty * (-speeds / 10).clamp(0, 1)
            torch.testing.assert_close(events[:, 1], expected)
            torch.testing.assert_close(events[:, 0], torch.ones(len(speeds)))
            # This changes credit, never contact classification, and scores once.
            self.assertTrue(obj.hit.all())
            self.assertEqual(obj.forward.tolist(), (speeds > 0).tolist())
            torch.testing.assert_close(self.step(obj, ball, contact=True, velocity=velocity),
                                       torch.zeros_like(events))
            obj.reset(torch.arange(len(speeds)), ball)
            no_contact = self.step(obj, ball, velocity=velocity)
            torch.testing.assert_close(no_contact, torch.zeros_like(events))
        # Omitted mode retains exactly the old positive-only forward definition.
        obj, ball = self.make(count=len(speeds))
        torch.testing.assert_close(self.step(obj, ball, contact=True, velocity=velocity)[:, 1],
                                   (speeds / 10).clamp(0, 1))

    def test_signed_direction_configuration_rejects_typos_and_invalid_penalties(self):
        for config in ({'forward_reward_mode': 'signed'}, {'backward_penalty_scale': -1.},
                       {'backward_penalty_scale': float('nan')},
                       {'backward_penalty_scale': float('inf')}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.make(config=config)

    def test_vertical_penalty_is_bounded_once_per_contact_and_opt_in(self):
        vz = torch.tensor([-20., -8., -4., 0., 4., 8., 12., 16., 30.])
        obj, ball = self.make(count=len(vz), weights={'pos': 1., 'vertical': .2})
        velocity = torch.zeros(len(vz), 3)
        velocity[:, 1], velocity[:, 2] = 2., vz
        # No contact means no penalty, even for extreme velocities.
        torch.testing.assert_close(self.step(obj, ball, velocity=velocity), torch.zeros(len(vz), 6))
        events = self.step(obj, ball, contact=True, velocity=velocity)
        torch.testing.assert_close(events[:, -1], torch.tensor([-1., -1., -.5, 0., 0., 0., -.5, -1., -1.]))
        torch.testing.assert_close(events[:, 0], torch.ones(len(vz)))
        torch.testing.assert_close(self.step(obj, ball, contact=True, velocity=velocity), torch.zeros_like(events))
        stats = obj.stats(torch.ones(len(vz), dtype=torch.bool), torch.zeros(len(vz), dtype=torch.bool))
        self.assertEqual(stats['return_vertical_outside_band'], 6)
        obj.reset(torch.tensor([0]), ball[:1])
        self.assertFalse(obj.vertical_outside_band[0])
        # Legacy configs keep both the old component shape and old stat keys.
        old, old_ball = self.make()
        self.assertEqual(self.step(old, old_ball, contact=True).shape, (2, 5))
        self.assertNotIn('return_vertical_outside_band', old.stats(
            torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)))

    def test_vertical_configuration_and_nonfinite_contact(self):
        for spec in ({'min_mps': 8, 'max_mps': 0}, {'min_mps': 1, 'max_mps': 1},
                     {'falloff_mps': 0}, {'max_mps': float('inf')}):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                self.make(config={'vertical_velocity': spec}, weights={'vertical': .2})
        obj, ball = self.make(config={'vertical_velocity': {'min_mps': 1., 'max_mps': 5., 'falloff_mps': 2.}},
                              weights={'vertical': .2})
        events = self.step(obj, ball, contact=True,
                           velocity=torch.tensor([[0., -2., 0.], [0., 2., float('nan')]]))
        # Backward hits cannot collect a positive vertical bonus; invalid hits score nothing.
        torch.testing.assert_close(events[:, -1], torch.tensor([-.5, 0.]))
        self.assertEqual(obj.hit.tolist(), [True, False])

    def test_net_and_first_measured_landing_include_out_target_feedback(self):
        obj, ball = self.make()
        self.step(obj, ball, contact=True, valid=True)
        crossed = torch.tensor([[0., 1., 2.]]).repeat(2, 1)
        events = self.step(obj, crossed)
        torch.testing.assert_close(events[:, 2], torch.ones(2))
        landing = torch.tensor([[0., 10., 0.], [6., 10., 0.]])
        events = self.step(obj, landing, bounce=True)
        self.assertEqual(events[0, 3].item(), 1.)
        self.assertGreater(events[1, 3].item(), 0.)
        self.assertLess(events[1, 3].item(), 1.)
        self.assertAlmostEqual(events[1, 4].item(), math.exp(-.005 * 36), places=6)
        self.assertEqual(obj.in_court.tolist(), [True, False])
        torch.testing.assert_close(self.step(obj, landing, bounce=True), torch.zeros(2, 5))
        stats = obj.stats(torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool))
        self.assertEqual(stats['return_landings'].item(), 2)
        self.assertEqual(stats['return_in'].item(), 1)
        self.assertEqual(stats['return_target_error_sum'].item(), 6.)

    def test_net_failure_and_ambiguous_bounce_do_not_earn_target(self):
        obj, ball = self.make()
        self.step(obj, ball, contact=True)
        self.step(obj, torch.tensor([[0., 1., .1], [10., 1., 2.]]))
        self.assertFalse(obj.crossed_net.any())
        landing = torch.tensor([[0., 10., 0.]]).repeat(2, 1)
        events = self.step(obj, landing, bounce=True)
        self.assertEqual(events[:, 4].sum().item(), 0.)
        self.assertFalse(obj.in_court.any())
        # Same-step contact/ground event cannot later score a second bounce.
        obj, ball = self.make()
        self.step(obj, ball, contact=True, bounce=True)
        torch.testing.assert_close(self.step(obj, landing, bounce=True), torch.zeros(2, 5))
        self.assertFalse(obj.landed.any())

    def test_partial_reset_no_incoming_or_default_estimate_landing_reward(self):
        obj, ball = self.make()
        events = self.step(obj, ball, bounce=True, valid=True)
        torch.testing.assert_close(events, torch.zeros(2, 5))
        self.step(obj, ball, contact=True)
        events = self.step(obj, ball, landing=torch.zeros_like(ball))
        torch.testing.assert_close(events, torch.zeros(2, 5))
        obj.reset(torch.tensor([0]), ball[:1])
        self.assertEqual(obj.hit.tolist(), [False, True])

    def test_schedules_zero_disabled_resume_and_configuration_validation(self):
        config = {'weights_schedule': {
            'forward': dict(start=0., end=1., start_frame=100, end_frame=300),
            'target': dict(enabled=False, start=0., end=1., start_frame=100, end_frame=300)},
            'scales_schedule': {'target': dict(start=.005, end=.05, start_frame=100, end_frame=300)}}
        obj, _ = self.make(config=config)
        obj.set_progress(200)
        self.assertEqual(obj.weights['forward'], .5)
        self.assertEqual(obj.weights['target'], 0.)
        self.assertAlmostEqual(obj.scales['target'], .0275)
        resumed, _ = self.make(config=config)
        resumed.set_progress(200)
        self.assertEqual(obj.schedule_metrics(), resumed.schedule_metrics())
        self.assertNotIn('curriculum/weight_target', obj.schedule_metrics())
        for bad in [{'weights_schedule': {'typo': dict(start=0, end=1, start_frame=0, end_frame=1)}},
                    {'forward_speed_mps': 0}, {'max_post_contact_s': -1}]:
            with self.subTest(config=bad), self.assertRaises(ValueError):
                self.make(config=bad)

    def test_controller_schedule_uses_stage_frame_only_in_training(self):
        config = {'weights_schedule': {'forward': dict(start=0., end=1., start_frame=100, end_frame=300)}}
        obj, _ = self.make(config=config)
        task = self.controller()
        task._return_curriculum, task._pos_reward_schedule = obj, None
        task._is_train = False
        task.set_reward_schedule_progress(200)
        self.assertEqual(obj.weights['forward'], 0.)
        task._is_train = True
        task.set_reward_schedule_progress(200)
        self.assertEqual(obj.weights['forward'], .5)

    def test_every_registered_reward_supports_weight_scheduling(self):
        config = {'weights_schedule': {key: dict(start=1., end=.5, start_frame=100, end_frame=300)
                                       for key in COMPONENTS},
                  'scales_schedule': {'phase': dict(start=5., end=0., start_frame=100, end_frame=300)}}
        obj, _ = self.make(config=config)
        obj.set_progress(200)
        self.assertTrue(all(value == .75 for value in obj.weights.values()))
        self.assertEqual(obj.scales['phase'], 2.5)
        # Reject aliases rather than silently ignoring a misspelled reward.
        with self.assertRaisesRegex(ValueError, 'weights.reach'):
            self.make(config={'weights_schedule': {'reach': config['weights_schedule']['pos']}})

    def controller(self):
        namespace = definitions(ROOT / 'env/tasks/physics_mvae_controller.py',
                                {'PhysicsMVAEController', 'compute_reward_reach', 'check_out_of_court'},
                                {'torch': torch, 'math': math, 'COMPONENTS': COMPONENTS})
        return object.__new__(namespace['PhysicsMVAEController'])

    def test_controller_reward_preserves_reaching_and_applies_scheduled_weights(self):
        task = self.controller()
        obj, ball = self.make(weights={'pos': 1., 'contact': .5, 'forward': 2.})
        task.cfg_v2p = {'reward_type': 'return_curriculum'}
        task._return_curriculum = obj
        task._reward_weights, task._reward_scales = obj.weights, obj.scales
        task._phase_pred = torch.full((2,), 3.)
        task._tar_action = torch.ones(2, dtype=torch.long)
        task._racket_pos = task._ball_pos = ball
        task._ball_vel = torch.tensor([[0., 2., 0.]]).repeat(2, 1)
        task._est_bounce_valid = torch.zeros(2, dtype=torch.bool)
        task._target_bounce_pos = torch.tensor([[0., 10., 0.]]).repeat(2, 1)
        task._mvae_player = NS(_swing_type=torch.full((2,), -1))
        task._physics_player = NS(task=NS(_has_racket_ball_contact_now=torch.ones(2, dtype=torch.bool),
                                        _has_bounce_now=torch.zeros(2, dtype=torch.bool), _bounce_pos=ball))
        task.rew_buf = torch.zeros(2)
        task._compute_reward(None)
        torch.testing.assert_close(task.rew_buf, torch.full((2,), 1.9))
        diagnostics = definitions(ROOT / 'utils/controller_evaluation.py', {'phase_metrics', '_number'},
                                  {'np': NS(pi=math.pi, exp=math.exp, isfinite=math.isfinite)})
        self.assertEqual(diagnostics['phase_metrics'](task)['phase_target_rad'], 3.)
        self.assertEqual(diagnostics['phase_metrics'](task)['phase_reward_factor'], 1.)
        task._tar_action[:] = 0
        task._compute_reward(None)
        torch.testing.assert_close(task.rew_buf, torch.zeros(2))
        # Optional vertical column is integrated by name/order and can be negative.
        vertical, _ = self.make(weights={'pos': 0., 'vertical': .2})
        task._return_curriculum, task._reward_weights = vertical, vertical.weights
        task._ball_vel = torch.tensor([[0., 2., -20.], [0., 2., 4.]])
        task._compute_reward(None)
        torch.testing.assert_close(task.rew_buf, torch.tensor([-.2, 0.]))
        self.assertEqual(task._sub_rewards_names.split(',')[-1], 'vertical_reward')

    def test_relaunch_waits_for_landing_or_timeout_without_losing_due_timer(self):
        task = self.controller()
        obj, ball = self.make(count=1)
        self.step(obj, ball, contact=True)
        task._return_curriculum = obj
        task._root_pos = torch.tensor([[0., -13., 1.]])
        task._court_min, task._court_max = torch.tensor([-5., -16.]), torch.tensor([5., -10.])
        task.obs_buf = torch.zeros(1, 3)
        task.progress_buf = torch.ones(1, dtype=torch.long)
        task._max_episode_length = 600
        task._terminate_buf = task.reset_buf = torch.zeros(1, dtype=torch.long)
        task._tar_time = torch.tensor([70])
        task._tar_time_total = torch.tensor([70])
        task._tar_action = torch.tensor([0])
        task._ball_pos = ball
        task._compute_stats = lambda: None
        task._physics_player = NS(task=NS(_has_racket_ball_contact=torch.ones(1, dtype=torch.bool)))
        task.cfg = {'env': {'enableEarlyTermination': False}}
        task._compute_reset()
        self.assertFalse(task._reset_reaction_buf.any())
        obj.contact_age[:] = 91
        task._tar_time[:] = 140
        task._compute_reset()
        self.assertTrue(task._reset_reaction_buf.all())
        # Also release the wait on the first observed landing.
        obj.contact_age[:] = 1
        obj.landing_seen[:] = True
        task._compute_reset()
        self.assertTrue(task._reset_reaction_buf.all())

    def test_logging_denominators_and_no_extra_panels_for_old_reward_types(self):
        namespace = definitions(ROOT / 'agents/v2p_agent.py', {'summarize_task_stats'}, {})
        summarize = namespace['summarize_task_stats']
        totals = {'cycles': 10, 'hits': 4, 'return_hits': 4, 'return_forward': 3,
                  'return_net': 2, 'return_in': 1, 'return_landings': 3,
                  'return_valid_estimates': 2, 'return_target_landings': 2,
                  'return_target_error_sum': 8, 'return_contact_vy_sum': 20, 'return_boundaries': 1}
        result = summarize({k: torch.tensor(v) for k, v in totals.items()})
        self.assertEqual(result['task/hit_rate'], .4)
        self.assertEqual(result['return/in_rate'], .1)
        self.assertEqual(result['return/forward_per_contact'], .75)
        self.assertEqual(result['return/estimator_valid_per_contact'], .5)
        self.assertEqual(result['return/target_error_m'], 4.)
        self.assertNotIn('task/estimated_in_rate', result)
        result = summarize({'cycles': torch.tensor(10), 'hits': torch.tensor(4)})
        self.assertEqual(result, {'task/cycles': 10., 'task/hit_rate': .4,
                                  'task/estimated_in_rate': 0.})
        result = summarize({'cycles': torch.tensor(10), 'return_hits': torch.tensor(0)})
        self.assertNotIn('return/forward_per_contact', result)
        result = summarize({'cycles': torch.tensor(10), 'return_hits': torch.tensor(4),
                            'return_vertical_outside_band': torch.tensor(3)})
        self.assertEqual(result['return/vertical_outside_band_per_contact'], .75)

    def test_2_2_direction_and_reach_schedules_and_fixed_evaluation_weights(self):
        cfg_root = ROOT / 'cfg/controller/usopen_train_stage_2'
        old = yaml.safe_load((cfg_root / 'high_level_2_1.yaml').read_text())
        new = yaml.safe_load((cfg_root / 'high_level_2_2.yaml').read_text())
        old_v, new_v = old['env']['vid2player'], new['env']['vid2player']
        for key, value in old_v.items():
            if key not in ('reward_weights', 'return_curriculum'):
                self.assertEqual(new_v[key], value, key)
        self.assertEqual(new['env']['physics'], old['env']['physics'])
        self.assertEqual(new['params']['network'], old['params']['network'])
        for key, value in old['params']['config'].items():
            if key not in ('name', 'description', 'max_epochs'):
                self.assertEqual(new['params']['config'][key], value, key)
        self.assertEqual(new['params']['config']['name'], 'high_level_2_2')
        self.assertEqual(new['params']['config']['max_epochs'], 300)
        obj, ball = self.make(config=new_v['return_curriculum'],
                              weights=new_v['reward_weights'].copy(), scales=new_v['reward_scales'].copy())
        # Without training progress updates, evaluation uses fixed final weights.
        self.assertEqual([obj.weights[k] for k in ('pos', 'contact', 'forward')], [.5, 1., 2.])
        for epoch, expected in [(0, [1., .25, 0.]), (10, [1., .325, 0.]),
                                (55, [.75, .6625, 1.]), (100, [.5, 1., 2.]), (200, [.5, 1., 2.])]:
            obj.set_progress(epoch * 2048 * 64)
            for key, value in zip(('pos', 'contact', 'forward'), expected):
                self.assertAlmostEqual(obj.weights[key], value, msg=f'{key} at epoch {epoch}')
            self.assertEqual([obj.weights[k] for k in ('net', 'court', 'target')], [0., 0., 0.])
        # Preserve the actual trained backward scale of 0.5.
        self.assertEqual(obj.backward_penalty_scale, .5)
        # Final contact + direction credit: -5 m/s -> 0.5; +2 m/s -> 1.4.
        events = self.step(obj, ball, contact=True,
                           velocity=torch.tensor([[0., -5., 0.], [0., 2., 0.]]))
        event_reward = obj.weights['contact'] * events[:, 0] + obj.weights['forward'] * events[:, 1]
        torch.testing.assert_close(event_reward, torch.tensor([.5, 1.4]))
        self.assertEqual(set(obj.schedule_metrics()), {'curriculum/weight_pos',
                                                     'curriculum/weight_contact', 'curriculum/weight_forward'})
        # Enable staged proposals explicitly; all reward changes remain zero up to 300.
        import copy
        proposal = copy.deepcopy(new_v['return_curriculum'])
        for name in ('vertical', 'net', 'court', 'target'):
            proposal['weights_schedule'][name]['enabled'] = True
        obj, _ = self.make(config=proposal, weights=new_v['reward_weights'].copy())
        for epoch, vertical, net in [(200, 0., 0.), (300, 0., 0.), (325, .1, .125),
                                     (350, .2, .25), (400, .2, .5), (425, .1, .75), (450, 0., 1.)]:
            obj.set_progress(epoch * 2048 * 64)
            self.assertAlmostEqual(obj.weights['vertical'], vertical)
            self.assertAlmostEqual(obj.weights['net'], net)

    def test_yaml_matches_run6_except_intentional_curriculum_and_finetuning(self):
        cfg_root = ROOT / 'cfg/controller'
        source = yaml.safe_load((cfg_root / 'usopen_train_stage_1/high_level_6.yaml').read_text())
        stage2 = yaml.safe_load((cfg_root / 'usopen_train_stage_2/high_level_2_1.yaml').read_text())
        v1, v2 = source['env']['vid2player'], stage2['env']['vid2player']
        for key, value in v1.items():
            if key not in ('reward_type', 'reward_weights', 'reward_scales', 'pos_reward_schedule'):
                self.assertEqual(v2[key], value, key)
        self.assertEqual(source['env']['physics'], stage2['env']['physics'])
        self.assertEqual(source['params']['network'], stage2['params']['network'])
        self.assertEqual(stage2['env']['episodeLength'], source['env']['episodeLength'])
        self.assertNotIn('pos_reward_schedule', v2)
        self.assertNotIn('net_height_m', v2['return_curriculum'])
        self.assertNotIn('ball_radius_m', v2['return_curriculum'])
        obj, _ = self.make(config=v2['return_curriculum'], weights=v2['reward_weights'], scales=v2['reward_scales'])
        obj.set_progress(200 * 2048 * 64)
        self.assertEqual(obj.weights['contact'], 1.)
        self.assertEqual(obj.weights['forward'], 1.)
        self.assertEqual(obj.weights['target'], 0.)
        self.assertEqual(obj.logged_components, ['pos', 'contact', 'forward'])
        config = stage2['params']['config']
        self.assertEqual(config['max_epochs'], 200)
        self.assertEqual(config['horizon_length'], 64)
        project = ROOT.parents[0]
        self.assertTrue((project / config['pretrained_model_cp']).is_file())


if __name__ == '__main__':
    unittest.main()
