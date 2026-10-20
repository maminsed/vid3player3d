"""Contact profile propagation and landing semantics, without Isaac Gym."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import yaml
import torch
from vid2player.utils.tennis_contacts import (
    resolve_tennis_contacts, contact_materials, ground_impact, apply_solver_config, racket_impact, preview_timing)
from vid2player.utils.tennis_materials import configure_ball_asset, configure_racket_asset
from vid2player.tests.test_tennis_materials import FakeGym

ROOT = Path(__file__).resolve().parents[1]


class TestTennisContacts(unittest.TestCase):
    def test_profile_and_legacy_fallback(self):
        profile = contact_materials({'tennis_contacts': {'model': 'hardcourt_v1'}, 'restitution': 0.})
        self.assertTrue(profile.enabled)
        self.assertAlmostEqual((profile.ball_restitution + profile.court_restitution) / 2, .77)
        self.assertAlmostEqual((profile.ball_restitution + profile.racket_restitution) / 2, .85)
        self.assertAlmostEqual((profile.ball_friction + profile.court_friction) / 2, .7)
        old = contact_materials({'restitution': .9, 'ball_restitution': 0., 'ball_friction': 0.})
        self.assertFalse(old.enabled)
        self.assertEqual(old.ball_restitution, 0.)
        self.assertEqual(old.ball_friction, 0.)
        self.assertEqual(old.racket_restitution, .9)
        self.assertFalse(resolve_tennis_contacts().enabled)

    def test_invalid_profiles_fail(self):
        for cfg in [False, {'model': 'typo'}, {'fricton': .5}, {'court_restitution': 1.1},
                    {'ball_friction': -1}, {'racket_contact_offset': 0}, {'court_friction': float('nan')}]:
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                resolve_tennis_contacts(cfg)

    def test_offsets_apply_to_ball_and_only_racket_head(self):
        gym = FakeGym()
        configure_ball_asset(gym, 'ball', .9, .4, .02)
        self.assertTrue(all(p.contact_offset == .02 and p.rest_offset == 0 for p in gym.props))
        gym = FakeGym()
        configure_racket_asset(gym, 'human', 1, .8, .6, .05)
        self.assertTrue(all(not hasattr(p, 'contact_offset') for p in gym.props[:4]))
        self.assertEqual(gym.props[4].contact_offset, .05)

    def test_rebound_event_and_ballistic_contact_position(self):
        radius, dt, hit_time = .0335, 1 / 60, .005
        previous = torch.zeros((4, 13), dtype=torch.float64)
        current = previous.clone()
        previous[0, 2] = radius + 5 * hit_time + .5 * 9.81 * hit_time**2
        previous[0, 7:10] = torch.tensor([20., 30., -5.])
        current[0, 2], current[0, 9] = radius + .77 * 5 * (dt-hit_time), 3.8
        previous[1, 2], previous[1, 9] = .1, -5.
        current[1, 2], current[1, 9] = .05, -5.1
        previous[2, 2], previous[2, 9] = 2., -5.
        current[2, 2], current[2, 9] = 2., 10.
        previous[3, 2], previous[3, 9] = .04, 3.
        current[3, 2], current[3, 9] = .08, 2.8
        hit, position = ground_impact(previous, current, radius, dt)
        self.assertEqual(hit.tolist(), [True, False, False, False])
        torch.testing.assert_close(position[0], torch.tensor([.1, .15, radius], dtype=torch.float64))
        self.assertTrue(torch.isfinite(position).all())

    def test_racket_region_event_uses_impulse_and_proximity(self):
        previous = torch.zeros((4, 13))
        current = previous.clone()
        previous[:, 1], current[:, 1] = .3, -.3
        previous[:, 2], current[:, 2] = 3., 3.
        racket = torch.tensor([[0., 0., 3.]]).repeat(4, 1)
        # A missed sensor sample with an impulse still counts, regardless of vy sign.
        previous[:, 8] = -36.
        current[:, 8] = -36.
        current[0, 7] = 10.
        # Opposing force components must not cancel the sensor test.
        force = torch.zeros((4, 3)); force[1] = torch.tensor([1., -1., 0.])
        # Smooth drag exactly explains this velocity change; do not infer impact.
        current[2, 8] = -41.
        expected = torch.zeros((4, 3)); expected[2, 1] = -5.
        # Strong body impulse far from the racket must be excluded.
        previous[3, 0], current[3, 0], current[3, 7] = 2., 2., 10.
        hit = racket_impact(previous, current, racket, racket, force, force, expected, .0335, .05, 1 / 60)
        self.assertEqual(hit.tolist(), [True, True, False, False])

    def test_preview_timing_preserves_30hz(self):
        for rate in [60, 240, 480, 960, 1920]:
            dt, inv = preview_timing(SimpleNamespace(dt=1 / rate))
            self.assertEqual(inv, rate // 30)
            self.assertAlmostEqual(dt * inv, 1 / 30)
        for dt in [0., float('nan'), .01]:
            with self.assertRaises(ValueError):
                preview_timing(SimpleNamespace(dt=dt))

    def test_all_stage_settings_and_low_level_forwarding(self):
        path = ROOT / 'env/utils/player_builder.py'
        definition = next(n for n in ast.parse(path.read_text()).body
                          if isinstance(n, ast.ClassDef) and n.name == 'PlayerBuilder')
        configs = []
        for stage in [1, 2, 3]:
            high = yaml.safe_load((ROOT / f'cfg/controller/usopen_train_stage_{stage}.yaml').read_text())
            high['env']['controller_cfg_name'] = 'test'
            low = {'env': {'plane': {}, 'asset': {}, 'controlFrequencyInv': 2}, 'sim': {'physx': {}}}
            ns = {'resolve_tennis_contacts': resolve_tennis_contacts,
                  'get_args': lambda: SimpleNamespace(test=False, play=False),
                  'load_cfg': lambda args: (low, {'params': {'config': {}}})}
            exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), 'exec'), ns)
            _, actual, _ = ns['PlayerBuilder'](SimpleNamespace(device_id=0), high).build_player_cfg()
            self.assertTrue(contact_materials(actual['env']['vid2player']).enabled)
            self.assertEqual(actual['env']['plane'], {'staticFriction': 1., 'dynamicFriction': 1., 'restitution': .5})
            generator = SimpleNamespace(dt=1 / 60, substeps=6, physx=SimpleNamespace())
            apply_solver_config(generator, high['env']['physics'])
            self.assertEqual(generator.substeps, actual['sim']['substeps'])
            self.assertEqual(generator.substeps, 12)
            dt, inv = preview_timing(generator)
            self.assertEqual(inv, 2)
            self.assertEqual(generator.physx.solver_type, 1)
            self.assertEqual(generator.physx.num_velocity_iterations, 0)
            self.assertEqual(generator.physx.contact_offset, .02)
            self.assertEqual(actual['env']['vid2player']['racket_dynamics']['model'], 'legacy')
            self.assertAlmostEqual(dt * actual['env']['controlFrequencyInv'], 1 / 30)
            for key, value in actual['sim']['physx'].items():
                self.assertEqual(getattr(generator.physx, key), value)
            configs.append((high['env']['physics'], high['env']['vid2player']['tennis_contacts']))
        self.assertTrue(all(c == configs[0] for c in configs))


if __name__ == '__main__':
    unittest.main()
