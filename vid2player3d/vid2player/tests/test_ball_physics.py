"""Physics invariants and legacy compatibility; runnable without Isaac Gym."""
import math
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
from vid2player.utils.tennis_contacts import resolve_tennis_contacts
import unittest
import xml.etree.ElementTree as ET

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vid2player.utils.ball_physics import (
    BALL_MASS_KG, ball_aerodynamic_force, configure_ball_options,
    resolve_ball_physics, validate_ball_asset, resolve_ball_asset,
)


class BallPhysicsTests(unittest.TestCase):
    def setUp(self):
        self.settings = resolve_ball_physics({'model': 'aerodynamic_v1'})
        self.v = torch.tensor([[0., 30., 0.]], dtype=torch.float64)
        self.zero = torch.zeros_like(self.v)

    def lift(self, omega, velocity=None):
        v = self.v if velocity is None else velocity
        return (ball_aerodynamic_force(v, omega, self.settings)
                - ball_aerodynamic_force(v, torch.zeros_like(v), self.settings))

    def test_top_back_and_sidespin(self):
        spin = 8 * 2 * math.pi
        top = self.lift(torch.tensor([[-spin, 0., 0.]], dtype=self.v.dtype))
        back = self.lift(torch.tensor([[spin, 0., 0.]], dtype=self.v.dtype))
        side = self.lift(torch.tensor([[0., 0., spin]], dtype=self.v.dtype))
        self.assertLess(top[0, 2], 0)
        torch.testing.assert_close(top, -back)
        self.assertLess(side[0, 0], 0)
        self.assertEqual(side[0, 2], 0)

    def test_si_magnitude(self):
        omega = torch.tensor([[-16*math.pi, 0., 0.]], dtype=torch.float64)
        surface_speed = self.settings.radius * 16 * math.pi
        cl = surface_speed / (30 + 2 * surface_speed)
        expected = self.settings.force_factor * cl * 30**2
        self.assertAlmostEqual(float(self.lift(omega)[0, 2]), -expected, places=12)

    def test_parallel_spin_and_parallel_component(self):
        torch.testing.assert_close(self.lift(self.v * 5), self.zero)
        transverse = torch.tensor([[-40., 0., 0.]], dtype=torch.float64)
        torch.testing.assert_close(self.lift(transverse), self.lift(transverse + self.v * 5))

    def test_stationary_ball_is_finite_and_has_no_aerodynamic_force(self):
        for model in ['legacy', 'aerodynamic_v1']:
            for omega in [self.zero, self.v * 100]:
                force = ball_aerodynamic_force(self.zero, omega, {'model': model})
                self.assertTrue(torch.isfinite(force).all())
                torch.testing.assert_close(force, self.zero)

    def test_drag_opposes_velocity_and_lift_does_no_work(self):
        gen = torch.Generator().manual_seed(123)
        v = torch.randn(100, 3, generator=gen, dtype=torch.float64) * 30
        omega = torch.randn(100, 3, generator=gen, dtype=torch.float64) * 100
        drag = ball_aerodynamic_force(v, torch.zeros_like(v), self.settings)
        lift = self.lift(omega, v)
        self.assertTrue(((drag * v).sum(-1) <= 0).all())
        torch.testing.assert_close((lift * v).sum(-1), torch.zeros(100, dtype=torch.float64), atol=1e-12, rtol=0)

    def test_rotation_invariance_including_vertical_flight(self):
        # An orthogonal rotation exchanging horizontal and vertical flight.
        rotation = torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]], dtype=torch.float64)
        v = torch.tensor([[30., 0., 0.]], dtype=torch.float64)
        omega = torch.tensor([[0., 80., 0.]], dtype=torch.float64)
        f = ball_aerodynamic_force(v, omega, self.settings)
        rotated = ball_aerodynamic_force(v @ rotation.T, omega @ rotation.T, self.settings)
        torch.testing.assert_close(rotated, f @ rotation.T)
        self.assertGreater(float(self.lift(omega, v @ rotation.T).norm()), 0)

    def test_corrected_ignores_legacy_scaling_and_sign_bookkeeping(self):
        omega = self.v.roll(1, -1)
        first = ball_aerodynamic_force(self.v, omega, self.settings, spin_scale=1)
        second = ball_aerodynamic_force(self.v, omega, self.settings, spin_scale=500,
                                        legacy_spin_sign=torch.tensor([-1.]))
        torch.testing.assert_close(first, second)

    def test_legacy_matches_original_live_and_generator_equations(self):
        gen = torch.Generator().manual_seed(456)
        v = torch.randn(100, 3, generator=gen, dtype=torch.float64) * 30
        omega = torch.randn(100, 3, generator=gen, dtype=torch.float64) * 60
        for signs in [None, torch.where(torch.arange(100) % 2 == 0, 1., -1.)]:
            for scale in [1., 5.]:
                speed = v.norm(dim=1, keepdim=True)
                direction = v / speed
                spin = omega.norm(dim=1, keepdim=True) / (2 * math.pi)
                if signs is not None:
                    spin *= signs[:, None]
                cl = 1 / (2 + abs(speed / (spin * scale + 1e-6)))
                cl *= torch.where(spin > 0, -torch.ones_like(cl), torch.ones_like(cl))
                factor = 1.21 * math.pi * .032**2 / 2
                down = torch.tensor([0., 0., -1.], dtype=v.dtype).expand_as(v)
                expected = (-factor * .55 * speed * v - factor * cl * speed**2
                            * torch.cross(torch.cross(direction, down, dim=-1), direction, dim=-1))
                actual = ball_aerodynamic_force(v, omega, spin_scale=scale, legacy_spin_sign=signs)
                torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)

    def test_config_toggle_restores_legacy_asset_defaults(self):
        default = SimpleNamespace(angular_damping=.5, max_angular_velocity=64.)
        legacy = resolve_ball_physics({'model': 'legacy', 'angular_damping': 0., 'max_angular_velocity': 1000.})
        configure_ball_options(default, legacy)
        self.assertEqual((default.angular_damping, default.max_angular_velocity), (.5, 64.))
        self.assertEqual(legacy.radius, .032)
        self.assertEqual(legacy.asset_file, 'tennis_ball.urdf')
        configure_ball_options(default, self.settings)
        self.assertEqual((default.angular_damping, default.max_angular_velocity), (0., 1000.))

    def test_corrected_asset_geometry_and_inertia_match_model(self):
        assets, asset_file = resolve_ball_asset('unused/legacy/path', self.settings)
        assets = Path(assets)
        validate_ball_asset(assets, asset_file, self.settings)
        xml = ET.parse(assets / self.settings.asset_file).getroot()
        self.assertEqual(float(xml.find('.//visual/geometry/sphere').get('radius')), self.settings.radius)
        expected_inertia = 2/3 * BALL_MASS_KG * self.settings.radius**2
        for axis in ['ixx', 'iyy', 'izz']:
            self.assertAlmostEqual(float(xml.find('.//inertia').get(axis)), expected_inertia, places=12)
        with self.assertRaises(FileNotFoundError):
            resolve_ball_asset('unused/custom/path', self.settings, 'missing.urdf')
        self.assertEqual(resolve_ball_asset('legacy/path', resolve_ball_physics()),
                         ('legacy/path', 'tennis_ball.urdf'))

    def test_finite_wall_v2_inertia(self):
        settings = resolve_ball_physics({'model': 'aerodynamic_v2'})
        root, name = resolve_ball_asset('unused', settings)
        xml = ET.parse(Path(root) / name).getroot()
        self.assertTrue(settings.corrected)
        self.assertAlmostEqual(float(xml.find('.//inertia').get('ixx')),
                               .55 * BALL_MASS_KG * settings.radius**2, places=13)
        velocity = torch.tensor([[0., 30., 0.]])
        omega = torch.tensor([[-50., 0., 0.]])
        torch.testing.assert_close(ball_aerodynamic_force(velocity, omega, settings),
                                   ball_aerodynamic_force(velocity, omega, self.settings))

    def test_stage_configs_select_corrected_model(self):
        for stage in [1, 2, 3]:
            config = yaml.safe_load((ROOT / f'vid2player/cfg/controller/usopen_train_stage_{stage}.yaml').read_text())
            settings = resolve_ball_physics(config['env']['vid2player']['ball_physics'])
            self.assertTrue(settings.corrected)
            self.assertEqual(settings.angular_damping, 0.)
            self.assertGreater(settings.max_angular_velocity, 80 * 2 * math.pi)

    def test_high_level_toggle_reaches_low_level_player(self):
        path = ROOT / 'vid2player/env/utils/player_builder.py'
        definition = next(n for n in ast.parse(path.read_text()).body
                          if isinstance(n, ast.ClassDef) and n.name == 'PlayerBuilder')
        for model in ['legacy', 'aerodynamic_v1']:
            block = {'model': model, 'angular_damping': 0., 'max_angular_velocity': 1000.}
            low = {'env': {}, 'sim': {'physx': {}}}
            ns = {'resolve_tennis_contacts': resolve_tennis_contacts, 'get_args': lambda: SimpleNamespace(test=False, play=False),
                  'load_cfg': lambda args: (low, {'params': {'config': {}}})}
            exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), 'exec'), ns)
            high = {'env': {'numEnvs': 1, 'controller_cfg_name': 'test',
                            'vid2player': {'ball_physics': block},
                            'physics': {'config': 'usopen_im', 'name': 'HumanoidSMPLIMMVAE'}}}
            builder = ns['PlayerBuilder'](SimpleNamespace(device_id=0), high)
            _, actual, _ = builder.build_player_cfg()
            received = resolve_ball_physics(actual['env']['vid2player']['ball_physics'])
            self.assertEqual(received.model, model)

    def test_invalid_config_is_rejected(self):
        for config in [False, {'model': 'typo'}, {'radius': .1}, {'angular_damping': -.1},
                       {'angular_damping': float('nan')}, {'max_angular_velocity': 0}]:
            with self.assertRaises(ValueError):
                resolve_ball_physics(config)


if __name__ == '__main__':
    unittest.main()
