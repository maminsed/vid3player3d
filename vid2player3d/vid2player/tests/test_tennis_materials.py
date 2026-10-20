"""Material initialization regression checks without IsaacGym."""
import unittest
import ast
from pathlib import Path
from types import SimpleNamespace
from vid2player.utils.tennis_contacts import resolve_tennis_contacts

from vid2player.utils.tennis_materials import configure_ball_asset, configure_racket_asset


class FakeGym:
    def __init__(self):
        self.props = [SimpleNamespace(restitution=0., friction=1., compliance=0.)
                      for _ in range(5)]
        self.saved = None

    def get_asset_rigid_shape_properties(self, asset):
        return self.props

    def set_asset_rigid_shape_properties(self, asset, props):
        self.saved = (asset, props)

    def get_asset_rigid_body_shape_indices(self, asset):
        return [SimpleNamespace(start=0, count=3), SimpleNamespace(start=3, count=2)]


class TestTennisMaterials(unittest.TestCase):
    def test_ball_material_written_to_asset(self):
        gym = FakeGym()
        configure_ball_asset(gym, 'ball', restitution=0., friction=0.)
        self.assertEqual(gym.saved[0], 'ball')
        self.assertTrue(all(p.restitution == 0 and p.friction == 0 for p in gym.props))

    def test_only_racket_head_changes_with_multiple_body_shapes(self):
        gym = FakeGym()
        configure_racket_asset(gym, 'humanoid', 1, restitution=.9, friction=.8)
        self.assertEqual(gym.saved[0], 'humanoid')
        self.assertTrue(all(p.restitution == 0 and p.friction == 1 for p in gym.props[:4]))
        self.assertEqual(gym.props[4].restitution, .9)
        self.assertEqual(gym.props[4].friction, .8)

    def test_zero_physics_overrides_reach_low_level_config(self):
        path = Path(__file__).resolve().parents[1] / 'env/utils/player_builder.py'
        tree = ast.parse(path.read_text())
        definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PlayerBuilder')
        low_cfg = {'env': {'plane': {'restitution': .5}},
                   'sim': {'physx': {'contact_offset': .02}}}
        train = {'params': {'config': {}}}
        ns = {'resolve_tennis_contacts': resolve_tennis_contacts, 'get_args': lambda: SimpleNamespace(test=False, play=False),
              'load_cfg': lambda args: (low_cfg, train)}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), 'exec'), ns)
        cfg = {'env': {'numEnvs': 1, 'vid2player': {}, 'controller_cfg_name': 'test',
                       'physics': {'config': 'usopen_im', 'name': 'HumanoidSMPLIMMVAE',
                                   'plane_restitution': 0., 'contact_offset': 0.,
                                   'rest_offset': 0., 'bounce_threshold_velocity': 0.,
                                   'num_velocity_iterations': 0}}}
        builder = ns['PlayerBuilder'](SimpleNamespace(device_id=0), cfg)
        _, actual, _ = builder.build_player_cfg()
        self.assertEqual(actual['env']['plane']['restitution'], 0.)
        for key in ('contact_offset', 'rest_offset', 'bounce_threshold_velocity', 'num_velocity_iterations'):
            self.assertEqual(actual['sim']['physx'][key], 0.)


if __name__ == '__main__':
    unittest.main()
