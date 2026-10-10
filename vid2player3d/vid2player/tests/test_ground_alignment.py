import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from vid2player.utils.ground_alignment import reset_height_offset


class TestGroundAlignment(unittest.TestCase):
    def test_mesh_surface_and_chunked_shapes(self):
        batches = []

        def smpl(**kwargs):
            root = kwargs['root_trans']
            beta = kwargs['betas'][:, 0]
            batches.append(len(root))
            vertices = root[:, None, :].repeat(1, 2, 1)
            # Test distinct shapes: soles lie beta metres below the root.
            vertices[:, :, 2] -= beta[:, None]
            return SimpleNamespace(vertices=vertices)

        root = torch.tensor([[0., 0., .8], [0., 0., 1.5], [0., 0., .7]])
        betas = torch.zeros(3, 10)
        betas[:, 0] = torch.tensor([1., 1., .9])
        rotations = torch.eye(3).repeat(3, 24, 1, 1)
        original = root.clone()
        offset = reset_height_offset(smpl, root, rotations, betas, batch_size=2)
        self.assertTrue(torch.allclose(offset, torch.tensor([.202, 0., .202]), atol=1e-6))
        self.assertEqual(batches, [2, 1])
        self.assertTrue(torch.equal(root, original))

    def test_nonfinite_mesh_rejected(self):
        smpl = lambda **kwargs: SimpleNamespace(vertices=torch.full((1, 1, 3), float('nan')))
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            reset_height_offset(smpl, torch.zeros(1, 3), torch.eye(3).repeat(1, 24, 1, 1), torch.zeros(1, 10))

    def test_targets_retain_offset_after_partial_reset(self):
        path = Path(__file__).resolve().parents[1] / 'env/tasks/humanoid_smpl_im_mvae.py'
        tree = ast.parse(path.read_text())
        definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'HumanoidSMPLIMMVAE')
        method = next(n for n in definition.body if isinstance(n, ast.FunctionDef) and n.name == '_set_target_motion_state')
        ns = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), ns)
        raw = torch.tensor([[0., -13., .8], [0., -13., 1.]])
        offset = torch.tensor([.2, .1])
        target = torch.tensor([[0., -13., 1.], [0., -13., 1.1]])

        def convert(root, rotations, **kwargs):
            self.assertTrue(torch.allclose(root, target))
            # A constant offset must not introduce a spurious target velocity.
            velocity = (root - kwargs['prev_root_pos']) * 30
            self.assertTrue(torch.allclose(velocity, torch.zeros_like(root)))
            return root, root.clone(), root.clone(), velocity, velocity, velocity, root[:, None], root[:, None]

        task = SimpleNamespace(cfg_v2p={}, _reset_height_offset=offset,
                               _mvae_player=SimpleNamespace(_root_pos=raw, _joint_rotmat=None),
                               _prev_target_root_pos=target, _prev_target_rb_rot=None,
                               _smpl_to_sim=convert)
        ns['_set_target_motion_state'](task)
        self.assertTrue(torch.equal(raw, torch.tensor([[0., -13., .8], [0., -13., 1.]])))
        self.assertTrue(torch.allclose(task._target_root_pos, target))
        task._target_root_pos[0] = 9
        ns['_set_target_motion_state'](task, env_ids=torch.tensor([1]))
        self.assertTrue(torch.equal(task._target_root_pos[0], torch.full((3,), 9.)))
        self.assertTrue(torch.allclose(task._target_root_pos[1], target[1]))

    def test_simulator_feedback_returns_to_decoder_frame(self):
        path = Path(__file__).resolve().parents[1] / 'env/tasks/humanoid_smpl_im_mvae.py'
        tree = ast.parse(path.read_text())
        definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'HumanoidSMPLIMMVAE')
        method = next(n for n in definition.body if isinstance(n, ast.FunctionDef) and n.name == '_update_state_from_sim')
        feedback = next(n for n in method.body if isinstance(n, ast.If) and 'update_mvae_from_sim' in ast.dump(n.test))
        body = torch.zeros(2, 25, 3)
        body[:, 0, 2] = torch.tensor([1.1, 1.2])
        player = SimpleNamespace(
            _root_pos=torch.zeros(2, 3), _root_vel=torch.zeros(2, 3),
            _conditions=torch.zeros(2, 1, 6),
            _root_pos_inds=torch.arange(3), _root_vel_inds=torch.arange(3, 6),
            _mvae=SimpleNamespace(normalize=lambda value, indices: value))
        task = SimpleNamespace(cfg_v2p={'update_mvae_from_sim': True},
                               _mvae_player=player, _rigid_body_pos=body,
                               _reset_height_offset=torch.tensor([.2, .1]),
                               _humanoid_root_states=torch.zeros(2, 13), dt=1/30, num_envs=2)
        code = compile(ast.Module(body=[feedback], type_ignores=[]), str(path), 'exec')
        for _ in range(2):
            exec(code, {'self': task})
            self.assertTrue(torch.allclose(player._root_pos[:, 2], torch.tensor([.9, 1.1])))
            self.assertTrue(torch.allclose(player._conditions[:, 0, 2], torch.tensor([.9, 1.1])))
        self.assertTrue(torch.equal(body[:, 0, 2], torch.tensor([1.1, 1.2])))


if __name__ == '__main__':
    unittest.main()
