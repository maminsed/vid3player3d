"""Controller checks on CPU without importing IsaacGym or creating a simulator."""
import ast
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import unittest
import tempfile
import torch

from vid2player.utils.training_io import check_training_output, write_epoch_metrics
from vid2player.utils.torch_transform import rotmat_to_rot6d, rot6d_to_rotmat, rot6d_to_angle_axis
from vid2player.utils.konia_transform import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis

ROOT = Path(__file__).resolve().parents[1]


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            if isinstance(node, ast.FunctionDef):
                node.decorator_list = []
            selected.append(node)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def test_phase_free_rewards_and_latched_contact():
    ns = definitions(ROOT / 'env/tasks/physics_mvae_controller.py',
                     {'compute_reward_reach', 'compute_reward_return_w_estimate'},
                     {'torch': torch, 'math': math})
    zeros = torch.zeros(2, 3)
    phase = torch.tensor([0., math.pi])
    swing = torch.full((2,), -1, dtype=torch.long)
    reward, _, _ = ns['compute_reward_reach'](
        phase, torch.ones(2), zeros, zeros, swing,
        {'pos': 5., 'phase': 0.}, {'pos': 1.})
    assert torch.equal(reward, torch.ones(2))
    contact = torch.ones(2, dtype=torch.bool)
    target = torch.tensor([[0., 10., 0.]]).repeat(2, 1)
    reward, components, _ = ns['compute_reward_return_w_estimate'](
        zeros, phase, swing, zeros, contact, target, torch.ones(2),
        torch.tensor([True, False]), target, {'pos': 50., 'phase': 0.},
        {'pos': .1, 'ball_pos': .9})
    assert torch.equal(components[:, 0], torch.ones(2))
    assert math.isclose(reward[0].item(), .1 + .9 * math.exp(-.1), rel_tol=1e-6)
    assert math.isclose(reward[1].item(), .1, rel_tol=1e-6)


def check_phase_free_residual_preserves_decoded_pose(training):
    ns = definitions(ROOT / 'players/mvae_player.py', {'MVAEPlayer'}, {
        'torch': torch, 'math': math,
        'rotmat_to_rot6d': rotmat_to_rot6d, 'rot6d_to_rotmat': rot6d_to_rotmat,
        'rot6d_to_angle_axis': rot6d_to_angle_axis,
        'angle_axis_to_rotation_matrix': angle_axis_to_rotation_matrix,
        'rotation_matrix_to_angle_axis': rotation_matrix_to_angle_axis})
    player = object.__new__(ns['MVAEPlayer'])
    player.cfg = {'player': 'djokovic', 'add_residual_dof': 'euler'}
    player.num_envs, player.device = 2, 'cpu'
    player._is_train, player._is_dual, player._predict_phase = training, False, False
    player._conditions = torch.zeros(2, 1, 288)
    player._root_pos = torch.zeros(2, 3)
    player._root_vel = torch.zeros(2, 3)
    player._root_pos_inds, player._root_vel_inds = torch.arange(3), torch.arange(3, 6)
    player._joint_pos_inds, player._joint_rot6d_inds = torch.arange(6, 75), torch.arange(144, 288)
    player._residual_joint_inds = torch.tensor([19, 21])
    player._joint_rotmat, player._joint_rot6d = torch.zeros(2, 24, 3, 3), torch.zeros(2, 24, 6)
    player._joint_rot = torch.zeros(2, 24, 3)
    player._mvae = SimpleNamespace(
        opt=SimpleNamespace(pose_feature=['root_pos'], no_condition_root_y=False, condition_root_x_only=False),
        normalize=lambda x, inds=None: x, unnormalize=lambda x: x)
    rotations = torch.zeros(2, 24, 3)
    rotations[:, 19, 0], rotations[:, 21, 1] = .2, -.3
    base = angle_axis_to_rotation_matrix(rotations)
    feature = torch.zeros(2, 288)
    feature[:, 144:] = rotmat_to_rot6d(base).reshape(2, -1)
    player._update_mvae_state(feature.clone(), residual=torch.zeros(2, 3))
    assert torch.allclose(player._joint_rotmat, base, atol=1e-5)
    residual = torch.tensor([[.4, -.4, .4]]).repeat(2, 1)
    player._update_mvae_state(feature.clone(), residual=residual)
    expected = rotations.clone()
    expected[:, 19, 0] += .1 * math.pi
    expected[:, 21, 1] -= .1 * math.pi
    expected[:, 21, 2] += .1 * math.pi
    assert torch.allclose(player._joint_rotmat, angle_axis_to_rotation_matrix(expected), atol=1e-5)


def check_output_protection_and_metrics(tmp_path):
    cfg = {'protect_existing_checkpoints': True}
    args = SimpleNamespace(train=True, resume=False, checkpoint=None)
    (tmp_path / 'old.pth').write_text('existing')
    with unittest.TestCase().assertRaises(FileExistsError):
        check_training_output(cfg, args, str(tmp_path))
    args.resume, args.checkpoint = True, 'latest'
    check_training_output(cfg, args, str(tmp_path))
    values = []
    writer = SimpleNamespace(add_scalar=lambda *args: values.append(args))
    write_epoch_metrics(writer, {'reward': torch.tensor(.5), 'epoch': 1}, 64)
    assert values == [('reward', .5, 64), ('epoch', 1, 64)]
    assert not (tmp_path / 'metrics.jsonl').exists()


def test_event_weighted_metrics():
    ns = definitions(ROOT / 'agents/v2p_agent.py', {'summarize_task_stats'}, {})
    result = ns['summarize_task_stats']({
        'cycles': torch.tensor(10), 'hits': torch.tensor(7),
        'estimated_in': torch.tensor(5), 'landing_error_sum': torch.tensor(10.)})
    assert result['task/hit_rate'] == .7
    assert result['task/estimated_in_rate'] == .5
    assert result['task/estimated_landing_error_m'] == 2.
    assert 'task/detected_in_rate' not in result


def check_racket_geometry():
    asset = ET.parse(ROOT / 'data/assets/smpl_mesh_humanoid_djokovic.xml')
    head = asset.find(".//body[@name='Racket']/geom[2]")
    endpoints = torch.tensor([float(x) for x in head.attrib['fromto'].split()])
    normal = endpoints[3:] - endpoints[:3]
    normal /= normal.norm()
    torch.testing.assert_close(normal, torch.tensor([0., 1., 1.]) / math.sqrt(2))


def check_batched_mvae_decode():
    player_ns = definitions(ROOT / 'players/mvae_player.py', {'MVAEPlayer'}, {'torch': torch})
    decoder_ns = definitions(ROOT / 'motion_vae/model.py', {'MixedDecoder'}, {
        'torch': torch, 'nn': torch.nn, 'F': torch.nn.functional})
    torch.manual_seed(10)
    decoder = decoder_ns['MixedDecoder'](8, 10, 3, 16, 1, 1, 4).eval()
    # Strided inputs exercise dual-player batches as well as an uneven last chunk.
    latents, conditions = torch.randn(46, 3)[::2], torch.randn(46, 8)[::2]
    player = object.__new__(player_ns['MVAEPlayer'])
    player.cfg = {'mvae_inference_batch_size': 7}
    for predict_phase in (False, True):
        calls = []
        def forward(z, c):
            calls.append(len(z))
            output = decoder(z, c)
            return (output[:, :-2], output[:, -2:]) if predict_phase else (output, None)
        mvae = SimpleNamespace(forward=forward)
        expected, expected_phase = forward(latents, conditions)
        calls.clear()
        actual, phase = player._decode_mvae(mvae, latents, conditions)
        assert calls == [7, 7, 7, 2]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        assert not actual.requires_grad
        if predict_phase:
            torch.testing.assert_close(phase, expected_phase, rtol=1e-5, atol=1e-6)
        else:
            assert phase is None
    player.cfg['mvae_inference_batch_size'] = 0
    calls.clear()
    player._decode_mvae(mvae, latents, conditions)
    assert calls == [23]
    player.cfg['mvae_inference_batch_size'] = -1
    with unittest.TestCase().assertRaises(ValueError):
        player._decode_mvae(mvae, latents, conditions)


class ControllerBaselineTests(unittest.TestCase):
    def test_batched_decoder(self):
        check_batched_mvae_decode()

    def test_rewards(self):
        test_phase_free_rewards_and_latched_contact()

    def test_residual_training(self):
        check_phase_free_residual_preserves_decoded_pose(True)

    def test_residual_evaluation(self):
        check_phase_free_residual_preserves_decoded_pose(False)

    def test_output_and_logging(self):
        with tempfile.TemporaryDirectory() as directory:
            check_output_protection_and_metrics(Path(directory))

    def test_metrics(self):
        test_event_weighted_metrics()

    def test_racket_geometry(self):
        check_racket_geometry()


if __name__ == '__main__':
    unittest.main()
