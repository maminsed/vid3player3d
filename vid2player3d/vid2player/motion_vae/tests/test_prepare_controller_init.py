"""Regression checks for controller seed selection and feature compatibility."""
import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
import torch
from vid2player.motion_vae.dataset import Video3DPoseDataset
from vid2player.motion_vae.prepare_controller_init import (
    _feature, _save_pose, _select, prepare, _readiness_scores, READINESS_CONFIG)


class ControllerInitTests(unittest.TestCase):
    def setUp(self):
        self.dataset = object.__new__(Video3DPoseDataset)
        self.dataset.opt = SimpleNamespace(
            pose_feature=['root_pos', 'root_velo', 'joint_pos', 'joint_velo', 'joint_rotmat'],
            frame_size=288, condition_root_x_only=False, no_condition_root_y=False,
            predict_phase=False)
        self.dataset.std = self.dataset.avg = None
        positions = np.zeros((8, 72), dtype=np.float32)
        positions[:, 1] = -12
        positions[:, 2] = .9
        positions[:, 0] = [0, .2, .21, .25, 8, 8.1, 8.101, 8.11]
        self.ready_joints = np.zeros((24, 3), dtype=np.float32)
        for left, right, x, y, z in (
                (1, 2, .12, 0, 0), (4, 5, .15, .07, -.4),
                (7, 8, .18, 0, -.8), (10, 11, .18, .12, -.85),
                (13, 14, .1, 0, .5), (16, 17, .2, 0, .5),
                (18, 19, .16, .15, .3), (20, 21, .06, .25, .25),
                (22, 23, .06, .30, .25)):
            self.ready_joints[left] = [-x, y, z]
            self.ready_joints[right] = [x, y, z]
        for joint, z in ((3, .15), (6, .3), (9, .45), (12, .65), (15, .8)):
            self.ready_joints[joint, 2] = z
        positions[:, 3:] = self.ready_joints[1:].reshape(-1)
        self.dataset.joint_pos_arr = positions
        self.dataset.joint_rotmat_arr = np.tile(np.eye(3, dtype=np.float32), (8, 24, 1, 1))
        # SMPL +Z forward -> court +Y, SMPL +Y up -> court +Z.
        self.dataset.joint_rotmat_arr[:, 0] = [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]
        self.dataset.joint_quat_arr = np.tile([1., 0., 0., 0.], (8, 24, 1))
        self.dataset.valid_arr = np.ones(8, dtype=bool)
        self.dataset.sequences = [dict(base=0, length=4, _video_name='ready'),
                                  dict(base=4, length=4, _video_name='outside')]

    def devices(self):
        return ['cpu', 'cuda:0'] if torch.cuda.is_available() else ['cpu']

    def test_auto_score_and_chunk_boundaries(self):
        for device in self.devices():
            for batch in (1, 2, 65536):
                score, index, sequence = _select(self.dataset, set(), device, batch_size=batch)
                self.assertEqual(index, 2)
                self.assertEqual(sequence['_video_name'], 'ready')
                delta = self.dataset.joint_pos_arr[2] - self.dataset.joint_pos_arr[1]
                expected = np.sum(delta[:3] ** 2) + np.mean(delta[3:] ** 2)
                total, _, diagnostics = _readiness_scores(
                    torch.from_numpy(self.dataset.joint_pos_arr[2:3]),
                    torch.from_numpy(self.dataset.joint_pos_arr[1:2]),
                    torch.from_numpy(self.dataset.joint_rotmat_arr[2:3, 0]))
                self.assertAlmostEqual(float(diagnostics['motion_score'][0]), float(expected), places=7)
                self.assertAlmostEqual(score, float(total[0]), places=6)

    def test_ready_posture_beats_each_undesirable_pose(self):
        for device in self.devices():
            ready = torch.as_tensor(self.dataset.joint_pos_arr[2:3], device=device).clone()
            rotation = torch.as_tensor(self.dataset.joint_rotmat_arr[2:3, 0], device=device)
            base, _, _ = _readiness_scores(ready, ready, rotation)
            for defect in ('back_to_net', 'asymmetry', 'straight_knees',
                           'deep_squat', 'hands_at_sides', 'hands_behind', 'fast_motion'):
                pose, previous, facing = ready.clone(), ready.clone(), rotation.clone()
                joints = pose[:, 3:].reshape(-1, 23, 3)
                if defect == 'back_to_net':
                    facing[:, :2] *= -1
                elif defect == 'asymmetry':
                    joints[:, 18, 2] += .4  # right elbow
                elif defect == 'straight_knees':
                    joints[:, [3, 4], 1] = 0
                elif defect == 'deep_squat':
                    joints[:, [3, 4], 1] = .35
                elif defect == 'hands_at_sides':
                    joints[:, 19] = joints.new_tensor([-.6, 0, 0])
                    joints[:, 20] = joints.new_tensor([.6, 0, 0])
                elif defect == 'hands_behind':
                    joints[:, [19, 20], 1] = -.3
                else:
                    previous[:, 0] -= .2
                if defect != 'fast_motion':
                    previous = pose.clone()
                score, _, _ = _readiness_scores(pose, previous, facing)
                self.assertGreater(float(score[0]), float(base[0]), defect)

    def test_facing_net_handles_both_halves_and_symmetry_is_heading_relative(self):
        for device in self.devices():
            pose = torch.as_tensor(self.dataset.joint_pos_arr[2:3], device=device).clone()
            rotation = torch.as_tensor(self.dataset.joint_rotmat_arr[2:3, 0], device=device).clone()
            _, near_components, _ = _readiness_scores(pose, pose, rotation)
            # Rotate the entire pose 180 degrees into the far half.
            pose[:, 1] *= -1
            pose[:, 3:].reshape(-1, 23, 3)[:, :, :2] *= -1
            rotation[:, :2] *= -1
            _, far_components, diagnostics = _readiness_scores(pose, pose, rotation)
            self.assertEqual(float(diagnostics['facing_net_deg'][0]), 0)
            for key in READINESS_CONFIG['weights']:
                self.assertAlmostEqual(float(near_components[key][0]), float(far_components[key][0]), places=6)

    def test_automatic_selection_prioritizes_readiness_over_stillness(self):
        # The low-motion winner (frame 2) faces away; frame 3 has small motion
        # but the complete ready stance, and should win instead.
        self.dataset.joint_rotmat_arr[2, 0, :2] *= -1
        for device in self.devices():
            _, index, _ = _select(self.dataset, set(), device)
            self.assertEqual(index, 3)

    def test_named_start_bypasses_court_filter_but_requires_valid_predecessor(self):
        self.dataset.valid_arr[4] = False
        for device in self.devices():
            _, index, sequence = _select(self.dataset, set(), device, name='outside', batch_size=1)
            self.assertEqual(index, 6)
            self.assertEqual(sequence['_video_name'], 'outside')
            with self.assertRaisesRegex(ValueError, 'no finite, valid training transition'):
                _select(self.dataset, {'ready'}, device, name='outside')

    def test_nonfinite_predecessor_is_excluded(self):
        self.dataset.joint_pos_arr[0, 3] = np.nan
        for device in self.devices():
            _, index, _ = _select(self.dataset, set(), device, name='ready')
            self.assertEqual(index, 2)

    def test_feature_matches_dataset_including_quaternions(self):
        for quaternion in (False, True):
            if quaternion:
                self.dataset.opt.pose_feature.append('joint_quat')
                self.dataset.opt.frame_size += 96
            expected = self.dataset.get_window(1, 1)['feature']
            for device in self.devices():
                actual = _feature(self.dataset, 2, device)
                self.assertEqual(actual.device.type, torch.device(device).type)
                np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=1e-6, atol=1e-7)

    def test_named_start_skips_nonfinite_rotation_features(self):
        self.dataset.joint_rotmat_arr[1, 0, 0, 0] = np.nan
        for device in self.devices():
            _, index, _ = _select(self.dataset, set(), device, name='ready')
            self.assertEqual(index, 2)

    def test_skip_counts_valid_candidates_across_chunks_and_sequences(self):
        self.dataset.joint_rotmat_arr[1, 0, 0, 0] = np.nan
        self.dataset.valid_arr[5] = False
        self.dataset.sequences[1]['_video_name'] = 'ready'
        for device in self.devices():
            for batch in (1, 2, 65536):
                # Valid candidates are 2, 3, and 7. Invalid transitions and
                # sequence boundaries must not consume the skip count.
                _, index, _ = _select(self.dataset, set(), device, name='ready',
                                       skip=2, batch_size=batch)
                self.assertEqual(index, 7)
                with self.assertRaisesRegex(ValueError, 'after skipping 3'):
                    _select(self.dataset, set(), device, name='ready', skip=3,
                            batch_size=batch)

    def test_skip_requires_name_and_nonnegative_integer(self):
        for skip in (0, 1):
            with self.assertRaisesRegex(ValueError, 'requires --name'):
                prepare('usopen', 'unused.pt', skip=skip, device='cpu')
        for skip in (-1, 1.5, True):
            with self.assertRaisesRegex(ValueError, 'nonnegative integer'):
                prepare('usopen', 'unused.pt', name='ready', skip=skip, device='cpu')

    def test_centring_translates_root_only(self):
        expected = self.dataset.get_window(1, 1)['feature'].copy()
        expected[:, 0] = 0
        original_positions = self.dataset.joint_pos_arr.copy()
        for device in self.devices():
            actual = _feature(self.dataset, 2, device, centre_x=True)
            np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(self.dataset.joint_pos_arr, original_positions)

    def test_saved_tensor_metadata_and_plot_use_centred_skipped_pose(self):
        opt = self.dataset.opt
        opt.load = Mock()
        opt.num_condition_frames = 1
        opt.checkpoint_dir = 'unused'
        opt.model_ver = 'test'
        opt.dataset_dir = 'unused'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'init.pt'
            with patch('vid2player.motion_vae.prepare_controller_init.MotionVAEOption', return_value=opt), \
                 patch('vid2player.motion_vae.prepare_controller_init.Video3DPoseDataset', return_value=self.dataset), \
                 patch('vid2player.motion_vae.prepare_controller_init.torch.load', return_value={}), \
                 patch('vid2player.motion_vae.prepare_controller_init._save_pose', wraps=_save_pose) as render:
                prepare('test', path, name='ready', device='cpu', centre_x=True, skip=1)
            expected = self.dataset.get_window(1, 1)['feature'].copy()
            expected[:, 0] = 0
            np.testing.assert_allclose(torch.load(path)[0].numpy(), expected, rtol=1e-6, atol=1e-7)
            metadata = json.loads(path.with_suffix('.json').read_text())
            self.assertEqual(metadata['frame_index'], 2)
            self.assertEqual(metadata['skipped_valid_frames'], 1)
            self.assertTrue(metadata['centre_x'])
            self.assertEqual(metadata['root_xyz_m'][0], 0)
            self.assertAlmostEqual(metadata['original_root_xyz_m'][0], .21)
            self.assertEqual(render.call_args[0][0][0].item(), 0)
            self.assertTrue(path.with_suffix('.png').exists())

    def test_pose_png_and_existing_sidecar_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'init.png'
            _save_pose(torch.from_numpy(self.dataset.joint_pos_arr[2]), path, 'Test pose')
            self.assertEqual(path.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            with self.assertRaises(FileExistsError):
                prepare('usopen', path.with_suffix('.pt'), device='cpu')


if __name__ == '__main__':
    unittest.main()
