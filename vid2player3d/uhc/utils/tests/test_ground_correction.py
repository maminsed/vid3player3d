"""Regression checks for shared correction and SMPL/pelvis coordinates.

Run from vid2player3d with:
    python -m unittest discover -s uhc/utils/tests -v
"""

import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import cv2
import joblib
import pandas as pd
from scipy.spatial.transform import Rotation
import torch

from uhc.utils import convert_amass_isaac_correct_ground as ground

PROJECT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_DIR.parent))
from vid2player3d.poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonTree


def camera_fixture():
    center = np.array([1., -30., 10.])
    forward = -center / np.linalg.norm(center)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    rotation = np.stack((right, np.cross(forward, right), forward))
    intrinsic = np.array([[700., 0., 320.], [0., 700., 180.], [0., 0., 1.]])
    return rotation, -rotation @ center, intrinsic


def project(points, rotation, translation, intrinsic):
    points = (points @ rotation.T + translation) @ intrinsic.T
    return points[:, :2] / points[:, 2:]


class CourtCameraTests(unittest.TestCase):
    def test_camera_pose_and_elevated_hip_reconstruction(self):
        r, t, k = camera_fixture()
        court = ground.court_points_meters()
        pixels = project(court, r, t, k)
        forward, _ = cv2.findHomography(ground.TP_COURT_POINTS, pixels)
        recovered_r, recovered_t, error = ground.estimate_court_camera(np.linalg.inv(forward), k, pixels)
        np.testing.assert_allclose(recovered_r, r, atol=2e-6)
        np.testing.assert_allclose(recovered_t, t, atol=2e-5)
        self.assertLess(error, 1e-4)
        hips = np.array([[2., -12., 1.], [-1., -10., 1.8], [0., 5., 0.8]])
        hip_pixels = project(hips, r, t, k)
        recovered = ground.intersect_rays_at_height(
            hip_pixels, np.tile(k, (3, 1, 1)), np.tile(recovered_r, (3, 1, 1)),
            np.tile(recovered_t, (3, 1)), hips[:, 2])
        np.testing.assert_allclose(recovered, hips, atol=1e-5)
        naive = cv2.perspectiveTransform(hip_pixels[:, None], np.linalg.inv(forward))[:, 0]
        naive = (naive - ground.TP_NET_CENTER) * ground.TP_METERS_PER_PIXEL
        self.assertGreater(np.linalg.norm(naive[0] - hips[0, :2]), 1.)

    def test_heading_aligns_axes_without_reflection_or_tilt(self):
        r, _, _ = camera_fixture()
        yaw = Rotation.from_euler('z', 1.2).as_matrix()
        original = Rotation.from_euler('xyz', [0.2, -0.3, 0.5]).as_matrix()
        incam = r @ yaw @ original
        rotations, _, residual = ground.court_heading_rotations(r[None], incam[None], original[None])
        np.testing.assert_allclose(rotations[0], yaw, atol=1e-12)
        np.testing.assert_allclose(rotations[0] @ [0., 0., 1.], [0., 0., 1.])
        self.assertAlmostEqual(np.linalg.det(rotations[0]), 1.)
        self.assertAlmostEqual(residual[0], 0.)

    def test_bad_calibration_and_behind_camera_rays_are_rejected(self):
        r, t, k = camera_fixture()
        with self.assertRaises(ValueError):
            ground.estimate_court_camera(np.zeros((3, 3)), k)
        with self.assertRaises(ValueError):
            ground.intersect_rays_at_height(np.array([[320., 180.]]), k[None], r[None], t[None], np.array([20.]))

    def test_video_name_normalization_keeps_dots(self):
        name = 'Arthur_Rinderknech_vs._Carlos_Alcaraz-scene007-000'
        self.assertEqual(ground.normalized_video_name('res_2/0-' + name + '.mp4'), name)
        self.assertEqual(ground.normalized_video_name(name), name)


class InputAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        (self.directory / 'preprocess').mkdir()
        self.name = 'Player_vs._Opponent-scene001-000'
        self.csv = self.directory / (self.name + '.csv')
        count, start = 4, 7
        r, t, k = camera_fixture()
        pixels = project(ground.court_points_meters(), r, t, k)
        h, _ = cv2.findHomography(pixels, ground.TP_COURT_POINTS)
        self.rows = pd.DataFrame({
            'global_frame': np.arange(start, start + count),
            'local_frame': np.arange(count),
            'output_video_name': ['res_2/0-' + self.name + '.mp4'] * count,
            'inv_matrix': [np.array2string(h, precision=16)] * count,
            'court_kps': [repr((pixels * 2).tolist())] * count,
        })
        self.rows.iloc[::-1].to_csv(self.csv, index=False)
        source = dict(path='/source/' + self.name + '.mp4', start_frame=start, end_frame=start + count, fps=30.)
        (self.directory / '0_input_video.json').write_text(json.dumps({'source': source}))
        export = Rotation.from_euler('x', np.pi / 2)
        self.entry = dict(pose_aa=np.zeros((count, 66)), trans_orig=np.tile([0., 0., .92], (count, 1)),
                          beta=np.zeros(10), gender='neutral', fps=30.)
        self.entry['pose_aa'][:, :3] = export.as_rotvec()
        self.amass = self.directory / (self.name + '_amass.pkl')
        joblib.dump({'demo_' + self.name: self.entry}, self.amass)
        full_k = k.copy()
        full_k[:2] *= 2
        results = dict(K_fullimg=torch.from_numpy(np.tile(full_k, (count, 1, 1))),
                       smpl_params_incam={'global_orient': torch.from_numpy(np.tile(
                           Rotation.from_matrix(r @ export.as_matrix()).as_rotvec(), (count, 1)))},
                       smpl_params_global=dict(global_orient=torch.zeros(count, 3), transl=torch.zeros(count, 3),
                                               body_pose=torch.zeros(count, 63), betas=torch.zeros(count, 10)))
        torch.save(results, str(self.directory / 'hmr4d_results.pt'))
        pose = torch.ones(count, 17, 3)
        pose[:, :, :2] = torch.tensor([600., 450.])
        torch.save(pose, str(self.directory / 'preprocess/vitpose.pt'))
        values = {cv2.CAP_PROP_FRAME_WIDTH: 1280, cv2.CAP_PROP_FRAME_HEIGHT: 720,
                  cv2.CAP_PROP_FRAME_COUNT: count, cv2.CAP_PROP_FPS: 30}
        video = Mock()
        video.get.side_effect = values.__getitem__
        patcher = patch('cv2.VideoCapture', return_value=video)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_exact_json_frame_join_resolution_and_path_prefix(self):
        _, context, metadata = ground.load_correction_inputs(self.directory, self.csv)
        self.assertEqual(metadata['camera']['source_frames'], [7, 8, 9, 10])
        np.testing.assert_allclose(context['hip_pixels'], np.tile([300., 225.], (4, 1)))
        np.testing.assert_allclose(context['intrinsic'][0], camera_fixture()[2])

    def test_missing_and_duplicate_frames_fail(self):
        for rows in (self.rows.iloc[:-1], pd.concat([self.rows, self.rows.iloc[:1]])):
            rows.to_csv(self.csv, index=False)
            with self.assertRaisesRegex(ValueError, 'exactly one row'):
                ground.load_correction_inputs(self.directory, self.csv)

    def test_stale_motion_artifact_fails(self):
        self.entry['trans_orig'][1, 0] += 1
        joblib.dump({'demo_' + self.name: self.entry}, self.amass)
        with self.assertRaisesRegex(ValueError, 'same exported motion'):
            ground.load_correction_inputs(self.directory, self.csv)


class GroundCorrectionTests(unittest.TestCase):
    def make_sequence(self, **kwargs):
        # A nonzero offset on every axis catches accidental use of pelvis
        # coordinates as SMPL translation, including horizontal corrections.
        offset = np.array([0.12, -0.08, 0.25], dtype=np.float32)
        translations = np.tile([0.4, 0.6, 1.2], (30, 1)).astype(np.float32)
        entry = {
            'pose_aa': np.zeros((30, 72), dtype=np.float32),
            'trans_orig': translations,
            'beta': np.zeros(10, dtype=np.float32),
            'gender': 'neutral',
        }
        local_translations = torch.zeros(24, 3)
        local_translations[0] = torch.from_numpy(offset)
        tree = SkeletonTree(
            ['Pelvis'] + [f'joint_{i}' for i in range(23)],
            torch.tensor([-1] + [0] * 23), local_translations,
        )
        # Stand-in geometry avoids requiring licensed SMPL assets or MuJoCo.
        template = torch.zeros(6890, 3)
        template[:, 2] = torch.linspace(-0.9, 0.9, 6890)

        joints = torch.from_numpy(np.tile(offset, (24, 1)))
        joints[1, :2] += torch.tensor([-0.1, 0.03])
        joints[2, :2] += torch.tensor([0.1, 0.03])
        joints[1:3, 2] -= 0.07

        def mesh(pose, th_betas, th_trans):
            rotations = torch.from_numpy(Rotation.from_rotvec(pose[:, :3]).as_matrix()).to(th_trans)
            def transform(points):
                return torch.einsum('nij,vj->nvi', rotations, (points - offset).to(th_trans)) + offset + th_trans[:, None, :]
            return transform(template), transform(joints)

        robot = SimpleNamespace(
            smpl_parser_n=SimpleNamespace(get_joints_verts=mesh),
            load_from_skeleton=Mock(), write_xml=Mock(), model_xml_path='unused.xml',
        )
        with patch.object(SkeletonTree, 'from_mjcf', return_value=tree):
            result = ground.correct_smpl_sequence(entry, robot, plot_results=False, **kwargs)
        np.testing.assert_array_equal(entry['trans_orig'], translations)
        return result, offset

    def test_mesh_and_skeleton_receive_same_displacement(self):
        sequence, offset = self.make_sequence()
        delta = sequence['corrected_root_trans'] - sequence['root_trans']
        expected_mesh = sequence['verts'].numpy() + delta[:, None, :]
        np.testing.assert_allclose(sequence['corrected_verts'], expected_mesh, atol=2e-7)
        np.testing.assert_allclose(
            sequence['corrected_trans'] + offset,
            sequence['corrected_root_trans'], atol=2e-7,
        )
        np.testing.assert_allclose(sequence['corrected_verts'][..., 2].min(dim=1)[0], 0, atol=2e-7)

    def test_horizontal_correction_rotates_body_and_preserves_z(self):
        baseline, _ = self.make_sequence()
        r, t, k = camera_fixture()
        target_root = baseline['corrected_root_trans'].copy()
        target_root[:, :2] = [2., -12.]
        yaw = Rotation.from_euler('z', np.pi / 2).as_matrix()
        hip_relative = np.array([0., 0.03, -0.07]) @ yaw.T
        pixels = project(target_root + hip_relative, r, t, k)
        context = dict(enabled=True, yaw_rotation=np.tile(yaw, (30, 1, 1)),
                       intrinsic=np.tile(k, (30, 1, 1)), camera_rotation=np.tile(r, (30, 1, 1)),
                       camera_translation=np.tile(t, (30, 1)), hip_pixels=pixels)
        sequence, offset = self.make_sequence(xy_context=context)
        np.testing.assert_allclose(sequence['corrected_root_trans'], target_root, atol=2e-7)
        np.testing.assert_allclose(sequence['corrected_verts'][..., 2], baseline['corrected_verts'][..., 2], atol=2e-7)
        expected = (sequence['verts'].numpy() - sequence['root_trans'][:, None]) @ yaw.T + target_root[:, None]
        np.testing.assert_allclose(sequence['corrected_verts'], expected, atol=5e-7)
        np.testing.assert_allclose(sequence['pose_aa'][:, 2], np.pi / 2)
        motion, _ = ground.build_motion_output(sequence, 'camera', 0, 0)
        state = SkeletonMotion.from_dict(motion)
        expected_quat = Rotation.from_matrix(yaw).as_quat()
        np.testing.assert_allclose(state.global_rotation[:, 0], np.tile(expected_quat, (30, 1)), atol=1e-6)
        np.testing.assert_allclose(sequence['corrected_trans'] + offset, target_root, atol=1e-7)

    def test_serialized_motion_and_render_mesh_keep_every_frame(self):
        sequence, offset = self.make_sequence()
        motion, render = ground.build_motion_output(sequence, 'example', 0, 0)
        reconstructed = SkeletonMotion.from_dict(motion)
        np.testing.assert_allclose(motion['trans'] + offset, motion['root_trans'], atol=2e-7)
        np.testing.assert_allclose(reconstructed.root_translation, motion['root_trans'])
        np.testing.assert_allclose(render['verts'], sequence['corrected_verts'])
        self.assertEqual(len(motion['pose_aa']), 30)
        self.assertAlmostEqual(motion['min_verts_h'], 0, places=6)

    def test_imports_ignore_cli_arguments_and_do_not_load_simulation(self):
        code = '''
import sys
sys.argv = ['caller', '--unrelated-option']
from uhc.utils import convert_amass_isaac_correct_ground as ground
from uhc.utils import plot_ground_contacts_simple as plot
assert plot.correct_smpl_sequence is ground.correct_smpl_sequence
assert plot.get_foot_vertices is ground.get_foot_vertices
assert plot.detect_ground_contacts_with_feet is ground.detect_ground_contacts_with_feet
assert 'mujoco_py' not in sys.modules
assert 'isaacgym' not in sys.modules
assert 'torch' not in sys.modules
'''
        env = dict(os.environ, PYTHONPATH=str(PROJECT_DIR), MPLBACKEND='Agg')
        with tempfile.TemporaryDirectory() as directory:
            env['MPLCONFIGDIR'] = str(Path(directory) / 'matplotlib')
            subprocess.run([sys.executable, '-c', code], cwd=directory, env=env, check=True)
            self.assertFalse((Path(directory) / 'data').exists())

    def test_diagnostic_plot_uses_explicit_output_directory(self):
        sequence, _ = self.make_sequence()
        with tempfile.TemporaryDirectory() as directory:
            ground.correct_ground_height(
                sequence['verts'].numpy(), sequence['root_trans'],
                sequence_name='folder/example', out_dir=directory, show_plots=False,
            )
            self.assertTrue((Path(directory) / 'ground_correction_folder_example.png').is_file())


if __name__ == '__main__':
    unittest.main()
