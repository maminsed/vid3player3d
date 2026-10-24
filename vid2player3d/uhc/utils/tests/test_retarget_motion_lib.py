"""CPU coverage for geometry, conditioning, and serialized multi-part libraries."""
from pathlib import Path
from contextlib import redirect_stderr
import io
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'vid2player3d'))

# The saved MotionLib imports Isaac Gym; initialize it before torch.
from isaacgym import gymapi  # noqa: F401
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from vid2player3d.uhc.utils.retarget_motion_lib import (
    BODY_NAMES, continuous_quaternions, contact_probabilities, fk, load_asset,
    main, resolve_settings, rotation_steps, SkinMetrics,
)
from vid2player3d.embodied_pose.utils.motion_lib import MotionLib
from vid2player3d.poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonMotion


class RetargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.settings = resolve_settings('djokovic_published_v1')
        cls.tree, _, _, _ = load_asset(cls.settings['asset_path'], 'cpu')

    def test_requested_racket_and_body_conditioning(self):
        self.assertEqual(self.settings['racket_dynamics']['model'], 'published_reference_v1')
        self.assertEqual(self.settings['racket_dynamics']['mass'], .270)
        self.assertEqual(self.settings['gender'], 'male')
        self.assertAlmostEqual(self.settings['effective_physics']['control_dt_s'], 1. / 30.)

    def test_quaternion_signs_do_not_create_motion(self):
        q = np.tile([0., 0., 0., 1.], (10, 24, 1))
        q[::2] *= -2
        normalized = continuous_quaternions(q)
        np.testing.assert_allclose(np.linalg.norm(normalized, axis=-1), 1)
        np.testing.assert_allclose(rotation_steps(normalized)[1:], 0)
        with self.assertRaises(ValueError):
            continuous_quaternions(np.zeros_like(q))

    def test_fk_matches_independent_matrix_composition(self):
        rng = np.random.default_rng(3)
        local = Rotation.from_rotvec(rng.normal(0, .3, (48, 3)))
        q = local.as_quat().reshape(2, 24, 4)
        matrices = local.as_matrix().reshape(2, 24, 3, 3)
        root = rng.normal(size=(2, 3))
        offsets = self.tree.local_translation.numpy()
        parents = self.tree.parent_indices.numpy()
        expected, world = np.zeros((2, 24, 3)), np.zeros_like(matrices)
        expected[:, 0], world[:, 0] = root, matrices[:, 0]
        for joint in range(1, 24):
            p = parents[joint]
            expected[:, joint] = expected[:, p] + np.einsum('nij,j->ni', world[:, p], offsets[joint])
            world[:, joint] = world[:, p] @ matrices[:, joint]
        pos, rot = fk(torch.tensor(q), torch.tensor(root), self.tree.local_translation, parents)
        np.testing.assert_allclose(pos.numpy(), expected, atol=1e-6)
        np.testing.assert_allclose(Rotation.from_quat(rot.numpy().reshape(-1, 4)).as_matrix(),
                                   world.reshape(-1, 3, 3), atol=1e-6)

    def test_contacts_saved_or_explicitly_inferred(self):
        positions = np.zeros((6, 24, 3))
        inferred, label = contact_probabilities({}, 'clip', positions, 30, self.settings)
        self.assertIn('inferred', label)
        self.assertTrue(inferred.all())
        saved = {'sequences': {'clip': {'ik': {'stationary_probability': np.zeros((6, 4))}}}}
        probabilities, label = contact_probabilities(saved, 'clip', positions, 30, self.settings)
        self.assertIn('saved', label)
        self.assertFalse(probabilities.any())
        saved['sequences']['clip']['ik']['stationary_probability'] = np.ones((5, 4))
        with self.assertRaises(ValueError):
            contact_probabilities(saved, 'clip', positions, 30, self.settings)

    def test_cpu_conversion_preserves_parts_frames_identity_and_upper_body(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / 'input', Path(folder) / 'output'
            source.mkdir()
            originals = []
            for part in range(2):
                n = 12
                q = np.tile([0., 0., 0., 1.], (n, 24, 1)).astype(np.float32)
                q[:, 0] = Rotation.from_euler('x', 90, degrees=True).as_quat()
                root = np.zeros((n, 3), dtype=np.float32)
                root[:, 0] = np.linspace(0, .03, n)
                root[:, 2] = 1.1
                if part == 1:
                    # Keep an upstream ground penetration and an airborne end.
                    root[:, 2] = np.linspace(.65, 1.45, n)
                    q[:, 2] = Rotation.from_euler('x', np.linspace(0, 30, n), degrees=True).as_quat()
                state = SkeletonState.from_rotation_and_root_translation(
                    self.tree, torch.from_numpy(q), torch.from_numpy(root), is_local=True)
                motion = SkeletonMotion.from_skeleton_state(state, fps=30).to_dict()
                name = f'clip_{part}'
                motion.update(seq_name=name, seq_idx=10 + part, beta=np.zeros(10), beta_idx=0,
                              gender='neutral', body_scale=1., min_verts_h=0., pose_aa=np.zeros((n, 72)))
                m = MotionLib({name: motion}, list(range(1, 24)), list(range(0, 70, 3)),
                              [3, 7, 18, 23], device='cpu', clean_up=True)
                originals.append(m)
                torch.save(m, source / f'mlib_part_{part:05d}.pth')
            main(['--input_dir', str(source), '--out_dir', str(output), '--device', 'cpu', '--batch-size', '4'])
            report = json.loads((output / 'diagnostics.json').read_text())
            self.assertEqual(len(report['sequences']), 2)
            self.assertTrue((output / 'metrics.png').stat().st_size > 1000)
            self.assertTrue((output / 'args.yaml').exists())
            with np.load(output / 'diagnostics.npz') as diagnostics:
                for record in report['sequences']:
                    prefix = record['array_prefix']
                    original_feet = diagnostics[prefix + 'source_skin_foot_height_m']
                    self.assertEqual(original_feet.shape, (12, 2))
                    self.assertTrue(np.isfinite(original_feet).all())
                    np.testing.assert_allclose(diagnostics[prefix + 'skin_foot_height_change_m'],
                                               diagnostics[prefix + 'skin_foot_height_m'] - original_feet)
                    np.testing.assert_allclose(diagnostics[prefix + 'skin_foot_height_m'].min(1),
                                               original_feet.min(1), atol=2e-6)
                    if record['name'] == 'clip_1':
                        self.assertLess(original_feet.min(), -.1)
                        self.assertGreater(original_feet[-1].min(), .1)
                # Re-skin the serialized output, independently of translated diagnostics.
                skin = SkinMetrics(self.settings, 'cpu', 4)
                final = torch.load(output / 'mlib_part_00001.pth', map_location='cpu')
                _, _, actual_feet, _ = skin.measure(final.lrs.numpy(), final.gts[:, 0].numpy(), final.gts.numpy())
                np.testing.assert_allclose(actual_feet.min(1),
                                           diagnostics[report['sequences'][1]['array_prefix'] + 'source_skin_foot_height_m'].min(1), atol=2e-6)
            self.assertFalse((output / 'comparison.mp4').exists())
            for part, original in enumerate(originals):
                m = torch.load(output / f'mlib_part_{part:05d}.pth', map_location='cpu')
                torch.testing.assert_close(m.gts[:, 0, :2], original.gts[:, 0, :2], rtol=0, atol=0)
                torch.testing.assert_close(m.lrs[:, 9:], original.lrs[:, 9:], rtol=0, atol=1e-7)
                self.assertEqual(int(m._motion_bodies[0, 0]), 1)
                np.testing.assert_allclose(m._motion_bodies[0, 1:].numpy(), self.settings['beta'], atol=1e-6)
                self.assertEqual(m._motion_seq_names, [f'clip_{part}'])
                self.assertEqual(int(m._motion_seq_ids[0]), 10 + part)
                self.assertEqual(tuple(m.dvs.shape), (12, 69))
            with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                main(['--input_dir', str(source), '--out_dir', str(output), '--device', 'cpu'])
            unchanged = Path(folder) / 'without_ik'
            main(['--input_dir', str(source), '--out_dir', str(unchanged), '--device', 'cpu', '--disable-ik', '--verbose'])
            import cv2
            capture = cv2.VideoCapture(str(unchanged / 'comparison.mp4'))
            try:
                self.assertTrue(capture.isOpened())
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 24)
                ok, frame = capture.read()
                self.assertTrue(ok)
                self.assertEqual(frame.shape, (480, 640, 3))
                self.assertGreater(frame[5:28, 10:240].std(), 10)  # Frame overlay decoded.
            finally:
                capture.release()
            verbose_report = json.loads((unchanged / 'diagnostics.json').read_text())
            self.assertEqual([r['video']['first_video_frame'] for r in verbose_report['sequences']], [0, 12])
            self.assertEqual([r['video']['video_frame_count'] for r in verbose_report['sequences']], [12, 12])
            with np.load(unchanged / 'diagnostics.npz') as diagnostics:
                for record in verbose_report['sequences']:
                    prefix = record['array_prefix']
                    # Pure translation reuses geometry without changing the
                    # separately saved pre-translation collision measurements.
                    np.testing.assert_allclose(diagnostics[prefix + 'collision_foot_height_m'],
                        diagnostics[prefix + 'raw_collision_foot_height_m'] +
                        diagnostics[prefix + 'root_z_change_m'][:, None], atol=2e-6)
            for part, original in enumerate(originals):
                result = torch.load(unchanged / f'mlib_part_{part:05d}.pth', map_location='cpu')
                torch.testing.assert_close(result.lrs, original.lrs, rtol=0, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
