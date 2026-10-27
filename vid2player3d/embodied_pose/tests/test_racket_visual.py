"""Racket rendering follows the configured asset and the physical body state."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from embodied_pose.utils.racket_visual import racket_lines, transform_lines, wrist_racket_lines, racket_face_normal
from embodied_pose.utils.debug_video import DebugVideo, close_debug_videos
from embodied_pose.utils.tools import create_vis_model_xml

ASSET = PROJECT/'vid2player/data/assets/smpl_mesh_humanoid_djokovic.xml'
NAMES = ['Pelvis', 'R_Wrist']


class RacketVisualTests(unittest.TestCase):
    def tearDown(self):
        close_debug_videos()

    def test_asset_geometry_and_rotated_face(self):
        lines = racket_lines(ASSET)
        self.assertEqual(len(lines), 2)
        np.testing.assert_allclose(lines[0], [[.5, 0, 0], [.15, 0, 0]])
        np.testing.assert_allclose(np.linalg.norm(lines[1], axis=-1), .15)
        np.testing.assert_allclose(lines[1] @ np.array([0, 1, 1]), 0, atol=1e-10)
        q = np.array([0., 0., np.sqrt(.5), np.sqrt(.5)])
        a = transform_lines(lines, np.array([1, 2, 3]), q)
        b = transform_lines(lines, np.array([1, 2, 3]), -q)
        np.testing.assert_allclose(a[0], [[1, 2.5, 3], [1, 2.15, 3]])
        for first, second in zip(a, b):
            np.testing.assert_allclose(first, second)

    def test_interactive_xml_preserves_all_three_rackets(self):
        from lxml import etree
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'viewer.xml'
            create_vis_model_xml(str(ASSET), str(path), num_actor=3)
            root = etree.parse(str(path))
            rackets = root.xpath("//body[contains(@name, 'Racket')]")
            self.assertEqual(len(rackets), 3)
            self.assertTrue(all(len(r.findall('geom')) == 2 for r in rackets))
            self.assertEqual(len(root.findall('.//joint')), 208)  # 69 per actor + default.

    def test_fixed_racket_geometry_uses_same_wrist_frame_for_both_poses(self):
        lines = racket_lines(ASSET)
        normal = racket_face_normal(ASSET)
        np.testing.assert_allclose(normal, [0, 2**-.5, 2**-.5])
        wrist = np.array([1., 2., 3.])
        wrist_q = np.array([0., 0., 0., 1.])
        offset = np.array([-.5, 0, 0])
        expected = wrist_racket_lines(lines, wrist, wrist_q, offset)
        np.testing.assert_allclose(expected[0][0], wrist)  # Handle meets wrist.
        # A rebased simulator frame must not rotate the XML outline by 90 deg.
        imported_q = np.array([-2**-.5, 0, 0, 2**-.5])
        old = transform_lines(lines, wrist + offset, imported_q)
        self.assertFalse(np.allclose(old[1], expected[1]))
        corrected = wrist_racket_lines(lines, wrist, -wrist_q, offset)
        for actual, reference in zip(corrected, expected):
            np.testing.assert_allclose(actual, reference)

    def test_headless_uses_physical_and_reference_rackets_only_when_present(self):
        task = SimpleNamespace(
            cfg={'env': {'num_rec_frames': 1}}, args=SimpleNamespace(rec_once=False),
            dt=1/30, body_names=NAMES, _racket_body_id=24,
            humanoid_files={0: str(ASSET)}, _racket_offset=torch.tensor([-.5, 0, 0]),
            _rigid_body_pos=torch.tensor([[[0., 0., 1.], [0., 0., 1.]]]),
            _rigid_body_rot=torch.tensor([0., 0., 0., 1.]).expand(1, 2, 4),
            _prev_target_rb_pos=torch.tensor([[[0., 0., 1.], [0., 0., 1.]]]),
            _prev_target_rb_rot=torch.tensor([[[0., 0., 0., 1.], [0., 0., 0., 1.]]]),
            _physical_body_pos=torch.zeros(1, 25, 3),
            _physical_body_rot=torch.tensor([0., 0., 0., 1.]).expand(1, 25, 4))
        task._physical_body_pos[0, 24] = torch.tensor([.75, 0, 1.])
        task._rigid_body_pos[0, 1, 0] = .75
        with tempfile.TemporaryDirectory() as tmp, patch('embodied_pose.utils.debug_video.imageio.get_writer') as writer:
            task.cfg['env']['rec_fname'] = str(Path(tmp)/'test.mp4')
            video = DebugVideo(task)
            video.append(task)
            frame = writer.return_value.append_data.call_args[0][0]
            self.assertTrue(np.any(np.all(frame == [35, 100, 220], axis=-1)))
            self.assertTrue(np.any(np.all(frame == [225, 140, 30], axis=-1)))
            task._racket_body_id = None
            plain = DebugVideo(task)
            plain.append(task)
            frame = writer.return_value.append_data.call_args[0][0]
            self.assertFalse(np.any(np.all(frame == [35, 100, 220], axis=-1)))
            self.assertFalse(np.any(np.all(frame == [225, 140, 30], axis=-1)))


if __name__ == '__main__':
    unittest.main()
