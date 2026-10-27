"""Tracking units and invariances, independent of the simulator."""
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from embodied_pose.utils.tracking_errors import body_tracking_errors, racket_tracking_errors
from embodied_pose.utils.evaluation_report import EvaluationReport


class TrackingErrorsTests(unittest.TestCase):
    def test_translation_is_not_a_joint_or_root_aligned_error(self):
        p = torch.zeros(2, 24, 3)
        q = torch.tensor([0., 0., 0., 1.]).expand(2, 24, 4)
        delta = torch.tensor([.3, .4, .12])
        r = body_tracking_errors(p + delta, q, p, -q)
        torch.testing.assert_close(r['root_error_m'], torch.full((2,), delta.norm()))
        torch.testing.assert_close(r['root_xy_error_m'], torch.full((2,), .5))
        torch.testing.assert_close(r['body_position_error_m'], r['root_error_m'])
        self.assertEqual(float(r['root_aligned_body_error_m'].max()), 0.)
        self.assertEqual(float(r['joint_rotation_error_deg'].max()), 0.)

    def test_joint_rotation_is_local_and_excludes_root(self):
        p = torch.zeros(1, 24, 3)
        q = torch.tensor([0., 0., 0., 1.]).expand(1, 24, 4).clone()
        target = q.clone()
        # R_Hand is a leaf; rotating it changes exactly one of 23 local joints.
        q[:, -1] = torch.tensor([2**-.5, 0, 0, 2**-.5])
        self.assertAlmostEqual(float(body_tracking_errors(p, q, p, target)['joint_rotation_error_deg']), 90/23, places=5)
        # Rotate every global frame equally: relative joint angles are unchanged.
        q[:] = torch.tensor([0., 0., 2**-.5, 2**-.5])
        self.assertAlmostEqual(float(body_tracking_errors(p, q, p, target)['joint_rotation_error_deg']), 0., places=5)

    def test_racket_face_and_center_use_xml_wrist_frame(self):
        p = torch.zeros(1, 3)
        identity = torch.tensor([[0., 0., 0., 1.]])
        q = torch.tensor([[2**-.5, 0., 0., 2**-.5]])
        offset = torch.tensor([-.5, 0, 0])
        normal = torch.tensor([0., 0., 1.])
        r = racket_tracking_errors(p, q, p, identity, offset, normal)
        self.assertAlmostEqual(float(r['racket_face_error_deg']), 90., places=4)
        self.assertEqual(float(r['racket_center_error_m']), 0.)
        r = racket_tracking_errors(p + torch.tensor([.3, .4, 0]), q, p, -q, offset, normal)
        self.assertEqual(float(r['racket_face_error_deg']), 0.)
        self.assertAlmostEqual(float(r['racket_center_error_m']), .5, places=5)

    def test_evaluation_saves_new_metrics_and_plot(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = SimpleNamespace(cfg={'env': {'evaluation_dir': tmp}}, dt=1/30,
                _rigid_body_pos=torch.ones(1, 24, 3), _prev_target_rb_pos=torch.ones(1, 24, 3),
                _humanoid_head_id=13, _termination_heights=torch.zeros(24),
                _reset_ref_motion_ids=torch.tensor([0]), _cur_ref_motion_times=torch.tensor([1.]),
                _motion_lib=SimpleNamespace(_motion_lengths=torch.tensor([1.])))
            report = EvaluationReport(task)
            tracking = {'root_xy_error_m': torch.tensor([.1]),
                        'joint_rotation_error_deg': torch.tensor([3.]),
                        'racket_face_error_deg': torch.tensor([5.])}
            report.append(0, 1, torch.ones(1), torch.ones(1), {'terminate': torch.zeros(1), 'tracking': tracking})
            report.close('completed')
            row = json.loads((Path(tmp)/'summary.json').read_text())['episodes'][0]
            self.assertEqual(row['mean_joint_rotation_error_deg'], 3.)
            self.assertEqual(row['mean_racket_face_error_deg'], 5.)
            self.assertTrue((Path(tmp)/'metrics.png').is_file())


if __name__ == '__main__':
    unittest.main()
