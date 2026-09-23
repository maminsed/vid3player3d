"""Geometric invariants and failure handling, without SMPL/GPU dependencies."""

import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from uhc.utils.contact_ik import (
    IKSettings, correct_contact_poses, contact_intervals, forward_kinematics)
from uhc.utils.debug_court_video import ik_overlay_lines


class ContactIKTests(unittest.TestCase):
    def setUp(self):
        self.n = 30
        self.parents = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                                 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21])
        self.rest = np.zeros((24, 3))
        self.rest[1] = [-.1, 0, 0]
        self.rest[2] = [.1, 0, 0]
        for hip, knee, ankle, toe in ((1, 4, 7, 10), (2, 5, 8, 11)):
            self.rest[knee] = self.rest[hip] + [0, 0, -.45]
            self.rest[ankle] = self.rest[knee] + [0, 0, -.45]
            self.rest[toe] = self.rest[ankle] + [0, .15, -.04]
        self.pose = np.zeros((self.n, 72))
        self.pose[:, [3, 6]] = .3
        self.pose[:, [12, 15]] = -.6
        self.pose[:, [21, 24]] = .3
        self.root = np.tile([0., 0., .9], (self.n, 1))
        self.root[:, 0] = np.linspace(0, .12, self.n)
        self.prob = np.ones((self.n, 4)) * .99
        self.height = np.zeros((self.n, 2))

    def solve(self, **kwargs):
        return correct_contact_poses(self.pose, self.root, self.rest, self.parents,
                                     self.prob, self.height, **kwargs)

    def test_flat_contact_locks_feet_and_preserves_root_upper_body_and_bone_lengths(self):
        original_root = self.root.copy()
        pose, diagnostics = self.solve()
        editable = {1, 2, 4, 5, 7, 8}
        for joint in set(range(24)) - editable:
            np.testing.assert_array_equal(pose[:, joint*3:joint*3+3], self.pose[:, joint*3:joint*3+3])
        np.testing.assert_array_equal(self.root, original_root)
        np.testing.assert_allclose(diagnostics['anchor_residual_m'][3:-3], 0, atol=1e-7)
        np.testing.assert_allclose(diagnostics['speed_after_mps'][4:-3], 0, atol=1e-6)
        # Entire foot is fixed for flat support, not only the ankle.
        feet = diagnostics['feet_after_m'][3:-3]
        np.testing.assert_allclose(feet, np.broadcast_to(feet[0], feet.shape), atol=1e-7)
        offsets = self.rest.copy()
        offsets[1:] -= self.rest[self.parents[1:]]
        rotations = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(self.n, 24, 3, 3)
        positions, _ = forward_kinematics(rotations, offsets, self.parents, self.root)
        for j in range(1, 24):
            np.testing.assert_allclose(np.linalg.norm(positions[:, j]-positions[:, self.parents[j]], axis=1),
                                       np.linalg.norm(offsets[j]), atol=1e-9)

    def test_toe_pivot_keeps_toe_anchor(self):
        self.prob[:, [0, 2]] = .1
        self.pose[:, [21, 24]] += np.linspace(0, .15, self.n)[:, None]
        _, d = self.solve()
        self.assertTrue((d['contact_mode'] == 2).all())
        np.testing.assert_allclose(d['anchor_residual_m'][3:-3], 0, atol=1e-7)
        self.assertGreater(np.linalg.norm(np.diff(d['feet_after_m'][3:-3, 0, 0], axis=0)), .001)

    def test_disabled_and_no_contact_are_exact_noops(self):
        pose, d = self.solve(enabled=False)
        np.testing.assert_array_equal(pose, self.pose)
        np.testing.assert_array_equal(d['foot_change_m'], 0)
        self.assertTrue((d['contact_mode'] != 0).all())
        self.prob[:] = .1
        pose, d = self.solve()
        np.testing.assert_array_equal(pose, self.pose)
        self.assertEqual(d['summary']['interval_count'], 0)

    def test_unreachable_contact_is_flagged_without_nan_or_root_changes(self):
        self.root[:, 0] = np.linspace(0, 3, self.n)
        root = self.root.copy()
        pose, d = self.solve()
        self.assertTrue(np.isfinite(pose).all())
        self.assertTrue(d['limited'][15:].all())
        self.assertGreater(d['anchor_residual_m'][20, 0], 1.)
        np.testing.assert_array_equal(self.root, root)
        np.testing.assert_array_equal(pose[:, :3], self.pose[:, :3])

    def test_straight_leg_singularity_remains_finite_and_bounded(self):
        self.pose[:] = 0
        self.root[:, 0] = 0
        pose, d = self.solve()
        self.assertTrue(np.isfinite(pose).all())
        self.assertTrue(d['limited'].all())
        self.assertLess(np.nanmax(d['anchor_residual_m']), .002)

    def test_hysteresis_short_contacts_and_airborne_frames(self):
        self.prob[:] = .1
        self.prob[1:3] = .99  # Too short.
        self.prob[6:10] = .99
        self.prob[10:14] = .6  # Remain active below enter threshold.
        self.height[12:] = .3  # Airborne despite stationarity predictions.
        modes, intervals = contact_intervals(self.prob, self.height, IKSettings())
        self.assertFalse(modes[:6].any())
        self.assertTrue((modes[6:12] == 3).all())
        self.assertFalse(modes[12:].any())
        self.assertEqual(len(intervals), 2)

    def test_overlay_reports_disabled_contacts_and_undefined_first_speed(self):
        _, d = self.solve(enabled=False)
        text = '\n'.join(ik_overlay_lines(0, 30, d, 100))
        self.assertIn('source 100', text)
        self.assertIn('IK OFF', text)
        self.assertIn('flat', text)
        self.assertIn('move 0.0 cm', text)
        self.assertIn('n/a -> n/a', text)


if __name__ == '__main__':
    unittest.main()
