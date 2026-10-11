"""GVHMR source ranges and low-level reference-time joins, without model configs."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vid2player.motion_vae.frame_alignment import load_gvhmr_range, export_source_frames
from vid2player.motion_vae.phase import align_phase


class FrameAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.clip = self.root/'A_vs._B'
        self.clip.mkdir()
        self.path = self.clip/'0_input_video.json'
        self.source = dict(path='/inputs/0-A_vs._B.mp4', start_frame=4, end_frame=20, fps=30.)
        self.write_source()

    def write_source(self):
        self.path.write_text(json.dumps(dict(source=self.source)))

    def test_root_and_direct_clip_matching_preserve_dots(self):
        for directory in (self.root, self.clip):
            metadata = load_gvhmr_range(directory, 'demo_A_vs._B.pk')
            self.assertEqual(metadata['start_frame'], 4)
            self.assertEqual(metadata['end_frame'], 20)
        with self.assertRaisesRegex(ValueError, 'Expected one'):
            load_gvhmr_range(self.root, 'demo_other_clip.pk')
        with self.assertRaisesRegex(ValueError, 'Expected one'):
            load_gvhmr_range(self.clip, 'demo_other_clip.pk')

    def test_wrong_source_and_invalid_range_or_fps_fail(self):
        for changes in (dict(path='/inputs/other.mp4'), dict(start_frame=-1),
                        dict(end_frame=4), dict(end_frame=20.5), dict(fps=60)):
            original = self.source.copy()
            self.source.update(changes); self.write_source()
            with self.assertRaises(ValueError):
                load_gvhmr_range(self.root, 'demo_A_vs._B.pk')
            self.source = original

    def test_legacy_post_step_offset_trimming_and_exclusive_end(self):
        metadata = load_gvhmr_range(self.root, 'demo_A_vs._B.pk')
        frames, mask, diagnostic = export_source_frames({}, 16, 2, 14, metadata)
        np.testing.assert_array_equal(frames, np.arange(7, 21))
        self.assertFalse(mask[-1])  # frame 20 is outside [4,20)
        self.assertTrue(mask[:-1].all())
        self.assertEqual(diagnostic['out_of_range_frames'], 1)
        frames, mask, _ = export_source_frames({}, 3, 0, 3, metadata)
        np.testing.assert_array_equal(frames, [5, 6, 7])  # early termination never stretches time
        override, _, _ = export_source_frames({}, 3, 0, 3, metadata, export_frame_offset=0)
        np.testing.assert_array_equal(override, [4, 5, 6])

    def test_recorded_timestamps_handle_random_resets_and_different_cadence(self):
        metadata = load_gvhmr_range(self.root, 'demo_A_vs._B.pk')
        data = dict(reference_times=np.array([5.5, 7.5, 9.5, 11.5])/30)
        frames, mask, diagnostic = export_source_frames(data, 4, 1, 2, metadata, export_frame_offset=999)
        np.testing.assert_allclose(frames, [11.5, 13.5])
        self.assertTrue(mask.all())
        self.assertEqual(diagnostic['mode'], 'recorded_reference_times')
        for invalid in ([0., 0.], [0., float('nan')], [0.], [-1., 0.]):
            with self.assertRaises(ValueError):
                export_source_frames(dict(reference_times=invalid), 2, 0, 2, metadata)

    def test_manual_mapping_preserved_and_ambiguous_options_rejected(self):
        frames, mask, _ = export_source_frames({}, 5, 2, 3)
        np.testing.assert_array_equal(frames, [2, 3, 4])
        frames, _, _ = export_source_frames({}, 5, 2, 3, source_start_frame=30)
        np.testing.assert_array_equal(frames, [32, 33, 34])
        with self.assertRaisesRegex(ValueError, 'cannot be combined'):
            export_source_frames({}, 5, 0, 5, dict(start_frame=4, end_frame=20, fps=30), source_start_frame=2)

    def test_fractional_csv_join_preserves_mask_and_wraps_angle(self):
        angles = np.array([2*np.pi-.1, .1, .2, .3])
        values = np.column_stack((np.sin(angles), np.cos(angles)))
        phases, mask = align_phase(np.array([0.5, 1., 1.5, 2.5, 4.]), np.arange(4),
                                   values, np.array([True, True, False, True]))
        np.testing.assert_allclose(phases[0], [0., 1.], atol=1e-6)
        np.testing.assert_array_equal(mask, [True, True, False, False, False])


if __name__ == '__main__':
    unittest.main()
