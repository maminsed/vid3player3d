"""CPU regressions for diagnostic video completion, compatibility, and merging."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from GVHMR.hmr4d.utils.video_io_utils import (
    get_debug_writer, valid_debug_video, merge_videos_horizontal, get_video_lwh,
    debug_video_size, read_video_np,
)


class DebugVideoTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, length=6, color=(120, 120, 120)):
        path = self.root / name
        with get_debug_writer(path, 64, 48, length) as writer:
            for _ in range(length):
                writer.write_frame(np.full((48, 64, 3), color, dtype=np.uint8))
        return path

    def test_interrupt_does_not_publish_partial_video(self):
        path = self.write('video.mp4')
        previous = path.read_bytes()
        with self.assertRaises(KeyboardInterrupt):
            with get_debug_writer(path, 64, 48, 6) as writer:
                writer.write_frame(np.zeros((48, 64, 3), dtype=np.uint8))
                raise KeyboardInterrupt()
        self.assertEqual(path.read_bytes(), previous)
        self.assertEqual(list(self.root.glob('.debug-*')), [])

    def test_short_write_is_not_published_or_reused(self):
        path = self.root / 'short.mp4'
        with self.assertRaisesRegex(RuntimeError, 'Incomplete debug video'):
            with get_debug_writer(path, 64, 48, 6) as writer:
                writer.write_frame(np.zeros((48, 64, 3), dtype=np.uint8))
        self.assertFalse(path.exists())
        short = self.write('existing.mp4', length=5)
        self.assertFalse(valid_debug_video(short, 6, 64, 48))

    def test_merge_and_corrupt_cache_recovery(self):
        a = self.write('a.mp4', color=(220, 20, 20))
        b = self.write('b.mp4', color=(20, 20, 220))
        merged = self.root / 'merged.mp4'
        merged.write_bytes(b'incomplete old output')
        merge_videos_horizontal([a, b], merged)
        self.assertTrue(valid_debug_video(merged, 6, 128, 48))
        self.assertEqual(get_video_lwh(merged), (6, 128, 48))
        frame = read_video_np(merged)[0]
        self.assertGreater(frame[:, :64, 0].mean(), frame[:, :64, 2].mean() + 150)
        self.assertGreater(frame[:, 64:, 2].mean(), frame[:, 64:, 0].mean() + 150)
        before = merged.stat().st_mtime_ns
        merge_videos_horizontal([a, b], merged)
        self.assertEqual(merged.stat().st_mtime_ns, before)

    def test_merge_rejects_different_lengths(self):
        a = self.write('a.mp4')
        b = self.write('b.mp4', length=5)
        with self.assertRaisesRegex(ValueError, 'equal frame counts'):
            merge_videos_horizontal([a, b], self.root / 'merged.mp4')

    def test_debug_size_keeps_aspect_and_does_not_upscale(self):
        self.assertEqual(debug_video_size(1920, 1080), (960, 540))
        self.assertEqual(debug_video_size(1080, 1920), (302, 540))
        self.assertEqual(debug_video_size(640, 480), (640, 480))


if __name__ == '__main__':
    unittest.main()
