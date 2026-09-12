"""CPU-only video/CLI regression checks; run from the repository root with unittest."""

from contextlib import redirect_stderr
from fractions import Fraction
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from GVHMR.hmr4d.utils import video_io_utils as video_io
from GVHMR.tools.demo.demo_amass import parse_args_to_cfg, prepare_input_video


class VideoTrimTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.mp4"
        self.output = self.root / "output.mp4"
        # Distinct frame identities survive lossy encoding, making off-by-one trims visible.
        self.frames = np.stack([np.full((48, 64, 3), 20 + i * 18, dtype=np.uint8) for i in range(12)])
        self.write_source()

    def write_source(self, fps=30):
        with video_io.get_writer(self.source, fps=fps, crf=14) as writer:
            for frame in self.frames:
                writer.write_frame(frame)

    def test_exact_ranges_and_opencv_compatibility(self):
        for start, end in [(0, -1), (0, 1), (3, 9), (11, 12), (7, -1)]:
            with self.subTest(start=start, end=end):
                stop = len(self.frames) if end == -1 else end
                video_io.trim_video(self.source, self.output, start, end)
                expected = self.frames[start:stop]
                np.testing.assert_allclose(video_io.read_video_np(self.output), expected, atol=3, rtol=0)
                self.assertEqual(video_io.get_video_lwh(self.output), (len(expected), 64, 48))
                self.assertEqual(video_io.get_video_fps(self.output), 30)
                cap = cv2.VideoCapture(str(self.output))
                try:
                    for frame in expected:
                        ok, bgr = cap.read()
                        self.assertTrue(ok)
                        np.testing.assert_allclose(bgr[..., ::-1], frame, atol=3, rtol=0)
                    self.assertFalse(cap.read()[0])
                finally:
                    cap.release()

    def test_standard_profile_and_color_metadata(self):
        tagged = self.root / "tagged.mp4"
        video_io.ffmpeg.output(
            video_io.ffmpeg.input(str(self.source)).video, str(tagged),
            vcodec="libx264", crf=14, pix_fmt="yuv420p", colorspace="bt709",
            color_range="tv", color_trc="bt709", color_primaries="bt709",
        ).run(quiet=True)
        video_io.trim_video(tagged, self.output, 2, 10)
        stream = video_io.ffmpeg.probe(str(self.output))["streams"][0]
        for key, value in {"codec_name": "h264", "profile": "High", "pix_fmt": "yuv420p",
                           "color_space": "bt709", "color_range": "tv",
                           "color_transfer": "bt709", "color_primaries": "bt709"}.items():
            self.assertEqual(stream[key], value)

    def test_non_keyframe_trim_with_b_frames_and_long_gop(self):
        # Encode a recoverable frame number into seven large black/white blocks.
        frames = np.full((90, 48, 64, 3), 64, dtype=np.uint8)
        for index, frame in enumerate(frames):
            for bit in range(7):
                frame[8:24, bit * 8:(bit + 1) * 8] = 235 if index & (1 << bit) else 16
        source = self.root / "long_gop.mp4"
        video_io.ffmpeg.output(
            video_io.ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgb24", s="64x48", framerate=30),
            str(source), vcodec="libx264", crf=14, pix_fmt="yuv420p", g=30, bf=3,
            sc_threshold=0, **{"profile:v": "high"},
        ).run(input=frames.tobytes(), quiet=True)
        info = video_io.ffmpeg.probe(str(source), select_streams="v:0", show_frames=None)["frames"]
        self.assertTrue(any(frame["pict_type"] == "B" for frame in info))
        self.assertFalse(info[17]["key_frame"])
        # Include cuts inside a GOP and across a keyframe, plus a final-frame cut.
        for start, end in [(17, 46), (29, 32), (89, 90)]:
            with self.subTest(start=start, end=end):
                video_io.trim_video(source, self.output, start, end)
                saved = video_io.read_video_np(self.output)
                frame_ids = [
                    sum((1 << bit) for bit in range(7)
                        if frame[10:22, bit * 8 + 2:bit * 8 + 6].mean() > 128)
                    for frame in saved
                ]
                self.assertEqual(frame_ids, list(range(start, end)))

    def test_invalid_ranges_and_in_place_write_leave_source_untouched(self):
        original = self.source.read_bytes()
        for start, end in [(-1, 4), (3, 3), (4, 3), (0, 13), (12, -1), (0, -2)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                video_io.trim_video(self.source, self.output, start, end)
        with self.assertRaises(ValueError):
            video_io.trim_video(self.source, self.source)
        self.assertEqual(self.source.read_bytes(), original)
        self.assertFalse(self.output.exists())

    def test_failed_frame_count_verification_preserves_previous_output(self):
        self.output.write_bytes(b"previous output")
        real_probe = video_io.ffmpeg.probe

        def wrong_count(path, **kwargs):
            result = real_probe(path, **kwargs)
            if "count_frames" in kwargs:
                result["streams"][0]["nb_read_frames"] = "2"
            return result

        with patch.object(video_io.ffmpeg, "probe", wrong_count):
            with self.assertRaisesRegex(RuntimeError, "decoded 2 frames; expected 3"):
                video_io.trim_video(self.source, self.output, 2, 5)
        self.assertEqual(self.output.read_bytes(), b"previous output")

    def test_cache_reuse_and_equal_length_range_changes(self):
        cached = self.root / "features.pt"
        cfg = SimpleNamespace(video_path=str(self.output), paths={"features": str(cached)})
        cached.write_bytes(b"old full-video features")
        prepare_input_video(cfg, self.source, 0, 4, 30)
        self.assertFalse(cached.exists())
        cached.write_bytes(b"valid features")
        saved_mtime = self.output.stat().st_mtime_ns
        prepare_input_video(cfg, self.source, 0, 4, 30)
        self.assertEqual(self.output.stat().st_mtime_ns, saved_mtime)
        self.assertTrue(cached.exists())
        prepare_input_video(cfg, self.source, 4, 8, 30)
        self.assertFalse(cached.exists())
        np.testing.assert_allclose(video_io.read_video_np(self.output), self.frames[4:8], atol=3, rtol=0)
        cached.write_bytes(b"valid features")
        self.output.write_bytes(b"damaged video")
        prepare_input_video(cfg, self.source, 4, 8, 30)
        self.assertFalse(cached.exists())
        np.testing.assert_allclose(video_io.read_video_np(self.output), self.frames[4:8], atol=3, rtol=0)

    def test_cli_rejects_non_30_fps_before_creating_output(self):
        for fps in [24, Fraction(30000, 1001), 60]:
            with self.subTest(fps=fps):
                self.write_source(fps=fps)
                argv = ["demo_amass", "--video", str(self.source), "--output_root", str(self.root / "results")]
                error = StringIO()
                with patch("sys.argv", argv), redirect_stderr(error), self.assertRaises(SystemExit) as raised:
                    parse_args_to_cfg()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("must have 30 fps", error.getvalue())
                self.assertFalse((self.root / "results").exists())

    def test_cli_valid_trim(self):
        argv = ["demo_amass", "--video", str(self.source), "--output_root", str(self.root / "results"),
                "--start_frame", "3", "--end_frame", "8"]
        with patch("sys.argv", argv):
            cfg, _ = parse_args_to_cfg()
        np.testing.assert_allclose(video_io.read_video_np(cfg.video_path), self.frames[3:8], atol=3, rtol=0)
        self.assertEqual(video_io.get_video_fps(cfg.video_path), 30)


if __name__ == "__main__":
    unittest.main()
