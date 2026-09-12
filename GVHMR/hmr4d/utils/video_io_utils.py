from fractions import Fraction
import tempfile
import imageio.v3 as iio
import numpy as np
import torch
from pathlib import Path
import shutil
import ffmpeg
from tqdm import tqdm
import cv2


def get_video_lwh(video_path):
    L, H, W, _ = iio.improps(video_path, plugin="pyav").shape
    return L, W, H


def get_video_fps(video_path):
    return iio.immeta(video_path, plugin="pyav")["fps"]


def read_video_np(video_path, start_frame=0, end_frame=-1, scale=1.0):
    """
    Args:
        video_path: str
    Returns:
        frames: np.array, (N, H, W, 3) RGB, uint8
    """
    # If video path not exists, an error will be raised by ffmpegs
    filter_args = []
    should_check_length = False

    # 1. Trim
    if not (start_frame == 0 and end_frame == -1):
        if end_frame == -1:
            filter_args.append(("trim", f"start_frame={start_frame}"))
        else:
            should_check_length = True
            filter_args.append(("trim", f"start_frame={start_frame}:end_frame={end_frame}"))

    # 2. Scale
    if scale != 1.0:
        filter_args.append(("scale", f"iw*{scale}:ih*{scale}"))

    # Excute then check
    frames = iio.imread(video_path, plugin="pyav", filter_sequence=filter_args)
    if should_check_length:
        assert len(frames) == end_frame - start_frame

    return frames


def get_video_reader(video_path):
    return iio.imiter(video_path, plugin="pyav")


def read_images_np(image_paths, verbose=False):
    """
    Args:
        image_paths: list of str
    Returns:
        images: np.array, (N, H, W, 3) RGB, uint8
    """
    if verbose:
        images = [cv2.imread(str(img_path))[..., ::-1] for img_path in tqdm(image_paths)]
    else:
        images = [cv2.imread(str(img_path))[..., ::-1] for img_path in image_paths]
    images = np.stack(images, axis=0)
    return images


def save_video(images, video_path, fps=30, crf=17):
    """
    Args:
        images: (N, H, W, 3) RGB, uint8
        crf: 17 is visually lossless, 23 is default, +6 results in half the bitrate
    0 is lossless, https://trac.ffmpeg.org/wiki/Encode/H.264#crf
    """
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy().astype(np.uint8)
    elif isinstance(images, list):
        images = np.array(images).astype(np.uint8)

    with iio.imopen(video_path, "w", plugin="pyav") as writer:
        writer.init_video_stream("libx264", fps=fps)
        writer._video_stream.options = {"crf": str(crf)}
        writer.write(images)


def get_writer(video_path, fps=30, crf=17):
    """Remember to .close()."""
    writer = iio.imopen(video_path, "w", plugin="pyav")
    writer.init_video_stream("libx264", fps=fps)
    writer._video_stream.options = {"crf": str(crf)}
    tb = Fraction(1, int(fps))
    # _video_stream is private API, but this is the least invasive patch
    if getattr(writer._video_stream.codec_context, "time_base", None) is None:
        writer._video_stream.codec_context.time_base = tb
    if getattr(writer._video_stream, "time_base", None) is None:
        writer._video_stream.time_base = tb
    return writer


def trim_video(video_path, out_video_path, start_frame=0, end_frame=-1, fps=30, crf=10):
    """Save zero-based [start_frame, end_frame) as compatible H.264; -1 means EOF.

    FFmpeg's trim filter selects decoded frame indices without keyframe seeking.
    The completed file is decoded by ffprobe to check its actual frame count
    before replacing any existing output. Audio is not used by the pipeline.
    """
    length, width, height = get_video_lwh(video_path)
    end_frame = length if end_frame == -1 else end_frame
    if not 0 <= start_frame < end_frame <= length:
        raise ValueError(f"Expected 0 <= start_frame < end_frame <= {length}, got {start_frame}:{end_frame}")
    out_video_path = Path(out_video_path)
    if Path(video_path).resolve() == out_video_path.resolve():
        raise ValueError("Input and output video paths must differ")
    if width % 2 or height % 2:
        raise ValueError("H.264 YUV420 requires even width and height; input dimensions will not be changed")
    expected_length = end_frame - start_frame
    source_stream = ffmpeg.probe(str(video_path), select_streams="v:0")["streams"][0]
    # Preserve declared source color metadata instead of going through RGB.
    color_options = {
        key: source_stream[key]
        for key in ("color_range", "color_space", "color_transfer", "color_primaries")
        if source_stream.get(key) not in (None, "unknown", "unspecified", "reserved")
    }
    if "color_space" in color_options:
        color_options["colorspace"] = color_options.pop("color_space")
    if "color_transfer" in color_options:
        color_options["color_trc"] = color_options.pop("color_transfer")
    # RGB's identity matrix is not valid metadata for the YUV420 output.
    if color_options.get("colorspace") == "gbr":
        color_options.pop("colorspace")
    with tempfile.TemporaryDirectory(prefix=".trim-", dir=out_video_path.parent) as temp_dir:
        temp_path = Path(temp_dir) / out_video_path.name
        video = (
            ffmpeg.input(str(video_path)).video
            .filter("trim", start_frame=start_frame, end_frame=end_frame)
            .filter("setpts", "PTS-STARTPTS")
        )
        output = ffmpeg.output(
            video, str(temp_path), vcodec="libx264", crf=crf, preset="fast",
            pix_fmt="yuv420p", movflags="+faststart", vsync=0,
            **{"profile:v": "high", **color_options},
        )
        try:
            ffmpeg.run(output, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            stream = ffmpeg.probe(str(temp_path), select_streams="v:0", count_frames=None)["streams"][0]
        except ffmpeg.Error as exc:
            detail = (exc.stderr or b"").decode(errors="replace")
            raise RuntimeError(f"FFmpeg failed to save or verify the trimmed video: {detail}") from exc
        actual_length = int(stream.get("nb_read_frames", 0))
        if actual_length != expected_length:
            raise RuntimeError(f"Saved video decoded {actual_length} frames; expected {expected_length}")
        if (stream["width"], stream["height"]) != (width, height):
            raise RuntimeError("Saved video dimensions do not match the input")
        # Downstream readers use container frame counts, so check that too.
        if get_video_lwh(temp_path) != (expected_length, width, height):
            raise RuntimeError("Saved video metadata does not match the requested trim")
        saved_fps = float(Fraction(stream["avg_frame_rate"]))
        if not np.isclose(saved_fps, fps, rtol=0, atol=1e-6):
            raise RuntimeError(f"Saved video has {saved_fps} fps; expected {fps}")
        if stream.get("profile") != "High" or stream.get("pix_fmt") != "yuv420p":
            raise RuntimeError("Saved video is not H.264 High/YUV420")
        temp_path.replace(out_video_path)


def copy_file(video_path, out_video_path, overwrite=True):
    if not overwrite and Path(out_video_path).exists():
        return
    shutil.copy(video_path, out_video_path)


def merge_videos_horizontal(in_video_paths: list, out_video_path: str):
    if len(in_video_paths) < 2:
        raise ValueError("At least two video paths are required for merging.")
    inputs = [ffmpeg.input(path) for path in in_video_paths]
    merged_video = ffmpeg.filter(inputs, "hstack", inputs=len(inputs))
    output = ffmpeg.output(merged_video, out_video_path)
    ffmpeg.run(output, overwrite_output=True, quiet=True)


def merge_videos_vertical(in_video_paths: list, out_video_path: str):
    if len(in_video_paths) < 2:
        raise ValueError("At least two video paths are required for merging.")
    inputs = [ffmpeg.input(path) for path in in_video_paths]
    merged_video = ffmpeg.filter(inputs, "vstack", inputs=len(inputs))
    output = ffmpeg.output(merged_video, out_video_path)
    ffmpeg.run(output, overwrite_output=True, quiet=True)
