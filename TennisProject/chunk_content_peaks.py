#!/usr/bin/env python3
import argparse
import csv
import re
import subprocess
from pathlib import Path

from scenedetect import ContentDetector, detect


PROJECT_DIR = Path("/pub2/amin/vid3player/TennisProject")
INPUT_DIR = Path("/pub2/amin/youtube_download/video_only_usopen/reencoded_2")
SCENE_STATS_DIR = PROJECT_DIR / "scene_stats"
RUN_LOG = PROJECT_DIR / "run.log"

FPS = 30
MAX_CHUNK_SECONDS = 15
SCENE_THRESHOLD = 13.5
SCENE_MIN_LEN_FRAMES = 8


def parse_chunk_path(chunk_path):
    match = re.match(r"(?P<base>.+)-scene(?P<scene>\d+)-(?P<chunk>\d+)\.mp4$", chunk_path.name)
    if not match:
        raise ValueError(f"Could not parse chunk filename: {chunk_path.name}")

    return {
        "base_name": match.group("base"),
        "scene_idx": int(match.group("scene")),
        "chunk_idx": int(match.group("chunk")),
    }


def timecode(seconds):
    millis = round((seconds - int(seconds)) * 1000)
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def probe_duration(video_path):
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def frame_range_from_run_log(chunk_path):
    if not RUN_LOG.exists():
        return None

    chunk_path_text = str(chunk_path)
    pattern = re.compile(r"frames (?P<start>\d+)-(?P<end>\d+).* -> (?P<path>.+)$")

    with RUN_LOG.open("r", encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            if chunk_path_text not in line:
                continue
            match = pattern.search(line.strip())
            if match:
                return int(match.group("start")), int(match.group("end")), "run.log"

    return None


def frame_range_from_detector(base_name, scene_idx, chunk_idx):
    source_path = INPUT_DIR / f"{base_name}.mp4"
    if not source_path.exists():
        raise FileNotFoundError(f"Could not find source video: {source_path}")

    detector = ContentDetector(
        threshold=SCENE_THRESHOLD,
        min_scene_len=SCENE_MIN_LEN_FRAMES,
    )
    scenes = detect(str(source_path), detector)

    if scene_idx >= len(scenes):
        raise IndexError(f"Scene {scene_idx} is out of range. Detected {len(scenes)} scenes.")

    scene_start, scene_end = scenes[scene_idx]
    max_chunk_frames = MAX_CHUNK_SECONDS * FPS
    start_frame = scene_start.frame_num + chunk_idx * max_chunk_frames
    end_frame = min(start_frame + max_chunk_frames, scene_end.frame_num)

    if start_frame >= scene_end.frame_num:
        raise IndexError(
            f"Chunk {chunk_idx} is out of range for scene {scene_idx}. "
            f"Scene frames: {scene_start.frame_num}-{scene_end.frame_num}."
        )

    return start_frame, end_frame, "detector"


def read_stats(base_name, start_frame, end_frame):
    stats_path = SCENE_STATS_DIR / f"{base_name}_scene_stats.csv"
    if not stats_path.exists():
        raise FileNotFoundError(f"Could not find stats CSV: {stats_path}")

    rows = []
    with stats_path.open("r", encoding="utf-8", newline="") as stats_file:
        for row in csv.DictReader(stats_file):
            frame = int(row["Frame Number"])
            if start_frame <= frame < end_frame:
                row["frame"] = frame
                row["content_val_float"] = float(row["content_val"])
                rows.append(row)

    return stats_path, rows


def local_peaks(rows, min_gap_frames):
    sorted_rows = sorted(rows, key=lambda row: row["content_val_float"], reverse=True)
    selected = []

    for row in sorted_rows:
        frame = row["frame"]
        if all(abs(frame - peak["frame"]) >= min_gap_frames for peak in selected):
            selected.append(row)

    return selected


def print_rows(title, rows, chunk_start_frame, limit):
    print(title)
    print("rank,source_frame,source_time,chunk_frame,chunk_time,content_val,delta_hue,delta_sat,delta_lum,delta_edges")

    for rank, row in enumerate(rows[:limit], start=1):
        source_frame = row["frame"]
        chunk_frame = source_frame - chunk_start_frame
        print(
            f"{rank},"
            f"{source_frame},"
            f"{row['Timecode']},"
            f"{chunk_frame},"
            f"{timecode(chunk_frame / FPS)},"
            f"{float(row['content_val']):.3f},"
            f"{float(row['delta_hue']):.3f},"
            f"{float(row['delta_sat']):.3f},"
            f"{float(row['delta_lum']):.3f},"
            f"{float(row['delta_edges']):.3f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Show the highest PySceneDetect content_val moments inside a chopped chunk."
    )
    parser.add_argument("chunk_path", help="Full path to a chopped MP4 chunk.")
    parser.add_argument("--top", type=int, default=20, help="Number of highest raw rows to print.")
    parser.add_argument(
        "--peak-gap-frames",
        type=int,
        default=15,
        help="Minimum frame gap between local peaks.",
    )
    args = parser.parse_args()

    chunk_path = Path(args.chunk_path).expanduser().resolve()
    parsed = parse_chunk_path(chunk_path)

    frame_range = frame_range_from_run_log(chunk_path)
    if frame_range is None:
        frame_range = frame_range_from_detector(
            parsed["base_name"],
            parsed["scene_idx"],
            parsed["chunk_idx"],
        )
    start_frame, end_frame, frame_range_source = frame_range

    stats_path, rows = read_stats(parsed["base_name"], start_frame, end_frame)
    if not rows:
        raise RuntimeError(
            f"No stats rows found for frames {start_frame}-{end_frame} in {stats_path}"
        )

    duration = probe_duration(chunk_path)
    top_rows = sorted(rows, key=lambda row: row["content_val_float"], reverse=True)
    peak_rows = local_peaks(rows, args.peak_gap_frames)

    print(f"chunk_path: {chunk_path}")
    print(f"source_video: {INPUT_DIR / (parsed['base_name'] + '.mp4')}")
    print(f"stats_csv: {stats_path}")
    print(f"scene_idx: {parsed['scene_idx']}")
    print(f"chunk_idx: {parsed['chunk_idx']}")
    print(f"source_frame_range: {start_frame}-{end_frame} ({frame_range_source})")
    print(f"source_time_range: {timecode(start_frame / FPS)}-{timecode(end_frame / FPS)}")
    if duration is not None:
        print(f"chunk_duration_probe: {duration:.3f}s")
    print()

    print_rows(f"Top {args.top} raw content_val rows", top_rows, start_frame, args.top)
    print()
    print_rows(f"Top {args.top} local peaks", peak_rows, start_frame, args.top)


if __name__ == "__main__":
    main()
