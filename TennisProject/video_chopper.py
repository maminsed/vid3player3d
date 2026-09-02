import os
import os.path as path
import ffmpeg
import json
from glob import glob
from scenedetect import detect, ContentDetector
from datetime import datetime
from time import perf_counter

INPUT_DIR = "/pub2/amin/youtube_download/video_only_usopen/reencoded_2/"
OUTPUT_DIR = "/pub2/amin/vid3player/TennisProject/chopped_2/"
SCENE_STATS_DIR = "/pub2/amin/vid3player/TennisProject/scene_stats/"
FPS = 30
MAX_CHUNK_SECONDS = 15
MIN_SCENE_SECONDS = 1
SCENE_THRESHOLD = 12.0
SCENE_MIN_LEN_FRAMES = 8
SCENE_EDGE_WEIGHT = 0.5
CRF = 18
PRESET = "slow"


def log(message):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def scene_detect(path_video, stats_file_path=None):
    """
    Split video to disjoint fragments based on color histograms.
    Returns list of [start_frame, end_frame, start_time_seconds, end_time_seconds].
    """
    detector = ContentDetector(
        threshold=SCENE_THRESHOLD,
        min_scene_len=SCENE_MIN_LEN_FRAMES,
    )
    scene_list = detect(path_video, detector, stats_file_path=stats_file_path)

    if scene_list == []:
        raise RuntimeError(f"Empty scenes for {path_video}")

    scenes = [
        [start.frame_num, end.frame_num, start.get_seconds(), end.get_seconds()]
        for start, end in scene_list
    ]
    return scenes


def main():
    run_start_time = perf_counter()
    resDict = {}
    max_chunk_frames = MAX_CHUNK_SECONDS * FPS
    min_scene_frames = MIN_SCENE_SECONDS * FPS

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SCENE_STATS_DIR, exist_ok=True)

    try:
        for input_path in os.listdir(INPUT_DIR):
            if not input_path.endswith(".mp4"):
                continue

            base_name = input_path[:-4]
            existing = glob(path.join(OUTPUT_DIR, f"{base_name}-scene*-*.mp4"))
            if existing:
                log(f"already processed skipping! {existing}")
                continue

            video_start_time = perf_counter()
            full_path = path.join(INPUT_DIR, input_path)
            log(f"\n\nProcessing: {input_path}")

            resDict[input_path] = []

            try:
                detect_start_time = perf_counter()
                stats_file_path = path.join(SCENE_STATS_DIR, f"{base_name}_scene_stats.csv")
                scenes = scene_detect(full_path, stats_file_path=stats_file_path)
                log(
                    f"Scene detection found {len(scenes)} scenes for {input_path} "
                    f"in {perf_counter() - detect_start_time:.2f}s; stats={stats_file_path}"
                )
            except Exception as e:
                log(f"Scene detection failed for {full_path}: {e}")
                resDict[input_path].append("scene detection failed")
                continue


            for scene_idx, (start_frame, end_frame, start_time, end_time) in enumerate(scenes):
                scene_start_time = perf_counter()
                log(f"At scene_idx: {scene_idx}")

                scene_frames = end_frame - start_frame
                duration = scene_frames / FPS
                if scene_frames <= min_scene_frames:
                    log(f"Skipping very short scene in {base_name}, idx={scene_idx}")
                    resDict[input_path].append(
                        f"scene_idx: {scene_idx} skipped (duration {duration:.2f}s). start_time: {start_time}, end_time: {end_time}"
                    )
                    continue

                try:
                    chunk_idx = 0
                    chunk_start = start_frame

                    while chunk_start < end_frame:
                        chunk_wall_start_time = perf_counter()
                        chunk_end = min(chunk_start + max_chunk_frames, end_frame)
                        chunk_duration = (chunk_end - chunk_start) / FPS

                        if chunk_duration <= MIN_SCENE_SECONDS:
                            break

                        chunk_output_path = path.join(
                            OUTPUT_DIR,
                            f"{base_name}-scene{scene_idx:03d}-{chunk_idx:03d}.mp4",
                        )
                        log(
                            f"  -> scene {scene_idx}, chunk {chunk_idx}: "
                            f"frames {chunk_start}-{chunk_end} "
                            f"(duration {chunk_duration:.2f}s) -> {chunk_output_path}"
                        )

                        in_stream = ffmpeg.input(full_path)
                        trimmed = (
                            in_stream
                            .trim(start_frame=chunk_start, end_frame=chunk_end)
                            .setpts("PTS-STARTPTS")
                        )

                        (
                            trimmed
                            .output(
                                chunk_output_path,
                                vcodec='libx264',
                                an=None,
                                pix_fmt='yuv420p',
                                crf=CRF,
                                preset=PRESET,
                                reset_timestamps=1
                            )
                            .run(quiet=True)
                        )

                        log(
                            f"Finished scene {scene_idx}, chunk {chunk_idx} "
                            f"in {perf_counter() - chunk_wall_start_time:.2f}s"
                        )
                        resDict[input_path].append(
                            f"scene_idx: {scene_idx}, chunk_idx: {chunk_idx} processed successfully "
                            f"(duration {chunk_duration:.2f}s)"
                        )

                        chunk_idx += 1
                        chunk_start = chunk_end

                    log(f"Finished scene {scene_idx} in {perf_counter() - scene_start_time:.2f}s")

                except Exception as e:
                    log(f"Error processing scene {scene_idx} in {base_name}: {e}")
                    resDict[input_path].append(
                        f"scene_idx: {scene_idx} failed with error: {e}"
                    )
            log(f"Finished video {input_path} in {perf_counter() - video_start_time:.2f}s")
    except Exception as e:
        log(f"Big Exception Occurred: {e}")
    finally:
        # Save results to JSON
        try:
            with open("chop_results.json", "w", encoding="utf-8") as f:
                json.dump(resDict, f, indent=2)
            log("Saved results to chop_results.json")
        except Exception as e:
            log(f"Failed to write chop_results.json: {e}")

        log(f"Total run time: {perf_counter() - run_start_time:.2f}s")
        return resDict


if __name__ == "__main__":
    log("Start")
    main()
    log("Done")
