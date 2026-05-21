import os
import os.path as path
import ffmpeg
import json
from scenedetect import detect, ContentDetector
from datetime import datetime


def scene_detect(path_video):
    """
    Split video to disjoint fragments based on color histograms.
    Returns list of [start_time_seconds, end_time_seconds] for each scene.
    """
    scene_list = detect(path_video, ContentDetector())

    if scene_list == []:
        raise RuntimeError(f"Empty scenes for {path_video}")

    scenes = [[x[0].get_seconds(), x[1].get_seconds()] for x in scene_list]
    return scenes


def main():
    DIR = "../youtube_download/video_only_usopen/reencoded/"
    MAX_CHUNK_SECONDS = 15
    resDict = {}

    os.makedirs("chopped", exist_ok=True)

    try:
        for input_path in os.listdir(DIR):
            if not input_path.endswith(".mp4"):
                continue

            full_path = path.join(DIR, input_path)
            print(f"\n\nProcessing: {input_path}")

            resDict[input_path] = []

            try:
                scenes = scene_detect(full_path)
            except Exception as e:
                print(f"Scene detection failed for {full_path}: {e}")
                resDict[input_path].append("scene detection failed")
                continue

            base_name = input_path[:-4]

            for scene_idx, (start_time, end_time) in enumerate(scenes):
                print(f"At scene_idx: {scene_idx}")

                duration = end_time - start_time
                if duration <= 1:  # Don't want less than 1 second long vids
                    print(f"Skipping very short scene in {base_name}, idx={scene_idx}")
                    resDict[input_path].append(
                        f"scene_idx: {scene_idx} skipped (duration {duration:.2f}s)"
                    )
                    continue

                try:
                    in_stream = ffmpeg.input(full_path)

                    # Trim to the scene boundaries
                    trimmed = (
                        in_stream
                        .trim(start=start_time, end=end_time)
                        .setpts('PTS-STARTPTS')
                    )

                    # Convert to CFR 30fps
                    trimmed_cfr = trimmed.filter('fps', fps=30)

                    # ----- CASE 1: Scene shorter than or equal to MAX_CHUNK_SECONDS -----
                    # Don't chop further; just one file with suffix 000.
                    if duration <= 20:
                        output_path = path.join(
                            'chopped',
                            f"{base_name}-scene{scene_idx:03d}-000.mp4"
                        )
                        print(
                            f"  -> scene {scene_idx}: {start_time:.2f}s - {end_time:.2f}s "
                            f"(duration {duration:.2f}s) -> {output_path}"
                        )
                        print('skipping chopping more due to size')

                        (
                            trimmed_cfr
                            .output(
                                output_path,
                                vcodec='libx264',
                                an=None,
                                pix_fmt='yuv420p',
                                movflags='+faststart',
                                crf=22,          # adjust if you need smaller/larger files
                                preset='slow',   # better compression at same quality
                                reset_timestamps=1
                            )
                            .run(quiet=True)
                        )

                    # ----- CASE 2: Scene longer than MAX_CHUNK_SECONDS -----
                    # Use segment muxer to further chop into <= MAX_CHUNK_SECONDS chunks.
                    else:
                        output_pattern = path.join(
                            'chopped',
                            f"{base_name}-scene{scene_idx:03d}-%03d.mp4"
                        )
                        print(
                            f"  -> scene {scene_idx}: {start_time:.2f}s - {end_time:.2f}s "
                            f"(duration {duration:.2f}s) -> {output_pattern}"
                        )

                        (
                            trimmed_cfr
                            .output(
                                output_pattern,
                                vcodec='libx264',
                                an=None,
                                pix_fmt='yuv420p',
                                movflags='+faststart',
                                crf=22,
                                preset='slow',
                                f='segment',
                                segment_time=MAX_CHUNK_SECONDS,
                                reset_timestamps=1,
                            )
                            .run(quiet=True)
                        )

                    resDict[input_path].append(
                        f"scene_idx: {scene_idx} processed successfully (duration {duration:.2f}s)"
                    )

                except Exception as e:
                    print(f"Error processing scene {scene_idx} in {base_name}: {e}")
                    resDict[input_path].append(
                        f"scene_idx: {scene_idx} failed with error: {e}"
                    )
            # break
    except Exception as e:
        print("Big Exception Occurred:", e)
    finally:
        # Save results to JSON
        try:
            with open("chop_results.json", "w", encoding="utf-8") as f:
                json.dump(resDict, f, indent=2)
            print("Saved results to chop_results.json")
        except Exception as e:
            print("Failed to write chop_results.json:", e)

        return resDict


if __name__ == "__main__":
    print(f"Start at {datetime.now().isoformat()}")
    main()
    print(f"Done at {datetime.now().isoformat()}")
