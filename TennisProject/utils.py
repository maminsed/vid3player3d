# from scenedetect.video_manager import VideoManager
# from scenedetect.scene_manager import SceneManager
# from scenedetect.stats_manager import StatsManager
# from scenedetect.detectors import ContentDetector
import ffmpeg
from scenedetect import detect, ContentDetector
import cv2

def scene_detect(path_video):
    """
    Split video to disjoint fragments based on color histograms
    We have already choped with video_chopper.py so we stick with ffmpeg probe.
    """
    scene_list = [] # detect(path_video, ContentDetector())

    if scene_list == []:
        probe = ffmpeg.probe(path_video)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        n_frames = int(video_info['nb_frames'])

        return [[0, n_frames - 1]]
    scenes = [[x[0].frame_num, x[1].frame_num]for x in scene_list]    
    return scenes


