import cv2
import numpy as np
import torch
from TennisProject.tracknet import BallTrackerNet
import torch.nn.functional as F
from TennisProject.postprocess import refine_kps
from TennisProject.homography import get_trans_matrix, refer_kps
from itertools import islice


def smooth_court_predictions(matrixes, keypoints, radius=3, strength=0.4):
    """Smooth one camera scene offline, keeping image-to-court maps consistent.

    Blend each frame with the mean of its centered window (including itself).
    Missing detections split windows and remain missing. Radius is in frames;
    strength=0 or radius=0 disables smoothing. Inputs are never modified.
    """
    if not isinstance(radius, int) or radius < 0:
        raise ValueError("radius must be a non-negative integer")
    if not 0 <= strength <= 1:
        raise ValueError("strength must be between 0 and 1")
    if len(matrixes) != len(keypoints):
        raise ValueError("Matrices and keypoints must have the same length")

    smoothed_matrices = list(matrixes)
    smoothed_points = list(keypoints)
    if radius == 0 or strength == 0:
        return smoothed_matrices, smoothed_points

    valid = [
        p is not None and m is not None
        and np.isfinite(p).all() and np.isfinite(m).all()
        for m, p in zip(matrixes, keypoints)
    ]
    continued=0
    avgleft=0
    avgright=0
    validCount=0
    for i, points in enumerate(keypoints):
        if not valid[i]:
            smoothed_matrices[i] = None
            smoothed_points[i] = None
            continued+=1
            continue

        left, right = i, i + 1
        while left > max(0, i - radius) and valid[left - 1]:
            left -= 1
        while right < min(len(keypoints), i + radius + 1) and valid[right]:
            right += 1
        if right - left == 1:
            continued+=1
            continue

        mean_points = np.mean(np.stack(keypoints[left:right]), axis=0)
        target = ((1 - strength) * points + strength * mean_points).astype(np.float32)

        # Averaging projected points need not preserve a single projective court.
        # Fit all 14 correspondences, then derive both outputs from that fit.
        forward, _ = cv2.findHomography(refer_kps, target, method=0)
        if forward is None or not np.isfinite(forward).all():
            continued+=1
            continue  # Retain the original matched pair if the fit fails.
        invertible, inverse = cv2.invert(forward)
        if not invertible or not np.isfinite(inverse).all():
            continued+=1
            continue
        projected = cv2.perspectiveTransform(refer_kps, forward)
        if not np.isfinite(projected).all():
            continued+=1
            continue
        smoothed_matrices[i] = inverse
        smoothed_points[i] = projected
        avgleft+=i-left
        avgright+=right-(i+1)
        validCount+=1
    print(f"debug: continued/valid: {continued/max(1,sum(valid)):.4f}. avgleft: {avgleft/max(1,validCount):.2f}. avgright={avgright/max(1,validCount):.2f}")
    return smoothed_matrices, smoothed_points


class CourtDetectorNet():
    def __init__(self, path_model=None,  device='cuda'):
        self.model = BallTrackerNet(out_channels=15).to(device)
        self.device = device
        if path_model:
            self.model.load_state_dict(torch.load(path_model, map_location=device))
            self.model = self.model.to(device)
            self.model.eval()
    @torch.inference_mode()
    def infer_model(self, frames, start,end, smoothing_radius=16, smoothing_strength=0.8):
        scaleX = 1
        scaleY = 1
        
        kps_res = []
        matrixes_res = []
        num_refined=0
        totalFrames=0
        for num_frame, image in enumerate(islice(frames, start,end), start): # tqdm
            totalFrames+=1
            inp_np = (image.astype(np.float32) / 255.)
            inp = torch.from_numpy(inp_np).permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)
            out = self.model(inp)[0]
            pred = F.sigmoid(out).cpu().numpy()

            points = []
            for kps_num in range(14):
                heatmap = (pred[kps_num]*255).astype(np.uint8)
                _, heatmap = cv2.threshold(heatmap, 170, 255, cv2.THRESH_BINARY)
                circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=2,
                                           minRadius=10, maxRadius=25)
                if circles is not None:
                    x_pred = circles[0][0][0]*scaleX
                    y_pred = circles[0][0][1]*scaleY
                    if kps_num not in [8, 12, 9]:
                        x_pred, y_pred, is_refined = refine_kps(image, int(y_pred), int(x_pred), crop_size=20)
                        num_refined+=int(is_refined)
                    points.append((x_pred, y_pred))                
                else:
                    points.append(None)

            matrix_trans = get_trans_matrix(points) 
            points = None
            if matrix_trans is not None:
                points = cv2.perspectiveTransform(refer_kps, matrix_trans)
                matrix_trans = cv2.invert(matrix_trans)[1]
            kps_res.append(points)
            matrixes_res.append(matrix_trans)
        print(f"debug: average number of refined points per frame: {num_refined/totalFrames:.2f}/14")
        return smooth_court_predictions(
            matrixes_res, kps_res, smoothing_radius, smoothing_strength
        )
