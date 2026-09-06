import cv2
import numpy as np
import os
from pathlib import Path
from sympy import Line
from scipy.spatial import distance
from sympy.geometry.point import Point2D

# Set COURT_DEBUG_DIR=/tmp/court_debug before running to save the first 100
# refinement crops per process (not frames). Unset it to disable image output.
COURT_DEBUG_DIR = os.environ.get('COURT_DEBUG_DIR')
SHOW_IMAGES=False
COURT_DEBUG_MAX_IMAGES = 100
_court_debug_count = 0

# Line detection settings; the debug mask shows the effect of the threshold.
LINE_BRIGHTNESS_THRESHOLD = 155
# I halved the crop_size
HOUGH_VOTE_THRESHOLD = 15 
MIN_LINE_LENGTH = 5
MAX_LINE_GAP = 15


def _save_refinement_debug(crop, raw_lines, merged_lines, original, refined,
                           origin, status):
    global _court_debug_count
    if not COURT_DEBUG_DIR or _court_debug_count >= COURT_DEBUG_MAX_IMAGES:
        return

    mask = cv2.threshold(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY),
                         LINE_BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    raw_view, merged_view = crop.copy(), crop.copy()
    for view, lines in ((raw_view, raw_lines), (merged_view, merged_lines)):
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = map(int, line)
            color = ((0, 255, 255), (255, 0, 255), (255, 255, 0))[i % 3]
            cv2.line(view, (x1, y1), (x2, y2), color, 1)

    panels = []
    for title, view in (
        ('Crop', crop),
        (f'Mask > {LINE_BRIGHTNESS_THRESHOLD}', cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)),
        (f'Raw: {len(raw_lines)}', raw_view),
        (f'Merged: {len(merged_lines)}', merged_view),
    ):
        panel = cv2.resize(view, (320, 320), interpolation=cv2.INTER_NEAREST)
        if title.startswith('Merged'):
            for point, color in ((original, (0, 0, 255)), (refined, (0, 255, 0))):
                center = (round((point[0] + 0.5) * 320 / crop.shape[1]),
                          round((point[1] + 0.5) * 320 / crop.shape[0]))
                # Different sizes keep both markers visible when unchanged.
                radius = 8 if color == (0, 0, 255) else 4
                cv2.circle(panel, center, radius, color, 2)
        panel = cv2.copyMakeBorder(panel, 30, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(panel, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1)
        panels.append(panel)
    montage = cv2.copyMakeBorder(np.hstack(panels), 0, 35, 0, 0, cv2.BORDER_CONSTANT)
    label = f'Crop origin (x,y)={origin} | {status} | red: input, green: output'
    cv2.putText(montage, label, (8, montage.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    output_dir = Path(COURT_DEBUG_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'refine_{_court_debug_count:05d}.png'
    if status != 'intersection accepted':
        if not cv2.imwrite(str(output_path), montage):
            raise OSError(f'Could not write court debug image: {output_path}')
    _court_debug_count += 1


def _is_valid_line(l, min_len_px=2.0):
    x1, y1, x2, y2 = map(float, l)
    dx, dy = x2 - x1, y2 - y1
    return (dx*dx + dy*dy) >= (min_len_px * min_len_px)


def line_intersection(line1, line2):
    """
    Find 2 lines intersection point
    """
    if not _is_valid_line(line1) or not _is_valid_line(line2):
        return None

    p11, p12 = Point2D(line1[0], line1[1]), Point2D(line1[2], line1[3])
    p21, p22 = Point2D(line2[0], line2[1]), Point2D(line2[2], line2[3])

    # Extra safety: SymPy will also reject p==p
    if p11 == p12 or p21 == p22:
        return None

    l1, l2 = Line(p11, p12), Line(p21, p22)

    intersection = l1.intersection(l2)
    point = None
    if len(intersection) > 0:
        if isinstance(intersection[0], Point2D):
            point = intersection[0].coordinates
    return point 

def refine_kps(img, x_ct, y_ct, crop_size=40):
    """Refine a point; inputs are (row, column), output is (x, y)."""
    refined_x_ct, refined_y_ct = x_ct, y_ct
    is_refined=False

    img_height, img_width = img.shape[:2]
    x_min = max(x_ct-crop_size, 0)
    x_max = min(img_height, x_ct+crop_size)
    y_min = max(y_ct-crop_size, 0)
    y_max = min(img_width, y_ct+crop_size)

    img_crop = img[x_min:x_max, y_min:y_max]
    lines = detect_lines(img_crop)
    raw_lines = lines
    status = 'unchanged: fewer than two raw lines'
    
    if len(lines) > 1:
        lines = merge_lines(lines)
        status = 'unchanged: merged count is not two'
        if len(lines) == 2:
            status = 'unchanged: no unique intersection'
            inters = line_intersection(lines[0], lines[1])
            if inters:
                status = 'unchanged: intersection outside crop'
                new_x_ct = int(inters[1])
                new_y_ct = int(inters[0])
                if new_x_ct > 0 and new_x_ct < img_crop.shape[0] and new_y_ct > 0 and new_y_ct < img_crop.shape[1]:
                    refined_x_ct = x_min + new_x_ct
                    refined_y_ct = y_min + new_y_ct
                    is_refined=True
                    status = 'intersection accepted'
    if SHOW_IMAGES:
        _save_refinement_debug(
            img_crop, raw_lines, lines,
            (y_ct - y_min, x_ct - x_min),
            (refined_y_ct - y_min, refined_x_ct - x_min),
            (y_min, x_min), status,
        )
    return refined_y_ct, refined_x_ct, is_refined

def detect_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, LINE_BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    lines = cv2.HoughLinesP(gray, 1, np.pi / 180, HOUGH_VOTE_THRESHOLD,
                          minLineLength=MIN_LINE_LENGTH, maxLineGap=MAX_LINE_GAP)
    lines = np.squeeze(lines) 
    if len(lines.shape) > 0:
        if len(lines) == 4 and not isinstance(lines[0], np.ndarray):
            lines = [lines]
    else:
        lines = []
    return lines

def merge_lines(lines):
    lines = sorted(lines, key=lambda item: item[0])
    mask = [True] * len(lines)
    new_lines = []

    for i, line in enumerate(lines):
        if mask[i]:
            for j, s_line in enumerate(lines[i + 1:]):
                if mask[i + j + 1]:
                    x1, y1, x2, y2 = line
                    x3, y3, x4, y4 = s_line
                    dist1 = distance.euclidean((x1, y1), (x3, y3))
                    dist2 = distance.euclidean((x2, y2), (x4, y4))
                    if dist1 < 10 and dist2 < 10:
                        line = np.array([int((x1+x3)/2), int((y1+y3)/2), int((x2+x4)/2), int((y2+y4)/2)])
                        mask[i + j + 1] = False
            new_lines.append(line)  
    return new_lines       

