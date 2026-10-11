"""Conservative weak phase labels from TennisProject image-space ball tracks.

The homography is a ground-plane projection, NOT a 3-D ball reconstruction.
Only hit timing is inferred; phase remains temporal between alternating hits.
"""
import ast
import csv
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

NET_CENTER = np.array([832.5, 1748.0])
METERS_PER_PIXEL = np.array([10.97 / 1093, -23.78 / 2374])
COURT_POINTS = np.array([(286, 561), (1379, 561), (286, 2935), (1379, 2935),
                        (423, 561), (423, 2935), (1242, 561), (1242, 2935),
                        (423, 1110), (1242, 1110), (423, 2386), (1242, 2386),
                        (832, 1110), (832, 2386)], dtype=float)


def canonical_name(value):
    name = Path(str(value)).name
    for suffix in ('.pk', '.pkl', '.csv', '.mp4'):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    for prefix in ('demo_', '0-'):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def find_csv(directory, export_path):
    name = canonical_name(export_path)
    matches = [p for p in Path(directory).glob('*.csv') if canonical_name(p) == name]
    if len(matches) > 1:
        raise ValueError('Ambiguous TennisProject CSV for ' + name)
    return matches[0] if matches else None


def project_ball(row):
    """Infer CSV-output->640x360 scale from court keypoints and H, then project.

    Ball/keypoints are output-video pixels; H expects 640x360 model pixels.
    Near/bottom player is negative court Y after the required Y-axis flip.
    """
    h = np.fromstring(row['inv_matrix'].replace('[', ' ').replace(']', ' '), sep=' ').reshape(3, 3)
    kps = np.asarray(ast.literal_eval(row['court_kps']), dtype=float)
    projected = np.column_stack((COURT_POINTS, np.ones(14))) @ np.linalg.inv(h).T
    projected = projected[:, :2] / projected[:, 2:]
    scale = np.median(projected / kps, axis=0)
    if not np.isfinite(scale).all() or np.any(scale <= 0) or np.max(np.abs(kps * scale - projected)) > 4:
        raise ValueError('CSV keypoints and homography disagree')
    pixels = np.array([float(row['x_ball']), float(row['y_ball'])])
    if not np.isfinite(pixels).all() or np.any(pixels <= 0):
        raise ValueError('Missing ball observation')
    point = h @ np.r_[pixels * scale, 1]
    if abs(point[2]) < 1e-8:
        raise ValueError('Ball projects to infinity')
    return (point[:2] / point[2] - NET_CENTER) * METERS_PER_PIXEL


def labels_between_hits(frames, observed, hits, side='fg'):
    """No extrapolation; missing observations never receive a phase target."""
    phase = np.zeros((len(frames), 2), dtype=np.float32)
    mask = np.zeros(len(frames), dtype=bool)
    for left, right in zip(hits[:-1], hits[1:]):
        if left['side'] == right['side'] or right['frame'] <= left['frame']:
            continue
        selected = (frames >= left['frame']) & (frames <= right['frame']) & observed
        # Our hit = pi; opponent hit = 0 (or 2pi).
        start = np.pi if left['side'] == side else 0.
        angle = start + np.pi * (frames[selected] - left['frame']) / (right['frame'] - left['frame'])
        phase[selected] = np.column_stack((np.sin(angle), np.cos(angle)))
        mask[selected] = True
    return phase, mask


def infer_csv_phase(path, side='fg', fps=30.):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    frames = np.array([int(r['global_frame']) for r in rows])
    if len(frames) and (np.diff(frames) <= 0).any():
        raise ValueError('CSV global_frame must be unique and increasing: ' + str(path))
    ball = np.full((len(rows), 2), np.nan)
    players = np.full((len(rows), 2, 2), np.nan)
    bounces = np.array([r.get('is_bounce', '').lower() == 'true' for r in rows])
    for i, row in enumerate(rows):
        try:
            ball[i] = project_ball(row)
        except (ValueError, SyntaxError, TypeError, KeyError, np.linalg.LinAlgError):
            pass
        for j, key in enumerate(('person_point_bottom', 'person_point_top')):
            try:
                players[i, j] = (np.asarray(ast.literal_eval(row[key]), dtype=float) - NET_CENTER) * METERS_PER_PIXEL
            except (ValueError, SyntaxError, TypeError, KeyError):
                pass
    observed = np.isfinite(ball).all(axis=1)
    # Bridge at most 0.2 s for hit detection only; original missing frames stay masked.
    # Longer gaps split detection runs, so unrelated trajectories are never joined.
    detection_ball = ball.copy()
    known = np.flatnonzero(observed)
    for left, right in zip(known[:-1], known[1:]):
        if 1 < right - left <= max(2, int(round(fps * .2)) + 1) and frames[right] - frames[left] == right - left:
            for axis in range(2):
                detection_ball[left:right+1, axis] = np.linspace(ball[left, axis], ball[right, axis], right-left+1)
    indices = np.flatnonzero(np.isfinite(detection_ball).all(axis=1))
    runs = np.split(indices, np.flatnonzero((np.diff(indices) != 1) | (np.diff(frames[indices]) != 1)) + 1)
    hits = []
    width = max(3, int(round(fps * .1)))
    def accept(index, method):
        if bounces[max(0, index - 1):index + 2].any():
            return
        distance = np.linalg.norm(players[index] - detection_ball[index], axis=1)
        if not np.isfinite(distance).any():
            return
        player = int(np.nanargmin(distance))
        if distance[player] > 6 or abs(detection_ball[index, 1]) < 4 or np.sign(detection_ball[index, 1]) != np.sign(players[index, player, 1]):
            return
        hits.append(dict(frame=int(frames[index]), side=('fg', 'bg')[player], method=method))
    for run in runs:
        if len(run) < 2 * width + 1:
            continue
        y = detection_ball[run, 1]
        smooth = np.array([np.median(y[max(0, i-1):i+2]) for i in range(len(y))])
        for sign in (1, -1):
            peaks, _ = find_peaks(sign * smooth, prominence=1.5, distance=max(1, int(fps * .3)))
            for peak in peaks:
                if peak < width or peak + width >= len(run):
                    continue
                before = smooth[peak] - smooth[peak-width]
                after = smooth[peak+width] - smooth[peak]
                if before * after < 0 and min(abs(before), abs(after)) >= .25:
                    accept(run[peak], 'sustained_y_reversal')
        # Serve/start: require observed stationary history followed by sustained departure.
        for i in range(width, len(run) - width):
            before = np.ptp(smooth[i-width:i+1])
            after = smooth[i+width] - smooth[i]
            if before < .15 and abs(after) > .6:
                outgoing = after * smooth[i] < 0
                if outgoing:
                    accept(run[i], 'stationary_to_departure')
                    break
    hits.sort(key=lambda h: h['frame'])
    dedup = []
    for hit in hits:
        if not dedup or hit['frame'] - dedup[-1]['frame'] >= fps * .3:
            dedup.append(hit)
    phase, mask = labels_between_hits(frames, observed, dedup, side)
    # If a long tracking gap could hide another shot, neither endpoint constrains
    # a trustworthy half-cycle. Leave the entire uncertain interval unlabelled.
    detection_valid = np.isfinite(detection_ball).all(axis=1)
    for left, right in zip(dedup[:-1], dedup[1:]):
        between = (frames >= left['frame']) & (frames <= right['frame'])
        selected_frames = frames[between]
        if (not detection_valid[between].all() or (np.diff(selected_frames) != 1).any()
                or right['frame'] - left['frame'] > fps * 3):
            mask[between & (frames > left['frame']) & (frames < right['frame'])] = False
    phase[~mask] = 0
    return frames, phase, mask, dict(csv=str(path), hits=dedup,
                                   observed_frames=int(observed.sum()), labelled_frames=int(mask.sum()),
                                   method='weak hit detection; temporal interpolation; no extrapolation or gap filling')


def align_phase(source_frames, csv_frames, phase, mask):
    values = np.zeros((len(source_frames), 2), dtype=np.float32)
    valid = np.zeros(len(source_frames), dtype=bool)
    lookup = {int(frame): i for i, frame in enumerate(csv_frames)}
    for i, frame in enumerate(source_frames):
        if not np.isfinite(frame):
            continue
        nearest = int(round(frame))
        if abs(frame - nearest) < 1e-5:
            j = lookup.get(nearest)
            if j is not None:
                values[i], valid[i] = phase[j], mask[j]
            continue
        # Random reset times need not land exactly on a video frame. Interpolate
        # only when BOTH adjacent frames have labels, preserving masked gaps.
        lower = int(np.floor(frame))
        left, right = lookup.get(lower), lookup.get(lower + 1)
        if left is None or right is None or not mask[left] or not mask[right]:
            continue
        weight = frame - lower
        vector = (1-weight) * phase[left] + weight * phase[right]
        norm = np.linalg.norm(vector)
        if norm > 1e-6:
            values[i], valid[i] = vector / norm, True
    return values, valid
