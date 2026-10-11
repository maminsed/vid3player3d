"""Map saved simulation poses to original TennisProject input-video frames."""
import json
from pathlib import Path

import numpy as np

from vid2player.motion_vae.phase import canonical_name


def load_gvhmr_range(directory, export_path):
    root = Path(directory)
    name = canonical_name(export_path)
    if (root / '0_input_video.json').is_file():
        candidates = [root] if canonical_name(root.name) == name else []
    else:
        candidates = [p for p in root.iterdir() if p.is_dir() and canonical_name(p.name) == name
                      and (p / '0_input_video.json').is_file()]
    if len(candidates) != 1:
        raise ValueError('Expected one GVHMR clip with 0_input_video.json for {}; found {}'.format(name, len(candidates)))
    path = candidates[0] / '0_input_video.json'
    source = json.loads(path.read_text())['source']
    start, end, fps = source['start_frame'], source['end_frame'], float(source['fps'])
    if (isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int)
            or not isinstance(end, int) or not 0 <= start < end):
        raise ValueError('Invalid inclusive-start/exclusive-end GVHMR frame range: ' + str(path))
    if not np.isclose(fps, 30):
        raise ValueError('TennisProject/GVHMR frame alignment requires 30 FPS: ' + str(path))
    if canonical_name(source['path']) != name:
        raise ValueError('GVHMR JSON source video does not match export: ' + str(path))
    return dict(metadata=str(path), start_frame=start, end_frame=end, fps=fps)


def export_source_frames(seq_data, export_length, trim_start, retained_length,
                         gvhmr_range=None, source_start_frame=None, export_frame_offset=1.):
    """Do not stretch a short/early-terminated export to fit the source range.

    Timestamps are seconds in the reference motion AFTER each saved sim step.
    Legacy GVHMR exports assume Start reset and one frame per control step;
    --export-frame-offset can override their first reference frame explicitly.
    Without GVHMR, retain the existing manual absolute-frame mapping.
    """
    if gvhmr_range is None:
        start = 0 if source_start_frame is None else source_start_frame
        frames = np.arange(retained_length, dtype=float) + start + trim_start
        return frames, np.ones(retained_length, dtype=bool), dict(
            mode='manual_absolute_frames', first_source_frame=float(frames[0]))
    if source_start_frame is not None:
        raise ValueError('--source-start-frame and --gvhmr_dir cannot be combined')
    reference_times = seq_data.get('reference_times')
    if reference_times is not None:
        times = np.asarray(reference_times, dtype=float)
        if times.ndim != 1 or len(times) < export_length or not np.isfinite(times).all() or (np.diff(times) <= 0).any() or (times < 0).any():
            raise ValueError('Export reference_times must contain one finite increasing timestamp per saved pose')
        relative = times[trim_start:trim_start + retained_length] * gvhmr_range['fps']
        mode = 'recorded_reference_times'
    else:
        if not np.isfinite(export_frame_offset) or export_frame_offset < 0:
            raise ValueError('--export-frame-offset must be finite and nonnegative')
        relative = np.arange(retained_length, dtype=float) + trim_start + export_frame_offset
        mode = 'legacy_reference_frame_offset'
    frames = gvhmr_range['start_frame'] + relative
    in_range = (frames >= gvhmr_range['start_frame']) & (frames < gvhmr_range['end_frame'])
    diagnostic = dict(gvhmr_range, mode=mode, first_source_frame=float(frames[0]),
                      last_source_frame=float(frames[-1]), out_of_range_frames=int((~in_range).sum()))
    if reference_times is None:
        diagnostic['export_frame_offset'] = export_frame_offset
        diagnostic['assumption'] = 'legacy export: Start reset; one 30 FPS frame per saved control step'
    return frames, in_range, diagnostic
