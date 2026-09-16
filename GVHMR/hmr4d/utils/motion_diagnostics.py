"""Conservative ViTPose leg-label repair and report-only HMR4D diagnostics."""
import csv
import json
from pathlib import Path

import numpy as np
import torch


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def detect_short_lr_swaps(keypoints, bbx_xys, max_frames=3, conf_thr=0.5,
                          error_ratio=0.25, min_gain=0.05, max_swapped_error=0.06):
    """Flag brief COCO17 leg identity reversals between agreeing neighbours.

    Coordinates are normalized by the tracked crop to remove camera translation
    and scale. At least two reliable left/right pairs must support the swap on
    EVERY frame, and at both boundaries. Swapped positions must fit the linear
    bridge between neighbours at least four times better, within 6% of crop size.
    Runs touching sequence edges or lasting longer than max_frames are ignored.
    This deliberately favours missed detections over flagging fast tennis turns.
    Returned frame ranges are zero-based and inclusive; inputs are never changed.
    """
    if not isinstance(max_frames, int) or max_frames < 1:
        raise ValueError("max_frames must be a positive integer")
    kp, boxes = _numpy(keypoints), _numpy(bbx_xys)
    if kp.ndim != 3 or kp.shape[1:] != (17, 3) or boxes.shape != (len(kp), 3):
        raise ValueError("Expected keypoints (F, 17, 3) and boxes (F, 3)")
    if not np.isfinite(boxes).all() or np.any(boxes[:, 2] <= 0):
        raise ValueError("Crop centres/scales must be finite with positive scales")
    xy = (kp[..., :2] - boxes[:, None, :2]) / boxes[:, None, 2:3]
    valid = np.isfinite(kp).all(-1) & (kp[..., 2] > conf_thr)
    candidates = []
    for limb, pairs in {
        # "arms": [[5, 6], [7, 8], [9, 10]],
        "legs": [[11, 12], [13, 14], [15, 16]]
    }.items():
        pairs = np.asarray(pairs)
        for start in range(1, len(kp) - 1):
            for size in range(1, min(max_frames, len(kp) - start - 1) + 1):
                stop = start + size
                # Use the same reliable pairs throughout the run and both anchors.
                reliable = valid[start - 1:stop + 1, pairs].all(axis=(0, 2))
                if reliable.sum() < 2:
                    continue
                ids = pairs[reliable]
                before, after = xy[start - 1, ids], xy[stop, ids]
                direct_bridge = np.linalg.norm(after - before, axis=-1).mean()
                flipped_bridge = np.linalg.norm(after[:, ::-1] - before, axis=-1).mean()
                if direct_bridge >= flipped_bridge:
                    continue  # Neighbours must agree on limb identity.
                observed = xy[start:stop, ids]
                weights = np.arange(1, size + 1)[:, None, None, None] / (size + 1)
                expected = before[None] * (1 - weights) + after[None] * weights
                direct = np.linalg.norm(observed - expected, axis=-1).mean(axis=-1)
                swapped = np.linalg.norm(observed[:, :, ::-1] - expected, axis=-1).mean(axis=-1)
                supports_swap = ((swapped < error_ratio * direct) &
                                 (direct - swapped > min_gain) &
                                 (swapped < max_swapped_error))
                supported_pairs = supports_swap.all(axis=0)
                if supported_pairs.sum() < 2:
                    continue
                # Hips may nearly overlap in projection. They need not
                # vote for a swap when both distal pairs provide strong evidence.
                ids = ids[supported_pairs]
                before, after = before[supported_pairs], after[supported_pairs]
                observed = observed[:, supported_pairs]
                direct, swapped = direct[:, supported_pairs], swapped[:, supported_pairs]
                # Both entry and exit must improve too, not just the average fit.
                good_boundaries = True
                for endpoint, anchor in [(observed[0], before), (observed[-1], after)]:
                    keep = np.linalg.norm(endpoint - anchor, axis=-1).mean()
                    flip = np.linalg.norm(endpoint[:, ::-1] - anchor, axis=-1).mean()
                    good_boundaries &= flip < error_ratio * keep
                if good_boundaries:
                    candidates.append({
                        "limb": limb, "start_frame": start, "end_frame": stop - 1,
                        "pairs": ids.tolist(), "original_error": float(direct.mean()),
                        "swapped_error": float(swapped.mean()),
                    })
    # Prefer the best explanation when candidate intervals overlap on one limb.
    selected = []
    for event in sorted(candidates, key=lambda e: e["swapped_error"] / e["original_error"]):
        if not any(e["limb"] == event["limb"] and
                   max(e["start_frame"], event["start_frame"]) <= min(e["end_frame"], event["end_frame"])
                   for e in selected):
            selected.append(event)
    return sorted(selected, key=lambda e: (e["start_frame"], e["limb"]))


def repair_short_lr_swaps(keypoints, bbx_xys, **detector_options):
    """Return a copy with only strongly supported leg pairs relabelled, plus an audit.

    Swap XY AND confidence together; never invent positions or boost scores.
    Detection always uses the original input, avoiding cascading corrections.
    Arms and unsupported pairs remain unchanged, even inside a detected interval.
    """
    events = detect_short_lr_swaps(keypoints, bbx_xys, **detector_options)
    repaired = keypoints.clone() if isinstance(keypoints, torch.Tensor) else np.array(keypoints, copy=True)
    for event in events:
        interval = slice(event["start_frame"], event["end_frame"] + 1)
        for left, right in event["pairs"]:
            repaired[interval, [left, right]] = keypoints[interval, [right, left]]
    return repaired, [dict(event, repaired=True) for event in events]


def save_lr_swap_report(events, path, fps=30.0, max_frames=3):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "diagnostic_only": not any(e.get("repaired", False) for e in events),
        "frame_indexing": "zero-based, inclusive ranges in output video",
        "fps": fps, "max_frames": max_frames,
        "events": [dict(e, start_seconds=e["start_frame"] / fps,
                        end_seconds=e["end_frame"] / fps) for e in events],
    }, indent=2) + "\n")


def rotation_steps(axis_angles):
    """SO(3) geodesic degrees from f-1 to f; frame 0 has no transition (NaN)."""
    from pytorch3d.transforms import axis_angle_to_matrix

    rotations = axis_angle_to_matrix(torch.as_tensor(_numpy(axis_angles), dtype=torch.float64))
    relative = rotations[1:] @ rotations[:-1].transpose(-1, -2)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    steps = torch.rad2deg(torch.acos(cosine)).numpy()
    return np.concatenate([np.full((1,) + steps.shape[1:], np.nan), steps], axis=0)


def final_motion_metrics(pred):
    """Use final SMPL parameters, including IK; angles are never subtracted as vectors."""
    metrics = {}
    for space in ("incam", "global"):
        params = pred[f"smpl_params_{space}"]
        metrics[f"root_{space}_deg"] = rotation_steps(params["global_orient"])
        transl = _numpy(params["transl"])
        metrics[f"translation_{space}_cm"] = np.r_[np.nan, np.linalg.norm(np.diff(transl, axis=0), axis=-1) * 100]
    body = _numpy(pred["smpl_params_incam"]["body_pose"])
    joint_steps = rotation_steps(body.reshape(len(body), 21, 3))
    # SMPL joint indices minus the root (body_pose starts at the left hip).
    for name, joints in {"legs": [1, 2, 4, 5, 7, 8, 10, 11],
                         "arms": [13, 14, 16, 17, 18, 19, 20, 21],
                         "torso": [3, 6, 9, 12, 15]}.items():
        metrics[f"{name}_max_deg"] = np.max(joint_steps[:, np.asarray(joints) - 1], axis=-1)
    return metrics


def spike_mask(values, minimum, radius=15):
    """Conservative local median + 6 robust standard deviations, plus an absolute floor."""
    values = np.asarray(values)
    flags = np.zeros(len(values), dtype=bool)
    for i, value in enumerate(values):
        neighbours = values[max(0, i - radius):min(len(values), i + radius + 1)]
        neighbours = neighbours[np.isfinite(neighbours)]
        if len(neighbours) < 5 or not np.isfinite(value):
            continue
        median = np.median(neighbours)
        threshold = max(minimum, median + 6 * 1.4826 * np.median(np.abs(neighbours - median)))
        flags[i] = value > threshold
    return flags


def save_hmr4d_spike_plot(pred, plot_path, fps=30.0, lr_swap_events=None):
    """Save final motion curves and numeric values; markers indicate candidates, not errors."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    metrics = final_motion_metrics(pred)
    length = len(metrics["root_incam_deg"])
    times = np.arange(length) / fps
    groups = [
        (["root_incam_deg", "root_global_deg"], "Body orientation change (degrees/frame)", 20),
        (["legs_max_deg", "arms_max_deg", "torso_max_deg"], "Largest joint rotation change (degrees/frame)", 30),
        (["translation_incam_cm", "translation_global_cm"], "Root translation change (cm/frame)", 10),
    ]
    flags = {key: spike_mask(metrics[key], minimum) for keys, _, minimum in groups for key in keys}
    fig = Figure(figsize=(15, 10), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(3, 1, sharex=True)
    for ax, (keys, ylabel, _) in zip(axes, groups):
        labelled = set()
        for key in keys:
            values = metrics[key]
            line, = ax.plot(times, values, label=key.replace("_", " "), linewidth=1)
            indices = np.flatnonzero(flags[key])
            ax.scatter(times[indices], values[indices], color=line.get_color(), s=24, marker="x")
            for i in sorted(indices, key=lambda i: values[i], reverse=True)[:6]:
                if i not in labelled:
                    ax.annotate(f"f{i}", (times[i], values[i]), xytext=(3, 6),
                                textcoords="offset points", fontsize=8)
                    labelled.add(i)
        for event in lr_swap_events or []:
            ax.axvspan(event["start_frame"] / fps, (event["end_frame"] + 1) / fps,
                       color="purple", alpha=0.15)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Output video time (seconds); f = zero-based output frame")
    fig.suptitle("Final HMR4D motion diagnostics (after postprocessing)\n"
                 "x = unusually large local change, not a confirmed error; purple = ViTPose L/R swap interval")
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    return {key: np.flatnonzero(mask).tolist() for key, mask in flags.items()}
