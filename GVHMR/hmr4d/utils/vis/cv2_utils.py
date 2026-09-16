import torch
import cv2
import numpy as np
from GVHMR.hmr4d.utils.wis3d_utils import get_colors_by_conf


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x.copy()
    elif isinstance(x, list):
        return np.array(x)
    return x.clone().cpu().numpy()


def draw_bbx_xys_on_image(bbx_xys, image, conf=True):
    assert isinstance(bbx_xys, np.ndarray)
    assert isinstance(image, np.ndarray)
    image = image.copy()
    lu_point = (bbx_xys[:2] - bbx_xys[2:] / 2).astype(int)
    rd_point = (bbx_xys[:2] + bbx_xys[2:] / 2).astype(int)
    color = (255, 178, 102) if conf == True else (128, 128, 128)  # orange or gray
    image = cv2.rectangle(image, lu_point, rd_point, color, 2)
    return image


def draw_bbx_xys_on_image_batch(bbx_xys_batch, image_batch, conf=None):
    """conf: if provided, list of bool"""
    use_conf = conf is not None
    bbx_xys_batch = to_numpy(bbx_xys_batch)
    assert len(bbx_xys_batch) == len(image_batch)
    image_batch_out = []
    for i in range(len(bbx_xys_batch)):
        if use_conf:
            image_batch_out.append(draw_bbx_xys_on_image(bbx_xys_batch[i], image_batch[i], conf[i]))
        else:
            image_batch_out.append(draw_bbx_xys_on_image(bbx_xys_batch[i], image_batch[i]))
    return image_batch_out


def draw_bbx_xyxy_on_image(bbx_xys, image, conf=True):
    bbx_xys = to_numpy(bbx_xys)
    image = to_numpy(image)
    color = (255, 178, 102) if conf == True else (128, 128, 128)  # orange or gray
    image = cv2.rectangle(image, (int(bbx_xys[0]), int(bbx_xys[1])), (int(bbx_xys[2]), int(bbx_xys[3])), color, 2)
    return image


def draw_bbx_xyxy_on_image_batch(bbx_xyxy_batch, image_batch, mask=None, conf=None):
    """
    Args:
        conf: if provided, list of bool, mutually exclusive with mask
        mask: whether to draw, historically used
    """
    if mask is not None:
        assert conf is None
    if conf is not None:
        assert mask is None
    use_conf = conf is not None
    bbx_xyxy_batch = to_numpy(bbx_xyxy_batch)
    image_batch = to_numpy(image_batch)
    assert len(bbx_xyxy_batch) == len(image_batch)
    image_batch_out = []
    for i in range(len(bbx_xyxy_batch)):
        if use_conf:
            image_batch_out.append(draw_bbx_xyxy_on_image(bbx_xyxy_batch[i], image_batch[i], conf[i]))
        else:
            if mask is None or mask[i]:
                image_batch_out.append(draw_bbx_xyxy_on_image(bbx_xyxy_batch[i], image_batch[i]))
            else:
                image_batch_out.append(image_batch[i])
    return image_batch_out


def draw_kpts(frame, keypoints, color=(0, 255, 0), thickness=2):
    frame_ = frame.copy()
    for x, y in keypoints:
        cv2.circle(frame_, (int(x), int(y)), thickness, color, -1)
    return frame_


def draw_kpts_with_conf(frame, kp2d, conf, thickness=2):
    """
    Args:
        kp2d: (J, 2),
        conf: (J,)
    """
    frame_ = frame.copy()
    conf = conf.reshape(-1)
    colors = get_colors_by_conf(conf)  # (J, 3)
    colors = colors[:, [2, 1, 0]].int().numpy().tolist()
    for j in range(kp2d.shape[0]):
        x, y = kp2d[j, :2]
        c = colors[j]
        cv2.circle(frame_, (int(x), int(y)), thickness, c, -1)
    return frame_


def draw_kpts_with_conf_batch(frames, kp2d_batch, conf_batch, thickness=2):
    """
    Args:
        kp2d_batch: (B, J, 2),
        conf_batch: (B, J)
    """
    assert len(frames) == len(kp2d_batch)
    assert len(frames) == len(conf_batch)
    frames_ = []
    for i in range(len(frames)):
        frames_.append(draw_kpts_with_conf(frames[i], kp2d_batch[i], conf_batch[i], thickness))
    return frames_


def draw_coco17_skeleton(img, keypoints, conf_thr=0):
    """Draw anatomical sides on RGB frames: left orange, right cyan."""
    keypoints = to_numpy(keypoints)
    img = img.copy()
    # fmt:off
    coco_skel = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]            
    # fmt:on
    left = {1, 3, 5, 7, 9, 11, 13, 15}
    right = {2, 4, 6, 8, 10, 12, 14, 16}
    left_color, right_color, center_color = (255, 155, 40), (0, 220, 255), (210, 210, 210)
    visible = np.isfinite(keypoints[:, :2]).all(axis=-1)
    if keypoints.shape[1] == 3:
        visible &= keypoints[:, 2] > conf_thr
    for a, b in coco_skel:
        color = left_color if a in left and b in left else (
            right_color if a in right and b in right else center_color)
        if visible[a] and visible[b]:
            cv2.line(img, tuple(keypoints[a, :2].astype(int)), tuple(keypoints[b, :2].astype(int)), color, 4)
    for j in np.flatnonzero(visible):
        color = left_color if j in left else right_color if j in right else center_color
        cv2.circle(img, tuple(keypoints[j, :2].astype(int)), 6, color, -1)
    return img


def draw_coco17_skeleton_batch(imgs, keypoints_batch, show_confidence=False, joint_conf_thr=None,
                             frame_indices=None, lr_swap_events=None):
    assert len(imgs) == len(keypoints_batch)
    keypoints_batch = to_numpy(keypoints_batch)
    thresholds = np.broadcast_to(0 if joint_conf_thr is None else joint_conf_thr, (17,))
    flagged = {}
    for event in lr_swap_events or []:
        for frame in range(event["start_frame"], event["end_frame"] + 1):
            flagged.setdefault(frame, []).append(event["limb"])
    imgs_out = []
    for i in range(len(imgs)):
        keypoints = keypoints_batch[i]
        # Show filled positions too, while reporting the original model scores.
        img = draw_coco17_skeleton(imgs[i], keypoints[:, :2] if show_confidence else keypoints, 0)
        frame_index = i if frame_indices is None else frame_indices[i]
        legend = "LEFT: orange | RIGHT: cyan (anatomical sides)"
        if frame_index in flagged:
            legend += " | L/R swap interval: " + ", ".join(flagged[frame_index])
        scale = min(0.8, img.shape[1] / 1400)
        origin = (8, img.shape[0] - 12)
        cv2.putText(img, legend, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, legend, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
        if show_confidence:
            names = ["nose", "L eye", "R eye", "L ear", "R ear", "L shoulder", "R shoulder",
                     "L elbow", "R elbow", "L wrist", "R wrist", "L hip", "R hip",
                     "L knee", "R knee", "L ankle", "R ankle"]
            frame_index = i if frame_indices is None else frame_indices[i]
            lines = [f"Frame {frame_index} | raw confidence / threshold (0 = off)"]
            lines += ["   ".join(f"{names[j]}: {keypoints[j, 2]:.2f}/{thresholds[j]:g}" for j in range(k, min(k + 3, 17)))
                      for k in range(0, 17, 3)]
            lines.append("Red rings = below interpolation threshold")
            scale = min(0.8, img.shape[1] / 1050)
            spacing = max(14, int(34 * scale))
            for row, line in enumerate(lines):
                origin = (8, (row + 1) * spacing)
                cv2.putText(img, line, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(img, line, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
            for j, (x, y, confidence) in enumerate(keypoints):
                if thresholds[j] > 0 and confidence <= thresholds[j] and np.isfinite([x, y]).all():
                    cv2.circle(img, (int(x), int(y)), 7, (255, 0, 0), 2)
        imgs_out.append(img)
    return imgs_out
