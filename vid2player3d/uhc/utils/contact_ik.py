"""Contact anchors and two-bone leg IK in SMPL's 24-joint order.

Only hip, knee and ankle local rotations change. Translation, root rotation,
bone lengths and the upper body are fixed. All geometry is in meters, Z-up.
Stationarity is a prediction, not a measured contact force.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class IKSettings:
    enter_probability: float = .8
    exit_probability: float = .5
    max_foot_height: float = .08
    min_foot_height: float = -.10
    min_contact_frames: int = 3
    blend_frames: int = 3
    min_knee_flexion_degrees: float = 5.
    max_knee_flexion_degrees: float = 155.
    max_joint_change_degrees: float = 60.


LEGS = ((1, 4, 7, 10), (2, 5, 8, 11))
MODE_NAMES = ('swing', 'ankle', 'toe', 'flat')


def forward_kinematics(local, offsets, parents, root):
    """Batched or single-frame FK; root is the actual SMPL pelvis position."""
    rotations = np.empty_like(local)
    positions = np.empty(local.shape[:-2] + (3,), dtype=np.float64)
    rotations[..., 0, :, :] = local[..., 0, :, :]
    positions[..., 0, :] = root
    for j in range(1, len(parents)):
        p = parents[j]
        rotations[..., j, :, :] = rotations[..., p, :, :] @ local[..., j, :, :]
        positions[..., j, :] = positions[..., p, :] + np.einsum(
            '...ij,j->...i', rotations[..., p, :, :], offsets[j])
    return positions, rotations


def contact_intervals(probabilities, foot_height, settings):
    """Hysteresis per ankle/toe; reject short intervals of each support mode."""
    modes = np.zeros((len(probabilities), 2), dtype=np.int8)
    for side in range(2):
        active = np.zeros(2, dtype=bool)
        for t in range(len(modes)):
            p = probabilities[t, side * 2:side * 2 + 2]
            active = p >= np.where(active, settings.exit_probability, settings.enter_probability)
            active &= settings.min_foot_height <= foot_height[t, side] <= settings.max_foot_height
            modes[t, side] = int(active[0]) + 2 * int(active[1])
    intervals = []
    for side in range(2):
        changes = np.r_[0, np.flatnonzero(np.diff(modes[:, side])) + 1, len(modes)]
        for start, end in zip(changes[:-1], changes[1:]):
            mode = int(modes[start, side])
            if mode and end - start >= settings.min_contact_frames:
                intervals.append((side, int(start), int(end), mode))
            else:
                modes[start:end, side] = 0
    return modes, intervals


def _align_vectors(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    cross = np.cross(a, b)
    sine, cosine = np.linalg.norm(cross), np.clip(a @ b, -1., 1.)
    if sine < 1e-9:
        if cosine > 0:
            return np.eye(3)
        axis = np.cross(a, np.eye(3)[np.argmin(np.abs(a))])
        return Rotation.from_rotvec(axis / np.linalg.norm(axis) * np.pi).as_matrix()
    return Rotation.from_rotvec(cross / sine * np.arctan2(sine, cosine)).as_matrix()


def _solve_leg(local, offsets, parents, root, leg, ankle_target, ankle_rotation, settings):
    hip, knee, ankle, _ = leg
    result = local.copy()
    positions, rotations = forward_kinematics(result, offsets, parents, root)
    h, k, a = positions[[hip, knee, ankle]]
    l1, l2 = np.linalg.norm(k - h), np.linalg.norm(a - k)
    vector = ankle_target - h
    distance = np.linalg.norm(vector)
    fallback = a - h if np.linalg.norm(a - h) > 1e-8 else k - h
    direction = vector / distance if distance > 1e-8 else fallback / np.linalg.norm(fallback)
    # Flexion 0 is straight. Keep a small bend and avoid fully folded legs.
    flex = np.deg2rad([settings.max_knee_flexion_degrees, settings.min_knee_flexion_degrees])
    bounds = np.sqrt(l1*l1 + l2*l2 + 2*l1*l2*np.cos(flex))
    reach = np.clip(distance, bounds[0], bounds[1])
    limited = abs(reach - distance) > 1e-6
    target = h + direction * reach
    # Preserve the input knee's bending side; never switch IK branches.
    bend = k - h - direction * ((k - h) @ direction)
    if np.linalg.norm(bend) < 1e-6:
        bend = rotations[hip][:, 0].copy()
        bend -= direction * (bend @ direction)
    if np.linalg.norm(bend) < 1e-6:
        bend = np.cross(direction, np.eye(3)[np.argmin(np.abs(direction))])
    bend /= np.linalg.norm(bend)
    along = (l1*l1 - l2*l2 + reach*reach) / (2*reach)
    desired_knee = h + direction * along + bend * np.sqrt(max(l1*l1 - along*along, 0.))
    hip_world = _align_vectors(k-h, desired_knee-h) @ rotations[hip]
    result[hip] = rotations[parents[hip]].T @ hip_world
    positions, rotations = forward_kinematics(result, offsets, parents, root)
    knee_world = _align_vectors(positions[ankle]-positions[knee], target-positions[knee]) @ rotations[knee]
    result[knee] = rotations[hip].T @ knee_world
    result[ankle] = knee_world.T @ ankle_rotation
    # Bound corrections relative to the observed pose; leave residual error
    # visible rather than forcing extreme rotations or moving the pelvis.
    for j in (hip, knee, ankle):
        delta = Rotation.from_matrix(result[j] @ local[j].T).as_rotvec()
        angle = np.linalg.norm(delta)
        cap = np.deg2rad(settings.max_joint_change_degrees)
        if angle > cap:
            result[j] = Rotation.from_rotvec(delta * cap / angle).as_matrix() @ local[j]
            limited = True
    return result, limited


def correct_contact_poses(pose_aa, pelvis, rest_joints, parents, probabilities,
                          foot_height, fps=30., enabled=True, settings=None):
    """Return corrected poses plus per-frame diagnostics, with fixed pelvis.

    Flat support fixes ankle position/orientation. Toe support fixes the toe
    while retaining the input foot orientation (allowing a pivot). Ankle-only
    support fixes the ankle and retains input foot orientation. Anchors stay at
    their touchdown heights; this pass does not redo grounding.
    """
    settings = settings or IKSettings()
    pose = np.asarray(pose_aa)
    n = len(pose)
    parents = np.asarray(parents, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    foot_height = np.asarray(foot_height, dtype=float)
    if (pose.shape != (n, 72) or n < 2 or np.shape(pelvis) != (n, 3)
            or np.shape(rest_joints) != (24, 3) or parents.shape != (24,)
            or probabilities.shape != (n, 4) or foot_height.shape != (n, 2)):
        raise ValueError('Invalid SMPL/contact IK input shapes')
    if not all(np.isfinite(x).all() for x in (pose, pelvis, rest_joints, probabilities, foot_height)):
        raise ValueError('Contact IK inputs must be finite')
    if fps <= 0 or not np.isfinite(fps) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('Invalid FPS or stationary probabilities')
    offsets = np.asarray(rest_joints).copy()
    offsets[1:] -= np.asarray(rest_joints)[parents[1:]]
    if any(np.linalg.norm(offsets[j]) < 1e-6 for leg in LEGS for j in leg[1:]):
        raise ValueError('Degenerate leg bone lengths')
    local = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(n, 24, 3, 3)
    before, global_rot = forward_kinematics(local, offsets, parents, pelvis)
    modes, intervals = contact_intervals(probabilities, foot_height, settings)
    output = pose.copy()
    anchors = np.full((n, 2, 3), np.nan)
    blend = np.zeros((n, 2))
    limited = np.zeros((n, 2), dtype=bool)
    for side, start, end, mode in intervals:
        leg = LEGS[side]
        hip, knee, ankle, toe = leg
        contact_joint = toe if mode == 2 else ankle
        anchor = before[start, contact_joint].copy()
        for t in range(start, end):
            anchors[t, side] = anchor
            weight = min(1., (t-start+1) / settings.blend_frames, (end-t) / settings.blend_frames)
            # Smoothstep at contact transitions.
            blend[t, side] = weight*weight*(3-2*weight) if enabled else 0.
            if not enabled:
                continue
            foot_rotation = global_rot[start if mode == 3 else t, ankle]
            ankle_target = anchor - foot_rotation @ offsets[toe] if mode == 2 else anchor
            solved, limited[t, side] = _solve_leg(
                local[t], offsets, parents, pelvis[t], leg, ankle_target, foot_rotation, settings)
            for j in (hip, knee, ankle):
                delta = Rotation.from_matrix(solved[j] @ local[t, j].T).as_rotvec()
                blended = Rotation.from_rotvec(delta * blend[t, side]).as_matrix() @ local[t, j]
                output[t, j*3:j*3+3] = Rotation.from_matrix(blended).as_rotvec()
    after, _ = forward_kinematics(
        Rotation.from_rotvec(output.reshape(-1, 3)).as_matrix().reshape(n, 24, 3, 3),
        offsets, parents, pelvis)
    feet = [7, 10, 8, 11]
    pre_feet, post_feet = before[:, feet].reshape(n, 2, 2, 3), after[:, feet].reshape(n, 2, 2, 3)
    toe_mode = modes == 2
    pre_point = np.where(toe_mode[..., None], pre_feet[:, :, 1], pre_feet[:, :, 0])
    post_point = np.where(toe_mode[..., None], post_feet[:, :, 1], post_feet[:, :, 0])
    residual = np.linalg.norm(post_point - anchors, axis=-1)
    # Speeds are backward differences; frame 0 is undefined, not zero speed.
    speed = []
    for feet_pos in (pre_feet, post_feet):
        s = np.full((n, 2, 2), np.nan)
        s[1:] = np.linalg.norm(np.diff(feet_pos[..., :2], axis=0), axis=-1) * fps
        speed.append(np.where(toe_mode, s[:, :, 1], s[:, :, 0]))
    stable = (modes[1:] != 0) & (modes[1:] == modes[:-1])
    summary = {
        'enabled': bool(enabled), 'settings': asdict(settings),
        'interval_count': len(intervals), 'contact_frames_left_right': (modes != 0).sum(0).tolist(),
        'limited_frames_left_right': limited.sum(0).tolist(),
        'root_translation_change_max_m': 0., 'root_rotation_change_max_rad': 0.,
        'mean_contact_speed_before_mps': float(speed[0][1:][stable].mean()) if stable.any() else None,
        'mean_contact_speed_after_mps': float(speed[1][1:][stable].mean()) if stable.any() else None,
        'max_anchor_residual_m': float(np.nanmax(residual)) if intervals else None,
        'contact_source': 'GVHMR stationary probabilities + pre-IK skinned foot height; predicted, not measured',
    }
    return output, {
        'summary': summary, 'contact_mode': modes, 'stationary_probability': probabilities,
        'foot_height_before_m': foot_height, 'anchor_m': anchors, 'blend_weight': blend,
        'limited': limited, 'anchor_residual_m': residual,
        'foot_change_m': np.linalg.norm(post_feet-pre_feet, axis=-1).max(-1),
        'speed_before_mps': speed[0], 'speed_after_mps': speed[1],
        'feet_before_m': pre_feet, 'feet_after_m': post_feet,
    }
