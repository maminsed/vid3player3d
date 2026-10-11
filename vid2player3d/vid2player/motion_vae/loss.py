"""Feature multipliers for normalized reconstruction MSE."""
import math

import torch


# Match Video3DPoseDataset.get_window, regardless of pose_feature list order.
FEATURE_DIMS = (
    ('root_pos', 3), ('root_velo', 3), ('joint_pos', 69),
    ('joint_velo', 69), ('joint_rotmat', 144), ('joint_quat', 96),
)


def build_recon_feature_weights(opt, device):
    """Return per-coordinate weights, or None for the exact legacy MSE path.

    Unspecified groups have multiplier 1. The denominator remains frame_size,
    so upweighting root features does not downweight local pose or alter KL.
    """
    configured = getattr(opt, 'recon_feature_weights', None) or {}
    if not configured:
        return None
    if not isinstance(configured, dict):
        raise ValueError('recon_feature_weights must be a feature-to-multiplier dictionary.')
    supported = {name for name, _ in FEATURE_DIMS}
    for name, weight in configured.items():
        if name not in supported or name not in opt.pose_feature:
            raise ValueError('Reconstruction weight names an unknown or disabled feature: ' + str(name))
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError('Reconstruction weight for {} must be finite and nonnegative.'.format(name))
    values = []
    for name, size in FEATURE_DIMS:
        if name not in opt.pose_feature:
            continue
        if name == 'root_pos':
            if opt.condition_root_x_only:
                size = 1
            elif opt.no_condition_root_y:
                size = 2
        values.extend([configured.get(name, 1.0)] * size)
    if len(values) != opt.frame_size:
        raise ValueError('Reconstruction feature dimensions do not match frame_size.')
    if not any(values):
        raise ValueError('At least one reconstruction feature weight must be positive.')
    if all(value == 1 for value in values):
        return None
    return torch.tensor(values, dtype=torch.float32, device=device)


def feature_squared_error(output, target, feature_weights=None):
    """Weight squared errors before the existing batch/feature/future reductions."""
    error = (output - target.detach()).pow(2)
    return error if feature_weights is None else error * feature_weights
