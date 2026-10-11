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


def trajectory_loss_settings(opt):
    """Validate the optional loss and locate displacement in the dataset layout."""
    enabled = getattr(opt, 'accumulate_trajectory_loss', False)
    weight = opt.weights.get('trajectory', 0)
    horizon = getattr(opt, 'trajectory_window_frames', 9)
    if not isinstance(enabled, bool):
        raise ValueError('accumulate_trajectory_loss must be a boolean.')
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
        raise ValueError("weights['trajectory'] must be finite and nonnegative.")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError('trajectory_window_frames must be a positive integer.')
    if not enabled or weight == 0:
        return False, None
    if 'root_velo' not in opt.pose_feature:
        raise ValueError('Accumulated trajectory loss requires root_velo.')
    root_size = 0
    if 'root_pos' in opt.pose_feature:
        root_size = 1 if opt.condition_root_x_only else (2 if opt.no_condition_root_y else 3)
    if opt.frame_size < root_size + 3:
        raise ValueError('Root displacement does not fit frame_size.')
    return True, slice(root_size, root_size + 3)


def accumulated_root_trajectory_loss(prediction, target, displacement_std):
    """MSE over all XYZ trajectory prefixes, batches, and predicted steps (m^2).

    Inputs are normalized displacements shaped [batch, steps, 3]. Multiplying
    their difference by checkpoint std recovers metres; the mean cancels.
    Both trajectories start at the same GT root. No detach on predictions:
    later prefix errors train every earlier contributing displacement.
    """
    if prediction.ndim != 3 or prediction.shape != target.shape or prediction.shape[-1] != 3 or prediction.shape[1] < 1:
        raise ValueError('Trajectory loss requires matching [batch, steps, 3] displacements.')
    error_m = (prediction.float() - target.detach().float()) * displacement_std
    return error_m.cumsum(dim=1).square().mean()


def masked_phase_loss(output, target, mask, future_weights):
    """Weighted mean over labelled batch/future positions and two coordinates.

    Renormalizing observed future weights preserves the all-labelled legacy
    reduction, without dilution by missing frames. Sanitize BEFORE subtraction
    so even NaN placeholders at unlabelled positions cannot poison gradients.
    """
    if mask is None:
        mask = torch.ones(target.shape[:-1], dtype=torch.bool, device=target.device)
    else:
        mask = mask.to(device=target.device, dtype=torch.bool)
    if mask.shape != target.shape[:-1]:
        raise ValueError('Phase mask must match batch/future dimensions.')
    safe_target = torch.where(mask.unsqueeze(-1), target.detach(), torch.zeros_like(target))
    safe_output = torch.where(mask.unsqueeze(-1), output, torch.zeros_like(output))
    weights = mask.to(output.dtype) * future_weights
    errors = (safe_output - safe_target).square().mean(dim=-1)
    return (errors * weights).sum() / weights.sum().clamp_min(torch.finfo(output.dtype).eps)
