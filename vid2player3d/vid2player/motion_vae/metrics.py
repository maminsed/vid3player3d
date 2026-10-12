"""Physical motion diagnostics shared by held-out evaluation and MP4 reports.

Velocities below are m/s. Dataset *_velo features are displacements per frame.
Pose errors use direct predicted joint positions, not forward kinematics.
"""
import numpy as np


METRICS = (
    'phase_mse', 'phase_weighted', 'phase_angular_error_rad',
    'recon', 'kl', 'recon_weighted', 'kl_weighted', 'recon_zero_latent',
    'trajectory_mse_m2', 'trajectory_weighted',
    'recon_root_pos', 'recon_root_velo', 'recon_joint_pos', 'recon_joint_velo',
    'recon_joint_rotmat', 'recon_joint_quat',
    'latent_joint_effect_m', 'root_error_m', 'root_xy_error_m',
    'latent_mu_rms', 'latent_posterior_std_mean',
    'local_mpjpe_m', 'world_mpjpe_m', 'left_wrist_error_m', 'right_wrist_error_m',
    'root_displacement_error_m', 'root_velocity_error_mps',
    'root_velocity_bias_x_mps', 'root_velocity_bias_y_mps', 'root_velocity_bias_z_mps',
    'joint_velocity_error_mps', 'root_consistency_m',
    'gt_root_speed_mps', 'pred_root_speed_mps',
    'gt_wrist_speed_mps', 'pred_wrist_speed_mps', 'pred_pose_wrist_speed_mps',
    'pose_wrist_velocity_error_mps',
)


def pool_latent_statistics(statistics):
    """Pool moments by frame count, retaining each latent dimension's scale."""
    statistics = [s for s in statistics if s and s['frames']]
    if not statistics:
        return None
    counts = np.asarray([s['frames'] for s in statistics], dtype=float)
    mean = np.average([s['mu_mean_per_dim'] for s in statistics], axis=0, weights=counts)
    square = np.average([np.square(s['mu_rms_per_dim']) for s in statistics], axis=0, weights=counts)
    posterior_std = np.average([s['posterior_std_mean_per_dim'] for s in statistics], axis=0, weights=counts)
    variance = np.average([s['posterior_variance_mean_per_dim'] for s in statistics], axis=0, weights=counts)
    return dict(frames=int(counts.sum()), mu_rms=float(np.sqrt(square.mean())),
                mu_mean_per_dim=mean.tolist(), mu_rms_per_dim=np.sqrt(square).tolist(),
                mu_std_per_dim=np.sqrt(np.maximum(0, square - mean**2)).tolist(),
                posterior_std_mean_per_dim=posterior_std.tolist(),
                posterior_std_mean=float(posterior_std.mean()),
                posterior_variance_mean_per_dim=variance.tolist(),
                z_rms_per_dim=np.sqrt(square + variance).tolist(),
                z_rms=float(np.sqrt((square + variance).mean())))


def add_motion_metrics(result, truth, fields, fps, opt):
    """Annotate each predicted frame; never differentiate across a reset jump."""
    predicted = result['predicted_features']
    gt_root, pred_root = result['roots']
    gt_local = truth[:, fields['joint_pos']].reshape(-1, 23, 3)
    pred_local = predicted[:, fields['joint_pos']].reshape(-1, 23, 3)
    local_error = np.linalg.norm(pred_local - gt_local, axis=-1)
    world_error = np.linalg.norm(result['joints'][1, :, 1:] - result['joints'][0, :, 1:], axis=-1)
    gt_disp = truth[:, fields['root_velo']]
    pred_disp = predicted[:, fields['root_velo']]
    # SMPL left/right wrists are joints 20/21, i.e. indices 19/20 without root.
    wrist_indices = [19, 20]
    for i, row in enumerate(result['rows']):
        for key in METRICS:
            row.setdefault(key, None)
        row['stationary'] = None
        row['fast_motion'] = None
        if row['recon'] is None:
            continue
        velocity_error = (pred_disp[i] - gt_disp[i]) * fps
        row.update(
            root_xy_error_m=float(np.linalg.norm(pred_root[i, :2] - gt_root[i, :2])),
            local_mpjpe_m=float(local_error[i].mean()),
            world_mpjpe_m=float(world_error[i].mean()),
            left_wrist_error_m=float(local_error[i, 19]),
            right_wrist_error_m=float(local_error[i, 20]),
            root_displacement_error_m=float(np.linalg.norm(pred_disp[i] - gt_disp[i])),
            root_velocity_error_mps=float(np.linalg.norm(velocity_error)),
            gt_root_speed_mps=float(np.linalg.norm(gt_disp[i, :2]) * fps),
            pred_root_speed_mps=float(np.linalg.norm(pred_disp[i, :2]) * fps),
        )
        for j, axis in enumerate('xyz'):
            row['root_velocity_bias_' + axis + '_mps'] = float(velocity_error[j])
        row['stationary'] = int(row['gt_root_speed_mps'] < opt.metric_stationary_speed_mps)
        if 'root_pos' in fields:
            row['root_consistency_m'] = float(np.linalg.norm(
                predicted[i, fields['root_pos']] - pred_root[i]))
        if 'joint_velo' in fields:
            gt_joint_vel = truth[i, fields['joint_velo']].reshape(23, 3)
            pred_joint_vel = predicted[i, fields['joint_velo']].reshape(23, 3)
            row['joint_velocity_error_mps'] = float(np.linalg.norm(
                (pred_joint_vel - gt_joint_vel) * fps, axis=-1).mean())
            # These are decoder velocity-head diagnostics, not reset-contaminated
            # numerical derivatives of the displayed trajectory.
            gt_wrist = (gt_joint_vel[wrist_indices] + gt_disp[i]) * fps
            pred_wrist = (pred_joint_vel[wrist_indices] + pred_disp[i]) * fps
            row['gt_wrist_speed_mps'] = float(np.linalg.norm(gt_wrist, axis=-1).max())
            row['pred_wrist_speed_mps'] = float(np.linalg.norm(pred_wrist, axis=-1).max())
            # Derive wrist motion from the displayed pose too: velocity and position
            # heads need not agree. A reset uses the GT preceding pose, not the last
            # displayed prediction, so a reset correction is never counted as speed.
            previous_local = gt_local[i - 1] if row.get('conditioned_from_gt') else pred_local[i - 1]
            pose_wrist = (pred_local[i, wrist_indices] - previous_local[wrist_indices] + pred_disp[i]) * fps
            row['pred_pose_wrist_speed_mps'] = float(np.linalg.norm(pose_wrist, axis=-1).max())
            row['pose_wrist_velocity_error_mps'] = float(np.linalg.norm(pose_wrist - gt_wrist, axis=-1).mean())
            row['fast_motion'] = int(row['gt_wrist_speed_mps'] >= opt.metric_fast_wrist_speed_mps)
    result['metric_thresholds'] = dict(stationary_speed_mps=opt.metric_stationary_speed_mps,
                                       fast_wrist_speed_mps=opt.metric_fast_wrist_speed_mps)


def summarize_rows(rows):
    """Frame-weighted means, with explicit denominators for subgroup metrics."""
    evaluated = [r for r in rows if r['recon'] is not None]
    labelled = sum(bool(r.get('phase_labelled')) for r in evaluated)
    summary = {'predicted_frames': len(evaluated), 'phase_labelled_frames': labelled,
               'phase_label_fraction': labelled / max(1, len(evaluated))}
    for key in METRICS:
        values = [r[key] for r in evaluated if r.get(key) is not None]
        if values:
            summary[key + '_mean'] = float(np.mean(values))
    for key in ('root_error_m', 'local_mpjpe_m', 'left_wrist_error_m', 'right_wrist_error_m'):
        values = [r[key] for r in evaluated if r.get(key) is not None]
        if values:
            summary[key + '_p95'] = float(np.percentile(values, 95))
    for label in ('stationary', 'fast_motion'):
        subset = [r for r in evaluated if r.get(label)]
        summary[label + '_frames'] = len(subset)
        for key in ('root_error_m', 'local_mpjpe_m', 'left_wrist_error_m',
                    'right_wrist_error_m', 'root_velocity_error_mps', 'pred_root_speed_mps',
                    'root_velocity_bias_x_mps', 'root_velocity_bias_y_mps',
                    'root_velocity_bias_z_mps', 'gt_wrist_speed_mps', 'pred_wrist_speed_mps',
                    'pred_pose_wrist_speed_mps', 'pose_wrist_velocity_error_mps'):
            values = [r[key] for r in subset if r.get(key) is not None]
            summary[label + '_' + key + '_mean'] = float(np.mean(values)) if values else None
    return summary
