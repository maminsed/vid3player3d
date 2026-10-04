"""Small, aggregate TensorBoard dashboard and static run metadata."""
import json
from pathlib import Path


# Primary cards are autoregressive posterior reconstructions. All values are
# means over evaluated frames except the mean of each rollout's endpoint error.
PRIMARY_METRICS = {
    'recon_weighted_mean': 'recon_loss',
    'kl_weighted_mean': 'kl_loss',
    'root_error_m_mean': 'root_error_m',
    'local_mpjpe_m_mean': 'local_joint_error_m',
    'root_endpoint_error_m_mean': 'root_endpoint_error_m',
}
SECONDARY_METRICS = {
    'world_mpjpe_m_mean': 'world_joint_error_m',
    'root_error_m_p95': 'root_error_p95_m',
    'root_velocity_error_mps_mean': 'root_velocity_error_mps',
    'root_velocity_bias_x_mps_mean': 'root_velocity_bias_x_mps',
    'root_velocity_bias_y_mps_mean': 'root_velocity_bias_y_mps',
    'root_consistency_m_mean': 'root_consistency_m',
    'stationary_pred_root_speed_mps_mean': 'stationary_root_speed_mps',
    'pose_wrist_velocity_error_mps_mean': 'wrist_velocity_error_mps',
}
TRAIN_TAGS = {'recon': 'recon_loss', 'kl': 'kl_loss', 'recon_phase': 'phase_loss'}


def train_tag(metric):
    return 'train/' + TRAIN_TAGS.get(metric, metric)


def _wrist_error(values, prefix=''):
    left = values.get(prefix + 'left_wrist_error_m_mean')
    right = values.get(prefix + 'right_wrist_error_m_mean')
    return None if left is None or right is None else (left + right) / 2


def validation_scalars(summaries):
    """Explicit allowlist: no per-video cards, counts, GT constants, or duplicates."""
    scalars = {}
    for scope_mode, values in summaries.items():
        scope, mode = scope_mode.split('/')
        label = 'val_full' if scope == 'whole_sequence' else 'val_' + scope[len('window_'):]
        diagnostic = 'diagnostics/' + label + '/'
        if mode == 'autoregressive':
            for source, name in PRIMARY_METRICS.items():
                scalars[label + '/' + name] = values.get(source)
            scalars[label + '/wrist_error_m'] = _wrist_error(values)
            for source, name in SECONDARY_METRICS.items():
                scalars[diagnostic + name] = values.get(source)
            scalars[diagnostic + 'fast_motion_wrist_error_m'] = _wrist_error(values, 'fast_motion_')
        elif mode == 'teacher_forced':
            for source, name in [('recon_weighted_mean', 'recon_loss'),
                                 ('root_error_m_mean', 'root_error_m'),
                                 ('local_mpjpe_m_mean', 'local_joint_error_m')]:
                scalars[diagnostic + 'one_step_' + name] = values.get(source)
            scalars[diagnostic + 'one_step_wrist_error_m'] = _wrist_error(values)
            scalars[diagnostic + 'latent_joint_effect_m'] = values.get('latent_joint_effect_m_mean')
            zero, posterior = values.get('recon_zero_latent_mean'), values.get('recon_mean')
            scalars[diagnostic + 'latent_recon_benefit'] = None if zero is None or posterior is None else zero - posterior
        # Zero failures is a model-dependent result, not a configuration constant.
        scalars[diagnostic + mode + '_failed_rollouts'] = values.get('failed_rollouts')
    return {tag: value for tag, value in scalars.items() if value is not None}


def write_run_config(model):
    """One JSON for resolved options, split identity, and model-independent facts."""
    opt = model.opt
    config = {key: value for key, value in vars(opt).items()
              if not key.startswith('_') and not callable(value)}
    metadata = dict(
        schema_version=1,
        config=config,
        validation_split=model.validation_split,
        training_data=dict(
            selected_sequences=len(model.dataset.sequences),
            valid_frames=int((model.dataset.valid_arr.astype(bool) & model.dataset.selected_arr.astype(bool)).sum()),
            available_training_windows=len(model.dataset.rollouts),
            sampled_windows_per_epoch=opt.nseqs,
            batches_per_epoch=len(model.trainset),
        ),
        validation_reference=model.validation.reference_stats if model.validation else None,
        tensorboard=dict(
            scalar_step='sampled training windows',
            validation_frequency_unit='completed training batches',
            primary_groups=['train', 'val_{}'.format(opt.validation_window_frames), 'val_full'],
            secondary_group='diagnostics',
            main_validation_mode='autoregressive conditioning, posterior-mean latent, integrated root',
            aggregation='frame-weighted means across all held-out videos; endpoint errors average rollouts; wrist error averages both wrists',
            latent_recon_benefit='zero-latent MSE minus posterior-mean MSE under teacher forcing; positive means latent helps',
            note='Training samples posterior latents and uses the curriculum; validation always encodes the target. No physics rollout.',
        ),
    )
    path = Path(model.checkpoint_dir) / 'run_config.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    try:
        temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False))
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
