"""Repeatable high-level policy validation, with bounded CPU-side reporting."""
import json
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from vid2player.utils.controller_evaluation import (
    BALL_PRESETS, RETURN_TARGETS, configure_eval_args, _cpu, _number)


def configure_validation_args(args):
    if args.eval_controller:
        raise ValueError('Choose either --validation or --eval_controller.')
    if args.horovod:
        raise ValueError('--validation runs on one device; omit --horovod.')
    if args.validation_shots_per_preset < 1:
        raise ValueError('--validation_shots_per_preset must be positive.')
    if args.validation_num_envs < 0:
        raise ValueError('--validation_num_envs must be nonnegative (0 means automatic).')
    if args.eval_ball_x is not None or args.eval_ball_height is not None:
        raise ValueError('--validation uses its complete suite; omit single-ball overrides.')
    configure_eval_args(args)
    limit = args.validation_num_envs or (8 if args.rl_device == 'cpu' else 90)
    args.num_envs = min(limit, 18*args.validation_shots_per_preset)
    if args.seed < 0:
        raise ValueError('--validation requires a nonnegative seed.')
    # IsaacGym defaults to GPU PhysX even with a CPU policy unless overridden.
    if args.rl_device == 'cpu':
        args.use_gpu = args.use_gpu_pipeline = False
        args.device = args.sim_device_type = args.sim_device = 'cpu'
        args.compute_device_id = args.device_id = 0


def select_validation_balls(filename, count):
    """Pick distinct nearest neighbours per preset from a fixed bounded sample."""
    if count < 1:
        raise ValueError('Shot count per preset must be positive.')
    data = np.load(filename, mmap_mode='r')
    if data.ndim != 2 or data.shape[1] != 307 or not len(data):
        raise ValueError('Expected incoming ball data with shape [N,307].')
    indices = np.linspace(0, len(data)-1, min(len(data), 8192), dtype=int)
    sampled = np.asarray(data[indices])
    trajectories = sampled[:, 7:].reshape(-1, 100, 3)
    nearest = np.abs(trajectories[:, :, 1]+13).argmin(axis=1)
    intercepts = trajectories[np.arange(len(indices)), nearest]
    valid = np.isfinite(sampled).all(axis=1) & (np.abs(intercepts[:, 1]+13) < .75)
    if valid.sum() < len(BALL_PRESETS)*count:
        raise ValueError('Not enough distinct finite baseline-reaching trajectories for this suite.')
    used, rows, cases = set(), [], []
    for preset, (x, z) in BALL_PRESETS.items():
        scores = (intercepts[:, 0]-x)**2 + 4*(intercepts[:, 2]-z)**2
        scores[~valid] = np.inf
        for selected in sorted(range(len(indices)), key=lambda i: (scores[i], int(indices[i]))):
            if selected in used or not np.isfinite(scores[selected]):
                continue
            used.add(selected)
            rows.append(sampled[selected].copy())
            cases.append({'preset': preset, 'source_row': int(indices[selected]),
                          'requested_intercept_xz_m': [x, z],
                          'preview_intercept_xyz_m': intercepts[selected].tolist(),
                          'selection_distance': float(scores[selected])})
            if sum(case['preset'] == preset for case in cases) == count:
                break
    return np.stack(rows), cases


def prepare_validation(args, cfg, cfg_train):
    if cfg['name'] not in ('MVAEControllerVis', 'PhysicsMVAEController') or cfg['env']['vid2player'].get('dual_mode'):
        raise ValueError('--validation supports single-player PhysicsMVAEController configs.')
    task = cfg['env']['vid2player']
    source = str(Path(task['ball_traj_file']).resolve())
    rows, balls = select_validation_balls(source, args.validation_shots_per_preset)
    output = Path(args.eval_output or ('out/controller_validation/' + getattr(args, 'cfg_name', args.cfg) + '_' +
                  Path(args.eval_checkpoint).stem + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    output.mkdir(parents=True, exist_ok=False)
    temporary = tempfile.TemporaryDirectory(prefix='controller_validation_')
    ball_file = Path(temporary.name)/'launches.npy'
    np.save(ball_file, rows)
    cfg['name'] = 'PhysicsMVAEController'
    cfg['env']['is_train'] = False
    cfg['env']['numEnvs'] = args.num_envs
    cfg_train['params']['config']['num_actors'] = args.num_envs
    # Permit measured outgoing flights even when training terminates on estimates.
    cfg['env']['enableEarlyTermination'] = False
    cfg['env']['episodeLength'] = max(cfg['env']['episodeLength'], args.eval_steps+2)
    task['test_mode'] = 'fixed_evaluation'
    task['ball_traj_file_test'] = str(ball_file)
    cfg['env'].pop('ball_traj_file_test', None)
    cfg_train['params']['config']['pretrained_model_cp'] = None
    cfg_train['params']['config']['device_name'] = args.rl_device
    cfg_train['params']['load_checkpoint'] = False
    cfg_train['params']['config']['load_checkpoint'] = False
    suite = [dict(ball, ball_index=i, target=target, target_xyz_m=list(xyz), seed=args.seed+i)
             for i, ball in enumerate(balls) for target, xyz in RETURN_TARGETS.items()]
    metadata = {'checkpoint': str(Path(args.eval_checkpoint).resolve()), 'arguments': vars(args),
                'source_file': source, 'suite': suite, 'config': cfg, 'training_config': cfg_train,
                'execution': {'protocol_version': 3,
                              'num_envs': args.num_envs, 'batches': (len(suite)+args.num_envs-1)//args.num_envs,
                              'method': 'Batched IsaacGym physics and policy inference; completed slots are excluded from metrics.'},
                'protocol': 'Distinct nearest rows from a fixed evenly spaced sample of at most 8192 incoming rows; '
                            'one combined actor reset per batch, mean actions, paired launch and initial pose across targets, '
                            'common initial motion pose across slots, fixed batch order; '
                            'batch RNG seeded by the last shot seed; stochastic recovery draws '
                            'depend on batch width (use the same concurrency when comparing checkpoints). '
                            'configured recovery, no timed relaunch. Early estimate/miss termination disabled; '
                            'physical and movement-boundary termination retained.',
                'definitions': {
                    'return_in_rate': 'First observed ground bounce strictly after contact inside singles court '
                                      '(-4.11<x<4.11, 0<y<11.89), divided by all shots. Net clearance not verified.',
                    'unresolved': 'Contact without an observed later landing before termination/step limit; not counted as in.',
                    'landing_error_m': 'XY distance of first observed post-contact landing to target, including out landings; missing excluded.',
                    'estimated_in': 'Lookup-table prediction at contact; reported separately from measured landing.',
                    'regularization': 'Squared residual action means and squared action-mean excess beyond +/-1, '
                                      'with configured weights. No PPO actor/critic training loss during validation.',
                    'selection': 'Presets label requested locations; actual preview intercepts and selection distances are stored. '
                                 'This is a fixed training-pool benchmark, not a held-out dataset.'}}
    (output/'validation_config.json').write_text(json.dumps(metadata, indent=2, default=str)+'\n')
    return output, suite, rows, temporary


def _empty_shot(case):
    return {'case': case, 'steps': [], 'contact_step': None, 'landing_xyz_m': None,
            'return_in': False, 'unresolved': False, 'landing_error_m': None,
            'min_ball_racket_distance_m': None, 'total_reward': 0., 'end_reason': 'step_limit'}


def _reset_slots(player, ids, ball_indices):
    """Reset only selected slots, without relaunching any other flight."""
    import torch

    task = player.task
    task._reset_reaction_buf[:] = False
    recovery = task._reset_recovery_buf.clone()
    task._reset_recovery_buf[:] = False
    task._reset_reaction_buf[ids] = True
    generator = task._physics_player.task._ball_generator
    generator.sample_idx[ids.to(generator.sample_idx.device)] = torch.as_tensor(
        ball_indices, device=generator.sample_idx.device, dtype=generator.sample_idx.dtype)
    obs = player.env_reset(ids)
    recovery[ids] = False
    task._reset_recovery_buf[:] = recovery
    return obs


def rollout_batch(player, args, cases):
    """Advance all slots together; retain metrics only until each shot finishes.

    A short final batch uses the existing simulator allocation. Unused/finished
    slots receive zero high-level actions and are reset if they terminate, so
    invalid states cannot leak into subsequent batched network calls.
    """
    import torch
    from vid2player.utils.config import set_seed

    task = player.task
    physics = task._physics_player.task
    width = task.num_envs
    if not 0 < len(cases) <= width:
        raise ValueError('Validation batch must fit the allocated environments.')
    # The original single-env evaluator always used initial motion row zero.
    # Preserve that pose for every slot rather than confounding targets with
    # different initialization rows selected by the training data loader.
    mvae = task._mvae_player
    mvae._init_data[:width] = mvae._init_data[:1].clone().expand_as(mvae._init_data[:width])
    task._reset_reaction_buf[:] = False
    task._reset_recovery_buf[:] = False
    # GPU state setters are deferred until simulate() and must be called only
    # once per step. Per-slot env_reset calls discard all but the final slot's
    # pending actor/DOF reset on the GPU pipeline (CPU applies them immediately).
    # Fixed-evaluation motion resets are deterministic; random targets/timers
    # drawn during reset are replaced below. Seed subsequent stochastic recovery
    # from the last real case, including when padding a short final batch.
    set_seed(cases[-1]['seed'], False)
    batch_cases = [cases[min(i, len(cases)-1)] for i in range(width)]
    _reset_slots(player, torch.arange(width, device=task.device),
                 [case['ball_index'] for case in batch_cases])
    task._tar_time_total[:] = args.eval_steps+1
    targets = [case['target_xyz_m'] for case in batch_cases]
    task._target_bounce_pos[:] = torch.tensor(targets, device=task.device)
    task._compute_observations()
    obs = player.obs_to_torch(task.obs_buf.to(player.device))
    player.get_batch_size(obs['obs'], 1)
    if player.is_rnn:
        player.init_rnn()
    low_player = task._physics_player
    if getattr(low_player, 'is_rnn', False):
        low_player.batch_size = width
        low_player.init_rnn()
    shots = [_empty_shot(case) for case in cases]
    active = np.zeros(width, dtype=bool)
    active[:len(cases)] = True
    aux_weight = task.cfg_v2p.get('aux_loss_specs', {}).get('dof_res', 0.)
    bounds_weight = player.config.get('bounds_loss_coef', 0.) or 0.
    weights = task.cfg_v2p.get('reward_weights', {})
    for step in range(args.eval_steps):
        with torch.no_grad():
            action = player.get_action(obs, is_determenistic=True).clone()
        means = _cpu(action)
        bad = ~np.isfinite(means).all(axis=1)
        for i in np.flatnonzero(active & bad):
            shots[i]['end_reason'] = 'nonfinite_action'
            active[i] = False
        if not active.any():
            break
        action[torch.as_tensor(~active, device=action.device)] = 0
        obs, reward, done, info = player.env_step(player.env, action)
        # Transfer each batched tensor once per step, not once per shot.
        ball, racket, root, bodies, contact, bounce_now, bounce, estimated = [
            _cpu(v) for v in (task._ball_pos, task._racket_pos, task._root_pos,
                             physics._rigid_body_pos, physics._has_racket_ball_contact,
                             physics._has_bounce_now, physics._bounce_pos, task._est_bounce_in)]
        rewards = _cpu(reward).reshape(width, -1)[:, 0]
        dones = _cpu(done).reshape(-1).astype(bool)
        finite = (np.isfinite(ball).all(axis=1) & np.isfinite(racket).all(axis=1)
                  & np.isfinite(root).all(axis=1) & np.isfinite(bodies).all(axis=(1, 2))
                  & np.isfinite(rewards))
        raw = None if info.get('sub_rewards') is None else _cpu(info['sub_rewards'])
        names = [] if raw is None else info['sub_rewards_names'].split(',')
        for i in np.flatnonzero(active):
            shot = shots[i]
            mean = means[i]
            residual = mean[task._num_mvae_action:task._num_mvae_action+task._num_res_dof_action]
            aux = float(np.square(residual).sum())
            bounds = float(np.square(np.maximum(np.abs(mean)-1, 0)).sum())
            value = _number(rewards[i])
            if value is not None:
                shot['total_reward'] += value
            distance = _number(np.linalg.norm(ball[i]-racket[i]))
            closest = shot['min_ball_racket_distance_m']
            if distance is not None:
                shot['min_ball_racket_distance_m'] = distance if closest is None else min(closest, distance)
            # Never count an incoming or ambiguous same-control-step bounce.
            if shot['landing_xyz_m'] is None and shot['contact_step'] is not None and bounce_now[i]:
                if np.isfinite(bounce[i]).all():
                    shot['landing_xyz_m'] = bounce[i].tolist()
            if shot['contact_step'] is None and contact[i]:
                shot['contact_step'] = step+1
            components = {} if raw is None else dict(zip(names, [_number(v) for v in raw[i]]))
            weighted = {k: None if v is None else _number(v*weights.get(k[:-7], 0.)) for k, v in components.items()}
            ended = dones[i] or not finite[i]
            if ended:
                outside = np.any(root[i, :2] < task.cfg_v2p['court_min']) or np.any(root[i, :2] > task.cfg_v2p['court_max'])
                shot['end_reason'] = 'nonfinite_state' if not finite[i] else ('movement_boundary' if outside else 'episode_termination')
            shot['steps'].append({'step': step+1, 'time_s': (step+1)*physics.dt, 'reward': value,
                                 'cumulative_reward': shot['total_reward'], 'ball_racket_distance_m': distance,
                                 'contact': shot['contact_step'] is not None, 'estimated_in': bool(estimated[i]),
                                 'reward_components': components, 'weighted_reward_components': weighted,
                                 'aux_dof_res_loss': _number(aux), 'bounds_loss': _number(bounds),
                                 'weighted_aux_dof_res_loss': _number(aux*aux_weight),
                                 'weighted_bounds_loss': _number(bounds*bounds_weight),
                                 'ball_xyz_m': [_number(v) for v in ball[i]]})
            if ended or shot['landing_xyz_m'] is not None:
                if not ended:
                    shot['end_reason'] = 'landing_observed'
                active[i] = False
        if not active.any() or step+1 == args.eval_steps:
            break
        # Park unsafe/completed slots before the next forward pass. Healthy
        # completed slots may keep simulating but never contribute more metrics.
        parked = np.flatnonzero(~active & (dones | ~finite | bad))
        if len(parked):
            _reset_slots(player, torch.as_tensor(parked, device=task.device), [0]*len(parked))
            for states in (getattr(player, 'states', None), getattr(low_player, 'states', None)):
                if states is not None:
                    for state in states:
                        state[:, torch.as_tensor(parked, device=state.device)] = 0
        task._reset_reaction_buf[:] = False
        task._reset_recovery_buf[torch.as_tensor(~active, device=task.device)] = False
        obs = player.env_reset(torch.empty(0, dtype=torch.long, device=task.device))
    for shot in shots:
        landing = shot['landing_xyz_m']
        shot['return_in'] = landing is not None and -4.11 < landing[0] < 4.11 and 0 < landing[1] < 11.89
        shot['unresolved'] = shot['contact_step'] is not None and landing is None
        if landing is not None:
            shot['landing_error_m'] = float(np.linalg.norm(np.array(landing[:2])-shot['case']['target_xyz_m'][:2]))
    return shots


def rollout_shot(player, args, case):
    """Single-environment compatibility wrapper around the batched protocol."""
    return rollout_batch(player, args, [case])[0]


def aggregate(shots):
    def mean(key):
        values = [shot[key] for shot in shots if shot[key] is not None]
        return float(np.mean(values)) if values else None
    n = len(shots)
    components = sorted({k for s in shots for r in s['steps'] for k in r['weighted_reward_components']})
    return {'shots': n, 'contacts': sum(s['contact_step'] is not None for s in shots),
            'contact_rate': sum(s['contact_step'] is not None for s in shots)/n,
            'return_in_rate': sum(s['return_in'] for s in shots)/n,
            'observed_landings': sum(s['landing_xyz_m'] is not None for s in shots),
            'unresolved': sum(s['unresolved'] for s in shots),
            'estimated_in_rate': sum(bool(s['steps']) and s['steps'][-1]['estimated_in'] and s['contact_step'] is not None
                                     for s in shots)/n,
            'mean_landing_error_m': mean('landing_error_m'),
            'mean_min_ball_racket_distance_m': mean('min_ball_racket_distance_m'),
            'mean_total_reward': mean('total_reward'),
            'mean_weighted_reward_component_totals': {
                k: float(np.mean([sum(r['weighted_reward_components'].get(k) or 0 for r in s['steps']) for s in shots]))
                for k in components},
            'mean_regularization_totals': {
                k: float(np.mean([sum(r[k] or 0 for r in s['steps']) for s in shots]))
                for k in ('weighted_aux_dof_res_loss', 'weighted_bounds_loss')},
            'termination_reasons': dict(Counter(s['end_reason'] for s in shots))}


def save_validation_plot(filename, shots, summary):
    from matplotlib import cm
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    figure = Figure(figsize=(16, 13), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 3).ravel()
    presets, targets = list(BALL_PRESETS), list(RETURN_TARGETS)
    cmap = cm.get_cmap('viridis').copy()
    cmap.set_bad('#555555')
    for ax, field, title in zip(axes[:3], ['contact_rate', 'return_in_rate', 'mean_landing_error_m'],
                               ['Contact rate (all shots)', 'Measured return-in rate (all shots)', 'Landing error (m; observed only)']):
        grid = np.array([[summary['by_preset_target'][p][t][field] for t in targets] for p in presets], dtype=float)
        im = ax.imshow(np.ma.masked_invalid(grid), aspect='auto', vmin=0,
                       vmax=1 if field.endswith('rate') else None, cmap=cmap)
        ax.set(xticks=range(3), xticklabels=targets, yticks=range(6), yticklabels=presets, title=title)
        for i in range(6):
            for j in range(3):
                ax.text(j, i, 'n/a' if not np.isfinite(grid[i, j]) else f'{grid[i,j]:.2f}', ha='center', va='center', color='white')
        figure.colorbar(im, ax=ax, shrink=.8)
    x = np.arange(1, len(shots)+1)
    for ax, field, title in zip(axes[3:6], ['total_reward', 'min_ball_racket_distance_m', 'landing_error_m'],
                               ['Episode reward', 'Closest ball–racket distance (m)', 'Landing-to-target error (m)']):
        ax.plot(x, [np.nan if s[field] is None else s[field] for s in shots], '.-')
        if all(s[field] is None for s in shots):
            ax.text(.5, .5, 'No observed values', transform=ax.transAxes, ha='center')
        ax.set(title=title, xlabel='Shot index')
        ax.grid(alpha=.25)
    names = sorted({k for s in shots for r in s['steps'] for k in r['weighted_reward_components']})
    for name in names:
        axes[6].plot(x, [sum(r['weighted_reward_components'].get(name) or 0 for r in s['steps']) for s in shots], label=name)
    axes[6].set(title='Weighted reward component totals', xlabel='Shot index')
    if names:
        axes[6].legend(fontsize=8)
    for key in ['weighted_aux_dof_res_loss', 'weighted_bounds_loss']:
        axes[7].plot(x, [sum(r[key] or 0 for r in s['steps']) for s in shots], label=key)
    axes[7].set(title='Regularization totals (no PPO update)', xlabel='Shot index')
    axes[7].legend(fontsize=8)
    reasons = summary['overall']['termination_reasons']
    axes[8].barh(list(reasons), list(reasons.values()))
    axes[8].set(title='Termination reasons', xlabel='Shots')
    overall = summary['overall']
    figure.suptitle(f"High-level validation: {len(shots)} shots | contact {overall['contact_rate']:.1%} | "
                   f"measured in {overall['return_in_rate']:.1%} | unresolved {overall['unresolved']}\n"
                   'Missing landings excluded from error; return-in requires observed bounce, net clearance not verified', fontsize=13)
    figure.savefig(filename, dpi=150)


def validate_controller(player, args, evaluation):
    import torch

    output, suite, rows, temporary = evaluation
    try:
        checkpoint = torch.load(args.eval_checkpoint, map_location=player.device)
        player.model.load_state_dict(checkpoint['model'], strict=True)
        if player.normalize_input:
            player.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        player.model.eval()
        shots = []
        width = player.task.num_envs
        for begin in range(0, len(suite), width):
            shots.extend(rollout_batch(player, args, suite[begin:begin+width]))
            print(f'Validation {len(shots)}/{len(suite)} shots ({width} concurrent environments)', flush=True)
        summary = {'overall': aggregate(shots),
                   'by_preset': {p: aggregate([s for s in shots if s['case']['preset'] == p]) for p in BALL_PRESETS},
                   'by_target': {t: aggregate([s for s in shots if s['case']['target'] == t]) for t in RETURN_TARGETS},
                   'by_preset_target': {p: {t: aggregate([s for s in shots if s['case']['preset'] == p and s['case']['target'] == t])
                                            for t in RETURN_TARGETS} for p in BALL_PRESETS}}
        (output/'values.json').write_text(json.dumps({'summary': summary, 'shots': shots, 'launch_rows': rows.tolist()},
                                                    separators=(',', ':'), allow_nan=False)+'\n')
        save_validation_plot(output/'metrics.png', shots, summary)
        print('Saved validation:', output.resolve())
        overall = summary['overall']
        print('Contact {:.1%}; measured in {:.1%}; observed landings {}; unresolved {}'.format(
            overall['contact_rate'], overall['return_in_rate'], overall['observed_landings'], overall['unresolved']))
    finally:
        temporary.cleanup()
