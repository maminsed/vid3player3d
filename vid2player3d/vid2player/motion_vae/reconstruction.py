"""Named MVAE reconstruction, display-free MP4, and per-frame diagnostics."""
import copy
import csv
from datetime import datetime
import json
from pathlib import Path
import re

import numpy as np
import torch

from vid2player.motion_vae.base import MotionVAEModel
from vid2player.motion_vae.dataset import Video3DPoseDataset
from vid2player.motion_vae.metrics import add_motion_metrics, summarize_rows
from vid2player.motion_vae.loss import build_recon_feature_weights, feature_squared_error


# The dataset concatenates features in this order, regardless of pose_feature order.
FEATURE_ORDER = ('root_pos', 'root_velo', 'joint_pos', 'joint_velo',
                 'joint_rotmat', 'joint_quat')
FEATURE_DIMS = (3, 3, 69, 69, 144, 96)
SMPL_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)
GT_COLOR = (245, 164, 66)
PRED_COLOR = (66, 190, 245)


def feature_slices(opt):
    if opt.condition_root_x_only or opt.no_condition_root_y:
        raise ValueError('Reconstruction recording currently requires full XYZ root features.')
    fields, offset = {}, 0
    for name, size in zip(FEATURE_ORDER, FEATURE_DIMS):
        if name in opt.pose_feature:
            fields[name] = slice(offset, offset + size)
            offset += size
    if offset != opt.frame_size or not {'root_velo', 'joint_pos'}.issubset(fields):
        raise ValueError('Recording requires root_velo and joint_pos, with a matching frame_size.')
    if opt.num_future_predictions != 1:
        raise ValueError('Reconstruction recording currently supports num_future_predictions=1.')
    return fields


def select_sequence(dataset, name):
    """Prefer exact manifest names; reject ambiguous filename/source-key aliases."""
    candidates = []
    sides = ('fg', 'bg') if dataset.opt.side == 'both' else (dataset.opt.side,)
    selected = {seq['base'] for seq in dataset.sequences}
    for video in dataset.manifest:
        for side in sides:
            for index, seq in enumerate(video['sequences'].get(side, [])):
                canonical = '{}:{}:{}'.format(video['name'], side, index)
                aliases = {video['name'], canonical, seq.get('name'), seq.get('source_key')}
                if seq.get('source_export'):
                    aliases.add(Path(seq['source_export']).stem)
                    aliases.add(Path(seq['source_export']).name)
                if name in aliases:
                    candidates.append((canonical, seq))
    exact = [item for item in candidates if item[0] == name]
    candidates = exact or candidates
    if len(candidates) != 1:
        examples = [item[0] for item in candidates] or [v['name'] for v in dataset.manifest[:8]]
        raise ValueError('Sequence {!r} {}. Use an exact name from manifest.json, or '
                         'VIDEO:fg:INDEX. Names: {}'.format(
                             name, 'is ambiguous' if candidates else 'was not found', ', '.join(examples)))
    canonical, seq = candidates[0]
    if seq['base'] not in selected:
        raise ValueError('{} is excluded by the experiment dataset filters.'.format(canonical))
    return canonical, seq


@torch.no_grad()
def reconstruct(model, raw_features, roots, source_frames, fps, resample=None, latent_diagnostics=True,
                latent_mode='posterior', actions=None, detailed_metrics=True):
    """Posterior-mean reconstruction with autoregressive conditioning and root integration.

    In posterior mode targets enter the encoder every step. Zero mode bypasses
    the encoder and feeds back its own predictions with z=0 at every step.
    Explicit-action mode also bypasses the encoder. Each call owns its history
    and root; callers choose the optional condition/root reset schedule.
    Losses measure the raw decoder output before replacing its root_pos feature
    with the integrated root, as is done during runtime motion generation.
    """
    if latent_mode not in ('posterior', 'zero', 'actions'):
        raise ValueError('Unknown latent_mode: ' + latent_mode)
    opt = model.opt
    fields = feature_slices(opt)
    recon_feature_weights = build_recon_feature_weights(opt, model.device)
    if resample is not None and resample <= 0:
        raise ValueError('resample must be positive')
    T = opt.num_condition_frames
    if T < 1 or len(raw_features) <= T:
        raise ValueError('The sequence needs more frames than num_condition_frames.')
    if latent_mode == 'actions':
        if actions is None:
            raise ValueError('Explicit-action mode requires actions.')
        actions = torch.as_tensor(actions, dtype=torch.float32, device=model.device)
        if actions.shape != (len(raw_features) - T, opt.latent_size) or not torch.isfinite(actions).all():
            raise ValueError('Actions must be finite with shape (predicted_frames, latent_size).')
    elif actions is not None:
        raise ValueError('Explicit actions require latent_mode=actions.')
    if not np.isfinite(raw_features).all() or not np.isfinite(roots).all():
        raise ValueError('Cannot reconstruct non-finite motion features or root positions.')
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Sequence fps must be positive and finite.')
    truth = model.normalize(torch.as_tensor(raw_features, dtype=torch.float32, device=model.device))
    condition = truth[:T].clone()
    predicted = np.array(raw_features, dtype=np.float32, copy=True)
    predicted_roots = np.array(roots, dtype=np.float32, copy=True)
    root = torch.as_tensor(roots[T - 1], dtype=torch.float32, device=model.device).clone()
    rows = []
    for i in range(len(truth)):
        reset = i >= T and resample is not None and (i - T) > 0 and (i - T) % resample == 0
        row = dict(frame=int(source_frames[i]), time_s=float(source_frames[i] / fps),
                   conditioned_from_gt=int(i < T or reset), recon=None, kl=None,
                   recon_weighted=None, kl_weighted=None,
                   recon_zero_latent=None, latent_joint_effect_m=None)
        if detailed_metrics:
            for field in fields:
                row['recon_' + field] = None
        if i >= T:
            if reset:
                condition = truth[i - T:i].clone()
                root = torch.as_tensor(roots[i - 1], dtype=torch.float32, device=model.device).clone()
            kl = None
            if latent_mode == 'posterior':
                mu, logvar = model.model.encode(truth[i:i + 1], condition.reshape(1, -1))
                latent = mu
                kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()).clamp_min(0)
            elif latent_mode == 'actions':
                latent = actions[i - T:i - T + 1]
            else:
                latent = truth.new_zeros((1, opt.latent_size))
            output = model.model.sample(latent, condition.reshape(1, -1)).reshape(-1)[:opt.frame_size]
            recon = (output - truth[i]).square().mean()
            weighted_recon = recon if recon_feature_weights is None else feature_squared_error(
                output, truth[i], recon_feature_weights).mean()
            if detailed_metrics:
                for field, indices in fields.items():
                    row['recon_' + field] = float((output[indices] - truth[i, indices]).square().mean())
            if (not torch.isfinite(output).all() or not torch.isfinite(weighted_recon)
                    or (kl is not None and not torch.isfinite(kl))):
                raise ValueError('Non-finite prediction/loss at source frame {}.'.format(source_frames[i]))
            raw_output = model.unnormalize(output)
            if latent_diagnostics and latent_mode == 'posterior':
                zero_output = model.model.sample(torch.zeros_like(mu), condition.reshape(1, -1)).reshape(-1)[:opt.frame_size]
                if not torch.isfinite(zero_output).all():
                    raise ValueError('Non-finite zero-latent prediction.')
                row['recon_zero_latent'] = float((zero_output - truth[i]).square().mean())
                zero_raw = model.unnormalize(zero_output)
                row['latent_joint_effect_m'] = float((raw_output[fields['joint_pos']] -
                    zero_raw[fields['joint_pos']]).reshape(23, 3).norm(dim=-1).mean())
            root = root + raw_output[fields['root_velo']]
            predicted_roots[i] = root.cpu().numpy()
            predicted[i] = raw_output.cpu().numpy()
            # Runtime uses integrated displacement, not the separately predicted root_pos.
            feedback = raw_output.clone()
            if 'root_pos' in fields:
                feedback[fields['root_pos']] = root
            condition = torch.cat((condition[1:], model.normalize(feedback).unsqueeze(0)))
            row.update(recon=float(recon), kl=float(kl) if kl is not None else None,
                       recon_weighted=float(weighted_recon) * opt.weights['recon'],
                       kl_weighted=float(kl) * opt.weights['kl'] if kl is not None else None)
        for label, pos in [('gt', roots[i]), ('pred', predicted_roots[i])]:
            row.update({label + '_root_' + axis: float(value) for axis, value in zip('xyz', pos)})
        row['root_error_m'] = float(np.linalg.norm(predicted_roots[i] - roots[i]))
        rows.append(row)
    # Show exactly the decoder's root-relative joint-position outputs.
    joints = np.zeros((2, len(truth), 24, 3), dtype=np.float32)
    for actor, features in enumerate((raw_features, predicted)):
        joints[actor, :, 1:] = features[:, fields['joint_pos']].reshape(-1, 23, 3)
    joints[0] += roots[:, None, :]
    joints[1] += predicted_roots[:, None, :]
    result = dict(rows=rows, joints=joints, roots=np.stack((roots, predicted_roots)),
                  predicted_features=predicted)
    add_motion_metrics(result, raw_features, fields, fps, opt)
    for row in rows:
        if any(value is not None and not np.isfinite(value) for value in row.values()):
            raise ValueError('Non-finite motion metric at source frame {}.'.format(row['frame']))
    return result


def court_lines():
    lines = []
    for x, extent in [(-5.485, 11.89), (-4.115, 11.89), (0, 6.4),
                      (4.115, 11.89), (5.485, 11.89)]:
        lines.append(np.array([[x, -extent, 0], [x, extent, 0]]))
    for y, width in [(-11.89, 5.485), (-6.4, 4.115), (6.4, 4.115), (11.89, 5.485)]:
        lines.append(np.array([[-width, y, 0], [width, y, 0]]))
    return lines


class CourtCamera:
    """One fixed perspective camera behind the center of the near baseline."""
    def __init__(self, width=768, height=720):
        self.width, self.height = width, height
        self.eye = np.array([0., -28., 9.])
        forward = np.array([0., -1., 0.]) - self.eye
        self.forward = forward / np.linalg.norm(forward)
        right = np.cross(self.forward, np.array([0., 0., 1.]))
        self.right = right / np.linalg.norm(right)
        self.up = np.cross(self.right, self.forward)
        self.focal = 0.5 * height / np.tan(np.deg2rad(45.) / 2)

    def project(self, points):
        delta = np.asarray(points) - self.eye
        depth = delta @ self.forward
        pixels = np.column_stack((self.width / 2 + self.focal * (delta @ self.right) / np.maximum(depth, .1),
                                  self.height / 2 - self.focal * (delta @ self.up) / np.maximum(depth, .1)))
        return pixels, depth


def render_frame(result, index, camera, title, resample):
    """One court with an opaque reconstruction and a translucent reference."""
    from PIL import Image, ImageDraw, ImageFont
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    try:
        font = ImageFont.truetype(font_path, 18)
    except OSError:
        font = ImageFont.load_default()
    frame = Image.new('RGB', (camera.width, camera.height), (29, 44, 53))
    # RGBA drawing onto RGB composites the reference against the actual scene.
    draw = ImageDraw.Draw(frame, 'RGBA')
    ghost_color = GT_COLOR + (89,)  # 35% opacity
    pred_color = PRED_COLOR + (255,)
    row = result['rows'][index]

    def pixels(points):
        xy, depth = camera.project(points)
        return [tuple(p) for p in xy], depth

    def line(points, color, width=2):
        xy, depth = pixels(points)
        if np.all(depth > .1):
            draw.line(xy, fill=color, width=width)

    floor = np.array([[-9, -18, 0], [9, -18, 0], [9, 18, 0], [-9, 18, 0]])
    draw.polygon(pixels(floor)[0], fill=(46, 98, 91, 255))
    court = np.array([[-5.485, -11.89, 0], [5.485, -11.89, 0],
                      [5.485, 11.89, 0], [-5.485, 11.89, 0]])
    draw.polygon(pixels(court)[0], fill=(54, 89, 140, 255))
    for segment in court_lines():
        line(segment, (230, 236, 242, 255))
    if index:
        for actor, color in enumerate((ghost_color, pred_color)):
            trail = result['roots'][actor, :index + 1].copy()
            trail[:, 2] = .015
            line(trail, color)
    # Depth-sort the net and both skeletons in their shared world coordinates.
    segments = []
    for x in np.linspace(-5.9, 5.9, 49):
        segments.append((np.array([[x, 0, 0], [x, 0, .95]]), (168, 183, 188, 255), 1))
    for z in np.linspace(0, .95, 8):
        segments.append((np.array([[-5.9, 0, z], [5.9, 0, z]]), (168, 183, 188, 255), 1))
    segments.append((np.array([[-5.9, 0, .95], [5.9, 0, .95]]), (250, 250, 250, 255), 3))
    for x in [-5.9, 5.9]:
        segments.append((np.array([[x, 0, 0], [x, 0, 1.07]]), (218, 226, 230, 255), 4))
    for actor, color in enumerate((ghost_color, pred_color)):
        points = result['joints'][actor, index]
        for j, parent in enumerate(SMPL_PARENTS):
            if parent >= 0:
                segments.append((points[[parent, j]], color, 7 if actor == 0 else 5))
    segments.sort(key=lambda s: float(((s[0] - camera.eye) @ camera.forward).mean()), reverse=True)
    for segment, segment_color, width in segments:
        line(segment, segment_color, width)
    # Cover anything projected above the scene with a legible fixed overlay.
    draw.rectangle((0, 0, camera.width, 132), fill=(18, 27, 36, 255))
    draw.text((16, 10), 'Ground truth (35% ghost)', fill=GT_COLOR + (255,), font=font)
    draw.text((390, 10), 'Reconstruction' if row['kl'] is not None else 'Action rollout', fill=pred_color, font=font)
    if row['recon_weighted'] is None:
        loss = 'Conditioning frame (no loss)'
    elif row['kl_weighted'] is None:
        loss = 'Reference MSE: {:.6f} | No encoder / KL'.format(row['recon_weighted'])
    else:
        loss = 'Recon: {:.6f}    KL: {:.6f}  (weighted)'.format(row['recon_weighted'], row['kl_weighted'])
    draw.text((16, 38), loss, fill='white', font=font)
    draw.text((16, 66), 'Frame {} | {:.2f}s | reset {}'.format(
        row['frame'], row['time_s'], 'every {} steps'.format(resample) if resample else 'at start only'),
        fill=(214, 222, 229, 255), font=font)
    if row.get('local_mpjpe_m') is not None:
        draw.text((16, 94), 'Root error: {:.2f} cm | Local joint error: {:.2f} cm{}'.format(
            row['root_error_m'] * 100, row['local_mpjpe_m'] * 100,
            ' | GT reset' if row['conditioned_from_gt'] else ''), fill=(248, 222, 137, 255), font=font)
    elif row['conditioned_from_gt']:
        draw.text((16, 94), 'Condition initialized from ground truth', fill=(248, 222, 137, 255), font=font)
    draw.rectangle((0, camera.height - 32, camera.width, camera.height), fill=(18, 27, 36, 255))
    draw.text((16, camera.height - 27), title[:145], fill='white', font=font)
    return np.asarray(frame)


# Recording-only columns. Shared training/validation diagnostics remain unchanged.
REPORT_COLUMNS = (
    'frame', 'time_s', 'conditioned_from_gt', 'recon',
    'root_error_m', 'local_mpjpe_m', 'left_wrist_error_m', 'right_wrist_error_m',
    'gt_root_speed_mps', 'pred_root_speed_mps',
    'gt_wrist_speed_mps', 'pred_wrist_speed_mps', 'pred_pose_wrist_speed_mps',
    'root_difference_from_zero_m', 'local_joint_difference_from_zero_m',
)
POSTERIOR_COLUMNS = (
    'kl', 'kl_weighted', 'recon_zero_latent', 'latent_joint_effect_m',
)


def report_rows(result):
    """Drop irrelevant/empty columns and repeated GT coordinates from CSVs."""
    rows = result['rows']
    posterior = any(row.get('kl') is not None for row in rows)
    keys = REPORT_COLUMNS + (POSTERIOR_COLUMNS if posterior else ())
    keys += tuple('pred_root_' + axis for axis in 'xyz')
    keys = [key for key in keys if any(row.get(key) is not None for row in rows)]
    return [{key: row.get(key) for key in keys} for row in rows]


def new_figure(rows, columns, size):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=size, constrained_layout=True)
    FigureCanvasAgg(figure)
    return figure, figure.subplots(rows, columns).ravel()


def plot_metrics(ax, rows, keys, title, ylabel):
    times = [row['time_s'] for row in rows]
    for key, label in keys:
        values = [row.get(key) for row in rows]
        if any(value is not None for value in values):
            ax.plot(times, [np.nan if value is None else value for value in values], label=label)
    ax.set(title=title, xlabel='Source time (s)', ylabel=ylabel)
    ax.grid(alpha=.25)
    if ax.lines:
        ax.legend(fontsize=7)


def save_report(directory, result, metadata):
    """One compact report per rollout; baseline data is never copied here."""
    rows = report_rows(result)
    with (directory / 'history.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Use only exported metrics and omit empty subgroup summaries.
    summary = summarize_rows(rows)
    metadata.update({key: value for key, value in summary.items()
                     if value is not None and not key.startswith(('stationary_', 'fast_motion_'))})
    evaluated = [row for row in rows if row['recon'] is not None]
    metadata['root_endpoint_error_m'] = evaluated[-1]['root_error_m']
    (directory / 'summary.json').write_text(json.dumps(metadata, indent=2))
    posterior = any(row.get('kl') is not None for row in evaluated)
    figure, axes = new_figure(3 if posterior else 2, 2, (10, 9 if posterior else 6))
    groups = [
        ((('root_error_m', 'Root'),), 'Root error versus reference', 'm'),
        ((('local_mpjpe_m', 'Mean joint'), ('left_wrist_error_m', 'Left wrist'),
          ('right_wrist_error_m', 'Right wrist')), 'Local pose error versus reference', 'm'),
        ((('gt_root_speed_mps', 'Reference'), ('pred_root_speed_mps', 'Predicted')),
         'Horizontal root speed', 'm/s'),
        ((('gt_wrist_speed_mps', 'Reference'), ('pred_wrist_speed_mps', 'Velocity head'),
          ('pred_pose_wrist_speed_mps', 'Displayed pose')), 'Maximum wrist speed', 'm/s'),
    ]
    if posterior:
        groups = [
            ((('recon', 'Posterior mean'), ('recon_zero_latent', 'One-step z=0')),
             'Normalized feature MSE (same history)', 'MSE'),
            ((('kl_weighted', 'Weighted KL'),), 'Posterior KL', 'Loss'),
        ] + groups
    for ax, (keys, title, ylabel) in zip(axes, groups):
        plot_metrics(ax, evaluated, keys, title, ylabel)
    figure.suptitle('{}\n{}; resets: {}'.format(
        metadata['name'], metadata.get('rollout_label', 'Posterior mean'),
        metadata['resample'] or 'initial state only'), fontsize=9)
    figure.savefig(str(directory / 'metrics.png'), dpi=120, pil_kwargs={'compress_level': 9})


def add_zero_comparison(result, zero, fields):
    """Compare independent histories, without another decoder evaluation."""
    joint_slice = fields['joint_pos']
    root_effect = np.linalg.norm(result['roots'][1] - zero['roots'][1], axis=-1)
    local = result['predicted_features'][:, joint_slice].reshape(-1, 23, 3)
    zero_local = zero['predicted_features'][:, joint_slice].reshape(-1, 23, 3)
    local_effect = np.linalg.norm(local - zero_local, axis=-1).mean(axis=-1)
    for i, row in enumerate(result['rows']):
        evaluated = row['recon'] is not None
        row['root_difference_from_zero_m'] = float(root_effect[i]) if evaluated else None
        row['local_joint_difference_from_zero_m'] = float(local_effect[i]) if evaluated else None
    evaluated = [row for row in result['rows'] if row['recon'] is not None]
    return dict(
        root_difference_from_zero_m_mean=float(np.mean([r['root_difference_from_zero_m'] for r in evaluated])),
        root_difference_from_zero_m_endpoint=float(root_effect[-1]),
        local_joint_difference_from_zero_m_mean=float(np.mean([r['local_joint_difference_from_zero_m'] for r in evaluated])),
    )


def save_experiments(directory, experiments, name):
    """Central comparisons with consistent colors and explicit reset schedules."""
    figure, axes = new_figure(4, 2, (12, 12))
    reference = next(iter(experiments.values()))['rows']
    evaluated_reference = [row for row in reference if row['recon'] is not None]
    panels = (
        ('recon', 'Normalized feature MSE versus reference', 'MSE'),
        ('root_error_m', 'Root error versus reference', 'm'),
        ('local_mpjpe_m', 'Local mean joint error versus reference', 'm'),
        ('pred_root_speed_mps', 'Horizontal root speed', 'm/s'),
        ('pred_pose_wrist_speed_mps', 'Maximum displayed-pose wrist speed', 'm/s'),
        ('root_difference_from_zero_m', 'Root difference from independent z=0', 'm'),
        ('local_joint_difference_from_zero_m', 'Local pose difference from independent z=0', 'm'),
    )
    court = axes[0]
    gt_roots = next(iter(experiments.values()))['roots'][0]
    court.plot(gt_roots[:, 0], gt_roots[:, 1], color='black', linestyle=':', label='Reference')
    colors = ('tab:blue', 'tab:gray', 'tab:orange', 'tab:green', 'tab:red')
    all_roots = [gt_roots]
    for (label, result), color in zip(experiments.items(), colors):
        roots = result['roots'][1]
        all_roots.append(roots)
        court.plot(roots[:, 0], roots[:, 1], color=color, label=label)
        court.scatter(*roots[-1, :2], color=color, marker='x', s=20)
        rows = [row for row in result['rows'] if row['recon'] is not None]
        for ax, (key, title, ylabel) in zip(axes[1:], panels):
            values = [row.get(key) for row in rows]
            if any(value is not None for value in values):
                ax.plot([row['time_s'] for row in rows], values, color=color, label=label)
            ax.set(title=title, xlabel='Source time (s)', ylabel=ylabel)
            ax.grid(alpha=.25)
    for ax, key in ((axes[4], 'gt_root_speed_mps'), (axes[5], 'gt_wrist_speed_mps')):
        if any(row.get(key) is not None for row in evaluated_reference):
            ax.plot([r['time_s'] for r in evaluated_reference], [r.get(key) for r in evaluated_reference],
                    color='black', linestyle=':', label='Reference')
    for segment in court_lines():
        court.plot(segment[:, 0], segment[:, 1], color='gray', linewidth=.6, alpha=.5)
    xy = np.concatenate(all_roots)[:, :2]
    margin = np.maximum(np.ptp(xy, axis=0) * .1, .25)
    court.set(xlim=(xy[:, 0].min() - margin[0], xy[:, 0].max() + margin[0]),
              ylim=(xy[:, 1].min() - margin[1], xy[:, 1].max() + margin[1]),
              title='Root trajectories (zoomed; cross: end)', xlabel='Court X (m)', ylabel='Court Y (m)')
    court.set_aspect('equal', adjustable='box')
    court.grid(alpha=.25)
    court.legend(fontsize=7)
    figure.suptitle(name + '\nRandom-action errors are reference differences, not reconstruction quality. '
                    'Reset schedules are shown in the legend.', fontsize=9)
    figure.savefig(str(directory / 'experiments.png'), dpi=120, pil_kwargs={'compress_level': 9})


def write_rollout_video(path, result, fps, title, resample=None):
    import imageio
    camera = CourtCamera()
    with imageio.get_writer(str(path), fps=fps, codec='libx264', quality=7,
                            ffmpeg_params=['-preset', 'fast'], macro_block_size=16) as writer:
        for index in range(len(result['rows'])):
            writer.append_data(render_frame(result, index, camera, title, resample))


def unit_action_sequences(predicted_frames, latent_size, seed=0, hold_frames=10):
    """Unit standard deviation per coordinate, not unit vector norm.

    Reuse a seeded draw stream so all schedules begin with the same action.
    This local RNG never changes training or posterior sampling RNG state.
    """
    if predicted_frames < 1 or latent_size < 1 or hold_frames < 1 or seed < 0:
        raise ValueError('Frame counts, latent size, and hold frames must be positive; seed must be nonnegative.')
    rng = np.random.default_rng(seed)
    draws = rng.standard_normal((predicted_frames, latent_size)).astype(np.float32)
    return {
        'fixed': np.repeat(draws[:1], predicted_frames, axis=0),
        'held': np.repeat(draws[::hold_frames], hold_frames, axis=0)[:predicted_frames],
        'per_frame': draws,
    }


def record_unit_actions(directory, model, features, roots, source_frames, fps, metadata,
                        zero, seed=0, hold_frames=10):
    """Test-only probes sharing one precomputed, independently rolled-out baseline."""
    sequences = unit_action_sequences(len(features) - model.opt.num_condition_frames,
                                      model.opt.latent_size, seed, hold_frames)
    directory.mkdir(parents=True, exist_ok=True)
    note = ('Independent autoregressive action rollout; GT initialization only; no encoder or KL. '
            'Source motion is a reference, not the intended target for random actions. '
            'Unit scale means N(0,1) per latent coordinate; no normalization or clipping. '
            'The parent reconstruction resample setting does not apply.')
    base_metadata = {key: metadata[key] for key in (
        'name', 'dataset_dir', 'checkpoint', 'frames', 'fps', 'weights', 'recon_feature_weights',
        'start_root', 'camera_position', 'camera_target') if key in metadata}
    base_metadata.update(resample=None, note=note, action_seed=seed,
                         zero_baseline='../zero/history.csv')
    comparisons, experiments = {}, {}
    fields = feature_slices(model.opt)
    for schedule, actions in sequences.items():
        result = reconstruct(model, features, roots, source_frames, fps,
                             latent_diagnostics=False, latent_mode='actions', actions=actions,
                             detailed_metrics=False)
        output_dir = directory / schedule
        output_dir.mkdir()
        np.save(output_dir / 'actions.npy', actions)
        comparison = add_zero_comparison(result, zero, fields)
        hold = len(actions) if schedule == 'fixed' else hold_frames if schedule == 'held' else 1
        label = 'Unit Gaussian: ' + schedule
        report_metadata = dict(base_metadata, rollout_label=label, action_schedule=schedule,
                              action_hold_frames=hold, actions_shape=list(actions.shape),
                              actions_first_source_frame=int(source_frames[model.opt.num_condition_frames]),
                              zero_baseline='../../zero/history.csv', **comparison)
        save_report(output_dir, result, report_metadata)
        write_rollout_video(output_dir / 'rollout.mp4', result, fps, label + ' | ' + metadata['name'])
        comparisons[schedule] = comparison
        experiments[label + ' (no resets)'] = result
    (directory / 'summary.json').write_text(json.dumps(dict(
        action_seed=seed, hold_frames=hold_frames, resample=None, note=note,
        zero_baseline='../zero/history.csv', comparisons_to_zero=comparisons), indent=2))
    return experiments


def record_reconstruction(opt, name, resample=None, nframes=None, suffix='',
                          latent_seed=0, latent_hold_frames=10, unit_actions=False):
    opt = copy.deepcopy(opt)
    opt.test_only = True
    feature_slices(opt)
    # No phase labels are needed to evaluate pose reconstruction and KL.
    data_opt = copy.deepcopy(opt)
    data_opt.predict_phase = False
    dataset = Video3DPoseDataset(data_opt)
    canonical, seq = select_sequence(dataset, name)
    length = seq['length'] - 1  # Keep one preceding frame for exact velocity features.
    if nframes is not None:
        length = min(length, nframes)
    base = seq['base']
    if length <= opt.num_condition_frames:
        raise ValueError('Sequence is too short for conditioning and at least one prediction.')
    if not dataset.valid_arr[base:base + length + 1].all():
        raise ValueError('Selected sequence contains invalid frames; choose a clean exported sequence.')
    features = dataset.get_window(base, length)['feature']
    roots = np.array(dataset.joint_pos_arr[base + 1:base + length + 1, :3], copy=True)
    source_frames = np.arange(seq['start'] + 1, seq['start'] + length + 1)
    fps = float(seq.get('fps', 30))
    model = MotionVAEModel(opt)
    result = reconstruct(model, features, roots, source_frames, fps, resample, detailed_metrics=False)
    zero_result = reconstruct(model, features, roots, source_frames, fps,
                              latent_diagnostics=False, latent_mode='zero', detailed_metrics=False)
    safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', canonical)[:160]
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory = Path(opt.result_dir) / (opt.model_ver + suffix) / (safe_name + '_' + stamp)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = dict(name=canonical, dataset_dir=opt.dataset_dir,
                    checkpoint=str(Path(model.checkpoint_dir) / 'latest.tar'),
                    frames=length, fps=fps, resample=resample, weights=opt.weights,
                    recon_feature_weights=opt.recon_feature_weights,
                    start_root=roots[0].tolist(), camera_position=[0, -28, 9], camera_target=[0, -1, 0],
                    note='Posterior mean latent; target enters encoder at every step. Resample resets '
                         'conditioning history and integrated root before the next prediction. '
                         'Joint positions are direct root-relative decoder outputs, not SMPL FK. '
                         'Recon is unweighted normalized feature MSE; recon_weighted includes feature '
                         'multipliers and the outer recon weight. KL is averaged over latent dimensions. '
                         'Loss is measured before root integration. Initial conditioning frames have no loss. '
                         'The first source frame is reserved for velocity computation.')
    comparison = add_zero_comparison(result, zero_result, feature_slices(opt))
    metadata.update(comparison, zero_baseline='zero/history.csv')
    # Store the uninterrupted zero baseline once, regardless of action schedules.
    zero_dir = directory / 'zero'
    zero_dir.mkdir()
    zero_metadata = {key: metadata[key] for key in (
        'name', 'dataset_dir', 'checkpoint', 'frames', 'fps', 'weights', 'recon_feature_weights',
        'start_root', 'camera_position', 'camera_target')}
    zero_metadata.update(resample=None, rollout_label='Always z=0',
                         note='Independent z=0 rollout; GT initialization only; no encoder or KL.')
    save_report(zero_dir, zero_result, zero_metadata)
    write_rollout_video(zero_dir / 'rollout.mp4', zero_result, fps, 'Always z=0 | ' + canonical)
    save_report(directory, result, metadata)
    video_path = directory / 'reconstruction.mp4'
    write_rollout_video(video_path, result, fps, canonical, resample)
    print('Reconstruction video: {}'.format(video_path.resolve()))
    print('Losses and root trajectories: {}'.format((directory / 'metrics.png').resolve()))
    print('Mean weighted recon: {:.6f}; KL: {:.6f}'.format(
        metadata['recon_mean'] * opt.weights['recon'], metadata['kl_weighted_mean']))
    print('Mean root error: {:.2f} cm; local joint error: {:.2f} cm; diagnostics: {}'.format(
        metadata['root_error_m_mean'] * 100, metadata['local_mpjpe_m_mean'] * 100,
        directory / 'metrics.png'))
    posterior_label = 'Posterior (resets every {} frames)'.format(resample) if resample else 'Posterior (no resets)'
    experiments = {posterior_label: result, 'z=0 (no resets)': zero_result}
    if unit_actions:
        experiments.update(record_unit_actions(
            directory / 'unit_actions', model, features, roots, source_frames,
            fps, metadata, zero_result, latent_seed, latent_hold_frames))
        print('Independent unit-scale action rollouts (GT initialization only): {}'.format(
            (directory / 'unit_actions').resolve()))
    save_experiments(directory, experiments, canonical)
    print('Central experiment comparison: {}'.format((directory / 'experiments.png').resolve()))
    return directory
