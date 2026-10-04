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
def reconstruct(model, raw_features, roots, source_frames, fps, resample=None, latent_diagnostics=True):
    """Posterior-mean reconstruction with autoregressive conditioning and root integration.

    Targets enter the encoder every step. Only condition/root resets are periodic.
    Losses measure the raw decoder output before replacing its root_pos feature
    with the integrated root, as is done during runtime motion generation.
    """
    opt = model.opt
    fields = feature_slices(opt)
    if resample is not None and resample <= 0:
        raise ValueError('resample must be positive')
    T = opt.num_condition_frames
    if T < 1 or len(raw_features) <= T:
        raise ValueError('The sequence needs more frames than num_condition_frames.')
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
        for field in fields:
            row['recon_' + field] = None
        if i >= T:
            if reset:
                condition = truth[i - T:i].clone()
                root = torch.as_tensor(roots[i - 1], dtype=torch.float32, device=model.device).clone()
            mu, logvar = model.model.encode(truth[i:i + 1], condition.reshape(1, -1))
            output = model.model.sample(mu, condition.reshape(1, -1)).reshape(-1)[:opt.frame_size]
            recon = (output - truth[i]).square().mean()
            for field, indices in fields.items():
                row['recon_' + field] = float((output[indices] - truth[i, indices]).square().mean())
            kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()).clamp_min(0)
            if not torch.isfinite(output).all() or not torch.isfinite(recon + kl):
                raise ValueError('Non-finite prediction/loss at source frame {}.'.format(source_frames[i]))
            raw_output = model.unnormalize(output)
            if latent_diagnostics:
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
            row.update(recon=float(recon), kl=float(kl),
                       recon_weighted=float(recon) * opt.weights['recon'],
                       kl_weighted=float(kl) * opt.weights['kl'])
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
    draw.text((390, 10), 'Reconstruction', fill=pred_color, font=font)
    loss = ('Conditioning frame (no loss)' if row['recon_weighted'] is None else
            'Recon: {:.6f}    KL: {:.6f}  (weighted)'.format(row['recon_weighted'], row['kl_weighted']))
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


def save_report(directory, result, metadata):
    rows = result['rows']
    with (directory / 'history.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    evaluated = [row for row in rows if row['recon'] is not None]
    metadata.update(summarize_rows(rows))
    metadata['root_endpoint_error_m'] = evaluated[-1]['root_error_m']
    metadata['metric_thresholds'] = result.get('metric_thresholds', {})
    (directory / 'summary.json').write_text(json.dumps(metadata, indent=2))
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(13, 17), constrained_layout=True)
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(5, 2)
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[1, 0])]
    times = [row['time_s'] for row in evaluated]
    for ax, key, label in zip(axes, ('recon_weighted', 'kl_weighted'), ('Reconstruction loss', 'KL loss')):
        ax.plot(times, [row[key] for row in evaluated])
        for row in evaluated:
            if row['conditioned_from_gt']:
                ax.axvline(row['time_s'], color='gray', linestyle=':', alpha=.35)
        ax.set(title=label + ' (weighted)', xlabel='Source time (s)', ylabel='Loss')
        ax.grid(alpha=.25)
    court = figure.add_subplot(grid[:2, 1])
    for segment in court_lines():
        court.plot(segment[:, 0], segment[:, 1], color='gray', linewidth=1)
    court.plot([-5.9, 5.9], [0, 0], color='gray', linewidth=3, label='Net')
    for roots, label, color in zip(result['roots'], ('Ground truth', 'Reconstruction'), (GT_COLOR, PRED_COLOR)):
        color = np.array(color) / 255
        court.plot(roots[:, 0], roots[:, 1], color=color, label=label)
        court.scatter(*roots[0, :2], color=color, marker='o')
        court.scatter(*roots[-1, :2], color=color, marker='x')
    court.set(title='Root trajectory (circle: start; cross: end)', xlabel='Court X (m)', ylabel='Court Y (m)')
    court.set_aspect('equal', adjustable='datalim')
    court.grid(alpha=.2)
    court.legend(fontsize=8)
    figure.suptitle('{}\nPosterior-mean reconstruction; resets: {}'.format(
        metadata['name'], metadata['resample'] or 'initial state only'), fontsize=9)
    groups = [
        (('root_error_m', 'root_consistency_m'), 'Root errors (m)'),
        (('local_mpjpe_m', 'world_mpjpe_m', 'left_wrist_error_m', 'right_wrist_error_m'), 'Joint errors (m)'),
        (('gt_root_speed_mps', 'pred_root_speed_mps'), 'Horizontal root speed (m/s)'),
        (('gt_wrist_speed_mps', 'pred_wrist_speed_mps', 'pred_pose_wrist_speed_mps'), 'Maximum wrist speed (m/s)'),
        (('root_velocity_bias_x_mps', 'root_velocity_bias_y_mps', 'root_velocity_bias_z_mps'), 'Signed root velocity error (m/s)'),
        (('recon', 'recon_zero_latent'), 'Normalized MSE: posterior mean vs zero latent'),
    ]
    for index, (keys, title) in enumerate(groups):
        ax = figure.add_subplot(grid[2 + index // 2, index % 2])
        for key in keys:
            values = [r.get(key) for r in evaluated]
            if any(v is not None for v in values):
                ax.plot(times, [np.nan if v is None else v for v in values], label=key)
        ax.set(title=title, xlabel='Source time (s)')
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    figure.savefig(str(directory / 'metrics.png'), dpi=140)


def record_reconstruction(opt, name, resample=None, nframes=None, suffix=''):
    import imageio
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
    result = reconstruct(model, features, roots, source_frames, fps, resample)
    safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', canonical)[:160]
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory = Path(opt.result_dir) / (opt.model_ver + suffix) / (safe_name + '_' + stamp)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = dict(name=canonical, dataset_dir=opt.dataset_dir,
                    checkpoint=str(Path(model.checkpoint_dir) / 'latest.tar'),
                    frames=length, fps=fps, resample=resample, weights=opt.weights,
                    start_root=roots[0].tolist(), camera_position=[0, -28, 9], camera_target=[0, -1, 0],
                    note='Posterior mean latent; target enters encoder at every step. Resample resets '
                         'conditioning history and integrated root before the next prediction. '
                         'Joint positions are direct root-relative decoder outputs, not SMPL FK. '
                         'Recon is normalized feature MSE; KL is averaged over latent dimensions. '
                         'Loss is measured before root integration. Initial conditioning frames have no loss. '
                         'The first source frame is reserved for velocity computation.')
    save_report(directory, result, metadata)
    camera = CourtCamera()
    video_path = directory / 'reconstruction.mp4'
    with imageio.get_writer(str(video_path), fps=fps, codec='libx264', quality=7,
                            ffmpeg_params=['-preset', 'fast'], macro_block_size=16) as writer:
        for index in range(length):
            writer.append_data(render_frame(result, index, camera, canonical, resample))
    print('Reconstruction video: {}'.format(video_path.resolve()))
    print('Losses and root trajectories: {}'.format((directory / 'metrics.png').resolve()))
    print('Mean weighted recon: {:.6f}; KL: {:.6f}'.format(
        metadata['recon_weighted_mean'], metadata['kl_weighted_mean']))
    print('Mean root error: {:.2f} cm; local joint error: {:.2f} cm; diagnostics: {}'.format(
        metadata['root_error_m_mean'] * 100, metadata['local_mpjpe_m_mean'] * 100,
        directory / 'metrics.png'))
    return directory
