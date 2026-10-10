"""Create an MVAE reset condition, metadata JSON, and initial-pose PNG.

Run from vid2player3d with PYTHONPATH=. CUDA is the default compute device.
Without --name, select a tennis-ready upright near-baseline training frame.
With --name, use the clip's first finite, valid within-sequence transition.
--skip N skips N valid candidates in that clip; --centre_x places the root at X=0.
"""
import argparse
import json
from pathlib import Path
import textwrap

import numpy as np
import torch

from vid2player.motion_vae.config import MotionVAEOption
from vid2player.motion_vae.dataset import Video3DPoseDataset
from vid2player.utils.torch_transform import rotmat_to_rot6d, angle_axis_to_rotation_matrix


SMPL_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)

# Dimensionless penalties; lower is better. These are pose-selection heuristics,
# not learned readiness probabilities. Distances are metres, angles are degrees.
READINESS_CONFIG = {
    'weights': {'motion': 0.25, 'facing_net': 2.0, 'symmetry': 1.0,
                'knee_bend': 1.0, 'hands_ready': 2.0},
    'motion_scale_m_per_frame': 0.02,
    'facing_scale_deg': 30.0,
    'symmetry_scale_m': 0.08,
    'knee_target_deg': 70.0, 'knee_scale_deg': 15.0,
    'hand_lateral_scale_m': 0.15,
    'hand_forward_range_m': [0.10, 0.50],
    'hand_height_above_root_range_m': [0.10, 0.50],
    'hand_position_scale_m': 0.15,
    'hand_separation_max_m': 0.25, 'hand_separation_scale_m': 0.15,
}


def _root_rotations(dataset, start, stop, device):
    if hasattr(dataset, 'joint_rotmat_arr'):
        return _tensor(dataset.joint_rotmat_arr[start:stop], device).reshape(-1, 24, 3, 3)[:, 0]
    angles = _tensor(dataset.joint_rot_arr[start:stop], device).reshape(-1, 24, 3)[:, 0]
    return angle_axis_to_rotation_matrix(angles)


def _readiness_scores(current, previous, root_rotations):
    """GPU-vectorized motion and ready-stance penalties in the MVAE world frame.

    SMPL's canonical +Z is forward; root rotation maps it into court XY.
    Symmetry/hand position are measured in the player's horizontal body frame.
    Knees target 20 degrees of flexion (zero means a straight leg).
    """
    config = READINESS_CONFIG
    delta = current - previous
    motion_raw = delta[:, :3].square().sum(dim=1) + delta[:, 3:].square().mean(dim=1)
    joints = torch.cat((torch.zeros_like(current[:, None, :3]),
                        current[:, 3:].reshape(-1, 23, 3)), dim=1)
    forward = root_rotations[:, :2, 2]
    forward = torch.nn.functional.normalize(forward, dim=1, eps=1e-6)
    lateral = torch.stack((-forward[:, 1], forward[:, 0]), dim=1)
    # Net is at Y=0: near-half players face +Y, far-half players face -Y.
    net_sign = torch.where(current[:, 1] <= 0, 1.0, -1.0)
    facing_deg = torch.rad2deg(torch.acos((forward[:, 1] * net_sign).clamp(-1, 1)))
    body = torch.stack(((joints[:, :, :2] * lateral[:, None]).sum(dim=2),
                        (joints[:, :, :2] * forward[:, None]).sum(dim=2),
                        joints[:, :, 2]), dim=2)
    left = body[:, [1, 4, 7, 10, 13, 16, 18, 20, 22]]
    right = body[:, [2, 5, 8, 11, 14, 17, 19, 21, 23]]
    reflected_right = right * right.new_tensor([-1, 1, 1])
    symmetry_mse = (left - reflected_right).square().mean(dim=(1, 2))
    thigh = joints[:, [4, 5]] - joints[:, [1, 2]]
    shin = joints[:, [7, 8]] - joints[:, [4, 5]]
    thigh = torch.nn.functional.normalize(thigh, dim=2, eps=1e-6)
    shin = torch.nn.functional.normalize(shin, dim=2, eps=1e-6)
    knee_deg = torch.rad2deg(torch.acos((thigh * shin).sum(dim=2).clamp(-1, 1)))
    hands = body[:, [20, 21]]

    def outside_range(values, bounds):
        return (bounds[0] - values).clamp(min=0) + (values - bounds[1]).clamp(min=0)

    separation = (hands[:, 0] - hands[:, 1]).norm(dim=1)
    hand_penalty = (
        (hands[:, :, 0] / config['hand_lateral_scale_m']).square().mean(dim=1)
        + (outside_range(hands[:, :, 1], config['hand_forward_range_m'])
           / config['hand_position_scale_m']).square().mean(dim=1)
        + (outside_range(hands[:, :, 2], config['hand_height_above_root_range_m'])
           / config['hand_position_scale_m']).square().mean(dim=1)
        + ((separation - config['hand_separation_max_m']).clamp(min=0)
           / config['hand_separation_scale_m']).square()) / 4
    components = {
        'motion': motion_raw / config['motion_scale_m_per_frame'] ** 2,
        'facing_net': (facing_deg / config['facing_scale_deg']).square(),
        'symmetry': symmetry_mse / config['symmetry_scale_m'] ** 2,
        'knee_bend': ((knee_deg - config['knee_target_deg']) / config['knee_scale_deg']).square().mean(dim=1),
        'hands_ready': hand_penalty,
    }
    total = sum(config['weights'][key] * value for key, value in components.items())
    diagnostics = {'motion_score': motion_raw, 'facing_net_deg': facing_deg,
                   'symmetry_rms_m': symmetry_mse.sqrt(),
                   'left_knee_bend_deg': knee_deg[:, 0], 'right_knee_bend_deg': knee_deg[:, 1],
                   'hand_separation_m': separation}
    return total, components, diagnostics


def _tensor(array, device):
    # Copy read-only memory maps before handing their storage to PyTorch.
    return torch.from_numpy(np.array(array, copy=True)).to(device)


def _feature(dataset, index, device, centre_x=False):
    """Match get_window(index - 1, 1), performing feature math on device."""
    positions = _tensor(dataset.joint_pos_arr[index - 1:index + 1], device)
    delta = positions[1:2] - positions[:1]
    fields = []
    for name in ('root_pos', 'root_velo', 'joint_pos', 'joint_velo',
                 'joint_rotmat', 'joint_quat'):
        if name not in dataset.opt.pose_feature:
            continue
        if name == 'root_pos':
            value = positions[1:2, :3]
        elif name == 'root_velo':
            value = delta[:, :3]
        elif name == 'joint_pos':
            value = positions[1:2, 3:]
        elif name == 'joint_velo':
            value = delta[:, 3:]
        elif name == 'joint_rotmat':
            rotations = _tensor(dataset.joint_rotmat_arr[index:index + 1], device)
            value = rotmat_to_rot6d(rotations.reshape(1, 24, 3, 3)).reshape(1, 144)
        else:
            value = _tensor(dataset.joint_quat_arr[index:index + 1], device).reshape(1, 96)
        fields.append(value)
    feature = torch.cat(fields, dim=1).float()
    if centre_x:
        if 'root_pos' not in dataset.opt.pose_feature:
            raise ValueError('--centre_x requires root_pos in pose_feature.')
        # Translate the pose, preserving velocities and root-relative joints.
        feature[:, 0] = 0
    return feature


def _validate_skip(name, skip):
    if skip is None:
        return 0
    if name is None:
        raise ValueError('--skip requires --name.')
    if isinstance(skip, bool) or not isinstance(skip, int) or skip < 0:
        raise ValueError('--skip must be a nonnegative integer.')
    return skip


def _named_sequences(dataset, name):
    # A video name may contain multiple sequences. Canonical names and export
    # aliases use the same disambiguation rules as reconstruction.
    videos = [seq for seq in dataset.sequences if seq['_video_name'] == name]
    if videos:
        return videos
    from vid2player.motion_vae.reconstruction import select_sequence
    _, selected = select_sequence(dataset, name)
    return [seq for seq in dataset.sequences if seq['base'] == selected['base']]


def _select(dataset, training, device, name=None, batch_size=65536, skip=None):
    """Bound GPU memory by processing within-sequence transition chunks."""
    remaining = _validate_skip(name, skip)
    sequences = _named_sequences(dataset, name) if name is not None else dataset.sequences
    best = None
    for sequence in sequences:
        if training and sequence['_video_name'] not in training:
            continue
        begin, end = sequence['base'], sequence['base'] + sequence['length']
        for start in range(begin + 1, end, batch_size):
            stop = min(start + batch_size, end)
            positions = _tensor(dataset.joint_pos_arr[start - 1:stop], device)
            flags = _tensor(dataset.valid_arr[start - 1:stop], device).bool()
            current, previous = positions[1:], positions[:-1]
            valid = flags[1:] & flags[:-1]
            valid &= torch.isfinite(current).all(dim=1) & torch.isfinite(previous).all(dim=1)
            if name is None:
                valid &= (current[:, 0].abs() < 5) & (current[:, 1] > -16) & (current[:, 1] < -10)
                valid &= (current[:, 2] > 0.65) & (current[:, 2] < 1.1)
            delta = current - previous
            scores = delta[:, :3].square().sum(dim=1) + delta[:, 3:].square().mean(dim=1)
            valid &= torch.isfinite(scores)
            if not valid.any().item():
                continue
            if name is not None:
                # Velocity features need a predecessor, so sequence frame zero
                # cannot itself be used as the condition.
                for offset in torch.nonzero(valid, as_tuple=False).flatten().tolist():
                    index = start + offset
                    feature = _feature(dataset, index, device)
                    if feature.shape == (1, dataset.opt.frame_size) and torch.isfinite(feature).all().item():
                        if remaining:
                            remaining -= 1
                            continue
                        return float(scores[offset].item()), index, sequence
                continue
            rotations = _root_rotations(dataset, start, stop, device)
            scores, _, _ = _readiness_scores(current, previous, rotations)
            valid &= torch.isfinite(rotations).flatten(1).all(dim=1) & torch.isfinite(scores)
            if not valid.any().item():
                continue
            scores = scores.masked_fill(~valid, float('inf'))
            offset = int(torch.argmin(scores).item())
            candidate = (float(scores[offset].item()), start + offset, sequence)
            if best is None or candidate[0] < best[0]:
                best = candidate
    if best is None:
        if name is not None:
            raise ValueError('Named clip has no finite, valid training transition after skipping {} valid candidates: {}'.format(
                0 if skip is None else skip, name))
        raise ValueError('No upright, near-baseline training frames found.')
    return best


def _save_pose(positions, path, title):
    # Rasterization/file I/O stays on CPU; all selection and feature math uses
    # the requested compute device. No simulator, decoder, or display required.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle
    from vid2player.motion_vae.reconstruction import court_lines

    positions = positions.detach().cpu().numpy()
    root = positions[:3]
    joints = np.vstack((root, positions[3:].reshape(23, 3) + root))
    figure = Figure(figsize=(15, 5))
    FigureCanvasAgg(figure)
    for panel, horizontal in enumerate((0, 1), start=1):
        axis = figure.add_subplot(1, 3, panel)
        for child, parent in enumerate(SMPL_PARENTS):
            if parent < 0:
                continue
            points = joints[[parent, child]]
            axis.plot(points[:, horizontal], points[:, 2], color='tab:blue', linewidth=2)
        axis.scatter(joints[:, horizontal], joints[:, 2], color='tab:orange', s=12)
        axis.axhline(0, color='gray', linestyle='--', linewidth=1)
        axis.set_aspect('equal', adjustable='box')
        axis.set_xlabel(('X' if horizontal == 0 else 'Y') + ' (m)')
        axis.set_ylabel('Z (m)')
        axis.set_title(('XZ' if horizontal == 0 else 'YZ') + ' projection')
        axis.grid(alpha=0.25)
    court = figure.add_subplot(1, 3, 3)
    # Use the reconstruction court's world coordinates without translating or
    # mirroring the root. Show its half, including space behind the baseline.
    far_half = root[1] > 0
    baseline = 11.89 if far_half else -11.89
    court.set_facecolor('#dce9dc')
    court.add_patch(Rectangle((-5.485, min(0, baseline)), 10.97, 11.89,
                              facecolor='#427aab', edgecolor='none'))
    for segment in court_lines():
        court.plot(segment[:, 0], segment[:, 1], color='white', linewidth=1.5)
    court.plot([-5.9, 5.9], [0, 0], color='#333333', linewidth=2)
    court.text(0, 0, 'Net', ha='center', va='bottom', fontsize=8)
    court.scatter(root[0], root[1], color='tab:orange', edgecolor='black',
                  s=65, zorder=3, label='Initial root')
    court.annotate('({:.2f}, {:.2f})'.format(*root[:2]), root[:2],
                   xytext=(0, -14 if far_half else 10), textcoords='offset points',
                   ha='center', fontsize=8)
    court.set_xlim(min(-6.5, float(root[0]) - 1), max(6.5, float(root[0]) + 1))
    if far_half:
        court.set_ylim(-0.5, max(16, float(root[1]) + 1.5))
    else:
        court.set_ylim(min(-16, float(root[1]) - 1.5), 0.5)
    court.set_aspect('equal', adjustable='box')
    court.set(title='Initial position — court top view', xlabel='X (m)', ylabel='Y (m)')
    court.legend(loc='upper right', fontsize=8)
    figure.suptitle(textwrap.fill(title, width=95) + '\nRoot XYZ: ' +
                   ', '.join('{:.3f}'.format(x) for x in root) + ' m', fontsize=10)
    figure.subplots_adjust(top=0.76, bottom=0.14, wspace=0.35)
    figure.savefig(str(path), dpi=150, bbox_inches='tight')


@torch.no_grad()
def prepare(exp, destination, name=None, device='cuda:0', centre_x=False, skip=None):
    skipped = _validate_skip(name, skip)
    destination = Path(destination)
    metadata_path = destination.with_suffix('.json')
    pose_path = destination.with_suffix('.png')
    if len({destination, metadata_path, pose_path}) != 3:
        raise ValueError('Output tensor path must differ from its .json and .png sidecars.')
    for path in (destination, metadata_path, pose_path):
        if path.exists():
            raise FileExistsError('Initialization artifact already exists: ' + str(path))
    device = torch.device(device)
    if device.type not in ('cuda', 'cpu'):
        raise ValueError('Use a CUDA device or cpu.')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Run on a GPU host, or explicitly use --device cpu.')
    opt = MotionVAEOption()
    opt.load(exp)
    if opt.num_condition_frames != 1 or opt.condition_root_x_only or opt.no_condition_root_y:
        raise ValueError('This initializer supports one condition frame and full XYZ root features.')
    if centre_x and 'root_pos' not in opt.pose_feature:
        raise ValueError('--centre_x requires root_pos in pose_feature.')
    checkpoint_path = Path(opt.checkpoint_dir) / opt.model_ver / 'latest.tar'
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    split = checkpoint.get('validation_split') or {}
    validation = split.get('validation_videos', [])
    training = set(split.get('training_videos', []))
    del checkpoint
    opt.nframes_seq = 1
    dataset = Video3DPoseDataset(opt, split='train', validation_videos=validation)
    _, index, sequence = _select(dataset, training, device, name=name, skip=skip)
    feature = _feature(dataset, index, device, centre_x=centre_x)
    if feature.shape != (1, opt.frame_size) or not torch.isfinite(feature).all().item():
        raise ValueError('Invalid initialization feature shape or values.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    positions = _tensor(dataset.joint_pos_arr[index], device)
    readiness, components, diagnostics = _readiness_scores(
        positions.unsqueeze(0), _tensor(dataset.joint_pos_arr[index - 1:index], device),
        _root_rotations(dataset, index, index + 1, device))
    original_root = positions[:3].tolist()
    if centre_x:
        positions[0] = 0
    _save_pose(positions, pose_path, '{} | sequence frame {}'.format(
        sequence['_video_name'], index - sequence['base']))
    # Keep the artifact portable across CUDA device indices and CPU inspection.
    torch.save(feature.unsqueeze(0).cpu(), str(destination))
    metadata = {
        'experiment': exp, 'checkpoint': str(checkpoint_path), 'dataset_dir': opt.dataset_dir,
        'video': sequence['_video_name'], 'frame_index': index,
        'sequence_frame_index': index - sequence['base'],
        'root_xyz_m': positions[:3].tolist(),
        'original_root_xyz_m': original_root, 'centre_x': centre_x,
        'skipped_valid_frames': skipped,
        'motion_score': float(diagnostics['motion_score'][0].item()),
        'readiness_score': float(readiness[0].item()),
        'score_components': {key: float(value[0].item()) for key, value in components.items()},
        'pose_diagnostics': {key: float(value[0].item()) for key, value in diagnostics.items()
                             if key != 'motion_score'},
        'scoring_config': READINESS_CONFIG, 'validation_videos_excluded': validation,
        'requested_name': name, 'compute_device': str(device), 'pose_png': str(pose_path),
        'selection': ('Finite, valid within-sequence training transition in the named clip after skipping {} valid candidates.'.format(skipped)
                      if name is not None else
                      'Minimum weighted tennis-readiness penalty among upright near-baseline training frames.'),
        'normalization': 'Raw features; MVAEPlayer applies checkpoint normalization at runtime.',
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    print('Saved', destination, 'shape', tuple(feature.unsqueeze(0).shape), 'from', metadata['video'])
    print('Saved pose', pose_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp', default='usopen')
    parser.add_argument('--out', default='data/vae/vae_init_condition_usopen_ready.pt')
    parser.add_argument('--name', help='Video/clip name, export alias, or VIDEO:fg:INDEX; use its first valid transition.')
    parser.add_argument('--centre_x', action='store_true', help='Center the selected root at court X=0, preserving Y/Z and velocities.')
    parser.add_argument('--skip', type=int, default=None, help='Skip N valid starting-frame candidates (requires --name).')
    parser.add_argument('--device', default='cuda:0', help='Compute device (default: cuda:0); cpu for debugging.')
    args = parser.parse_args()
    if args.skip is not None and (args.name is None or args.skip < 0):
        parser.error('--skip requires --name and a nonnegative integer.')
    prepare(args.exp, args.out, name=args.name, device=args.device,
            centre_x=args.centre_x, skip=args.skip)
