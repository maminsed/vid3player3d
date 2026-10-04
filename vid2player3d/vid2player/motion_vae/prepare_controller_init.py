"""Create a reproducible, low-motion controller seed from MVAE training data.

Run from vid2player3d with PYTHONPATH=. and CUDA_VISIBLE_DEVICES=''.
This loads only CPU tensors; it never creates a simulator or loads the decoder.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vid2player.motion_vae.config import MotionVAEOption
from vid2player.motion_vae.dataset import Video3DPoseDataset


def prepare(exp, destination):
    destination = Path(destination)
    metadata_path = destination.with_suffix('.json')
    if destination.exists() or metadata_path.exists():
        raise FileExistsError('Initialization artifacts already exist: ' + str(destination))
    opt = MotionVAEOption()
    opt.load(exp)
    if opt.num_condition_frames != 1 or opt.condition_root_x_only or opt.no_condition_root_y:
        raise ValueError('This initializer supports one condition frame and full XYZ root features.')
    checkpoint_path = Path(opt.checkpoint_dir) / opt.model_ver / 'latest.tar'
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    split = checkpoint.get('validation_split') or {}
    validation = split.get('validation_videos', [])
    opt.nframes_seq = 1
    dataset = Video3DPoseDataset(opt, split='train', validation_videos=validation)
    training = set(split.get('training_videos', []))
    best = None
    for sequence in dataset.sequences:
        if training and sequence['_video_name'] not in training:
            continue
        begin, end = sequence['base'], sequence['base'] + sequence['length']
        ids = np.arange(begin + 1, end)
        positions = dataset.joint_pos_arr[ids]
        valid = dataset.valid_arr[ids] & dataset.valid_arr[ids - 1]
        valid &= np.isfinite(positions).all(axis=1)
        valid &= (np.abs(positions[:, 0]) < 5) & (positions[:, 1] > -16) & (positions[:, 1] < -10)
        valid &= (positions[:, 2] > 0.65) & (positions[:, 2] < 1.1)
        ids = ids[valid]
        if not len(ids):
            continue
        delta = dataset.joint_pos_arr[ids] - dataset.joint_pos_arr[ids - 1]
        score = (delta[:, :3] ** 2).sum(axis=1) + (delta[:, 3:] ** 2).mean(axis=1)
        index = int(np.argmin(score))
        candidate = (float(score[index]), int(ids[index]), sequence)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise ValueError('No upright, near-baseline training frames found.')
    score, index, sequence = best
    feature = dataset.get_window(index - 1, 1)['feature']
    if feature.shape != (1, opt.frame_size) or not np.isfinite(feature).all():
        raise ValueError('Invalid initialization feature shape or values.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(feature.copy()).float().unsqueeze(0), str(destination))
    metadata = {
        'experiment': exp, 'checkpoint': str(checkpoint_path), 'dataset_dir': opt.dataset_dir,
        'video': sequence['_video_name'], 'frame_index': index,
        'sequence_frame_index': index - sequence['base'],
        'root_xyz_m': dataset.joint_pos_arr[index, :3].tolist(),
        'motion_score': score, 'validation_videos_excluded': validation,
        'selection': 'Minimum root and mean local-joint squared displacement among upright near-baseline training frames.',
        'normalization': 'Raw features; MVAEPlayer applies checkpoint normalization at runtime.',
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    print('Saved', destination, 'shape', (1, 1, opt.frame_size), 'from', metadata['video'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp', default='usopen')
    parser.add_argument('--out', default='data/vae/vae_init_condition_usopen_ready.pt')
    args = parser.parse_args()
    prepare(args.exp, args.out)
