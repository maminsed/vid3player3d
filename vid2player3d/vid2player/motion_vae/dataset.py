from vid2player.utils.io import *
from vid2player.utils.pose import SMPLPose
from vid2player.utils.common import concat
from vid2player.utils.torch_transform import rotmat_to_rot6d

from torch.utils.data import Dataset
import os
import random
import copy
import torch


def encode_action(action, action_dim):
    action = torch.as_tensor(action, dtype=torch.long)
    if action_dim is None:
        raise ValueError("action_dim must be set to encode actions.")
    action_dim = int(action_dim)
    if action.numel() > 0:
        min_action = int(action.min().item())
        max_action = int(action.max().item())
        if min_action < 0 or max_action >= action_dim:
            raise ValueError(
                "Action labels must be in [0, {}), got min={} max={}.".format(
                    action_dim, min_action, max_action))

    encoded = torch.zeros(action.shape + (action_dim,), device=action.device, dtype=torch.float32)
    if action.numel() > 0:
        encoded.scatter_(-1, action.unsqueeze(-1), 1.0)
    return encoded


class Video3DPoseDataset(Dataset):

    def __init__(
        self,
        opt,
        split='all',
        validation_videos=(),
        nframes_seq=None,
    ):
        if split not in ('all', 'train', 'validation'):
            raise ValueError('Unknown dataset split: ' + split)
        self.opt = copy.deepcopy(opt)
        self.manifest = load_json(
            os.path.join(opt.dataset_dir, 'manifest.json'))
        self.joint_pos_arr = np.load(
            os.path.join(opt.dataset_dir, 'joint_pos.npy'), mmap_mode='r')
        self.joint_rot_arr = np.load(
            os.path.join(opt.dataset_dir, 'joint_rot.npy'), mmap_mode='r')
        if 'joint_velo' in opt.pose_feature:
            self.joint_rot_arr = np.load(
                os.path.join(opt.dataset_dir, 'joint_rot.npy'), mmap_mode='r')
        if 'joint_rotmat' in opt.pose_feature:
            self.joint_rotmat_arr = np.load(
                os.path.join(opt.dataset_dir, 'joint_rotmat.npy'), mmap_mode='r')
        if 'joint_quat' in opt.pose_feature:
            self.joint_quat_arr = np.load(
                os.path.join(opt.dataset_dir, 'joint_quat.npy'), mmap_mode='r')
        self.valid_arr = np.load(
            os.path.join(opt.dataset_dir, 'valid.npy'))

        self.sequences = []
        self.selected_arr = np.zeros_like(self.valid_arr)
        if opt.predict_phase:
            self.phase_arr = np.zeros((self.valid_arr.shape[0], 2), dtype=float)
            self.phase_rad_arr = np.zeros(self.valid_arr.shape[0], dtype=float)
            self.phase_mask_arr = np.zeros(self.valid_arr.shape[0], dtype=bool)
            phase_path = os.path.join(opt.dataset_dir, 'phase.npy')
            mask_path = os.path.join(opt.dataset_dir, 'phase_mask.npy')
            explicit_phase = os.path.exists(phase_path)
            if explicit_phase != os.path.exists(mask_path):
                raise ValueError('phase.npy and phase_mask.npy must be supplied together.')
            if explicit_phase:
                self.phase_arr = np.load(phase_path)
                self.phase_mask_arr = np.load(mask_path).astype(bool)
                if self.phase_arr.shape != (len(self.valid_arr), 2) or self.phase_mask_arr.shape != self.valid_arr.shape:
                    raise ValueError('Phase arrays must align with valid.npy.')
                if not np.isfinite(self.phase_arr[self.phase_mask_arr]).all():
                    raise ValueError('Labelled phases must be finite.')
                self.phase_arr = self.phase_arr.copy()
                self.phase_arr[~self.phase_mask_arr] = 0
                self.phase_mask_arr &= self.valid_arr.astype(bool)
                self.phase_rad_arr = np.mod(np.arctan2(self.phase_arr[:, 0], self.phase_arr[:, 1]), 2*np.pi)
        
        num_videos = 0
        betas = []
        # Fliter sequences
        for video in self.manifest:
            if split == 'train' and video['name'] in validation_videos: continue
            if split == 'validation' and video['name'] not in validation_videos: continue
            if video['background'] not in opt.background: continue
            if video['gender'] not in opt.gender: continue
            if opt.sport == 'tennis':
                if video.get('is_orig') == True and 'orig' not in opt.split_annotation: continue
                if video.get('is_orig') == False and 'weak' not in opt.split_annotation: continue
            print(video['name'])

            if opt.side == 'both':
                seqs_candidate = video['sequences']['fg'] + video['sequences']['bg']
            elif opt.side == 'fg':
                seqs_candidate = video['sequences']['fg']
            elif opt.side == 'bg':
                seqs_candidate = video['sequences']['bg']

            num_videos += 1
            for seq in seqs_candidate:
                if opt.player_handness is not None:
                    if seq['handness'] not in opt.player_handness: continue
                else:
                    if video.get('is_orig') or seq['player'] is not None:
                        # These videos have player identity annotated
                        if opt.player_name is not None and seq['player'] not in opt.player_name: continue
                sequence_side = 'fg' if seq in video['sequences']['fg'] else 'bg'
                seq = seq.copy()
                seq['_video_name'] = video['name']
                
                if opt.predict_phase:
                    if not explicit_phase:
                        # Legacy hit annotations: interpolate only bracketed frames.
                        points = video.get('points_annotation', [])
                        point_idx = seq.get('point_idx')
                        hits = points[point_idx].get('keyframes', []) if point_idx is not None and 0 <= point_idx < len(points) else []
                        from vid2player.motion_vae.phase import labels_between_hits
                        frames = np.arange(seq['start'], seq['start'] + seq['length'])
                        events = [dict(frame=h['fid'], side='fg' if h['fg'] else 'bg') for h in hits]
                        phase, mask = labels_between_hits(frames, np.ones(len(frames), dtype=bool), events, sequence_side)
                        sl = slice(seq['base'], seq['base'] + seq['length'])
                        self.phase_arr[sl], self.phase_mask_arr[sl] = phase, mask
                        self.phase_rad_arr[sl] = np.mod(np.arctan2(phase[:, 0], phase[:, 1]), 2*np.pi)
                    seq['has_phase'] = bool(self.phase_mask_arr[seq['base']:seq['base']+seq['length']].any())

                self.sequences += [seq]
                self.selected_arr[seq['base'] : seq['base']+seq['length']] = 1
                betas += [seq['beta']]
              
        num_valid_frames = np.logical_and(self.valid_arr, self.selected_arr).sum()
        print(f"Loaded {len(self.sequences)} motion sequences from {num_videos} videos containing {num_valid_frames} frames")

        if opt.predict_phase:
            print('Phase labels: {} selected frames'.format(int((self.phase_mask_arr & self.selected_arr.astype(bool)).sum())))

        self.std, self.avg = None, None

        if not self.sequences:
            raise ValueError('No motion sequences pass the {} dataset filters.'.format(split))
        if opt.database_ratio != 1.0:
            count = int(len(self.sequences) * opt.database_ratio)
            if count < 1 or count > len(self.sequences):
                raise ValueError('database_ratio must retain at least one sequence and be <= 1.')
            self.sequences = self.sequences[:count]
            self.selected_arr[:] = 0
            for seq in self.sequences:
                self.selected_arr[seq['base']:seq['base'] + seq['length']] = 1

        self.seq_weights = np.array(
            [seq['length'] for seq in self.sequences], dtype=float)
        self.seq_weights /= np.sum(self.seq_weights)

        self.init_rollouts(opt.nframes_seq if nframes_seq is None else nframes_seq)


    def init_rollouts(self, nframes_seq):
        self.nframes_seq = nframes_seq # nframes_seq might be different from opt.nframes_seq
        self.rollouts = []
        for seq in self.sequences:
            for x in range(seq['base'], seq['base'] + seq['length'] - nframes_seq):
                if self.valid_arr[x : x + nframes_seq + 1].sum() == nframes_seq + 1:
                    self.rollouts += [x]
        
        random.shuffle(self.rollouts)
        print("Init {} rollouts".format(len(self.rollouts)))
    

    def get_normalization_stats(self):
        opt = self.opt
        feature_all = None
        if 'root_pos' in opt.pose_feature:
            root_pos = self.joint_pos_arr[:, :3]
            if opt.condition_root_x_only:
                root_pos = root_pos[:, 0:1]
            elif opt.no_condition_root_y:
                root_pos = root_pos[:, [0, 2]]
            feature_all = concat(feature_all, root_pos, axis=1)
        if 'root_velo' in opt.pose_feature:
            root_velo = self.joint_pos_arr[:, :3] - np.roll(self.joint_pos_arr[:, :3], 1, axis=0)
            feature_all = concat(feature_all, root_velo, axis=1)
        if 'joint_pos' in opt.pose_feature:
            joint_pos = self.joint_pos_arr[:, 3:]
            feature_all = concat(feature_all, joint_pos, axis=1)
        if 'joint_velo' in opt.pose_feature:
            joint_velo = self.joint_pos_arr[:, 3:] - np.roll(self.joint_pos_arr[:, 3:], 1, axis=0)
            feature_all = concat(feature_all, joint_velo, axis=1)
        if 'joint_quat' in opt.pose_feature:
            joint_quat = self.joint_quat_arr.reshape(-1, 24*4)
            feature_all = concat(feature_all, joint_quat, axis=1)
        if 'joint_rotmat' in opt.pose_feature:
            joint_rotmat = rotmat_to_rot6d(torch.from_numpy(
                self.joint_rotmat_arr.reshape(-1, 24, 3, 3).copy())).numpy().reshape(-1, 24*6)
            feature_all = concat(feature_all, joint_rotmat, axis=1)

        # A transition needs two valid frames from the SAME selected sequence.
        # In particular, np.roll above must never normalize cross-clip jumps.
        transitions = np.zeros(len(self.valid_arr), dtype=bool)
        for seq in self.sequences:
            begin, end = seq['base'], seq['base'] + seq['length']
            transitions[begin + 1:end] |= (self.valid_arr[begin + 1:end].astype(bool)
                                          & self.valid_arr[begin:end - 1].astype(bool))
        feature_all = feature_all[transitions & self.selected_arr.astype(bool)]
        if len(feature_all) == 0 or not np.isfinite(feature_all).all():
            raise ValueError('Normalization needs finite, valid within-sequence transitions.')

        std = np.std(feature_all, axis=0)
        std[std == 0] = 1.0
        avg = np.average(feature_all, axis=0)
        self.std = std
        self.avg = avg
    
    
    def set_normalization_stats(self, avg, std):
        self.avg = avg
        self.std = std


    def __len__(self):
        return self.opt.nseqs


    def __getitem__(self, idx):
        # Required
        # root velocity 3: do we need to share the same coordinate of the last frame
        # joint position 23 x 3
        # joint velocity 23 x 3: do we need this if we are considering more than one condition frames
        # root and joint orientation 24 x 6 first two columns of the rotation matrix 

        # Optional
        # joint angular velocity 24 x 3
        # global root position 3
        # contact  
        opt = self.opt
        if opt.test_only and opt.batch_size <= len(self.rollouts):
            start = self.rollouts[idx]
        else:
            start = random.choices(self.rollouts, k=1)[0]
        return self.get_window(start, self.nframes_seq)

    def get_window(self, start, nframes):
        """Read frames start+1..start+nframes, using start for velocities."""
        opt = self.opt
        end = start + nframes + 1
        L = nframes

        feature = None
        # the actual frame starts at start + 1
        if 'root_pos' in opt.pose_feature:
            root_pos = self.joint_pos_arr[start+1:end, :3]
            if opt.condition_root_x_only:
                root_pos = root_pos[:, 0:1]
            elif opt.no_condition_root_y:
                root_pos = root_pos[:, [0, 2]]
            feature = concat(feature, root_pos, axis=1)
        if 'root_velo' in opt.pose_feature:
            root_velo = self.joint_pos_arr[start+1:end, :3] - self.joint_pos_arr[start:end-1, :3]
            feature = concat(feature, root_velo, axis=1)
        if 'joint_pos' in opt.pose_feature:
            joint_pos = self.joint_pos_arr[start+1:end, 3:]
            feature = concat(feature, joint_pos, axis=1)
        if 'joint_velo' in opt.pose_feature:
            joint_velo = self.joint_pos_arr[start+1:end, 3:] - self.joint_pos_arr[start:end-1, 3:]
            feature = concat(feature, joint_velo, axis=1)
        if 'joint_rotmat' in opt.pose_feature:
            joint_rotmat = rotmat_to_rot6d(torch.from_numpy(
                self.joint_rotmat_arr[start+1:end].reshape(L, 24, 3, 3).copy())).numpy().reshape(L, 24*6)
            feature = concat(feature, joint_rotmat, axis=1)
        if 'joint_quat' in opt.pose_feature:
            joint_quat = self.joint_quat_arr[start+1:end].reshape(-1, 24*4)
            feature = concat(feature, joint_quat, axis=1)

        if self.std is not None:
            feature = (feature - self.avg) / self.std
        data_dict = {
            'feature': feature,
            'start': start,
        }
        if opt.predict_phase:
            data_dict['phase'] = self.phase_arr[start+1:end]
            data_dict['phase_mask'] = self.phase_mask_arr[start+1:end]
        return data_dict
    

    def sample_first_frame(self):
        opt = self.opt
        data = self.__getitem__(0)
        T = self.opt.num_condition_frames
        # the actual frame starts at data['start'] + 1
        frame = {
            'root_pos': self.joint_pos_arr[data['start'] + T, :3].copy(),
            'joint_pos': self.joint_pos_arr[data['start'] + T, 3:].copy(),
            'joint_rot': self.joint_rot_arr[data['start'] + T].copy(),
            'condition': data['feature'][:T]
        }
        return frame
