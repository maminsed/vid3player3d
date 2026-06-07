import torch
import os
import numpy as np
import time
import joblib

from scipy.spatial.transform import Rotation as sRot

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.players import rescale_actions, unsqueeze_obs
from rl_games.common.player import BasePlayer

import embodied_pose.learning.common_player as common_player


class ImitatorPlayer(common_player.CommonPlayer):
    def __init__(self, config):
        print("\nInitializing ImitatorPlayer...")
        BasePlayer.__init__(self, config)
        self.args = config['args']
        self.task = self.env.task
        self.horizon_length = config['horizon_length']
        self.num_actors = config['num_actors']
        self.network = config['network']
        self.network_path = config['network_path']
        
        print("Setting up action space...")
        self._setup_action_space()
        self.mask = [False]

        self.clip_actions = False

        self.normalize_input = self.config['normalize_input']
        
        print("Building network configuration...")
        net_config = self._build_net_config()
        print(f"Network config: {net_config}")
        
        print("Building network...")
        self._build_net(net_config)
        print("Network built")

        self.task.register_model(self.model)

        pretrained_model_cp = config.get('pretrained_model_cp', None)
        if self.args.checkpoint == 'base' and pretrained_model_cp is not None and not config.get('load_checkpoint', False):
            print(f"\nLoading pretrained model from: {pretrained_model_cp}")
            if type(pretrained_model_cp) is list:
                for cp in pretrained_model_cp:
                    self.load_pretrained(cp)
            else:
                self.load_pretrained(pretrained_model_cp)
        else:
            print(f"Not loading pretrained model from: {pretrained_model_cp}")

        # Export dataset is optional - only initialize if explicitly requested
        print(f"DEBUG: Full task cfg keys: {list(self.env.task.cfg.keys())}")
        print(f"DEBUG: env cfg keys: {list(self.env.task.cfg.get('env', {}).keys())}")
        
        # Check both locations for the export_dataset flag
        self._export_dataset = self.env.task.cfg.get('export_dataset', False)
        if not self._export_dataset:
            self._export_dataset = self.env.task.cfg.get('env', {}).get('export_dataset', False)
        
        print(f"DEBUG: export_dataset flag = {self._export_dataset}")
        self.motion_export_data = None
        if self._export_dataset:
            print("Motion export enabled - will collect motion data during simulation")
            self.motion_export_data = {
                'pose_aa': [],
                'trans_orig': []
            }
        else:
            print("Motion export disabled - no data will be collected")

        print("ImitatorPlayer initialization complete")

    def restore(self, cp_name):
        print(f"\nRestoring checkpoint: {cp_name}")
        if cp_name is not None and cp_name != "base":
            cp_path = os.path.join(self.network_path, f"{self.config['name']}_{cp_name}.pth")
            print(f"Loading checkpoint from: {cp_path}")
            checkpoint = torch.load(cp_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            if self.normalize_input:
                self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
            print("Checkpoint loaded successfully")
        else:
            print('No checkpoint provided.')

    def load_pretrained(self, cp_path):
        print(f"\nLoading pretrained model from: {cp_path}")
        checkpoint = torch.load(cp_path, map_location=self.device)

        if 'pytorch-lightning_version' in checkpoint:
            model_state_dict = self.model.a2c_network.state_dict()
            print('loading model from Lighting checkpoint...')
            print('Shared keys in model and lightning checkpoint:', [key for key in checkpoint['state_dict'] if key in model_state_dict])
            print('Lightning keys not found in current model:', [key for key in checkpoint['state_dict'] if key not in model_state_dict])
            print('Keys not found in lightning checkpoint:', [key for key in model_state_dict if key not in checkpoint['state_dict']])
            self.model.a2c_network.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model_state_dict = self.model.state_dict()
            missing_checkpoint_keys = [key for key in model_state_dict if key not in checkpoint['model']]
            print('loading model from EmbodyPose checkpoint...')
            print('Shared keys in model and checkpoint:', [key for key in model_state_dict if key in checkpoint['model']])
            print('Keys not found in current model:', [key for key in checkpoint['model'] if key not in model_state_dict])
            print('Keys not found in checkpoint:', missing_checkpoint_keys)

            discard_pretrained_sigma = self.config.get('discard_pretrained_sigma', False)
            if discard_pretrained_sigma:
                checkpoint_keys = list(checkpoint['model'])
                for key in checkpoint_keys:
                    if 'sigma' in key:
                        checkpoint['model'].pop(key)

            self.set_weights(checkpoint)

            load_rlgame_norm = len([key for key in missing_checkpoint_keys if 'running_obs' in key]) > 0
            if load_rlgame_norm:
                if 'running_mean_std' in checkpoint:
                    print('loading running_obs from rl game running_mean_std')
                    self.model.a2c_network.running_obs.n = checkpoint['running_mean_std']['count'].long()
                    self.model.a2c_network.running_obs.mean[:] = checkpoint['running_mean_std']['running_mean'].float()
                    self.model.a2c_network.running_obs.var[:] = checkpoint['running_mean_std']['running_var'].float()
                    self.model.a2c_network.running_obs.std[:] = torch.sqrt(self.model.a2c_network.running_obs.var)
                elif 'a2c_network.running_obs.running_mean' in checkpoint['model']:
                    print('loading running_obs from rl game in model')
                    obs_len = checkpoint['model']['a2c_network.running_obs.running_mean'].shape[0]
                    self.model.a2c_network.running_obs.n = checkpoint['model']['a2c_network.running_obs.count'].long()
                    self.model.a2c_network.running_obs.mean[:obs_len] = checkpoint['model']['a2c_network.running_obs.running_mean'].float()
                    self.model.a2c_network.running_obs.var[:obs_len] = checkpoint['model']['a2c_network.running_obs.running_var'].float()
                    self.model.a2c_network.running_obs.std[:obs_len] = torch.sqrt(self.model.a2c_network.running_obs.var[:obs_len])

    def set_weights(self, weights):
        if hasattr(self.model, 'load_weights'):
            self.model.load_weights(weights['model'])
        else:
            self.model.load_state_dict(weights['model'], strict=False)
        
        if self.normalize_input:
            self.running_mean_std.load_state_dict(weights['running_mean_std'])

    def _build_net_config(self):
        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_actors,
            'smpl_rest_joints': self.task.smpl_rest_joints,
            'smpl_parents': self.task.smpl_parents,
            'smpl_children': self.task.smpl_children
        }
        if hasattr(self.task, 'num_con_planes'):
            config['num_con_planes'] = self.task.num_con_planes
            config['num_con_bodies'] = self.task.num_con_bodies
        return config

    def get_action_values(self, obs):
        processed_obs = self._preproc_obs(obs['obs'])
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs,
            'rnn_states' : self.rnn_states,
            't': obs['t']
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
            if self.has_central_value:
                states = obs['states']
                input_dict = {
                    'is_train': False,
                    'states' : states,
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value
        if self.normalize_value:
            res_dict['values'] = self.value_mean_std(res_dict['values'], True)
        return res_dict

    def get_action(self, obs_dict, is_determenistic = False):
        print("\nGetting action:")
        obs = obs_dict['obs']
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        print(f"Processed obs shape: {obs.shape}")
        
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs,
            't': obs_dict['t'],
            'global_t_offset': obs_dict['global_t_offset'],
            'rnn_states' : self.states
        }
        print("Running model inference...")
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if is_determenistic:
            current_action = mu
        else:
            current_action = action
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        print(f"Action shape: {current_action.shape}")
        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action

    def env_step(self, env, actions):
        print("\nEnvironment step details:")
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
            print(f"Actions shape: {actions.shape}")
        else:
            print(f"Actions shape: {actions.shape}, device: {actions.device}")
            
        print("Calling env.step()...")
        obs, rewards, dones, infos = env.step(actions)
        print(f"Environment returned:")
        print(f"Obs shape: {obs.shape if hasattr(obs, 'shape') else 'No shape'}")
        print(f"Rewards shape: {rewards.shape if hasattr(rewards, 'shape') else 'No shape'}")
        print(f"Dones shape: {dones.shape if hasattr(dones, 'shape') else 'No shape'}")
        print(f"Info keys: {infos.keys() if isinstance(infos, dict) else 'No info dict'}")
        
        if isinstance(infos, dict):
            if 'terminate' in infos:
                print(f"Terminate flags: {infos['terminate']}")
            if 'sub_rewards' in infos:
                print("Sub rewards:")
                for i, (name, value) in enumerate(zip(infos['sub_rewards_names'].split(','), infos['sub_rewards'])):
                    print(f"  {name}: {value}")

        if hasattr(obs, 'dtype') and obs.dtype == np.float64:
            obs = np.float32(obs)
        if self.value_size > 1:
            rewards = rewards[0]
        if self.is_tensor_obses:
            obs_dict = {'obs': obs}
            return obs_dict, rewards.to(self.device), dones.to(self.device), infos
        else:
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return self.obs_to_torch(obs), torch.from_numpy(rewards), torch.from_numpy(dones), infos

    def run(self):
        print("\nStarting ImitatorPlayer.run()...")
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        print(f"Initial parameters:")
        print(f"n_games: {n_games}")
        print(f"n_game_life: {n_game_life}")
        print(f"max_steps: {self.max_steps}")

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn
        for game_idx in range(n_games):
            print(f"\nStarting new game iteration ({game_idx + 1}/{n_games})...")
            if games_played >= n_games:
                print("Games played reached limit, but continuing for debug...")
                break  # Disabled for debug

            # Resample motion for new game
            if hasattr(self.task, '_motion_lib') and hasattr(self.task, '_reset_ref_motion_ids'):
                weights_from_length = self.task.cfg['env'].get('motion_weights_from_length', False)
                new_motion_ids = self.task._motion_lib.sample_motions(1, weights_from_lenth=weights_from_length)
                self.task._reset_ref_motion_ids[:] = new_motion_ids
                print(f"Resampled motion_ids: {self.task._reset_ref_motion_ids}")

            obs_dict = self.env_reset()
            batch_size = 1
            batch_size = self.get_batch_size(obs_dict['obs'], batch_size)
            print(f"Batch size: {batch_size}")

            if need_init_rnn:
                print("Initializing RNN...")
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

            print_game_res = False

            done_indices = []

            print("Initializing visualization...")
            self.task.render_vis(init=True)

            prev_dones = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

            for n in range(self.max_steps):
                print(f"\nStep {n}/{self.max_steps}")
                t = n % self.task.context_length
                if n > 0 and t == 0:
                    print("Resetting context...")
                    self.task._init_context(self.task._reset_ref_motion_ids, self.task._cur_ref_motion_times)
                
                obs_dict['t'] = t
                obs_dict['global_t_offset'] = n - t
                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(obs_dict, masks, is_determenistic)
                else:
                    action = self.get_action(obs_dict, is_determenistic)

                print("Stepping environment...")
                obs_dict, r, done, info = self.env_step(self.env, action)
                print(f"Reward: {r}, Done: {done}")

                cr += r
                steps += 1

                print("Rendering visualization...")
                self.task.render_vis()
  
                self._post_step(info)

                # Only collect motion data if export is enabled
                if self._export_dataset and self.motion_export_data is not None:
                    print(f"DEBUG: Collecting motion data frame {len(self.motion_export_data['pose_aa'])}")
                    self._collect_motion_data()

                if render:
                    self.env.render(sync_frame_time=True)

                step_dones = done * (1 - prev_dones)
                all_done_indices = step_dones.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                games_played += done_count

                if done_count > 0:
                    print(f"Done count: {done_count}")
                    if self.is_rnn:
                        for s in self.states:
                            s[:,all_done_indices,:] = s[:,all_done_indices,:] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if isinstance(info, dict):
                        if 'battle_won' in info:
                            print_game_res = True
                            game_res = info.get('battle_won', 0.5)
                        if 'scores' in info:
                            print_game_res = True
                            game_res = info.get('scores', 0.5)
                    if self.print_stats:
                        if print_game_res:
                            print('reward:', cur_rewards/done_count, 'steps:', cur_steps/done_count, 'w:', game_res)
                        else:
                            print('reward:', cur_rewards/done_count, 'steps:', cur_steps/done_count)

                    sum_game_res += game_res
                    if batch_size//self.num_agents == 1 or games_played >= n_games:
                        print("Breaking due to batch size or games played, but continuing for debug...")
                        print(f"debug: 383: n_games: {n_games} games_played: {games_played}")
                        break  # Disabled for debug
                
                done_indices = done_indices[:, 0]

                prev_dones = done.clone()
                if done[0]:
                    print("Environment signaled done, but continuing for debug...")
                    break  # Disabled for debug

        print("\nFinal statistics:")
        print(f"Total rewards: {sum_rewards}")
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life)

        # Only save motion data if export is enabled
        if self._export_dataset and self.motion_export_data is not None:
            print(f"DEBUG: About to save motion data with {len(self.motion_export_data['pose_aa'])} frames")
            self._save_motion_data()
        else:
            print(f"DEBUG: Not saving motion data - export_dataset={self._export_dataset}, motion_export_data is None={self.motion_export_data is None}")

        return

    def _collect_motion_data(self):
        env_id = 0
        
        root_state = self.env.task._humanoid_root_states[env_id]
        root_pos = root_state[0:3].cpu().numpy()
        root_rot_quat = root_state[3:7].cpu().numpy()
        
        dof_pos = self.env.task._dof_pos[env_id].cpu().numpy()
        
        # Get motion ID for this environment
        motion_id = self.env.task._reset_ref_motion_ids[env_id]
        
        # Get the skeleton information from the first motion (they should all have the same skeleton)
        # We'll use the motion library's internal data to get the skeleton
        motion_lib = self.env.task._motion_lib
        
        # Get the names of the simulated DOFs
        sim_dof_names = self.env.task.dof_names

        # Use standard SMPL joint names (these are the same across all SMPL models)
        smpl_joint_names = [
            'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee',
            'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax',
            'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder',
            'R_Elbow', 'R_Wrist', 'R_Hand'
        ]

        # Initialize full pose with zeros
        num_smpl_joints = len(smpl_joint_names)
        full_pose_aa_new = np.zeros((num_smpl_joints, 3))

        # Convert simulated root rotation quaternion to axis-angle
        root_rot_aa = sRot.from_quat(root_rot_quat).as_rotvec()
        full_pose_aa_new[0] = root_rot_aa
        
        # Let's build a mapping from smpl joint name to its dof indices
        dof_map = {}
        for i, dof_name in enumerate(sim_dof_names):
            parts = dof_name.split('_')
            joint_name = '_'.join(parts[:-1])
            if joint_name not in dof_map:
                dof_map[joint_name] = {}
            axis = parts[-1]
            dof_map[joint_name][axis] = i
        
        # Now fill the full pose array
        for i, joint_name in enumerate(smpl_joint_names):
            if i == 0: continue # Skip root
            if joint_name in dof_map:
                axis_vals = np.zeros(3)
                if 'x' in dof_map[joint_name]: axis_vals[0] = dof_pos[dof_map[joint_name]['x']]
                if 'y' in dof_map[joint_name]: axis_vals[1] = dof_pos[dof_map[joint_name]['y']]
                if 'z' in dof_map[joint_name]: axis_vals[2] = dof_pos[dof_map[joint_name]['z']]
                full_pose_aa_new[i] = axis_vals

        self.motion_export_data['pose_aa'].append(full_pose_aa_new.flatten())
        self.motion_export_data['trans_orig'].append(root_pos)

    def _save_motion_data(self):
        print("\n" + "="*60)
        print("SAVING EXPORTED MOTION DATA")
        print("="*60)
        
        motion_id = self.env.task._reset_ref_motion_ids[0].item()
        
        # Get metadata from the motion library
        motion_lib = self.env.task._motion_lib
        
        # Get fps from the motion library
        fps = 30.0  # Default
        if hasattr(motion_lib, '_motion_fps') and motion_lib._motion_fps is not None:
            try:
                fps = motion_lib._motion_fps[motion_id].item()
            except:
                pass
        
        # Get betas from the motion lib's motion data
        betas = np.zeros(10) # Default
        gender = 'neutral'  # Default
        
        # Try to get metadata from motion_bodies if available
        if hasattr(motion_lib, '_motion_bodies') and motion_lib._motion_bodies is not None:
            try:
                motion_body = motion_lib._motion_bodies[motion_id].cpu().numpy()
                if len(motion_body) >= 11:  # Should have at least 11 values (gender + 10 betas)
                    betas = motion_body[1:11]  # Skip gender, take betas
            except:
                pass
        
        # Try to get gender from motion_bodies if available
        if hasattr(motion_lib, '_motion_bodies') and motion_lib._motion_bodies is not None:
            try:
                motion_body = motion_lib._motion_bodies[motion_id].cpu().numpy()
                if len(motion_body) > 0:
                    gender_val = motion_body[0]
                    if gender_val == 0:
                        gender = 'neutral'
                    elif gender_val == 1:
                        gender = 'male'
                    elif gender_val == 2:
                        gender = 'female'
            except:
                pass

        # Create exports directory if it doesn't exist
        export_dir = "exports"
        os.makedirs(export_dir, exist_ok=True)

        output_data = {
            'pose_aa': np.array(self.motion_export_data['pose_aa']),
            'trans_orig': np.array(self.motion_export_data['trans_orig']),
            'beta': betas,
            'gender': gender,
            'fps': fps
        }

        filename = f"motion_export_motion{motion_id}.pkl"
        full_path = os.path.join(export_dir, filename)
        
        joblib.dump({f"demo_export_{motion_id}": output_data}, full_path)
        
        print(f"✓ Motion data successfully exported!")
        print(f"📁 File location: {os.path.abspath(full_path)}")
        print(f"📊 Data summary:")
        print(f"   - Frames exported: {len(self.motion_export_data['pose_aa'])}")
        print(f"   - Motion ID: {motion_id}")
        print(f"   - Gender: {gender}")
        print(f"   - FPS: {fps}")
        print(f"   - File size: {os.path.getsize(full_path) / 1024:.1f} KB")
        print("="*60)
