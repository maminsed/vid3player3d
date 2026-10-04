class MotionVAEOption(object):
   
    # Dataset
    dataset_dir = 'tennis_dataset'
    sport = 'tennis'
    gender = ['mens']
    background = ['usopen']
    player_name = ['Federer']
    player_handness = None
    side = 'fg'
    split_annotation = ['orig', 'weak']
    database_ratio = 1.0

    pose_feature = ['root_pos', 'root_velo', 'joint_rotmat', 'joint_pos', 'joint_velo']
    update_joint_pos = False
    predict_phase = False
    condition_root_x_only = False
    no_condition_root_y = False
    
    # Network
    frame_size = None
    latent_size = 32
    hidden_size = 256
    num_condition_frames = 1
    num_future_predictions = 1
    num_experts = 6

    # Train
    gpu_ids = [0]
    base_opt_ver = None
    model_base_ver = None
    nframes_seq = 10
    nseqs = 50000
    curriculum_schedule = None
    mixed_phase_schedule = None
    weights = {'recon': 1, 'kl': 1, 'recon_phase': 10}
    softmax_future = False

    batch_size = 64
    num_threads = 8
    n_epochs = 500
    n_epochs_decay = 500
    log_freq = 2000
    vis_freq = 1e9
    save_freq_epoch = 100
    lr = 0.0001
    checkpoint_dir = 'results/motionVAE'
    continue_train = False
    use_amp = False
    no_log = False

    # Validation: individual manifest-video holdout; steps count completed batches.
    validation_num_videos = 0  # 0 disables holdout and periodic evaluation.
    validation_seed = 1234
    validation_freq_steps = 3000  # Batches; one epoch has nseqs // batch_size batches.
    validation_window_frames = 100  # Predicted frames, plus conditioning history.
    validation_windows_per_video = 3
    metric_stationary_speed_mps = 0.1  # Ground-truth horizontal root speed.
    metric_fast_wrist_speed_mps = 2.0  # Either ground-truth world-space wrist.

    # Test
    test_only = False
    result_dir = 'out/motionVAE'
    infer_racket = False


    def __init__(self):
        # Add all class attributes as instance attributes
        for key in sorted(dir(self)):
            if not key.startswith('__'):
                setattr(self, key, getattr(self, key))


    def update(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])


    def print(self):
        for key in sorted(self.__dict__):
            if not key.startswith('_'):
                print("Option: {:30s} = {}".format(key, self.__dict__[key]))


    def load(self, version):
        stack = [motion_vae_opt_dict[version]]
        while 'base_opt_ver' in stack[-1]:
            stack.append(motion_vae_opt_dict[stack[-1]['base_opt_ver']])
        stack = stack[::-1]
        for opt_update in stack:
            self.update(**opt_update)


motion_vae_opt_dict = {
'federer': {
    'model_ver'                             : 'federer',
    'sport'                                 : 'tennis',
    'background'                            : ['usopen'],
    'split_annotation'                      : ['orig', 'weak'],
    'gender'                                : ['mens'],
    'player_name'                           : ['Federer'],
    'player_handness'                       : None,
    'side'                                  : 'fg',
    'pose_feature'                          : ['root_pos', 'root_velo', 'joint_rotmat', 'joint_pos', 'joint_velo'],
    'update_joint_pos'                      : False,
    'predict_phase'                         : True,
    'frame_size'                            : 6 + 24*6 + 23*3 + 23*3,
    'num_condition_frames'                  : 1,
    'num_future_predictions'                : 1,
    'nframes_seq'                           : 10,
    'batch_size'                            : 100,
    'nseqs'                                 : 50000,
    'softmax_future'                        : True,
    'curriculum_schedule'                   : [0.1, 0.2],
    'mixed_phase_schedule'                  : [(0, 1), (0.5, 0.1)],
    'weights'                               : {'recon': 1, 'kl': 0.5, 'recon_phase': 10},
    'n_epochs'                              : 250,
    'n_epochs_decay'                        : 250,
    'save_freq_epoch'                       : 50,
}, 

'federer_physics': {
    'model_ver'                             : 'federer_physics',
    'base_opt_ver'                          : 'federer',
    'dataset_dir'                           : 'tennis_dataset_physics',
    'predict_phase'                         : False,
    'mixed_phase_schedule'                  : None,
},

'usopen': {
    'model_ver'                             : 'usopen_3',
    'base_opt_ver'                          : 'federer',
    'dataset_dir'                           : 'exports/vae_input/tennis_motion_dataset_2',
    'predict_phase'                         : False,
    'mixed_phase_schedule'                  : None,
    'continue_train'                        : False,
    'save_freq_epoch'                       : 50,
    'nseqs'                                 : 30_000,
    'validation_num_videos'                 : 2,
    'validation_freq_steps'                 : 3000,
    'validation_window_frames'              : 100,
    'validation_windows_per_video'          : 3,
},

'djokovic': {
    'model_ver'                             : 'djokovic',
    'base_opt_ver'                          : 'federer',
    'player_name'                           : ['Djokovic'],
},

'nadal': {
    'model_ver'                             : 'nadal',
    'base_opt_ver'                          : 'federer',
    'player_name'                           : ['Nadal'],
},

}
