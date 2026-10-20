from rl_games.torch_runner import Runner
from rl_games.common import env_configurations, vecenv

from vid2player.players.im_player import ImitatorPlayer
from vid2player.models.im_models import ImitatorModel
from vid2player.models.im_network_builder import ImitatorBuilder
from vid2player.models.im_network_builder_dual import ImitatorBuilderDual
from vid2player.utils.config import get_args, parse_sim_params, load_cfg
from vid2player.env.tasks.vec_task_wrappers import VecTaskPythonWrapper
from vid2player.env.tasks.humanoid_smpl_im_mvae import HumanoidSMPLIMMVAE
from vid2player.env.tasks.humanoid_smpl_im_mvae_dual import HumanoidSMPLIMMVAEDual

from vid2player.utils.tennis_contacts import resolve_tennis_contacts

import numpy as np

args = None
cfg = None
cfg_train = None


def parse_task(args, cfg, cfg_train, sim_params):
    # create native task and pass custom config
    device_id = args.device_id
    rl_device = args.rl_device

    cfg["seed"] = cfg_train.get("seed", -1)
    cfg_task = cfg["env"]
    cfg_task["seed"] = cfg["seed"]
    print(cfg['name'])
    try:
        task = eval(cfg['name'])(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            device_type=args.device,
            device_id=device_id,
            headless=args.headless)
    except NameError as e:
        print(e)
        warn_task_name()
    env = VecTaskPythonWrapper(task, rl_device, cfg_train['params']['config'].get("clip_observations", np.inf), cfg_train['params']['config'].get("clip_actions_val", np.inf))
    return task, env

def warn_task_name():
    raise Exception(
        "Unrecognized task!\nTask should be one of: [BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, ShadowHandLSTM, ShadowHandFFOpenAI, ShadowHandFFOpenAITest, ShadowHandOpenAI, ShadowHandOpenAITest, Ingenuity]")

def create_rlgpu_env(**kwargs):
    use_horovod = cfg_train['params']['config'].get('multi_gpu', False)
    if use_horovod:
        import horovod.torch as hvd

        rank = hvd.rank()
        print("Horovod rank: ", rank)

        cfg_train['params']['seed'] = cfg_train['params']['seed'] + rank

        args.device = 'cuda'
        args.device_id = rank
        args.rl_device = 'cuda:' + str(rank)

        cfg['rank'] = rank
        cfg['rl_device'] = 'cuda:' + str(rank)

    sim_params = parse_sim_params(args, cfg, cfg_train)
    task, env = parse_task(args, cfg, cfg_train, sim_params)

    print(env.num_envs)
    print(env.num_actions)
    print(env.num_obs)
    print(env.num_states)

    frames = kwargs.pop('frames', 1)
    if frames > 1:
        env = wrappers.FrameStack(env, frames, False)
    return env


class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)
        self.use_global_obs = (self.env.num_states > 0)

        self.full_state = {}
        self.full_state["obs"] = self.reset()
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
        return

    def step(self, action):
        next_obs, reward, is_done, info = self.env.step(action)

        # todo: improve, return only dictinary
        self.full_state["obs"] = next_obs
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state, reward, is_done, info
        else:
            return self.full_state["obs"], reward, is_done, info

    def reset(self, env_ids=None):
        self.full_state["obs"] = self.env.reset(env_ids)
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state
        else:
            return self.full_state["obs"]

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space

        if self.use_global_obs:
            info['state_space'] = self.env.state_space
            print(info['action_space'], info['observation_space'], info['state_space'])
        else:
            print(info['action_space'], info['observation_space'])

        return info

class PlayerBuilder():
    def __init__(self, task, config):
        self.task = task
        self.config = config
        self.physics_config = config['env']['physics']

    def build_player_cfg(self):
        args = get_args()
        evaluation = (getattr(self.config.get('args'), 'eval_controller', False)
                      or getattr(self.config.get('args'), 'validation', False))
        if evaluation:
            # get_args reparses CLI here; retain the evaluator's forced display
            # settings even if --headless was omitted from the user's command.
            args.headless = True
            args.graphics_device_id = -1
            args.record = args.record_scenepic = False
            if args.rl_device == 'cpu':
                args.use_gpu = args.use_gpu_pipeline = False
                args.device = args.sim_device_type = args.sim_device = 'cpu'
        test_controller = args.test or args.play
        if evaluation:
            test_controller = True
        args.tmp = False
        args.test = True
        args.play = True
        args.train = False
        # Outer --resume restores the high-level agent only. The frozen player
        # must still load its own configured pretrained checkpoint.
        args.resume = False
        args.resume_id = None
        args.cfg = self.physics_config['config']
        args.cfg_dir = 'embodied_pose'
        if args.cfg == 'humanoid_amass_ft_tennis_v1':
            args.cfg_dir = 'vid2player'
            
        args.experiment = self.physics_config.get('experiment', 'Base')
        args.checkpoint = self.physics_config.get('checkpoint', 'base')
        args.num_envs = self.config['env']['numEnvs']
        args.compute_device_id = self.task.device_id
        cfg, cfg_train = load_cfg(args)
        if evaluation:
            cfg_train['params']['config']['device_name'] = args.rl_device
        if not test_controller:
            cfg['name'] = self.physics_config['name']
        else:
            cfg['name'] = self.physics_config['test_name']
        cfg['env']['is_train'] = not test_controller
        cfg['env']['vid2player'] = self.config['env']['vid2player']
        cfg['env']['controller_cfg_name'] = self.config['env']['controller_cfg_name']

        if cfg['env'].get('ball_traj_file_test'):
            cfg['env']['vid2player']['ball_traj_file_test'] = cfg['env']['ball_traj_file_test']
            cfg['env']['vid2player']['test_mode'] = 'single_cycle' if 'single' in cfg['env']['ball_traj_file_test'] else 'multiple_cycle'
        
        if self.physics_config.get('pretrained_model_cp'):
            cfg_train['params']['config']['pretrained_model_cp'] = self.physics_config['pretrained_model_cp']
        if self.physics_config.get('dual_model_cp'):
            cfg_train['params']['config']['dual_model_cp'] = self.physics_config['dual_model_cp']
        if self.physics_config.get('assetFileName'):
            cfg['env']['asset']['assetFileName'] = self.physics_config['assetFileName']
        if self.physics_config.get('has_racket_collision'):
            cfg['env']['has_racket_collision'] = self.physics_config['has_racket_collision']
        if self.physics_config.get('controlFrequencyInv'):
            cfg['env']['controlFrequencyInv'] = self.physics_config['controlFrequencyInv']
        if self.physics_config.get('residual_force_scale'):
            cfg['env']['residual_force_scale'] = self.physics_config['residual_force_scale']
        if self.physics_config.get('plane_restitution') is not None:
            cfg['env']['plane']['restitution'] = self.physics_config['plane_restitution']

        if self.physics_config.get('dt'):
            cfg['sim']['dt'] = self.physics_config['dt']
        if self.physics_config.get('substeps'):
            cfg['sim']['substeps'] = self.physics_config['substeps']
        for key in ('solver_type', 'max_depenetration_velocity', 'friction_offset_threshold', 'friction_correlation_distance'):
            if self.physics_config.get(key) is not None:
                cfg['sim']['physx'][key] = self.physics_config[key]
        if self.physics_config.get('num_position_iterations'):
            cfg['sim']['physx']['num_position_iterations'] = self.physics_config['num_position_iterations']
        if self.physics_config.get('num_velocity_iterations') is not None:
            cfg['sim']['physx']['num_velocity_iterations'] = self.physics_config['num_velocity_iterations']
        if self.physics_config.get('contact_offset') is not None:
            cfg['sim']['physx']['contact_offset'] = self.physics_config['contact_offset']
        if self.physics_config.get('rest_offset') is not None:
            cfg['sim']['physx']['rest_offset'] = self.physics_config['rest_offset']
        if self.physics_config.get('bounce_threshold_velocity') is not None:
            cfg['sim']['physx']['bounce_threshold_velocity'] = self.physics_config['bounce_threshold_velocity']

        # The contact profile owns the court material in both live and generation.
        contacts = resolve_tennis_contacts(self.config['env']['vid2player'].get('tennis_contacts'))
        if contacts.enabled:
            cfg['env']['plane'].update(staticFriction=contacts.court_friction,
                                       dynamicFriction=contacts.court_friction,
                                       restitution=contacts.court_restitution)
        return args, cfg, cfg_train

    def build_player(self):
        global args
        global cfg
        global cfg_train
        args, cfg, cfg_train = self.build_player_cfg()

        vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
        env_configurations.register('rlgpu', {
            'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs),
            'vecenv_type': 'RLGPU'})

        runner = Runner()
        runner.player_factory.register_builder('pose_im', lambda **kwargs: ImitatorPlayer(**kwargs))  # testing agent
        runner.model_builder.model_factory.register_builder('pose_im', lambda network, **kwargs: ImitatorModel(
            network))  # network wrapper
        runner.model_builder.network_factory.register_builder('pose_im', lambda **kwargs: ImitatorBuilder())
        runner.model_builder.network_factory.register_builder('pose_im_dual', lambda **kwargs: ImitatorBuilderDual())
        runner.load(cfg_train)
        runner.reset()
        player = runner.create_player()
        if cfg_train['params']['config'].get('pretrained_model_cp') is None:
            player.restore(args.checkpoint)
        return player
