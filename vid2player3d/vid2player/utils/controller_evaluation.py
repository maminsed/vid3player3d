"""Small, repeatable physics evaluations with CPU-rendered court videos.

Rendering uses PIL and ffmpeg; no viewer, OpenGL or display server is required.
Simulation and policy inference still use the configured IsaacGym device.
"""
import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BALL_PRESETS = {
    'left_low': (-2., .9), 'left_high': (-2., 1.4),
    'center_low': (0., .9), 'center_high': (0., 1.4),
    'right_low': (2., .9), 'right_high': (2., 1.4),
}
RETURN_TARGETS = {'left': (-3.,10.,0.), 'center': (0.,10.,0.), 'right': (3.,10.,0.)}


def configure_eval_args(args):
    if not args.eval_checkpoint or not Path(args.eval_checkpoint).is_file():
        raise ValueError('--eval_checkpoint must name an existing .pth file; evaluation never falls back to another policy.')
    if args.eval_steps < 1:
        raise ValueError('--eval_steps must be positive.')
    if args.eval_ball not in BALL_PRESETS:
        raise ValueError('--eval_ball must be one of: ' + ', '.join(BALL_PRESETS))
    if args.eval_target not in RETURN_TARGETS:
        raise ValueError('--eval_target must be left, center or right.')
    for value in (args.eval_ball_x, args.eval_ball_height):
        if value is not None and not np.isfinite(value):
            raise ValueError('Incoming ball coordinates must be finite.')
    if args.eval_ball_height is not None and args.eval_ball_height <= 0:
        raise ValueError('--eval_ball_height must be positive.')
    args.test, args.play, args.train = True, True, False
    args.resume = False
    args.headless, args.graphics_device_id = True, -1
    args.num_envs = 1
    args.record = args.record_scenepic = False
    if args.seed is None:
        args.seed = 123


def select_ball_case(filename, preset='center_low', x=None, height=None):
    """Select one paired launch/preview row nearest the requested intercept."""
    desired_x, desired_z = BALL_PRESETS[preset]
    desired_x = desired_x if x is None else x
    desired_z = desired_z if height is None else height
    data = np.load(filename, mmap_mode='r')
    if data.ndim != 2 or data.shape[1] != 307 or not len(data):
        raise ValueError('Expected incoming ball data with shape [N,307].')
    indices = np.linspace(0, len(data)-1, min(len(data),8192), dtype=int)
    sampled = np.asarray(data[indices])
    trajectories = sampled[:,7:].reshape(-1,100,3)
    nearest = np.abs(trajectories[:,:,1]+13).argmin(axis=1)
    intercepts = trajectories[np.arange(len(indices)),nearest]
    valid = np.isfinite(sampled).all(axis=1) & (np.abs(intercepts[:,1]+13)<.75)
    score = (intercepts[:,0]-desired_x)**2 + 4*(intercepts[:,2]-desired_z)**2
    score[~valid] = np.inf
    selected = int(score.argmin())
    if not np.isfinite(score[selected]):
        raise ValueError('No finite incoming trajectory reaching the near baseline was found.')
    case = {'preset':preset, 'source_row':int(indices[selected]),
            'source_file':str(filename), 'requested_intercept_xz_m':[desired_x,desired_z],
            'preview_intercept_xyz_m':intercepts[selected].tolist(),
            'launch_xyz_m':sampled[selected,:3].tolist(),
            'launch_velocity_mps':sampled[selected,3:6].tolist(),
            'launch_spin_hz':float(sampled[selected,6])}
    return sampled[selected:selected+1].copy(), case


def prepare_evaluation(args, cfg, cfg_train):
    if cfg['name'] not in ('MVAEControllerVis','PhysicsMVAEController') or cfg['env']['vid2player'].get('dual_mode'):
        raise ValueError('Headless evaluation supports single-player PhysicsMVAEController configs.')
    rows,case = select_ball_case(cfg['env']['vid2player']['ball_traj_file'],args.eval_ball,
                                args.eval_ball_x,args.eval_ball_height)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = Path(args.eval_output or ('out/controller_eval/'+getattr(args, 'cfg_name', args.cfg)+'_'+
                                     Path(args.eval_checkpoint).stem+'_'+stamp))
    output.mkdir(parents=True,exist_ok=False)
    temporary = tempfile.TemporaryDirectory(prefix='controller_launch_')
    ball_file = Path(temporary.name)/'launch.npy'
    np.save(ball_file,rows)
    cfg['name'] = 'PhysicsMVAEController'
    cfg['env']['is_train'] = False
    task = cfg['env']['vid2player']
    task['test_mode'] = 'fixed_evaluation'
    task['ball_traj_file_test'] = str(ball_file)
    cfg_train['params']['config']['pretrained_model_cp'] = None
    cfg_train['params']['load_checkpoint'] = False
    cfg_train['params']['config']['load_checkpoint'] = False
    metadata = {'checkpoint':str(Path(args.eval_checkpoint).resolve()),'seed':args.seed,
                'steps':args.eval_steps,'ball':case,'return_target_xyz_m':RETURN_TARGETS[args.eval_target],
                'camera':{'eye':[0,-28,9],'target':[0,-1,0],'vertical_fov_degrees':45,'size':[768,720]},
                'protocol':'One fixed single-ball flight, centered root reset, mean policy actions, configured recovery behavior, no timed relaunch.',
                'launch_data':'rollout.npz:launch_row (temporary simulator input is removed after evaluation)',
                'config':cfg,'training_config':cfg_train}
    (output/'evaluation_config.json').write_text(json.dumps(metadata,indent=2,default=str)+'\n')
    return output,case,temporary


def _font(size=18):
    try:
        return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default()


class CourtRenderer:
    """Fixed perspective court with simulated skeleton, racket and ball."""
    def __init__(self, body_names, court_min, court_max):
        names = {name: i for i, name in enumerate(body_names)}
        chains = [['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe'],
                  ['Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe'],
                  ['Pelvis', 'Torso', 'Spine', 'Chest', 'Neck', 'Head'],
                  ['Chest', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand'],
                  ['Chest', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']]
        self.edges = [(names[a], names[b]) for chain in chains for a, b in zip(chain, chain[1:])
                      if a in names and b in names]
        self.font = _font()
        # Same eye, target, 45-degree vertical FOV and image size as the
        # MVAE reconstruction CourtCamera. Centered behind the near baseline.
        self.camera = np.array([10., -28., 9.])
        forward = np.array([0., -1., 0.]) - self.camera
        self.forward = forward / np.linalg.norm(forward)
        right = np.cross(self.forward, [0., 0., 1.])
        self.right = right / np.linalg.norm(right)
        self.up = np.cross(self.right, self.forward)
        self.focal = .5*720/np.tan(np.deg2rad(45.)/2)
        self.background = Image.new('RGB', (768, 720), (27, 62, 47))
        draw = ImageDraw.Draw(self.background)
        draw.polygon(self.project([[-5.485,-11.89,0], [5.485,-11.89,0],
                                   [5.485,11.89,0], [-5.485,11.89,0]]), fill=(35,85,132))
        for x in [-5.485, -4.115, 4.115, 5.485]:
            self.line(draw, [[x,-11.89,0], [x,11.89,0]], 'white')
        for y in [-11.89, 11.89]:
            self.line(draw, [[-5.485,y,0], [5.485,y,0]], 'white')
        for y in [-6.4,6.4]:
            self.line(draw, [[-4.115,y,0], [4.115,y,0]], 'white')
        self.line(draw, [[0,-6.4,0], [0,6.4,0]], 'white')
        for z in [.2,.4,.6,.8,1.07]:
            self.line(draw, [[-6.395,0,z], [6.395,0,z]], (195,195,195))
        for x in [-6.395,6.395]:
            self.line(draw, [[x,0,0], [x,0,1.07]], 'white', 3)
        a,b = court_min,court_max
        self.line(draw, [[a[0],a[1],.01], [b[0],a[1],.01], [b[0],b[1],.01],
                         [a[0],b[1],.01], [a[0],a[1],.01]], (230,140,75))

    def project(self, points):
        delta = np.asarray(points) - self.camera
        depth = np.maximum(delta @ self.forward, .1)
        pixels = np.column_stack((384 + self.focal * (delta @ self.right) / depth,
                                  360 - self.focal * (delta @ self.up) / depth))
        return [tuple(p) for p in pixels]

    def line(self, draw, points, color, width=2):
        if np.isfinite(points).all():
            draw.line(self.project(points), fill=color, width=width)

    def render(self, bodies, ball, racket, normal, target, trail, lines):
        frame = self.background.copy()
        draw = ImageDraw.Draw(frame)
        ring = None
        for a,b in self.edges:
            self.line(draw, bodies[[a,b]], (105,200,255), 4)
        if np.isfinite(racket).all() and np.isfinite(normal).all():
            normal = normal / max(np.linalg.norm(normal), 1e-8)
            axis = np.cross(normal, [0.,0.,1.])
            if np.linalg.norm(axis) < .01:
                axis = np.cross(normal, [0.,1.,0.])
            axis /= max(np.linalg.norm(axis), 1e-8)
            other = np.cross(normal, axis)
            theta = np.linspace(0, 2*np.pi, 24)
            ring = racket + .15*(np.cos(theta)[:,None]*axis + np.sin(theta)[:,None]*other)
            self.line(draw, ring, (255,185,90), 3)
        if len(trail) > 1:
            self.line(draw, trail, (220,220,105))
        for pos,color,radius in [(target,(255,170,70),6), (ball,(245,255,60),5)]:
            if np.isfinite(pos).all():
                x,y = self.project([pos])[0]
                draw.ellipse((x-radius,y-radius,x+radius,y+radius), fill=color)
        draw.rectangle((12,12,650,22+len(lines)*24), fill=(12,20,28))
        for i,line in enumerate(lines):
            draw.text((22,20+i*24), line, font=self.font, fill='white')
        draw.text((18,690), 'Blue: body | yellow: ball | orange: racket / target / boundary', font=self.font, fill='white')
        return frame


def _cpu(value):
    return value.detach().cpu().numpy().copy()


def _number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def save_metrics_plot(filename, metrics, fps):
    """Save one static reward/loss figure for the complete evaluation flight."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    times = np.asarray([row['time_s'] for row in metrics])
    components = sorted({name for row in metrics for name in row['weighted_reward_components']})
    panels = [('Step reward', 'Reward', [('reward',[row['reward'] for row in metrics])]),
              ('Cumulative reward', 'Total reward', [('total',[row['cumulative_reward'] for row in metrics])]),
              ('Weighted reward components', 'Reward contribution',
               [(name,[row['weighted_reward_components'].get(name) for row in metrics]) for name in components]),
              ('Weighted regularization losses', 'Loss contribution',
               [('auxiliary',[row['weighted_aux_dof_res_loss'] for row in metrics]),
                ('bounds',[row['weighted_bounds_loss'] for row in metrics])])]
    figure = Figure(figsize=(12,8), constrained_layout=True)
    FigureCanvasAgg(figure)
    contact = next((row['time_s'] for row in metrics if row['has_contact']),None)
    for axis,(title,label,curves) in zip(figure.subplots(2,2).ravel(),panels):
        for name,values in curves:
            values = np.asarray([np.nan if value is None else value for value in values])
            axis.plot(times,values,label=name,linewidth=1.8)
        if contact is not None:
            axis.axvline(contact,color='gray',linestyle='--',label='first contact')
        axis.set(title=title,xlabel='Time (s)',ylabel=label)
        axis.set_xlim(0,max(float(times[-1]),1/fps))
        axis.grid(True,alpha=.25)
        if curves or contact is not None:
            axis.legend(fontsize=8)
    figure.suptitle('Evaluation rewards and regularization losses over time\n'
                   'Auxiliary/bounds losses include configured weights; no PPO actor/critic update',fontsize=12)
    figure.savefig(filename,dpi=150)


def evaluate_controller(player, args, evaluation):
    import torch
    import imageio.v2 as imageio
    from vid2player.utils.config import set_seed

    output,case,temporary = evaluation
    checkpoint = torch.load(args.eval_checkpoint,map_location=player.device)
    player.model.load_state_dict(checkpoint['model'],strict=True)
    if player.normalize_input:
        player.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
    player.model.eval()
    task = player.task
    physics = task._physics_player.task
    names = physics.gym.get_asset_rigid_body_names(physics.humanoid_asset)
    renderer = CourtRenderer(names,task.cfg_v2p['court_min'],task.cfg_v2p['court_max'])
    fps = 1./physics.dt
    set_seed(args.seed,False)
    physics._ball_generator.sample_idx[:] = 0
    task._reset_reaction_buf[:] = True
    task._reset_recovery_buf[:] = False
    obs = player.env_reset(torch.arange(1,device=task.device))
    player.get_batch_size(obs['obs'],1)
    if player.is_rnn:
        player.init_rnn()
    task._tar_time_total[:] = args.eval_steps+1
    task._target_bounce_pos[:] = torch.tensor(RETURN_TARGETS[args.eval_target],device=task.device)
    task._compute_observations()
    obs = player.obs_to_torch(task.obs_buf.to(player.device))
    cumulative,closest,contact_step = 0.,None,None
    metrics,states,trail = [],[],[]
    end_reason = 'step_limit'
    bounds_weight = player.config.get('bounds_loss_coef',0.) or 0.
    aux_weight = task.cfg_v2p.get('aux_loss_specs',{}).get('dof_res',0.)
    writer = imageio.get_writer(str(output/'video.mp4'),fps=fps,codec='libx264',quality=7,macro_block_size=16)
    try:
        for step in range(args.eval_steps):
            action = player.get_action(obs,is_determenistic=True)
            means = _cpu(action)[0]
            bounds_loss = float(np.square(np.maximum(np.abs(means)-1,0)).sum())
            residual = means[task._num_mvae_action:task._num_mvae_action+task._num_res_dof_action]
            aux_loss = float(np.square(residual).sum())
            obs,reward,done,info = player.env_step(player.env,action)
            bodies,ball,racket,normal,target,root = [ _cpu(value)[0] for value in
                [physics._rigid_body_pos,task._ball_pos,task._racket_pos,task._racket_normal,
                 task._target_bounce_pos,task._root_pos] ]
            value = _number(reward.reshape(-1)[0].item())
            if value is not None:
                cumulative += value
            distance = _number(np.linalg.norm(ball-racket))
            if distance is not None:
                closest = distance if closest is None else min(closest,distance)
            contact = bool(physics._has_racket_ball_contact[0].item())
            if contact and contact_step is None:
                contact_step = step+1
            finite = np.isfinite(bodies).all() and np.isfinite(ball).all()
            ended = bool(done.reshape(-1)[0].item()) or not finite
            if ended:
                outside = np.any(root[:2]<np.asarray(task.cfg_v2p['court_min'])) or np.any(root[:2]>np.asarray(task.cfg_v2p['court_max']))
                end_reason = 'nonfinite_state' if not finite else ('movement_boundary' if outside else 'episode_termination')
            components = {}
            if info.get('sub_rewards') is not None:
                components = dict(zip(info['sub_rewards_names'].split(','),
                                      [_number(x) for x in _cpu(info['sub_rewards'])[0]]))
            weights = task.cfg_v2p.get('reward_weights',{})
            weighted = {name:None if val is None else val*weights.get(name[:-7],0.)
                        for name,val in components.items()}
            row = {'step':step+1,'time_s':(step+1)/fps,'reward':value,'cumulative_reward':cumulative,
                   'reward_components':components,'weighted_reward_components':weighted,
                   'aux_dof_res_loss':_number(aux_loss),'weighted_aux_dof_res_loss':_number(aux_weight*aux_loss),
                   'bounds_loss':_number(bounds_loss),'weighted_bounds_loss':_number(bounds_weight*bounds_loss),
                   'total_regularization_loss':_number(aux_weight*aux_loss+bounds_weight*bounds_loss),
                   'ball_racket_distance_m':distance,'has_contact':contact,
                   'estimated_in':bool(task._est_bounce_in[0].item()),
                   'root_xyz_m':[_number(x) for x in root], 'ball_xyz_m':[_number(x) for x in ball],
                   'terminated':ended}
            metrics.append(row)
            states.append({'body_xyz':bodies,'ball_xyz':ball,'racket_xyz':racket,'policy_mean':means})
            trail.append(ball)
            dist_text = 'n/a' if distance is None else f'{distance:.3f}m'
            best_text = 'n/a' if closest is None else f'{closest:.3f}m'
            value_text = 'n/a' if value is None else f'{value:.4f}'
            lines = [f"{case['preset']} | {Path(args.eval_checkpoint).stem}",
                     f"Step {step+1} | {(step+1)/fps:.2f}s | reward {value_text} | total {cumulative:.3f}",
                     f"Ball-racket {dist_text} | best {best_text} | contact {'YES' if contact else 'NO'}",
                     f"Aux {aux_loss:.4f} | bounds {bounds_loss:.5f} | est-in {row['estimated_in']}"]
            frame = renderer.render(bodies,ball,racket,normal,target,trail[-12:],lines)
            writer.append_data(np.asarray(frame))
            if ended:
                break
            task._reset_reaction_buf[:] = False
            obs = player.env_reset(torch.empty(0,dtype=torch.long,device=task.device))
    finally:
        writer.close()
        temporary.cleanup()
    # One accompanying data file: actual motion and the paired launch/preview.
    source = np.load(case['source_file'],mmap_mode='r')
    report = {'fps':fps,'total_reward':cumulative,'steps':len(metrics),'contact_step':contact_step,
              'min_ball_racket_distance_m':closest,'end_reason':end_reason,
              'loss_definitions':{'aux_dof_res_loss':'Squared residual action means, summed over elbow/wrist outputs.',
                                  'bounds_loss':'Squared excess of action means beyond +/-1, summed over actions.',
                                  'ppo_actor_loss':None,'ppo_critic_loss':None,
                                  'note':'Evaluation performs no PPO update: actor/critic training losses are not computed. Regularization losses are reported unweighted and weighted.'},
              }
    columns = ['step','time_s','reward','cumulative_reward','aux_dof_res_loss',
               'weighted_aux_dof_res_loss','bounds_loss','weighted_bounds_loss',
               'total_regularization_loss','ball_racket_distance_m','has_contact','estimated_in','terminated']
    component_names = sorted({name for row in metrics for name in row['reward_components']})
    metric_values = [[np.nan if row[key] is None else row[key] for key in columns]+
                     [row['reward_components'].get(name,np.nan) for name in component_names]+
                     [row['weighted_reward_components'].get(name,np.nan) for name in component_names]
                     for row in metrics]
    metric_names = columns+['raw/'+name for name in component_names]+['weighted/'+name for name in component_names]
    np.savez_compressed(output/'rollout.npz',launch_row=np.asarray(source[case['source_row']]),
                        metrics=np.asarray(metric_values,dtype=float),metric_names=np.asarray(metric_names),
                        **{key:np.stack([state[key] for state in states]) for key in states[0]})
    config_path = output/'evaluation_config.json'
    config = json.loads(config_path.read_text())
    config['evaluation_results'] = report
    config_path.write_text(json.dumps(config,indent=2,allow_nan=False)+'\n')
    save_metrics_plot(output/'metrics.png',metrics,fps)
    print('Saved single-ball evaluation:',output.resolve())
    print('Total reward {:.4f}; contact step {}'.format(cumulative,contact_step))
