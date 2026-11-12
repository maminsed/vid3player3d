"""CPU coverage for case selection, recording and the evaluation loop."""
import json
import ast
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import imageio.v2 as imageio
from vid2player.utils import controller_viewer

from vid2player.utils.controller_evaluation import (
    configure_eval_args, prepare_evaluation, select_ball_case, CourtRenderer, evaluate_controller, phase_metrics,
    save_metrics_plot, contact_interval_events)


def offline_generator_class():
    # Avoid importing IsaacGym/GUI dependencies for CPU sampler checks.
    path = Path(__file__).resolve().parents[1]/'utils/tennis_ball.py'
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node,ast.ClassDef) and node.name=='TennisBallGeneratorOffline']
    namespace = {'torch':torch,'np':np}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace['TennisBallGeneratorOffline']


class ControllerEvaluationTests(unittest.TestCase):
    def test_multiple_contact_evidence_intervals_keep_registration_and_rejection(self):
        candidate = dict(proximity_pass=True, impulse_evidence=True, sensor_evidence=False)
        samples = [dict(candidate, accepted_contact=True),
                   dict(candidate, accepted_contact=False, contact_already_latched=True),
                   dict(candidate, accepted_contact=False, ground_impact=True),
                   dict(proximity_pass=False, impulse_evidence=True)]
        events = contact_interval_events(samples, 3, 1/30, 1/120, 7)
        self.assertEqual([e['id'] for e in events], [7,8,9])
        self.assertEqual([e['status'] for e in events],
                         ['registered','ignored_already_latched','rejected_ground_impact'])
        self.assertEqual(sum(e['registered'] for e in events),1)
        self.assertAlmostEqual(events[0]['time_start_s'],2/30)
        self.assertAlmostEqual(events[1]['time_end_s'],2/30+2/120)
        grazes = contact_interval_events([dict(proximity_pass=True,sensor_evidence=True,
                                               impulse_evidence=False,accepted_contact=False)],
                                        3,1/30,1/120,7)
        self.assertEqual(grazes[0]['status'],'rejected_velocity_change')
        self.assertFalse(grazes[0]['registered'])

    def test_low_level_player_preserves_evaluation_display_settings(self):
        from vid2player.utils.tennis_contacts import resolve_tennis_contacts
        path = Path(__file__).resolve().parents[1]/'env/utils/player_builder.py'
        definition = next(node for node in ast.parse(path.read_text()).body
                          if isinstance(node, ast.ClassDef) and node.name == 'PlayerBuilder')
        for validation, headless, graphics in [(False, False, 0), (False, True, -1), (True, True, -1)]:
            with self.subTest(validation=validation, headless=headless):
                outer = SimpleNamespace(eval_controller=not validation, validation=validation,
                                        headless=headless, graphics_device_id=graphics)
                high = {'args': outer, 'env': {'physics': {'config': 'low', 'test_name': 'HumanoidSMPLIMMVAE'},
                        'numEnvs': 1, 'vid2player': {}, 'controller_cfg_name': 'test'}}
                # The nested builder reparses CLI, so start with intentionally
                # different display settings and verify the outer settings win.
                parsed = SimpleNamespace(test=False, play=False, rl_device='cpu',
                                         headless=not headless, graphics_device_id=3)
                low = {'env': {}, 'sim': {}}
                namespace = {'get_args': lambda: parsed,
                             'load_cfg': lambda args: (low, {'params': {'config': {}}}),
                             'resolve_tennis_contacts': resolve_tennis_contacts}
                exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), 'exec'), namespace)
                args, cfg, _ = namespace['PlayerBuilder'](SimpleNamespace(device_id=0), high).build_player_cfg()
                self.assertEqual(args.headless, headless)
                self.assertEqual(args.graphics_device_id, graphics)
                self.assertEqual(cfg['env'].get('viewerFollow', True), validation)

    def test_plot_marks_contact_or_explicitly_reports_none(self):
        from matplotlib.figure import Figure
        rows = [dict(time_s=(i+1)/30, reward=.1, cumulative_reward=.1*(i+1), phase_rad=3.,
                     phase_target_rad=np.pi, phase_reward_factor=.9, is_reaction=True,
                     ball_racket_distance_m=.2, has_contact=False) for i in range(3)]
        for has_contact in [False, True]:
            rows[-1]['has_contact'] = has_contact
            with patch.object(Figure, 'savefig', autospec=True) as save:
                save_metrics_plot('unused.png', rows, 30)
            figure = save.call_args.args[0]
            self.assertEqual(len(figure.axes), 4)
            title = figure._suptitle.get_text()
            if has_contact:
                self.assertIn('First racket contact: 0.10s', title)
                np.testing.assert_allclose(figure.axes[3].get_xlim(), [-.4, .6])
                for axis in figure.axes:
                    markers = [line for line in axis.lines if line.get_label().startswith('first racket contact')]
                    self.assertEqual(len(markers), 1)
                    self.assertEqual(markers[0].get_color(), 'crimson')
            else:
                self.assertIn('No racket contact detected', title)
                self.assertIn('No racket contact detected',
                              [text.get_text() for text in figure.axes[3].texts])

    def test_plot_reward_contributions_for_both_stages(self):
        from matplotlib.figure import Figure
        cases = [
            {'pos_reward': [.1, .3, 0.]},
            {'pos_reward': [.01, .03, .1], 'ball_pos_reward': [0., .2, .8]},
            {'pos_reward': [.05, .15, 0.], 'contact_reward': [0., .5, 0.],
             'forward_reward': [0., 1.2, 0.], 'vertical_reward': [0., -.2, 0.],
             'orientation_reward': [.2, .4, 0.], 'target_reward': [0., 0., 0.]},
        ]
        for components in cases:
            with self.subTest(components=list(components)):
                rows = [dict(time_s=(i+1)/30, reward=sum(v[i] for v in components.values()),
                             cumulative_reward=0., phase_rad=3., phase_target_rad=np.pi,
                             ball_racket_distance_m=.2, has_contact=False,
                             weighted_reward_components={k: v[i] for k, v in components.items()})
                        for i in range(3)]
                with patch.object(Figure, 'savefig', autospec=True) as save:
                    save_metrics_plot('unused.png', rows, 30)
                axis = save.call_args.args[0].axes[2]
                self.assertIn('Reward components', axis.get_title())
                self.assertEqual([line.get_label() for line in axis.lines], list(components))
                for line, values in zip(axis.lines, components.values()):
                    np.testing.assert_allclose(line.get_xdata(), [r['time_s'] for r in rows])
                    np.testing.assert_allclose(line.get_ydata(), values)
                np.testing.assert_allclose(sum(line.get_ydata() for line in axis.lines),
                                           [r['reward'] for r in rows])
                rows[1]['has_contact'] = True
                with patch.object(Figure, 'savefig', autospec=True) as save:
                    save_metrics_plot('unused.png', rows, 30)
                zoom = save.call_args.args[0].axes[3]
                np.testing.assert_allclose(zoom.get_xlim(), [2/30-.5, 2/30+.5])
                for full, close in zip(axis.lines, zoom.lines):
                    self.assertEqual(full.get_label(), close.get_label())
                    np.testing.assert_allclose(full.get_ydata(), close.get_ydata())

    def test_offline_sampler_device_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            data=np.arange(6*307,dtype=np.float32).reshape(6,307)
            filename=Path(directory)/'balls.npy';np.save(filename,data)
            generator=offline_generator_class()(filename,sample_random=False,num_envs=6)
            generator.sample_idx[:]=torch.arange(6)
            conversions=[]
            # A device-boundary sentinel verifies explicit conversion even on a
            # CPU host: direct indexing with this object fails like CUDA IDs.
            class DeviceIds:
                def detach(self):
                    return self
                def to(self,device,dtype):
                    conversions.append((device,dtype))
                    return torch.tensor([4,1],device=device,dtype=dtype)
            result=generator.generate(2,True,env_ids=DeviceIds())
            self.assertEqual(conversions,[(torch.device('cpu'),torch.long)])
            np.testing.assert_array_equal(result[0].numpy(),data[[4,1],7:].reshape(2,100,3))
            np.testing.assert_array_equal(result[1].numpy(),data[[4,1],:3])
            self.assertEqual(generator.sample_idx.tolist(),[0,2,2,3,5,5])
            generator.generate(1,env_ids=torch.tensor([5]))
            self.assertEqual(generator.sample_idx[5].item(),0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_offline_sampler_cuda_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            data=np.arange(6*307,dtype=np.float32).reshape(6,307)
            filename=Path(directory)/'balls.npy';np.save(filename,data)
            cls=offline_generator_class()
            generator=cls(filename,sample_random=False,num_envs=6)
            generator.sample_idx[:]=torch.arange(6)
            result=generator.generate(2,True,env_ids=torch.tensor([4,1],device='cuda'))
            np.testing.assert_array_equal(result[0].numpy(),data[[4,1],7:].reshape(2,100,3))
            self.assertTrue(all(value.device.type=='cpu' for value in result))
            generator=cls(filename,sample_random=True,num_envs=6)
            generator.generate(2,True,start_pos=torch.tensor([[0.,-13.,1.],[0.,13.,1.]],device='cuda'))

    def test_case_selection_and_output_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = np.zeros((6,307), dtype=np.float32)
            for i in range(6):
                traj = np.zeros((100,3), dtype=np.float32)
                traj[:,0],traj[:,1],traj[:,2] = [-2,0,2][i//2],np.linspace(13,-15,100),[.9,1.4][i%2]
                data[i,:7] = [0,13,1,0,-15,1,5]
                data[i,7:] = traj.ravel()
            path = root/'balls.npy';np.save(path,data)
            for i,name in enumerate(['left_low','left_high','center_low','center_high','right_low','right_high']):
                rows,case = select_ball_case(path,name)
                self.assertEqual(case['source_row'],i)
                np.testing.assert_array_equal(rows,data[i:i+1])
            _,case = select_ball_case(path,'left_low',x=2.,height=1.4)
            self.assertEqual(case['source_row'],5)
            args = SimpleNamespace(eval_checkpoint=str(root/'policy.pth'),eval_steps=4,
                seed=None,eval_output=str(root/'eval'),cfg='test',eval_ball='center_low',
                eval_ball_x=None,eval_ball_height=None,eval_target='center',
                headless=True,graphics_device_id=0)
            (root/'policy.pth').write_bytes(b'placeholder')
            configure_eval_args(args)
            self.assertTrue(args.headless and args.test and not args.train)
            self.assertEqual(args.graphics_device_id,-1)
            args.headless,args.graphics_device_id = False,0
            configure_eval_args(args)
            self.assertFalse(args.headless)
            self.assertEqual(args.graphics_device_id,0)
            self.assertEqual(args.num_envs,1)
            cfg = {'name':'MVAEControllerVis','env':{'vid2player':{'ball_traj_file':str(path)}}}
            training = {'params':{'config':{}}}
            output,case,temporary = prepare_evaluation(args,cfg,training)
            self.assertEqual(cfg['name'],'PhysicsMVAEController')
            self.assertEqual([x.name for x in output.iterdir()],['evaluation_config.json'])
            self.assertTrue(Path(temporary.name,'launch.npy').is_file())
            temporary.cleanup()
            with self.assertRaises(FileExistsError):
                prepare_evaluation(args,cfg,training)
            args.eval_ball='invalid'
            with self.assertRaises(ValueError):
                configure_eval_args(args)

    def test_mvae_camera_equivalence(self):
        path=Path(__file__).resolve().parents[1]/'motion_vae/reconstruction.py'
        nodes=[node for node in ast.parse(path.read_text()).body
               if isinstance(node,ast.ClassDef) and node.name=='CourtCamera']
        ns={'np':np}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
        mvae=ns['CourtCamera']()
        camera=CourtRenderer(['Pelvis'],[-5,-16],[5,-10])
        points=np.array([[0,-13,1],[-5,-11.89,0],[5,11.89,0],[0,0,1]])
        np.testing.assert_allclose(camera.project(points),mvae.project(points)[0])
        self.assertEqual(camera.background.size,(768,720))

    def test_recording_loop(self):
        self.recording_loop()

    def test_live_recording_defers_video_and_preserves_samples(self):
        self.recording_loop(viewer_steps=4)

    def test_viewer_close_saves_partial_and_empty_rollouts(self):
        for count in (0, 2):
            with self.subTest(count=count):
                self.recording_loop(viewer_steps=count)

    def recording_loop(self, viewer_steps=None):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/'eval';output.mkdir()
            model=torch.nn.Linear(1,1)
            checkpoint=root/'policy.pth';torch.save({'model':model.state_dict()},checkpoint)
            source=root/'source.npy';np.save(source,np.zeros((1,307),dtype=np.float32))
            (output/'evaluation_config.json').write_text('{}')
            args=SimpleNamespace(eval_checkpoint=str(checkpoint),eval_steps=5,seed=123,eval_target='right')
            physics=SimpleNamespace(dt=1/30,humanoid_asset=object(),
                gym=SimpleNamespace(get_asset_rigid_body_names=lambda _:['Pelvis','Torso','Spine','Chest','Neck','Head']),
                _rigid_body_pos=torch.zeros(1,6,3),_has_racket_ball_contact=torch.zeros(1,dtype=torch.bool),
                _has_bounce_now=torch.zeros(1,dtype=torch.bool),_bounce_pos=torch.zeros(1,3),
                _ball_generator=SimpleNamespace(sample_idx=torch.zeros(1,dtype=torch.long)))
            from vid2player.utils.racket_diagnostics import racket_ball_snapshot
            from vid2player.utils.tennis_contacts import resolve_tennis_contacts
            from vid2player.utils.racket_dynamics import resolve_racket_dynamics
            from vid2player.utils.ball_physics import resolve_ball_physics
            vector=lambda x,y,z:SimpleNamespace(x=x,y=y,z=z)
            prop=SimpleNamespace(mass=.27,com=vector(.18,0,0),
                inertia=SimpleNamespace(x=vector(.0012,0,0),y=vector(0,.0123,0),z=vector(0,0,.01349)))
            physics.gym.get_actor_rigid_body_properties=lambda *args:[prop]
            physics.envs=[object()];physics.humanoid_handles=[object()]
            physics.sim_params=SimpleNamespace(dt=1/60,substeps=12)
            physics._tennis_contacts=resolve_tennis_contacts({'model':'hardcourt_v1'})
            physics._racket_dynamics=resolve_racket_dynamics()
            physics._ball_physics=resolve_ball_physics()
            physics._contact_env_ids=torch.tensor([0]);physics._contact_racket_ids=torch.tensor(0)
            physics._rigid_body_pos_origin=torch.zeros(1,1,3)
            physics._rigid_body_rot_origin=torch.tensor([[[0.,0.,0.,1.]]])
            physics._rigid_body_vel_origin=torch.zeros(1,1,3)
            physics._rigid_body_ang_vel_origin=torch.zeros(1,1,3)
            physics._ball_root_states=torch.zeros(1,13)
            physics._ball_root_states[:,6]=1
            task=SimpleNamespace(device='cpu',_physics_player=SimpleNamespace(task=physics),
                cfg_v2p={'court_min':[-5,-16],'court_max':[5,-10],
                         'reward_type':'reach', 'reward_weights':{'pos':2.},'aux_loss_specs':{'dof_res':.01}},
                _reward_weights={'pos':1.},
                _phase_pred=torch.tensor([3.]), _reward_scales={'phase':5.},
                _tar_action=torch.ones(1,dtype=torch.long),
                _mvae_player=SimpleNamespace(_swing_type=torch.tensor([1])),
                _num_mvae_action=32,_num_res_dof_action=3,
                _reset_reaction_buf=torch.ones(1,dtype=torch.bool),_reset_recovery_buf=torch.zeros(1,dtype=torch.bool),
                _tar_time_total=torch.zeros(1,dtype=torch.long),_target_bounce_pos=torch.zeros(1,3),
                _ball_pos=torch.zeros(1,3),_racket_pos=torch.zeros(1,3),_racket_normal=torch.zeros(1,3),
                _est_bounce_in=torch.zeros(1,dtype=torch.bool),_root_pos=torch.tensor([[0.,-13.,1.]]),
                obs_buf=torch.zeros(1,1),_compute_observations=lambda:None)
            task._racket_normal[:,1]=1
            physics._rigid_body_pos[:,:,1]=-13
            physics._rigid_body_pos[:,:,2]=torch.linspace(.8,1.8,6)
            task._racket_pos[:,1:]=torch.tensor([-13.,1.])
            steps=[];resets=[];cleanups=[]
            def reset(ids):
                resets.append(ids.tolist())
                return {'obs':task.obs_buf}
            def step(env,action):
                n=len(steps)+1;steps.append(n)
                for interval in range(2):
                    pre=racket_ball_snapshot(physics)
                    physics._ball_root_states[:,8]=n + interval/2
                    post=racket_ball_snapshot(physics)
                    physics._racket_diagnostic_samples.append({
                        **{'pre_'+k:v for k,v in pre.items()},
                        **{'post_'+k:v for k,v in post.items()},
                        'accepted_contact':torch.tensor([n==2 and interval==0]),
                        'sensor_evidence':torch.tensor([False]),
                        'impulse_evidence':torch.tensor([n==2 and interval==0])})
                task._phase_pred[:]=2. + n*.2
                task._ball_pos[:,1:]=torch.tensor([-13.,1.])
                task._ball_pos[:,0]=.4/n
                physics._has_racket_ball_contact[:]=n>=2
                task._est_bounce_in[:]=n>=2
                done=torch.tensor([n>=4])
                if n>=4:
                    task._root_pos[0,0]=6
                task._reset_reaction_buf[:]=done
                return {'obs':task.obs_buf},torch.tensor([[.1]]),done,{
                    'sub_rewards':torch.tensor([[.1]]),'sub_rewards_names':'pos_reward'}
            means=torch.zeros(1,35);means[0,32]=.2;means[0,0]=1.1
            player=SimpleNamespace(device='cpu',model=model,normalize_input=False,task=task,is_rnn=False,
                config={'bounds_loss_coef':10.},env=object(),env_reset=reset,get_batch_size=lambda obs,n:n,
                obs_to_torch=lambda obs:{'obs':obs},get_action=lambda *a,**k:means,env_step=step)
            case={'preset':'right_high','source_row':0,'source_file':str(source)}
            temporary=SimpleNamespace(cleanup=lambda:cleanups.append(True))
            config_stub=SimpleNamespace(set_seed=lambda *args:torch.manual_seed(args[0]))
            modules = {'vid2player.utils.config':config_stub}
            if viewer_steps is not None:
                physics.viewer = object()
                physics.gym.viewer_camera_look_at = lambda *a: None
                modules['isaacgym'] = SimpleNamespace(gymapi=SimpleNamespace(Vec3=vector))
            render = CourtRenderer.render
            rendered_after_steps = []
            def render_frame(*args):
                rendered_after_steps.append(len(steps))
                return render(*args)
            with patch.dict(sys.modules,modules), \
                 patch.object(controller_viewer, 'EvaluationViewer') as viewer_type, \
                 patch.object(CourtRenderer,'render',side_effect=render_frame,autospec=True):
                viewer_type.return_value.wait_for_step.side_effect = [True] * (viewer_steps or 0) + [False]
                evaluate_controller(player,args,(output,case,temporary))
            report=json.loads((output/'evaluation_config.json').read_text())['evaluation_results']
            data=np.load(output/'rollout.npz')
            if viewer_steps is not None:
                self.assertEqual(rendered_after_steps, [viewer_steps] * viewer_steps)
                self.assertFalse(physics._evaluation_viewer_active)
                if viewer_steps < 4:
                    self.assertEqual(report['steps'],viewer_steps)
                    self.assertEqual(report['end_reason'],'viewer_closed')
                    self.assertEqual(len(steps),viewer_steps)
                    self.assertEqual(data['metrics'].shape[0],viewer_steps)
                    self.assertEqual(cleanups,[True])
                    self.assertTrue((output/'metrics.png').is_file())
                    return
            first=dict(zip(data['metric_names'].tolist(),data['metrics'][0]))
            metric_columns = dict(zip(data['metric_names'].tolist(), data['metrics'].T))
            np.testing.assert_array_equal(metric_columns['registered_contacts_this_step'], [0,1,0,0])
            np.testing.assert_array_equal(metric_columns['registered_contacts_total'], [0,1,1,1])
            self.assertEqual(report['registered_contacts_total'], 1)
            self.assertEqual(report['contact_evidence_intervals'][0]['id'], 1)
            self.assertEqual(report['contact_evidence_intervals'][0]['status'], 'registered')
            self.assertAlmostEqual(report['contact_evidence_intervals'][0]['time_end_s'], .05)
            np.testing.assert_allclose(data['physics_time_end_s'], np.arange(1,9)/60)
            self.assertEqual(report['steps'],4)
            self.assertEqual(report['contact_step'],2)
            self.assertEqual(data['racket_state'].shape,(4,13))
            self.assertEqual(data['body_state'].shape,(4,1,13))
            self.assertEqual(data['physics_pre_body_state'].shape,(8,1,13))
            self.assertEqual(data['physics_pre_ball_state'].shape,(8,13))
            np.testing.assert_array_equal(data['physics_control_step'],[1,1,2,2,3,3,4,4])
            np.testing.assert_array_equal(data['physics_interval_in_control_step'],[0,1]*4)
            np.testing.assert_allclose(data['ball_state'][:,8],[1.5,2.5,3.5,4.5])
            self.assertEqual(data['physics_accepted_contact'].sum(),1)
            self.assertTrue(data['physics_accepted_contact'][2])
            self.assertEqual(data['physics_pre_ball_state'][0,8],0.)
            metadata=json.loads(data['diagnostics_metadata_json'].item())
            self.assertEqual(metadata['racket_com_local_m'],[.18,0,0])
            self.assertFalse(physics._record_racket_diagnostics)
            self.assertEqual(report['end_reason'],'movement_boundary')
            self.assertAlmostEqual(report['total_reward'],.4)
            self.assertAlmostEqual(first['aux_dof_res_loss'],.04)
            self.assertAlmostEqual(first['weighted_aux_dof_res_loss'],.0004)
            self.assertAlmostEqual(first['bounds_loss'],.01,places=6)
            self.assertAlmostEqual(first['weighted_bounds_loss'],.1,places=6)
            self.assertAlmostEqual(first['total_regularization_loss'],.1004,places=6)
            self.assertAlmostEqual(first['weighted/pos_reward'],.1)
            self.assertAlmostEqual(first['phase_rad'],2.2,places=6)
            self.assertAlmostEqual(first['phase_target_rad'],np.pi)
            self.assertAlmostEqual(first['phase_reward_factor'],np.exp(-5*(2.2-np.pi)**2),places=6)
            self.assertEqual(first['is_reaction'],1.)
            self.assertEqual(resets[0],[0])
            self.assertTrue(all(not ids for ids in resets[1:]))
            self.assertEqual(cleanups,[True])
            self.assertEqual({p.name for p in output.iterdir()},
                {'video.mp4','evaluation_config.json','rollout.npz','metrics.png'})
            np.testing.assert_array_equal(task._target_bounce_pos.numpy(),[[3.,10.,0.]])
            reader=imageio.get_reader(output/'video.mp4')
            self.assertEqual(reader.get_data(0).shape,(720,768,3))
            reader.close()
            from PIL import Image
            with Image.open(output/'metrics.png') as plot:
                self.assertEqual(plot.size,(1800,1200))

    def test_phase_diagnostics_match_rewards(self):
        from vid2player.tests.test_controller_baseline import definitions, ROOT
        import math
        ns=definitions(ROOT/'env/tasks/physics_mvae_controller.py',
                       {'compute_reward_reach','compute_reward_return','compute_reward_return_w_estimate'},
                       {'torch':torch,'math':math})
        zeros=torch.zeros(1,3);no=torch.zeros(1,dtype=torch.bool)
        for kind in ['reach','return','return_w_estimate']:
            for swing in [-1,1,2]:
                for phase in [0.,3.,np.pi,6.2]:
                    task=SimpleNamespace(cfg_v2p={'reward_type':kind},_phase_pred=torch.tensor([phase]),
                        _reward_scales={'phase':5.},_tar_action=torch.ones(1),
                        _mvae_player=SimpleNamespace(_swing_type=torch.tensor([swing]),
                            _swing_type_cycle=torch.tensor([swing])))
                    if kind=='reach':
                        reward,*_=ns['compute_reward_reach'](task._phase_pred,task._tar_action,zeros,zeros,
                            task._mvae_player._swing_type,task._reward_scales,{'pos':1.})
                    elif kind=='return':
                        reward,*_=ns['compute_reward_return'](task._phase_pred,zeros,zeros,no,no,zeros,zeros,
                            task._mvae_player._swing_type,task._reward_scales,{'pos':1.})
                    else:
                        reward,*_=ns['compute_reward_return_w_estimate'](zeros,task._phase_pred,
                            task._mvae_player._swing_type_cycle,zeros,no,zeros,torch.zeros(1),no,zeros,
                            task._reward_scales,{'pos':1.})
                    self.assertAlmostEqual(phase_metrics(task)['phase_reward_factor'],reward.item(),places=6)
        # A disabled phase head still records its placeholder honestly; zero scale is neutral.
        task.cfg_v2p['predict_phase']=False
        task._reward_scales['phase']=0.
        self.assertEqual(phase_metrics(task)['phase_reward_factor'],1.)
        self.assertFalse(phase_metrics(task)['phase_enabled'])


if __name__=='__main__':
    unittest.main()
