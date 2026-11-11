"""Meaningful impact geometry, opt-in compatibility and curriculum checks on CPU."""
import math
from types import SimpleNamespace as NS
import unittest

import torch
import yaml

from vid2player.tests.test_controller_baseline import ROOT, definitions
from vid2player.utils.contact_quality import (
    RacketContactQuality, StringBedGeometry, read_string_bed_geometry, resolve_contact_quality)
from vid2player.utils.return_curriculum import ReturnCurriculum, COMPONENTS
from vid2player.utils.racket_diagnostics import racket_state_snapshot, racket_ball_snapshot
from vid2player.utils.tennis_contacts import racket_impact


class ContactQualityTests(unittest.TestCase):
    def make_quality(self, count=1, settings=None, geometry=None, com=(0., 0., 0.), margin=.07):
        return RacketContactQuality(count, 'cpu', settings,
            geometry or StringBedGeometry((0.,0.,0.), (0.,1.,0.), .15, .02),
            com, .0335, margin)

    def states(self, count=1):
        ball, racket = torch.zeros(count,13), torch.zeros(count,13)
        racket[:,6] = 1.
        ball[:,1], ball[:,8] = .1, -10.
        return ball, racket

    def curriculum(self, count=1, config=None, weights=None):
        obj = ReturnCurriculum(count, 'cpu', config or {},
            weights if weights is not None else {'contact':1., 'clean_contact':.5, 'contact_angle':0.},
            {}, ball_radius=.0335)
        ball = torch.tensor([[0.,-10.,1.]]).repeat(count,1)
        obj.reset(torch.arange(count),ball)
        return obj, ball

    def step(self, obj, ball, scores=None, contact=True):
        count = len(ball)
        velocity = ball.new_tensor([[0.,-5.,-4.]]).repeat(count,1)
        return obj.step(ball, velocity, torch.full((count,),contact), torch.zeros(count,dtype=torch.bool),
                        ball, ball, torch.zeros(count,dtype=torch.bool), contact_quality=scores)

    def test_asset_geometry_uses_physical_head_not_grip_or_com(self):
        geom = read_string_bed_geometry(ROOT/'data/assets/smpl_mesh_humanoid_djokovic.xml')
        self.assertEqual(geom.center,(0.,0.,0.))
        self.assertAlmostEqual(geom.radius,.15)
        self.assertAlmostEqual(geom.half_thickness,math.sqrt(2)*.015)
        torch.testing.assert_close(torch.tensor(geom.normal),torch.tensor([0.,2**-.5,2**-.5]))

    def test_swept_center_face_partial_edge_handle_and_remote_impacts(self):
        quality = self.make_quality(6)
        ball,racket = self.states(6)
        # Identical impacts with different radial positions. A ball near the
        # center but far from the plane also cannot earn clean-hit credit.
        ball[:,0] = torch.tensor([0.,.075,.1125,.15,.25,0.])
        ball[-1,1] = 2.
        quality.record(torch.ones(6,dtype=torch.bool),ball,racket,racket,1/60)
        expected = torch.tensor([1.,1.,.5,0.,0.,0.])
        torch.testing.assert_close(quality.scores['clean_contact'],expected,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(quality.scores['contact_angle'],expected,atol=1e-6,rtol=1e-6)
        # No accepted event means later intervals do not overwrite the first.
        quality.record(torch.zeros(6,dtype=torch.bool),ball+10,racket,racket,1/60)
        torch.testing.assert_close(quality.scores['clean_contact'],expected,atol=1e-6,rtol=1e-6)
        quality.reset(torch.tensor([0,2]))
        torch.testing.assert_close(quality.scores['clean_contact'],torch.tensor([0.,1.,0.,0.,0.,0.]))
        quality.reset()
        self.assertEqual(quality.scores['contact_angle'].sum(),0)

    def test_speculative_plane_region_is_graded_not_full_credit_far_from_face(self):
        quality = self.make_quality(3)
        ball,racket = self.states(3)
        # Parallel path inside, halfway through, and outside the contact margin.
        ball[:,1] = torch.tensor([.0535,.0885,.1335])
        ball[:,7:10] = torch.tensor([0.,0.,1.])
        quality.record(torch.ones(3,dtype=torch.bool),ball,racket,racket,1/60)
        torch.testing.assert_close(quality.scores['clean_contact'],torch.tensor([1.,.5,0.]),atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(quality.scores['contact_angle'],torch.zeros(3))

    def test_angle_plateau_taper_either_face_and_zero_speed(self):
        angles = torch.tensor([0.,15.,30.,60.,90.])
        quality = self.make_quality(5)
        ball,racket = self.states(5)
        ball[:,1] = .0535
        ball[:,8] = -10*torch.cos(angles*math.pi/180)
        ball[:,9] = 10*torch.sin(angles*math.pi/180)
        quality.record(torch.ones(5,dtype=torch.bool),ball,racket,racket,1/60)
        torch.testing.assert_close(quality.scores['contact_angle'],torch.tensor([1.,1.,1.,.5,0.]),atol=1e-6,rtol=1e-6)
        # Either contacting side scores identically, independent of normal sign.
        ball[:,1] *= -1; ball[:,8] *= -1
        quality.record(torch.ones(5,dtype=torch.bool),ball,racket,racket,1/60)
        torch.testing.assert_close(quality.scores['contact_angle'],torch.tensor([1.,1.,1.,.5,0.]),atol=1e-6,rtol=1e-6)
        ball[:,7:10] = 0
        quality.record(torch.ones(5,dtype=torch.bool),ball,racket,racket,1/60)
        torch.testing.assert_close(quality.scores['contact_angle'],torch.zeros(5))
        # Tightening leniency in YAML changes credit without changing clean contact.
        strict = self.make_quality(5,settings={'angle_tolerance_deg':0.,'angle_falloff_deg':90.})
        ball[:,8] = 10*torch.cos(angles*math.pi/180)
        ball[:,9] = 10*torch.sin(angles*math.pi/180)
        strict.record(torch.ones(5,dtype=torch.bool),ball,racket,racket,1/60)
        torch.testing.assert_close(strict.scores['contact_angle'],1-angles/90,atol=1e-6,rtol=1e-6)

    def test_relative_surface_velocity_includes_angular_motion_about_actual_com(self):
        ball,racket = self.states()
        ball[:,1] = .0535
        ball[:,8] = -10.; ball[:,9] = 10.
        # COM offset +X, omega +Y makes a +Z surface speed at the center.
        racket[:,11] = 10.
        moving = self.make_quality(com=(1.,0.,0.))
        moving.record(torch.tensor([True]),ball,racket,racket,1/60)
        self.assertAlmostEqual(moving.scores['contact_angle'][0].item(),1.,places=6)
        static = self.make_quality()
        static.record(torch.tensor([True]),ball,racket,racket,1/60)
        self.assertAlmostEqual(static.scores['contact_angle'][0].item(),.75,places=6)
        # World ball velocity is unchanged: matching COM translation removes the
        # tangential relative component too.
        racket[:,10:13] = 0.; racket[:,9] = 10.
        static.record(torch.tensor([True]),ball,racket,racket,1/60)
        self.assertAlmostEqual(static.scores['contact_angle'][0].item(),1.,places=6)

    def test_geometry_is_invariant_to_world_rotation_and_translation(self):
        quality = self.make_quality()
        ball,racket = self.states()
        # Local normal +Y rotated to world -X; translation is arbitrary.
        racket[:,5:7] = math.sqrt(.5)
        racket[:,:3] = torch.tensor([3.,-12.,2.])
        ball[:,:3] = racket[:,:3]+torch.tensor([-.1,0.,0.])
        ball[:,7:10] = torch.tensor([10.,0.,0.])
        quality.record(torch.tensor([True]),ball,racket,racket,1/60)
        self.assertAlmostEqual(quality.scores['clean_contact'][0].item(),1.,places=6)
        self.assertAlmostEqual(quality.scores['contact_angle'][0].item(),1.,places=6)

    def test_nonfinite_geometry_scores_zero_and_bad_settings_fail(self):
        for spec in [{'bad_key':1}, {'full_credit_radius_fraction':1.},
                     {'zero_credit_radius_fraction':1.1}, {'full_credit_radius_fraction':-.1},
                     {'angle_tolerance_deg':-1.}, {'angle_tolerance_deg':90.},
                     {'angle_falloff_deg':0.}, {'angle_tolerance_deg':60.},
                     {'angle_tolerance_deg':float('nan')}, {'angle_tolerance_deg':True}]:
            with self.subTest(spec=spec),self.assertRaises(ValueError): resolve_contact_quality(spec)
        quality = self.make_quality(3)
        ball,racket = self.states(3)
        ball[0,0] = float('nan'); racket[1,7] = float('inf')
        post = racket.clone(); post[2,0] = float('nan')
        quality.record(torch.ones(3,dtype=torch.bool),ball,racket,post,1/60)
        for value in quality.scores.values(): torch.testing.assert_close(value,torch.zeros(3))

    def test_quality_is_additive_once_per_launch_and_does_not_gate_penalties(self):
        obj,ball = self.curriculum(2,config={'forward_reward_mode':'signed_linear','backward_penalty_scale':.5},
            weights={'contact':1.,'forward':2.,'vertical':.2,'clean_contact':.5,'contact_angle':.25})
        scores = {'clean_contact':torch.tensor([0.,1.]),'contact_angle':torch.tensor([0.,.5])}
        events = self.step(obj,ball,scores)
        self.assertEqual(obj.components,COMPONENTS+('vertical','clean_contact','contact_angle'))
        torch.testing.assert_close(events[:,0],torch.ones(2))
        torch.testing.assert_close(events[:,1],torch.full((2,),-.25))
        torch.testing.assert_close(events[:,-3],torch.full((2,),-.5))
        torch.testing.assert_close(events[:,-2:],torch.tensor([[0.,0.],[1.,.5]]))
        torch.testing.assert_close(self.step(obj,ball,scores),torch.zeros_like(events))
        stats = obj.stats(torch.ones(2,dtype=torch.bool),torch.zeros(2,dtype=torch.bool))
        self.assertEqual(stats['return_clean_contact_sum'],1.)
        self.assertEqual(stats['return_contact_angle_sum'],.5)
        obj.reset(torch.tensor([1]),ball[1:])
        self.assertEqual(obj.quality_totals['clean_contact'][1],0.)
        torch.testing.assert_close(self.step(obj,ball,scores)[:,-2],torch.tensor([0.,1.]))
        # Angle-only opt-in must work without the clean-contact reward column.
        angle,ball = self.curriculum(weights={'contact_angle':1.})
        self.assertEqual(angle.components,COMPONENTS+('contact_angle',))
        with self.assertRaises(ValueError): self.step(angle,ball)
        self.assertEqual(self.step(angle,ball,{'contact_angle':torch.ones(1)})[0,-1],1.)

    def test_weight_schedules_ramp_both_rewards_on_and_off_and_eval_stays_fixed(self):
        config = {'weights_schedule': {key:{'segments':[
            {'start':0.,'end':1.,'start_frame':0,'end_frame':100},
            {'start':1.,'end':0.,'start_frame':200,'end_frame':300}]}
            for key in ('clean_contact','contact_angle')}}
        obj,_ = self.curriculum(config=config,weights={'clean_contact':.5,'contact_angle':0.})
        self.assertEqual(obj.weights['clean_contact'],.5)
        self.assertEqual(obj.weights['contact_angle'],0.)
        for frame,expected in [(0,0.),(50,.5),(100,1.),(150,1.),(250,.5),(300,0.),(400,0.)]:
            obj.set_progress(frame)
            for key in ('clean_contact','contact_angle'):
                self.assertAlmostEqual(obj.weights[key],expected)
                self.assertAlmostEqual(obj.schedule_metrics()['curriculum/weight_'+key],expected)

    def test_2_5_is_a_new_200_epoch_experiment_with_no_vertical_or_angle_credit(self):
        cfg_root = ROOT/'cfg/controller/usopen_train_stage_2'
        source = yaml.safe_load((cfg_root/'high_level_2_4.yaml').read_text())
        cfg = yaml.safe_load((cfg_root/'high_level_2_5.yaml').read_text())
        v = cfg['env']['vid2player']
        obj,_ = self.curriculum(config=v['return_curriculum'],weights=v['reward_weights'].copy())
        self.assertEqual(obj.weights['vertical'],0.)
        self.assertEqual(obj.weights['contact_angle'],0.)
        self.assertEqual(obj.weights['clean_contact'],.5)
        self.assertEqual(obj.weights['forward'],2.)
        for epoch in [0,50,100,150,200,250,500]:
            obj.set_progress(epoch*2048*64)
            self.assertEqual(obj.weights['vertical'],0.)
            self.assertEqual(obj.weights['contact_angle'],0.)
            self.assertEqual([obj.weights[k] for k in ('net','court','target')],[0.,0.,0.])
            if epoch >= 100:
                self.assertEqual(obj.weights['clean_contact'],.5)
                self.assertEqual(obj.weights['forward'],2.)
        self.assertNotIn('vertical',obj.logged_components)
        self.assertNotIn('contact_angle',obj.logged_components)
        self.assertEqual(cfg['params']['config']['name'],'high_level_2_5')
        self.assertEqual(cfg['params']['config']['max_epochs'],200)
        self.assertEqual(cfg['env']['physics'],source['env']['physics'])
        for key,value in source['env']['vid2player'].items():
            if key not in ('return_curriculum','reward_weights'):
                self.assertEqual(v[key],value,key)
        self.assertEqual(cfg['params']['config']['pretrained_model_cp'],source['params']['config']['pretrained_model_cp'])
        # If later explicitly enabled, resume uses the existing stage-local clock.
        continuation = v['return_curriculum']
        continuation['weights_schedule']['vertical']['enabled'] = True
        continuation['weights_schedule']['contact_angle']['enabled'] = True
        obj,_ = self.curriculum(config=continuation,weights=v['reward_weights'].copy())
        for epoch,vertical,angle in [(200,0.,0.),(225,.1,.125),(250,.2,.25)]:
            obj.set_progress(epoch*2048*64)
            self.assertAlmostEqual(obj.weights['vertical'],vertical)
            self.assertAlmostEqual(obj.weights['contact_angle'],angle)

    def test_all_previous_yamls_retain_original_component_shapes_and_no_geometry_requirement(self):
        for path in (ROOT/'cfg/controller').glob('**/*.yaml'):
            cfg = yaml.safe_load(path.read_text()); v = cfg.get('env',{}).get('vid2player',{})
            if v.get('reward_type') != 'return_curriculum' or path.stem == 'high_level_2_5': continue
            obj,_ = self.curriculum(config=v.get('return_curriculum'),weights=v['reward_weights'].copy())
            self.assertFalse(obj.needs_contact_quality,path)
            expected = COMPONENTS+(('vertical',) if 'vertical' in v['reward_weights'] or
                'vertical' in v.get('return_curriculum',{}).get('weights_schedule',{}) else ())
            self.assertEqual(obj.components,expected,path)

    def test_controller_integrates_geometry_scores_by_name_and_logs_quality(self):
        scope = definitions(ROOT/'env/tasks/physics_mvae_controller.py',
            {'PhysicsMVAEController','compute_reward_reach'}, {'torch':torch,'math':math})
        task = object.__new__(scope['PhysicsMVAEController'])
        obj,ball = self.curriculum(weights={'pos':0.,'contact':1.,'forward':2.,'vertical':.2,
                                           'clean_contact':.5,'contact_angle':.25})
        task.cfg_v2p = {'reward_type':'return_curriculum'}
        task._return_curriculum = obj
        task._reward_weights,task._reward_scales = obj.weights,obj.scales
        task._phase_pred = torch.tensor([3.]); task._tar_action = torch.ones(1)
        task._ball_pos = task._racket_pos = ball
        task._ball_vel = torch.tensor([[0.,2.,4.]])
        task._est_bounce_valid = torch.zeros(1,dtype=torch.bool)
        task._target_bounce_pos = ball; task._mvae_player = NS(_swing_type=torch.tensor([-1]))
        task._physics_player = NS(task=NS(_has_racket_ball_contact_now=torch.tensor([True]),
            _has_bounce_now=torch.tensor([False]),_bounce_pos=ball,
            _contact_quality=NS(scores={'clean_contact':torch.ones(1),'contact_angle':torch.tensor([.5])})))
        task.rew_buf = torch.zeros(1)
        task._compute_reward(None)
        torch.testing.assert_close(task.rew_buf,torch.tensor([2.025]))
        self.assertEqual(task._sub_rewards_names.split(',')[-2:],['clean_contact_reward','contact_angle_reward'])
        summarize = definitions(ROOT/'agents/v2p_agent.py',{'summarize_task_stats'}, {})['summarize_task_stats']
        stats = obj.stats(torch.tensor([True]),torch.tensor([False]))
        result = summarize(stats)
        self.assertEqual(result['return/clean_contact_per_contact'],1.)
        self.assertEqual(result['return/contact_angle_per_contact'],.5)

    def test_physics_loop_latches_first_interval_quality_without_changing_detection(self):
        scope = definitions(ROOT/'env/tasks/humanoid_smpl_im_mvae.py',{'HumanoidSMPLIMMVAE'},
            {'torch':torch, 'HumanoidSMPL':object, 'racket_state_snapshot':racket_state_snapshot,
             'racket_ball_snapshot':racket_ball_snapshot, 'racket_impact':racket_impact,
             'ground_impact':lambda *a:(torch.tensor([False]),torch.zeros(1,3)),
             'gymtorch':NS(unwrap_tensor=lambda x:x), 'gymapi':NS(ENV_SPACE=0)})
        step = scope['HumanoidSMPLIMMVAE']._physics_step
        results = []
        for enabled in (False,True):
            ball,racket = self.states()
            ball[:,2] = racket[:,2] = 1.
            ball[:,1] = .15
            state = NS(control_freq_inv=2, _tennis_contacts=NS(enabled=True,racket_contact_offset=.05),
                _contact_env_ids=torch.tensor([0]),_contact_racket_ids=torch.tensor(0),
                _ball_root_states=ball, _rigid_body_pos_origin=racket[:,:3].reshape(1,1,3),
                _rigid_body_rot_origin=racket[:,3:7].reshape(1,1,4),
                _rigid_body_vel_origin=racket[:,7:10].reshape(1,1,3),
                _rigid_body_ang_vel_origin=racket[:,10:13].reshape(1,1,3),
                forces=torch.zeros(1,2,3),torques=torch.zeros(1,2,3),
                _contact_forces=torch.zeros(1,2,3),_contact_forces_sum=torch.zeros(1,2,3),
                _ball_physics=NS(radius=.0335),sim_params=NS(dt=1/60),sim=None,_is_train=True,
                _has_bounce=torch.tensor([False]),_has_bounce_now=torch.tensor([False]),
                _bounce_pos=torch.zeros(1,3),_has_ground_impact_now=torch.tensor([False]),
                _ground_impact_pos=torch.zeros(1,3),_has_racket_ball_contact=torch.tensor([False]),
                _has_racket_ball_contact_now=torch.tensor([False]),
                apply_external_force_to_ball=lambda *a:None,render=lambda:None)
            if enabled: state._contact_quality = self.make_quality()
            calls = []
            def simulate(*a):
                calls.append(1)
                state._ball_root_states[:,8] = 2.
                state._ball_root_states[:,9] -= 9.81/60
                # Second interval has very different geometry. It must not
                # overwrite the first contact's scores used by the controller.
                state._ball_root_states[:,0] = 0. if len(calls)==1 else 1.
            noop = lambda *a:None
            state.gym = NS(simulate=simulate,fetch_results=noop,
                apply_rigid_body_force_tensors=noop,refresh_actor_root_state_tensor=noop,
                refresh_rigid_body_state_tensor=noop,refresh_net_contact_force_tensor=noop)
            step(state)
            results.append((state._has_racket_ball_contact.clone(),state._ball_root_states.clone()))
            if enabled:
                self.assertAlmostEqual(state._contact_quality.scores['clean_contact'][0].item(),1.,places=6)
                self.assertAlmostEqual(state._contact_quality.scores['contact_angle'][0].item(),1.,places=6)
                # Another step without a fresh hit must not reuse these scores.
                step(state)
                self.assertEqual(state._contact_quality.scores['clean_contact'][0],0.)
        for old,new in zip(*results): torch.testing.assert_close(old,new)

    def test_physics_configuration_reads_live_com_and_rejects_legacy_detector(self):
        scope = definitions(ROOT/'env/tasks/humanoid_smpl_im_mvae.py',{'HumanoidSMPLIMMVAE'},
            {'HumanoidSMPL':object, 'RacketContactQuality':RacketContactQuality,
             'read_string_bed_geometry':read_string_bed_geometry})
        configure = scope['HumanoidSMPLIMMVAE'].configure_contact_quality
        state = NS(_tennis_contacts=NS(enabled=True,ball_contact_offset=.02,racket_contact_offset=.05),
            cfg_v2p={},humanoid_files={0:ROOT/'data/assets/smpl_mesh_humanoid_djokovic.xml'},
            _racket_body_id_true=0,envs=[None],humanoid_handles=[None],num_envs=1,device='cpu',
            _ball_physics=NS(radius=.0335),gym=NS(get_actor_rigid_body_properties=lambda *a:[NS(com=NS(x=.18,y=0.,z=0.))]))
        configure(state,resolve_contact_quality())
        torch.testing.assert_close(state._contact_quality.com,torch.tensor([.18,0.,0.]))
        self.assertAlmostEqual(state._contact_quality.contact_margin,.07)
        state._tennis_contacts.enabled = False
        with self.assertRaises(ValueError): configure(state,resolve_contact_quality())


if __name__ == '__main__': unittest.main()
