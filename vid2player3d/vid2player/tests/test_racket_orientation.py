"""Useful face orientation, bounded preparation, first impact and live wiring."""
import math
from types import SimpleNamespace as NS
import unittest
import torch

from vid2player.tests.test_controller_baseline import ROOT, definitions
from vid2player.utils.contact_quality import StringBedGeometry
from vid2player.utils.racket_orientation import RacketOrientationReward, resolve_orientation
from vid2player.utils.racket_diagnostics import racket_state_snapshot
from vid2player.utils.return_curriculum import ReturnCurriculum
from vid2player.utils.tennis_contacts import racket_impact


class RacketOrientationTests(unittest.TestCase):
    def fixture(self, count=1, settings=None, dt=1/30):
        helper = RacketOrientationReward(count,'cpu',settings,
            StringBedGeometry((0.,0.,0.),(0.,1.,0.),.15,.02),.0335,dt)
        racket = torch.zeros(count,13); racket[:,6] = 1
        racket[:,:3] = torch.tensor([0.,-12.,1.])
        ball = racket.clone(); ball[:,8] = -8.
        helper.prepare(torch.full((count,),.1),torch.ones(count,dtype=torch.bool),racket[:,:3],
                       torch.tensor([[0.,10.,0.]]).repeat(count,1),torch.ones(count,dtype=torch.bool),
                       torch.zeros(count,dtype=torch.bool))
        return helper, racket, ball

    def test_useful_upright_face_beats_horizontal_and_normal_sign_does_not_matter(self):
        helper,racket,ball = self.fixture(3)
        # +Y, -Y, +Z face normals respectively.
        racket[1,5:7] = torch.tensor([1.,0.])
        racket[2,3:7] = torch.tensor([math.sqrt(.5),0.,0.,math.sqrt(.5)])
        score = helper.score(racket,ball,ball[:,:3])
        torch.testing.assert_close(score[:2],torch.ones(2))
        self.assertLess(score[2].item(),.3)
        # Racket velocity does not redefine the desired return impulse (unlike incidence).
        racket[:,7:13] = 100
        torch.testing.assert_close(helper.score(racket,ball,ball[:,:3]),score)

    def test_leniency_plateau_and_falloff(self):
        helper,racket,ball = self.fixture(5)
        # Choose incoming velocity so desired impulse is exactly +Y.
        outgoing = (helper.target-ball[:,:3])/helper.settings.flight_time_s
        outgoing[:,2] += .0335/helper.settings.flight_time_s+.5*9.81*helper.settings.flight_time_s
        ball[:,7:10] = outgoing-torch.tensor([0.,20.,0.])
        theta = torch.tensor([0.,15.,30.,60.,90.])*math.pi/180
        racket[:,3] = torch.sin(theta/2); racket[:,6] = torch.cos(theta/2)
        torch.testing.assert_close(helper.score(racket,ball,ball[:,:3]),
                                   torch.tensor([1.,1.,1.,.5,0.]),atol=2e-6,rtol=2e-6)

    def test_preparation_rewards_near_miss_but_not_waiting_far_away_or_recovery(self):
        helper,racket,ball = self.fixture(5)
        helper.eta[:] = torch.tensor([.1,1.,-.2,.1,.1])
        racket[3,0] = 2.
        helper.valid[4] = False
        reward = helper.step(racket,ball)
        self.assertGreater(reward[0],0)
        torch.testing.assert_close(reward[1:],torch.zeros(4))
        self.assertFalse(helper.contact_seen.any())

    def test_per_shot_budget_reset_and_contact_consumption(self):
        helper,racket,ball = self.fixture(2)
        reward = torch.zeros(2)
        # Even repeatedly revisiting the best timing cannot farm >.5 preparation.
        helper.eta[:] = helper.dt
        for _ in range(100): reward += helper.step(racket,ball)
        torch.testing.assert_close(reward,torch.full((2,),.5))
        helper.begin_step()
        helper.record_contact(torch.tensor([True,False]),ball,racket)
        reward += helper.step(racket,ball)
        torch.testing.assert_close(reward,torch.tensor([1.,.5]))
        self.assertEqual(helper.step(racket,ball)[0],0)
        helper.begin_step()
        helper.record_contact(torch.tensor([True,False]),ball,racket)
        self.assertEqual(helper.step(racket,ball)[0],0)
        helper.reset(torch.tensor([0]))
        torch.testing.assert_close(helper.preparation_paid,torch.tensor([0.,.5]))
        helper.record_contact(torch.tensor([True,False]),ball,racket)
        self.assertEqual(helper.step(racket,ball)[0],.5)

    def test_impact_uses_incoming_pose_without_clean_or_prediction_gate(self):
        helper,racket,ball = self.fixture()
        helper.valid[:] = False
        helper.record_contact(torch.tensor([True]),ball,racket)
        # Rebounded velocity and a bad post-impact pose cannot overwrite impact credit.
        ball[:,8] = 100
        racket[:,3:7] = torch.tensor([math.sqrt(.5),0.,0.,math.sqrt(.5)])
        self.assertEqual(helper.step(racket,ball)[0],.5)

    def test_preparation_credit_is_timestep_normalized(self):
        totals=[]
        for dt in [1/30,1/60]:
            helper,racket,ball = self.fixture(dt=dt)
            total=0.
            # Entire triangular window: time gate has unit-normalized area.
            for frame in range(round(.30/dt)+1):
                helper.eta[:] = .20+dt-frame*dt
                total += helper.step(racket,ball).item()
            totals.append(total)
        self.assertAlmostEqual(totals[0],.5,places=5)
        self.assertAlmostEqual(totals[1],.5,places=5)

    def test_invalid_settings_and_nonfinite_states(self):
        for settings in [False,{'typo':1},{'preparation_s':0},{'flight_time_s':-1},
                         {'angle_tolerance_deg':-1},{'angle_tolerance_deg':90},
                         {'angle_tolerance_deg':45},{'angle_falloff_deg':0},
                         {'max_distance_m':float('nan')},{'flight_time_s':True}]:
            with self.subTest(settings=settings),self.assertRaises(ValueError):
                resolve_orientation(settings)
        helper,racket,ball = self.fixture(2)
        racket[0,0] = float('nan');ball[1,8] = float('inf')
        torch.testing.assert_close(helper.step(racket,ball),torch.zeros(2))

    def test_curriculum_adds_preparation_on_misses_and_logs_per_shot(self):
        obj=ReturnCurriculum(2,'cpu',{'weights_schedule':{'orientation':{'segments':[
            {'start':0.,'end':1.,'start_frame':0,'end_frame':100},
            {'start':1.,'end':0.,'start_frame':200,'end_frame':300}]}}},
            {'orientation':1.},{},ball_radius=.0335)
        ball=torch.tensor([[0.,-12.,1.]]).repeat(2,1)
        zeros=torch.zeros(2,dtype=torch.bool)
        args=(ball,torch.zeros_like(ball),zeros,zeros,ball,ball,zeros)
        with self.assertRaises(ValueError): obj.step(*args)
        result=obj.step(*args,orientation_reward=torch.tensor([.2,.3]))
        torch.testing.assert_close(result[:,-1],torch.tensor([.2,.3]))
        self.assertFalse(obj.hit.any())
        stats=obj.stats(torch.ones(2,dtype=torch.bool),zeros)
        stats['cycles']=torch.tensor(2)
        summarize=definitions(ROOT/'agents/v2p_agent.py',{'summarize_task_stats'}, {})['summarize_task_stats']
        self.assertAlmostEqual(summarize(stats)['return/orientation_per_shot'],.25)
        for frame,weight in [(0,0.),(50,.5),(100,1.),(250,.5),(300,0.)]:
            obj.set_progress(frame); self.assertAlmostEqual(obj.weights['orientation'],weight)
        obj.reset(torch.tensor([0]),ball[:1])
        torch.testing.assert_close(obj.orientation_total,torch.tensor([0.,.3]))

    def test_controller_integrates_actual_pose_and_named_reward(self):
        helper,racket,ball=self.fixture()
        namespace=definitions(ROOT/'env/tasks/physics_mvae_controller.py',
            {'PhysicsMVAEController','compute_reward_reach'},
            {'torch':torch,'math':math,'racket_state_snapshot':racket_state_snapshot})
        task=object.__new__(namespace['PhysicsMVAEController'])
        obj=ReturnCurriculum(1,'cpu',{}, {'orientation':2.},{},ball_radius=.0335)
        task.cfg_v2p={'reward_type':'return_curriculum'}
        task._return_curriculum=obj;task._racket_orientation=helper
        task._reward_weights,task._reward_scales=obj.weights,obj.scales
        task._phase_pred=torch.tensor([3.]);task._tar_action=torch.ones(1,dtype=torch.long)
        task._racket_pos=task._ball_pos=ball[:,:3];task._ball_vel=ball[:,7:10]
        task._est_bounce_valid=torch.tensor([False]);task._target_bounce_pos=helper.target
        task._mvae_player=NS(_swing_type=torch.tensor([-1]));task.rew_buf=torch.zeros(1)
        player=NS(_has_racket_ball_contact_now=torch.tensor([False]),_has_bounce_now=torch.tensor([False]),
            _bounce_pos=ball[:,:3],_ball_root_states=ball,_contact_env_ids=torch.tensor([0]),
            _contact_racket_ids=torch.tensor(0),_rigid_body_pos_origin=racket[:,:3].reshape(1,1,3),
            _rigid_body_rot_origin=racket[:,3:7].reshape(1,1,4),
            _rigid_body_vel_origin=racket[:,7:10].reshape(1,1,3),
            _rigid_body_ang_vel_origin=racket[:,10:13].reshape(1,1,3))
        task._physics_player=NS(task=player)
        task._compute_reward(None)
        self.assertGreater(task.rew_buf.item(),0)
        self.assertEqual(task._sub_rewards_names.split(',')[-1],'orientation_reward')
        self.assertAlmostEqual(task.rew_buf.item(),2*task._sub_rewards[0,-1].item())

    def test_live_physics_ignores_graze_then_captures_same_speed_reversal(self):
        helper,racket,ball=self.fixture()
        ball[:,1] += .15
        namespace=definitions(ROOT/'env/tasks/humanoid_smpl_im_mvae.py',{'HumanoidSMPLIMMVAE'},
            {'torch':torch,'HumanoidSMPL':object,'racket_state_snapshot':racket_state_snapshot,
             'racket_impact':racket_impact,
             'ground_impact':lambda *a:(torch.tensor([False]),torch.zeros(1,3)),
             'gymtorch':NS(unwrap_tensor=lambda x:x),'gymapi':NS(ENV_SPACE=0)})
        state=NS(control_freq_inv=2,_racket_orientation=helper,
            _tennis_contacts=NS(enabled=True,racket_contact_offset=.05,min_velocity_change_mps=2.),
            _contact_env_ids=torch.tensor([0]),_contact_racket_ids=torch.tensor(0),
            _ball_root_states=ball,_rigid_body_pos_origin=racket[:,:3].reshape(1,1,3),
            _rigid_body_rot_origin=racket[:,3:7].reshape(1,1,4),
            _rigid_body_vel_origin=racket[:,7:10].reshape(1,1,3),
            _rigid_body_ang_vel_origin=racket[:,10:13].reshape(1,1,3),
            forces=torch.zeros(1,2,3),torques=torch.zeros(1,2,3),
            _contact_forces=torch.ones(1,2,3)*1000,_contact_forces_sum=torch.zeros(1,2,3),
            _ball_physics=NS(radius=.0335),sim_params=NS(dt=1/60),sim=None,_is_train=True,
            _has_bounce=torch.tensor([False]),_has_bounce_now=torch.tensor([False]),
            _bounce_pos=torch.zeros(1,3),_has_ground_impact_now=torch.tensor([False]),
            _ground_impact_pos=torch.zeros(1,3),_has_racket_ball_contact=torch.tensor([False]),
            _has_racket_ball_contact_now=torch.tensor([False]),
            apply_external_force_to_ball=lambda *a:None,render=lambda:None)
        before=[]
        def simulate(*args):
            before.append(bool(state._has_racket_ball_contact[0]))
            state._ball_root_states[:,9] -= 9.81/60
            if len(before)==2:
                state._ball_root_states[:,8] *= -1
                # Actual racket rotates after impact; pre-impact orientation must win.
                state._rigid_body_rot_origin[:] = torch.tensor([math.sqrt(.5),0.,0.,math.sqrt(.5)])
        noop=lambda *a:None
        state.gym=NS(simulate=simulate,fetch_results=noop,apply_rigid_body_force_tensors=noop,
            refresh_actor_root_state_tensor=noop,refresh_rigid_body_state_tensor=noop,
            refresh_net_contact_force_tensor=noop)
        namespace['HumanoidSMPLIMMVAE']._physics_step(state)
        self.assertEqual(before,[False,False])
        self.assertTrue(state._has_racket_ball_contact_now[0])
        self.assertTrue(helper.contact_seen[0])
        self.assertAlmostEqual(helper.step(racket,ball)[0].item(),.5)


if __name__ == '__main__': unittest.main()
