"""Contact evidence and surface-motion diagnostics without IsaacGym."""
import math
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch

from vid2player.utils.racket_diagnostics import racket_geometry, racket_ball_snapshot
from vid2player.utils.tennis_contacts import racket_impact


class TestRacketDiagnostics(unittest.TestCase):
    def test_physics_recorder_preserves_detection_and_copies_intervals(self):
        # Execute the real stepping method against a tiny simulated state buffer.
        # This checks integration without importing IsaacGym or running a rollout.
        path = Path(__file__).resolve().parents[1] / 'env/tasks/humanoid_smpl_im_mvae.py'
        cls = next(n for n in ast.parse(path.read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == 'HumanoidSMPLIMMVAE')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_physics_step')
        scope = {'racket_ball_snapshot': racket_ball_snapshot, 'racket_impact': racket_impact,
                 'ground_impact': lambda *args: (torch.tensor([False]), torch.zeros(1, 3)),
                 'gymtorch': NS(unwrap_tensor=lambda x: x), 'gymapi': NS(ENV_SPACE=0)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), scope)
        results = []
        for record in (False, True):
            state = NS(control_freq_inv=2, _record_racket_diagnostics=record,
                       _racket_diagnostic_samples=[], _tennis_contacts=NS(enabled=True, racket_contact_offset=.05),
                       _contact_env_ids=torch.tensor([0]), _contact_racket_ids=torch.tensor(0),
                       _ball_root_states=torch.zeros(1, 13), _rigid_body_pos_origin=torch.zeros(1, 1, 3),
                       _rigid_body_rot_origin=torch.tensor([[[0., 0., 0., 1.]]]),
                       _rigid_body_vel_origin=torch.zeros(1, 1, 3),
                       _rigid_body_ang_vel_origin=torch.zeros(1, 1, 3),
                       forces=torch.zeros(1, 2, 3), torques=torch.zeros(1, 2, 3),
                       _contact_forces=torch.zeros(1, 2, 3), _contact_forces_sum=torch.zeros(1, 2, 3),
                       _ball_physics=NS(radius=.0335), sim_params=NS(dt=1/60), sim=None,
                       _is_train=True, _has_bounce=torch.tensor([False]),
                       _has_bounce_now=torch.tensor([False]), _bounce_pos=torch.zeros(1, 3),
                       _has_ground_impact_now=torch.tensor([False]), _ground_impact_pos=torch.zeros(1, 3),
                       _has_racket_ball_contact=torch.tensor([False]),
                       _has_racket_ball_contact_now=torch.tensor([False]),
                       apply_external_force_to_ball=lambda *a: None, render=lambda: None)
            state._ball_root_states[:, 2] = 1.
            state._rigid_body_pos_origin[:, :, 2] = 1.
            def simulate(*args):
                state._ball_root_states[:, 8] += 3.
                state._ball_root_states[:, 9] -= 9.81 / 60
            noop = lambda *a: None
            state.gym = NS(simulate=simulate, fetch_results=noop,
                           apply_rigid_body_force_tensors=noop, refresh_actor_root_state_tensor=noop,
                           refresh_rigid_body_state_tensor=noop, refresh_net_contact_force_tensor=noop)
            scope['_physics_step'](state)
            results.append((state._has_racket_ball_contact.clone(), state._ball_root_states.clone()))
            if record:
                first, second = state._racket_diagnostic_samples
                self.assertTrue(first['accepted_contact'][0])
                self.assertFalse(second['accepted_contact'][0])
                self.assertTrue(second['contact_already_latched'][0])
                self.assertEqual(first['pre_ball_state'][0, 8], 0.)
                self.assertEqual(first['post_ball_state'][0, 8], 3.)
                self.assertEqual(second['post_ball_state'][0, 8], 6.)
        for baseline, recorded in zip(*results):
            torch.testing.assert_close(baseline, recorded)
        # An incoming bounce already latched must not hide an outgoing rebound.
        state._has_bounce[:] = True
        state._has_ground_impact_now[:] = False
        position = torch.tensor([[2., 3., .0335]])
        scope['ground_impact'] = lambda *args: (torch.tensor([True]), position)
        scope['_physics_step'](state)
        self.assertTrue(state._has_ground_impact_now[0])
        torch.testing.assert_close(state._ground_impact_pos, position)

    def test_rotation_and_velocity_use_com_offset(self):
        racket = torch.zeros(1, 13)
        # Rotate local +X to world +Y; COM is offset along that direction.
        racket[:, 5:7] = math.sqrt(.5)
        racket[:, 7] = 2.
        racket[:, 12] = 3.
        ball = torch.zeros(1, 13)
        ball[:, 1] = 2.
        result = racket_geometry({'ball_state': ball, 'racket_state': racket},
                                 [1., 0., 0.], [1., 0., 0.])
        torch.testing.assert_close(result['racket_face_normal'], torch.tensor([[0., 1., 0.]]))
        torch.testing.assert_close(result['racket_com_xyz'], torch.tensor([[0., 1., 0.]]))
        # v_surface = (2,0,0) + (0,0,3) cross (0,1,0).
        torch.testing.assert_close(result['racket_velocity_at_ball_center_mps'],
                                   torch.tensor([[-1., 0., 0.]]))

    def test_evidence_separates_sensors_impulse_and_proximity(self):
        pre = torch.zeros(4, 13)
        post = pre.clone()
        post[1:3, 8] = 3.
        racket = torch.zeros(4, 3)
        racket[2, 0] = 1.  # Impulse exists, but not near the racket.
        force = torch.zeros(4, 3)
        force[0, 0] = 1.
        args = (pre, post, racket, racket, force, force, torch.zeros(4, 3), .0335, .05, 1/60)
        original = racket_impact(*args)
        hit, evidence = racket_impact(*args, return_details=True)
        torch.testing.assert_close(hit, original)
        self.assertEqual(hit.tolist(), [True, True, False, False])
        self.assertEqual(evidence['sensor_evidence'].tolist(), [True, False, False, False])
        self.assertEqual(evidence['impulse_evidence'].tolist(), [False, True, True, False])
        self.assertEqual(evidence['proximity_pass'].tolist(), [True, True, False, True])
        self.assertAlmostEqual(evidence['swept_distance_m'][2].item(), 1.)


if __name__ == '__main__':
    unittest.main()
