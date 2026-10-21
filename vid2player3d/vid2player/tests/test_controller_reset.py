"""Reset kinematics and event isolation without IsaacGym bindings."""
import math
import unittest
from types import SimpleNamespace as NS

import torch
import torch.nn.functional as F

from vid2player.tests.test_controller_baseline import ROOT, definitions
from vid2player.utils.torch_transform import quat_apply, quat_mul


def rotate_xyzw(q, p):
    return quat_apply(q[:, [3, 0, 1, 2]], p)


def multiply_xyzw(a, b):
    return quat_mul(a[:, [3, 0, 1, 2]], b[:, [3, 0, 1, 2]])[:, [1, 2, 3, 0]]


def conjugate_xyzw(q):
    return torch.cat((-q[:, :3], q[:, 3:]), dim=1)


def reset_task():
    ns = definitions(ROOT/'env/tasks/humanoid_smpl_im_mvae.py', {'HumanoidSMPLIMMVAE'}, {
        'HumanoidSMPL': object, 'torch': torch, 'math': math, 'F': F,
        'quat_rotate': rotate_xyzw, 'quat_mul': multiply_xyzw,
        'quat_conjugate': conjugate_xyzw,
        'quat_rotate_inverse': lambda q, p: rotate_xyzw(conjugate_xyzw(q), p)})
    task = object.__new__(ns['HumanoidSMPLIMMVAE'])
    task.num_envs, task.device = 3, 'cpu'
    task._num_humanoid_bodies, task._racket_body_id = 24, 24
    # Distinct parent frames also exercise left/right racket attachments.
    task._racket_wrist_body_id = torch.tensor([22, 17, 22])
    task._rigid_body_pos = torch.zeros(3, 25, 3)
    task._rigid_body_rot = torch.zeros(3, 25, 4)
    task._rigid_body_rot[:, :, 3] = 1
    task._rigid_body_pos[:, 24, 0] = torch.tensor([-.5, .5, -.5])
    task._cache_racket_attachment()
    for name in ('_rigid_body_vel', '_rigid_body_ang_vel', '_prev_rigid_body_ang_vel'):
        setattr(task, name, torch.ones(3, 25, 3))
    task._humanoid_root_states = torch.zeros(3, 13)
    task._dof_pos, task._dof_vel = torch.zeros(3, 69), torch.zeros(3, 69)
    task._prev_target_rb_rot = torch.zeros(3, 24, 4)
    for name in ('_prev_target_root_pos', '_root_pos', '_root_vel', '_target_root_pos'):
        setattr(task, name, torch.zeros(3, 3))
    task._pd_target_dof_pos = torch.zeros(3, 69)
    return task


def pose():
    pos = torch.tensor([[0., -13., 1.]]).repeat(3, 24, 1)
    rot = torch.zeros(3, 24, 4)
    rot[:, :, 2:] = math.sqrt(.5)  # 90 degrees around Z
    return dict(root_pos=pos[:, 0], root_rot=rot[:, 0], dof_pos=torch.zeros(3, 69),
                root_vel=torch.zeros(3, 3), root_ang_vel=torch.zeros(3, 3),
                dof_vel=torch.zeros(3, 69), rb_pos=pos, rb_rot=rot)


class ControllerResetTests(unittest.TestCase):
    def test_reset_restores_racket_before_simulation(self):
        task = reset_task()
        task._rigid_body_pos[:, 24] = 100  # previous shot, or actor creation pose
        task._set_env_state(torch.tensor([1]), **{k: v[1:2] for k, v in pose().items()})
        torch.testing.assert_close(task._rigid_body_pos[1, 24], torch.tensor([0., -12.5, 1.]))
        torch.testing.assert_close(task._rigid_body_rot[1, 24], pose()['rb_rot'][1, 17])
        torch.testing.assert_close(task._rigid_body_pos[[0, 2], 24], torch.full((2, 3), 100.))

    def test_reset_pose_independent_of_prior_shot_and_batch_width(self):
        batched, sequential = reset_task(), reset_task()
        batched._rigid_body_pos[:] = torch.randn(3, 25, 3)
        sequential._rigid_body_pos[:] = torch.randn(3, 25, 3)*100
        batched._set_env_state(torch.arange(3), **pose())
        for i in range(3):
            sequential._set_env_state(torch.tensor([i]), **{k: v[i:i+1] for k, v in pose().items()})
        torch.testing.assert_close(batched._rigid_body_pos, sequential._rigid_body_pos)
        torch.testing.assert_close(batched._rigid_body_rot, sequential._rigid_body_rot)
        expected = torch.tensor([[0., -13.5, 1.], [0., -12.5, 1.], [0., -13.5, 1.]])
        torch.testing.assert_close(batched._rigid_body_pos[:, 24], expected)

    def test_ball_reset_clears_only_selected_contact_events(self):
        task = reset_task()
        task._ball_pos = torch.zeros(3, 3)
        task._ball_vel = torch.zeros(3, 3)
        task._ball_root_states = torch.zeros(3, 13)
        task._bounce_pos = torch.ones(3, 3)
        for name in ('_has_bounce', '_has_bounce_now', '_has_racket_ball_contact', '_has_racket_ball_contact_now'):
            setattr(task, name, torch.ones(3, dtype=torch.bool))
        task._ball_generator = NS(generate=lambda n, **kw: (
            torch.zeros(n, 100, 3), torch.zeros(n, 3), torch.tensor([[0., -25., 1.]]).repeat(n, 1), torch.ones(n)))
        task._reset_balls(torch.tensor([1]))
        for name in ('_has_bounce', '_has_bounce_now', '_has_racket_ball_contact', '_has_racket_ball_contact_now'):
            self.assertEqual(getattr(task, name).tolist(), [True, False, True])


if __name__ == '__main__':
    unittest.main()
