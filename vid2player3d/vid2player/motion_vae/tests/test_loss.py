"""Checks for feature weighting, legacy loss equivalence, and loss gradients."""
import unittest

import torch

from vid2player.motion_vae.config import MotionVAEOption
from vid2player.motion_vae.loss import build_recon_feature_weights, feature_squared_error


class ReconstructionLossTests(unittest.TestCase):
    def setUp(self):
        self.opt = MotionVAEOption()
        self.opt.load('usopen')

    def weights(self, configured):
        self.opt.recon_feature_weights = configured
        return build_recon_feature_weights(self.opt, 'cpu')

    def test_default_and_unit_weights_preserve_exact_loss_and_gradients(self):
        output = torch.randn(2, 3, 288, requires_grad=True)
        target = torch.randn_like(output)
        future = torch.softmax(torch.linspace(1, 0, 3), dim=0)
        old_loss = (output - target.detach()).pow(2).mean(dim=(0, -1)).mul(future).sum()
        old_grad, = torch.autograd.grad(old_loss, output)
        for configured in ({}, {'root_velo': 1.0, 'root_pos': 1.0}):
            weights = self.weights(configured)
            self.assertIsNone(weights)
            loss = feature_squared_error(output, target, weights).mean(dim=(0, -1)).mul(future).sum()
            grad, = torch.autograd.grad(loss, output)
            self.assertTrue(torch.equal(loss, old_loss))
            self.assertTrue(torch.equal(grad, old_grad))

    def test_root_displacement_gradient_is_scaled_without_changing_local_gradient(self):
        weights = self.weights({'root_velo': 5.0})
        output = torch.ones(2, 3, 288, requires_grad=True)
        target = torch.zeros_like(output, requires_grad=True)
        future = torch.tensor([.6, .3, .1])
        old = feature_squared_error(output, target).mean(dim=(0, -1)).mul(future).sum()
        weighted = feature_squared_error(output, target, weights).mean(dim=(0, -1)).mul(future).sum()
        self.assertAlmostEqual(weighted.item(), (285 + 3 * 5) / 288, places=6)
        old_grad, = torch.autograd.grad(old, output)
        grad, target_grad = torch.autograd.grad(weighted, (output, target), allow_unused=True)
        self.assertIsNone(target_grad)
        torch.testing.assert_close(grad[..., 3:6], old_grad[..., 3:6] * 5)
        self.assertTrue(torch.equal(grad[..., :3], old_grad[..., :3]))
        self.assertTrue(torch.equal(grad[..., 6:], old_grad[..., 6:]))

    def test_layout_uses_dataset_order_and_reduced_root_dimensions(self):
        self.opt.pose_feature = ['joint_rotmat', 'root_velo', 'root_pos']
        self.opt.condition_root_x_only = True
        self.opt.frame_size = 148
        weights = self.weights({'root_pos': 2, 'root_velo': 7, 'joint_rotmat': 3})
        self.assertEqual(weights.tolist(), [2] + [7] * 3 + [3] * 144)
        self.opt.condition_root_x_only = False
        self.opt.no_condition_root_y = True
        self.opt.frame_size = 149
        weights = self.weights({'root_pos': 2, 'root_velo': 7, 'joint_rotmat': 3})
        self.assertEqual(weights.tolist(), [2] * 2 + [7] * 3 + [3] * 144)

    def test_invalid_weights_fail(self):
        for configured in ({'root': 5}, {'joint_quat': 2}, {'root_velo': -1},
                           {'root_velo': float('nan')}, {'root_velo': float('inf')},
                           {'root_velo': True}, {'root_velo': '5'}, [5],
                           {name: 0 for name in self.opt.pose_feature}):
            with self.subTest(configured=configured), self.assertRaises(ValueError):
                self.weights(configured)
        self.opt.frame_size = 12
        with self.assertRaises(ValueError):
            self.weights({'root_velo': 5})

    def test_usopen_5_changes_only_feature_weights_relative_to_usopen_4(self):
        baseline = MotionVAEOption()
        baseline.load('usopen_4')
        experiment = MotionVAEOption()
        experiment.load('usopen_5')
        self.assertEqual(baseline.recon_feature_weights, {})
        self.assertEqual(experiment.recon_feature_weights, {'root_velo': 5.0})
        ignored = {'model_ver', 'base_opt_ver', 'recon_feature_weights'}
        for name in vars(baseline).keys() - ignored:
            if not callable(getattr(baseline, name)):
                self.assertEqual(getattr(experiment, name), getattr(baseline, name), name)


if __name__ == '__main__':
    unittest.main()
