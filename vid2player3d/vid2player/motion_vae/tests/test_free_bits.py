"""Free-bits reductions, gradients, and training integration on CPU."""
import unittest

import torch

from vid2player.motion_vae.config import MotionVAEOption
from vid2player.motion_vae.loss import diagonal_gaussian_kl, kl_free_bits_settings


class FreeBitsTests(unittest.TestCase):
    def test_disabled_matches_legacy_value_and_gradients_exactly(self):
        mu = torch.randn(4, 3, requires_grad=True)
        logvar = torch.randn(4, 3, requires_grad=True)
        legacy = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum().clamp(max=0)
        legacy /= logvar.numel()
        actual = diagonal_gaussian_kl(mu, logvar)
        self.assertTrue(torch.equal(actual, legacy))
        for old, new in zip(torch.autograd.grad(legacy, (mu, logvar)),
                            torch.autograd.grad(actual, (mu, logvar))):
            self.assertTrue(torch.equal(old, new))

    def test_clamps_batch_mean_per_dimension_before_latent_mean(self):
        # Per-example KLs are [[0, 0], [0, 2]]; batch means [0, 1].
        mu = torch.tensor([[0., 0.], [0., 2.]], requires_grad=True)
        logvar = torch.zeros_like(mu, requires_grad=True)
        loss = diagonal_gaussian_kl(mu, logvar, .5)
        self.assertEqual(loss.item(), .75)
        grad, = torch.autograd.grad(loss, (mu,))
        torch.testing.assert_close(grad, torch.tensor([[0., 0.], [0., .5]]))

    def test_below_floor_has_no_kl_gradient(self):
        mu = torch.full((4, 3), .1, requires_grad=True)
        logvar = torch.full_like(mu, .1, requires_grad=True)
        loss = diagonal_gaussian_kl(mu, logvar, .1)
        self.assertAlmostEqual(loss.item(), .1)
        for grad in torch.autograd.grad(loss, (mu, logvar)):
            self.assertTrue(torch.equal(grad, torch.zeros_like(grad)))

    def test_above_floor_matches_standard_kl_and_gradients(self):
        mu = torch.full((4, 3), 2., requires_grad=True)
        logvar = torch.full_like(mu, .2, requires_grad=True)
        legacy = diagonal_gaussian_kl(mu, logvar)
        actual = diagonal_gaussian_kl(mu, logvar, .1)
        torch.testing.assert_close(actual, legacy)
        for old, new in zip(torch.autograd.grad(legacy, (mu, logvar)),
                            torch.autograd.grad(actual, (mu, logvar))):
            torch.testing.assert_close(old, new)

    def test_enabled_kl_uses_float32_for_half_precision_inputs(self):
        mu = torch.full((2, 3), 300., dtype=torch.float16)
        logvar = torch.zeros_like(mu)
        loss = diagonal_gaussian_kl(mu, logvar, .1)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertEqual(loss.item(), 45000.)

    def test_config_defaults_validation_and_experiment_inheritance(self):
        opt = MotionVAEOption()
        self.assertEqual(kl_free_bits_settings(opt), 0.)
        opt.load('usopen_6_3')
        self.assertEqual(kl_free_bits_settings(opt), .1)
        self.assertEqual(opt.weights['kl'], .5)
        baseline = MotionVAEOption()
        baseline.load('usopen_6_1')
        self.assertEqual(baseline.kl_free_bits, 0.)
        for invalid in (-1, float('nan'), float('inf'), True, None, '0.1'):
            opt.kl_free_bits = invalid
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                kl_free_bits_settings(opt)

    def test_training_weights_floored_kl_but_logs_raw_separately(self):
        from test_checkpoint import CheckpointTests
        fixture = CheckpointTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        # Make every posterior dimension stay below the floor during the batch.
        model = fixture.model(kl_free_bits=10.)
        losses = model.train_batch(fixture.batch)
        self.assertAlmostEqual(losses['kl'], 10. * model.opt.weights['kl'])
        self.assertGreaterEqual(losses['kl_raw'], 0.)
        self.assertLess(losses['kl_raw'], 10.)
        self.assertTrue(model.optimizer.state)


if __name__ == '__main__':
    unittest.main()
