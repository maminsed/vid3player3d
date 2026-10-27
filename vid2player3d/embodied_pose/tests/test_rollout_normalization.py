"""CPU checks for PPO normalization consistency and the legacy default."""
from pathlib import Path
import sys
import unittest

import torch
import yaml

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from embodied_pose.models.running_norm import RunningNorm
from embodied_pose.utils.rollout_normalization import RolloutNormalization


class RolloutNormalizationTests(unittest.TestCase):
    def norm(self):
        norm = RunningNorm(2)
        norm.load_state_dict({'n': torch.tensor(34603008000), 'mean': torch.zeros(2),
                              'var': torch.ones(2), 'std': torch.ones(2)})
        return norm

    def policy(self, norm, gain=1.):
        def evaluate(x):
            mu = norm(x) * gain
            return mu, torch.full_like(mu, -1.756)
        return evaluate

    def test_legacy_default_matches_original_formula(self):
        norm = self.norm().train()
        x = torch.tensor([[1., 2.], [3., 8.]])
        old = {k: v.clone() for k, v in norm.state_dict().items()}
        batch_var, batch_mean = torch.var_mean(x, dim=0, unbiased=False)
        weight = old['n'].float() / (len(x) + old['n']).float()
        var = weight * old['var'] + (1 - weight) * batch_var + weight * (1 - weight) * (batch_mean - old['mean']).square()
        mean = weight * old['mean'] + (1 - weight) * batch_mean
        expected = ((x - mean) / (var.sqrt() + 1e-8)).clamp(-5, 5)
        self.assertTrue(torch.equal(norm(x), expected))
        self.assertEqual(int(norm.n), int(old['n']) + len(x))
        self.assertEqual(set(norm.state_dict()), set(old))

    def test_frozen_during_rollout_and_repeated_ppo_forwards(self):
        norm = self.norm()
        adapter = RolloutNormalization(norm)
        state = {k: v.clone() for k, v in norm.state_dict().items()}
        x = torch.tensor([[1., 2.], [3., 4.]])
        adapter.observe(x)
        expected = norm.eval()(x)
        for _ in range(6):
            self.assertTrue(torch.equal(norm.train()(x), expected))
        for key, value in state.items():
            self.assertTrue(torch.equal(norm.state_dict()[key], value))
        metrics = adapter.commit(self.policy(norm))
        self.assertGreater(metrics['normalization/update_fraction'], 0)
        self.assertEqual(int(norm.n), int(state['n']) + len(x))
        self.assertLessEqual(metrics['normalization/policy_kl'], .001)
        self.assertLessEqual(metrics['normalization/max_probe_kl'], .05)

    def test_streaming_unique_observations_and_blended_moments(self):
        norm = self.norm()
        adapter = RolloutNormalization(norm, rate=.01, max_policy_kl=100, max_probe_kl=100)
        x = torch.tensor([[1., 2.], [3., 4.], [8., 2.], [-3., 9.]])
        adapter.observe(x[:1]); adapter.observe(x[1:]); adapter.observe(x[:0])
        self.assertEqual(adapter.count, len(x))
        torch.testing.assert_close(adapter.mean, x.mean(0))
        torch.testing.assert_close(adapter.m2 / adapter.count, x.var(0, unbiased=False))
        result = adapter.commit(self.policy(norm))
        self.assertEqual(result['normalization/update_fraction'], .01)
        torch.testing.assert_close(norm.mean, .01 * x.mean(0))
        torch.testing.assert_close(norm.var, .99 * torch.ones(2) + .01 * x.var(0, unbiased=False) + .0099 * x.mean(0).square())
        self.assertEqual(adapter.count, 0)

    def test_backtrack_and_rollback_on_excessive_policy_drift(self):
        norm = self.norm()
        x = torch.tensor([[1., 2.], [3., 4.]])
        adapter = RolloutNormalization(norm, rate=1., max_backtracks=20)
        adapter.observe(x)
        metrics = adapter.commit(self.policy(norm))
        self.assertGreater(metrics['normalization/update_fraction'], 0)
        self.assertLess(metrics['normalization/update_fraction'], 1.)
        state = {k: v.clone() for k, v in norm.state_dict().items()}
        adapter = RolloutNormalization(norm, rate=1., max_backtracks=0)
        adapter.observe(x * 100)
        self.assertEqual(adapter.commit(self.policy(norm))['normalization/update_fraction'], 0)
        for key, value in state.items():
            self.assertTrue(torch.equal(norm.state_dict()[key], value))

    def test_zero_variance_checkpoint_and_resume(self):
        norm = self.norm()
        norm.var.zero_(); norm.std.zero_()
        x = torch.tensor([[1., -2.], [1., -2.]])
        adapter = RolloutNormalization(norm)
        adapter.observe(x)
        metrics = adapter.commit(self.policy(norm))
        self.assertTrue(torch.isfinite(norm(x)).all())
        self.assertLessEqual(metrics['normalization/policy_kl'], .001)
        restored = self.norm()
        restored.load_state_dict(norm.state_dict(), strict=True)
        resumed = RolloutNormalization(restored)
        adapter.observe(x); resumed.observe(x)
        self.assertEqual(adapter.commit(self.policy(norm)), resumed.commit(self.policy(restored)))
        for key in norm.state_dict():
            self.assertTrue(torch.equal(norm.state_dict()[key], restored.state_dict()[key]))

    def test_v2_experiment_is_separate_and_preserves_physics(self):
        cfg = PROJECT/'embodied_pose/cfg'
        old = yaml.safe_load((cfg/'tennis_im_training_2_ft.yaml').read_text())
        new = yaml.safe_load((cfg/'tennis_im_training_2_ft_v2.yaml').read_text())
        self.assertEqual(old['sim'], new['sim'])
        for key in old['env']:
            self.assertEqual(old['env'][key], new['env'][key])
        a, b = old['params']['config'], new['params']['config']
        self.assertEqual(a['pretrained_running_obs_count'], 10000)
        self.assertNotIn('observation_normalization', a)
        self.assertNotIn('pretrained_running_obs_count', b)
        self.assertEqual(a['pretrained_model_cp'], b['pretrained_model_cp'])
        self.assertNotEqual(a['name'], b['name'])
        self.assertNotIn('checkpoint_epochs', b)
        self.assertEqual(b['max_epochs'], 400)
        self.assertGreater(b['save_frequency'], 0)


if __name__ == '__main__':
    unittest.main()
