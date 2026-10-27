"""Adapt observation moments between PPO iterations, with a policy drift limit."""
import math

import torch


class RolloutNormalization:
    def __init__(self, normalizer, rate=0.01, max_policy_kl=0.001,
                 max_probe_kl=0.05, probe_envs_per_step=64, max_backtracks=12):
        if not 0 < rate <= 1:
            raise ValueError('rollout normalization rate must be in (0, 1]')
        if not all(math.isfinite(x) and x > 0 for x in (max_policy_kl, max_probe_kl)):
            raise ValueError('rollout normalization KL limits must be finite and positive')
        if probe_envs_per_step < 1 or max_backtracks < 0:
            raise ValueError('invalid rollout normalization probe/backtrack settings')
        self.normalizer = normalizer
        self.rate = rate
        self.max_policy_kl = max_policy_kl
        self.max_probe_kl = max_probe_kl
        self.probe_envs_per_step = int(probe_envs_per_step)
        self.max_backtracks = int(max_backtracks)
        normalizer.update_on_forward = False
        self.begin()

    def begin(self):
        self.count = 0
        self.mean = self.m2 = None
        self.probes = []

    @torch.no_grad()
    def observe(self, observations):
        """Called only for action inputs, excluding already finished environments."""
        x = observations.detach()
        size = x.shape[0]
        if not size:
            return
        var, mean = torch.var_mean(x, dim=0, unbiased=False)
        if self.count == 0:
            self.mean, self.m2 = mean, var * size
        else:
            delta = mean - self.mean
            total = self.count + size
            self.m2 += var * size + delta.square() * (self.count * size / total)
            self.mean += delta * (size / total)
        self.count += size
        # Evenly sample environments at every timestep, retaining only a small
        # probe instead of a second full [env, time, observation] buffer.
        indices = torch.linspace(0, size - 1, min(size, self.probe_envs_per_step),
                                 device=x.device).long()
        self.probes.append(x[indices].clone())

    @torch.no_grad()
    def commit(self, policy):
        """policy(x) returns Gaussian means and log standard deviations.

        Compare the same post-PPO weights under old and candidate normalization.
        Backtrack the moment blend until both average and worst probe KL pass.
        No candidates change weights or consume random numbers.
        """
        metrics = {'normalization/update_fraction': 0.,
                   'normalization/policy_kl': 0.,
                   'normalization/max_probe_kl': 0.}
        if not self.count:
            return metrics
        norm = self.normalizer
        if norm.n.item() <= 0:
            raise ValueError('rollout_ema requires initialized observation statistics from a checkpoint')
        old_mean, old_var, old_std = [v.clone() for v in (norm.mean, norm.var, norm.std)]
        probes = torch.cat(self.probes)
        old_mu, old_logstd = policy(probes)
        batch_var = self.m2 / self.count
        rate = self.rate
        accepted = False
        try:
            for _ in range(self.max_backtracks + 1):
                norm.mean.copy_(torch.lerp(old_mean, self.mean, rate))
                norm.var.copy_((1 - rate) * old_var + rate * batch_var +
                               rate * (1 - rate) * (self.mean - old_mean).square())
                norm.std.copy_(norm.var.clamp_min(0).sqrt())
                mu, logstd = policy(probes)
                # Exact diagonal Gaussian KL, without rl_games' epsilon bias.
                kl = (logstd - old_logstd + 0.5 * (
                    torch.exp(2 * (old_logstd - logstd)) +
                    (old_mu - mu).square() * torch.exp(-2 * logstd) - 1)).sum(-1).clamp_min(0)
                mean_kl, max_kl = kl.mean().item(), kl.max().item()
                if (math.isfinite(mean_kl) and math.isfinite(max_kl) and
                        mean_kl <= self.max_policy_kl and max_kl <= self.max_probe_kl):
                    accepted = True
                    # In this EMA mode n records seen observations; it does not
                    # weight future blends. Preserve the checkpoint's prior n.
                    norm.n.add_(self.count)
                    metrics = {'normalization/update_fraction': rate,
                               'normalization/policy_kl': mean_kl,
                               'normalization/max_probe_kl': max_kl}
                    break
                rate *= 0.5
        finally:
            if not accepted:
                norm.mean.copy_(old_mean)
                norm.var.copy_(old_var)
                norm.std.copy_(old_std)
            self.begin()
        return metrics
