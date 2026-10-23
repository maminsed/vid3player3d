"""Scheduled return shaping and once-per-launch physical events, without a simulator.

Landing rewards use the first measured post-contact bounce, including out balls.
The outgoing lookup estimator is diagnostic only; its validity never gates these
landing rewards. Net clearance is a linearly interpolated control-step estimate.
"""
import math
import torch

from vid2player.utils.reward_schedule import LinearRewardSchedule, PiecewiseRewardSchedule
from vid2player.utils.tennis_court import NET_HEIGHT_M


COMPONENTS = ('pos', 'contact', 'forward', 'net', 'court', 'target')
OPTIONAL_COMPONENTS = ('vertical',)


class ReturnCurriculum:
    def __init__(self, count, device, config, weights, scales, dt=1/30, *, ball_radius):
        if {'net_height_m', 'ball_radius_m'} & set(config):
            raise ValueError('Return geometry comes from tennis_court and the active BallPhysics; '
                             'remove net_height_m/ball_radius_m from return_curriculum')
        self.weights, self.scales = weights, scales
        # Append opt-in terms so older configs retain their exact component arrays.
        self.components = COMPONENTS + tuple(k for k in OPTIONAL_COMPONENTS
            if k in weights or k in config.get('weights_schedule', {}))
        unknown = set(weights) - set(self.components)
        if unknown:
            raise ValueError('Unknown return reward weights: ' + str(sorted(unknown)))
        for name in self.components:
            weights.setdefault(name, 0.)
        for name, default in [('pos', 5.), ('phase', 5.), ('court', .02), ('target', .005)]:
            scales.setdefault(name, default)
        if not all(math.isfinite(v) and v >= 0 for v in weights.values()):
            raise ValueError('Return reward weights must be finite and nonnegative')
        if not all(math.isfinite(scales[k]) and scales[k] > 0 for k in ('pos', 'court', 'target')):
            raise ValueError('Position and landing scales must be finite and positive')
        if not math.isfinite(scales['phase']) or scales['phase'] < 0:
            raise ValueError('Phase scale must be finite and nonnegative')
        self.schedules = {}
        for group, destination in [('weights', weights), ('scales', scales)]:
            self.schedules[group] = {}
            for name, spec in config.get(group + '_schedule', {}).items():
                if name not in destination:
                    raise ValueError('Unknown return schedule ' + group + '.' + name)
                schedule_type = PiecewiseRewardSchedule if 'segments' in spec else LinearRewardSchedule
                schedule = schedule_type(spec, allow_zero=(group == 'weights' or name == 'phase'))
                if not isinstance(spec.get('enabled', True), bool):
                    raise ValueError('Schedule enabled must be boolean')
                if spec.get('enabled', True):
                    self.schedules[group][name] = schedule
        self.logged_components = [k for k in self.components if weights[k] or k in self.schedules['weights']]
        # Opt-in signed feedback preserves the exact positive-only 2_1 reward.
        self.forward_reward_mode = config.get('forward_reward_mode', 'positive_linear')
        if self.forward_reward_mode not in ('positive_linear', 'signed_linear'):
            raise ValueError('forward_reward_mode must be positive_linear or signed_linear')
        self.backward_penalty_scale = float(config.get('backward_penalty_scale', 1.))
        if not math.isfinite(self.backward_penalty_scale) or self.backward_penalty_scale < 0:
            raise ValueError('backward_penalty_scale must be finite and nonnegative')
        vertical = config.get('vertical_velocity', {})
        self.vertical_min = float(vertical.get('min_mps', 0.))
        self.vertical_max = float(vertical.get('max_mps', 8.))
        self.vertical_falloff = float(vertical.get('falloff_mps', 8.))
        if (not all(math.isfinite(v) for v in (self.vertical_min, self.vertical_max, self.vertical_falloff))
                or self.vertical_min >= self.vertical_max or self.vertical_falloff <= 0):
            raise ValueError('Vertical velocity requires finite min_mps < max_mps and positive falloff_mps')
        self.forward_speed = float(config.get('forward_speed_mps', 10.))
        self.net_height = NET_HEIGHT_M
        self.ball_radius = float(ball_radius)
        self.max_post_contact_s = float(config.get('max_post_contact_s', 3.))
        self.dt = dt
        if not all(math.isfinite(v) and v > 0 for v in
                   (self.forward_speed, self.net_height, self.ball_radius, self.max_post_contact_s, dt)):
            raise ValueError('Return speed, net height and ball radius must be finite and positive')
        self.previous_ball = torch.zeros(count, 3, device=device)
        for name in ('hit', 'forward', 'crossed_net', 'landing_seen', 'landed', 'in_court', 'estimate_valid',
                     'vertical_outside_band'):
            setattr(self, name, torch.zeros(count, dtype=torch.bool, device=device))
        self.contact_vy = torch.zeros(count, device=device)
        self.contact_age = torch.zeros(count, device=device, dtype=torch.long)
        self.target_error = torch.zeros(count, device=device)

    def set_progress(self, frame):
        for group, destination in [('weights', self.weights), ('scales', self.scales)]:
            for name, schedule in self.schedules[group].items():
                destination[name] = schedule.value(frame)

    def schedule_metrics(self):
        return {'curriculum/' + group[:-1] + '_' + name: destination[name]
                for group, destination in [('weights', self.weights), ('scales', self.scales)]
                for name in self.schedules[group]}

    def reset(self, ids, ball):
        self.previous_ball[ids] = ball
        for name in ('hit', 'forward', 'crossed_net', 'landing_seen', 'landed', 'in_court', 'estimate_valid',
                     'vertical_outside_band'):
            getattr(self, name)[ids] = False
        self.contact_vy[ids] = self.target_error[ids] = 0
        self.contact_age[ids] = 0

    def waiting_for_landing(self):
        return self.hit & ~self.landing_seen & (self.contact_age * self.dt < self.max_post_contact_s)

    def step(self, ball, velocity, contact_now, bounce_now, bounce_pos, target, estimate_valid):
        """Return event components; repeated contacts/bounces cannot earn them again."""
        had_contact = self.hit.clone()
        self.contact_age[had_contact] += 1
        finite = torch.isfinite(ball).all(-1) & torch.isfinite(velocity).all(-1)
        first_contact = contact_now & ~had_contact & finite
        vy = torch.nan_to_num(velocity[:, 1], nan=0., posinf=0., neginf=0.)
        self.contact_vy = torch.where(first_contact, vy, self.contact_vy)
        self.forward |= first_contact & (vy > 0)
        self.estimate_valid |= first_contact & estimate_valid
        self.hit |= first_contact
        vz = torch.nan_to_num(velocity[:, 2], nan=0., posinf=0., neginf=0.)
        vertical_excess = (self.vertical_min - vz).clamp_min(0) + (vz - self.vertical_max).clamp_min(0)
        self.vertical_outside_band |= first_contact & (vertical_excess > 0)

        # Interpolate at y=0. Use shared net height plus the active ball radius, conservative across
        # the varying tennis net height; this is not a physical net collider.
        delta_y = ball[:, 1] - self.previous_ball[:, 1]
        fraction = (-self.previous_ball[:, 1] / delta_y.clamp_min(1e-6)).clamp(0, 1)
        crossing = self.previous_ball + fraction[:, None] * (ball - self.previous_ball)
        # If crossing and bounce share a recorded interval their order is unknown.
        # Reject clearance conservatively rather than credit a bounced-over ball.
        net = (had_contact & ~self.landing_seen & ~bounce_now & ~self.crossed_net & finite
               & (self.previous_ball[:, 1] < 0) & (ball[:, 1] >= 0)
               & (crossing[:, 0].abs() < 4.11)
               & (crossing[:, 2] > self.net_height + self.ball_radius))
        self.crossed_net |= net

        # A contact/bounce in the same control step is ambiguous: close that
        # flight without scoring it, rather than later scoring its second bounce.
        landing = (had_contact & bounce_now & ~self.landing_seen
                   & torch.isfinite(bounce_pos).all(-1))
        self.landing_seen |= self.hit & bounce_now
        self.landed |= landing
        xy = torch.nan_to_num(bounce_pos[:, :2], nan=1e6, posinf=1e6, neginf=-1e6)
        low = xy.new_tensor([-4.11, 0.])
        high = xy.new_tensor([4.11, 11.89])
        inside = ((xy > low) & (xy < high)).all(-1)
        self.in_court |= landing & self.crossed_net & inside
        court_error_sq = ((low - xy).clamp_min(0) + (xy - high).clamp_min(0)).square().sum(-1)
        target_error_sq = (xy - target[:, :2]).square().sum(-1)
        self.target_error = torch.where(landing, target_error_sq.sqrt(), self.target_error)
        court = torch.where(landing, torch.exp(-self.scales['court'] * court_error_sq), 0.)
        # Imperfect/out returns still receive graded target feedback. No hard
        # in-court gate; require clearance so net-failing flights cannot score it.
        target_reward = torch.where(landing & self.crossed_net,
                                    torch.exp(-self.scales['target'] * target_error_sq), 0.)
        self.previous_ball.copy_(ball)
        forward_reward = (vy / self.forward_speed).clamp(0, 1)
        if self.forward_reward_mode == 'signed_linear':
            forward_reward = forward_reward - self.backward_penalty_scale * (
                -vy / self.forward_speed).clamp(0, 1)
        events = [first_contact.float(), first_contact * forward_reward,
                  net.float(), court, target_reward]
        if 'vertical' in self.components:
            # A penalty avoids granting a free bonus to flat backward/weak hits.
            # Zero inside a broad band; at most -1 beyond it. No flight modifications.
            events.append(-first_contact.float() * (vertical_excess / self.vertical_falloff).clamp(0, 1))
        return torch.stack(events, dim=-1)

    def stats(self, completed, boundary):
        hits = completed & self.hit
        landings = completed & self.landed
        target_landings = landings & self.crossed_net
        stats = {
            'return_hits': hits.sum(),
            'return_forward': (completed & self.forward).sum(),
            'return_net': (completed & self.crossed_net).sum(),
            'return_in': (completed & self.in_court).sum(),
            'return_landings': landings.sum(),
            'return_valid_estimates': (completed & self.estimate_valid).sum(),
            'return_target_landings': target_landings.sum(),
            'return_target_error_sum': self.target_error[target_landings].sum(),
            'return_contact_vy_sum': self.contact_vy[hits].sum(),
            'return_boundaries': (completed & boundary).sum(),
        }
        if 'vertical' in self.components:
            stats['return_vertical_outside_band'] = (hits & self.vertical_outside_band).sum()
        return stats
