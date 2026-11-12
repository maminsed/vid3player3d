"""Opt-in, bounded preparation/impact feedback from the simulated string bed.

The desired impulse is v_out - v_in. v_out is an approximate ballistic return
to the broad landing target, not a prediction of aerodynamic/spin dynamics.
No actor observations or commanded poses are used or changed here.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import math
import torch

from vid2player.utils.racket_diagnostics import rotate_vector_xyzw


@dataclass(frozen=True)
class OrientationSettings:
    gate_mode: str = 'racket_distance'
    score_mode: str = 'positive'
    angle_tolerance_deg: float = 30.
    angle_falloff_deg: float = 60.
    angle_zero_deg: float = 60.
    angle_penalty_full_deg: float = 90.
    preparation_s: float = .20
    late_tolerance_s: float = .10
    full_credit_before_s: float = 0.
    full_credit_after_s: float = 0.
    max_distance_m: float = .60
    flight_time_s: float = 1.40
    lateral_reach_m: float = 1.50
    forward_min_m: float = -.60
    forward_max_m: float = 1.50
    min_height_m: float = .45
    max_height_m: float = 2.50
    reach_falloff_m: float = .50
    movement_speed_mps: float = 4.
    max_preview_error_m: float = 1.


def resolve_orientation(config=None):
    if isinstance(config, OrientationSettings):
        result = config
    else:
        if config is not None and not isinstance(config, Mapping):
            raise ValueError('return_curriculum.orientation must be a mapping')
        try:
            result = OrientationSettings(**({} if config is None else config))
        except TypeError as error:
            raise ValueError('Invalid orientation options: ' + str(error)) from error
    if result.gate_mode not in ('racket_distance', 'player_reach'):
        raise ValueError('orientation.gate_mode must be racket_distance or player_reach')
    if result.score_mode not in ('positive', 'signed'):
        raise ValueError('orientation.score_mode must be positive or signed')
    for name, value in vars(result).items():
        if name in ('gate_mode', 'score_mode'):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('orientation.' + name + ' must be finite and numeric')
        if value < 0 and name in ('angle_tolerance_deg', 'full_credit_before_s', 'full_credit_after_s'):
            raise ValueError('orientation.' + name + ' must be nonnegative')
        if value <= 0 and name not in ('angle_tolerance_deg', 'full_credit_before_s',
                                      'full_credit_after_s', 'forward_min_m'):
            raise ValueError('orientation.' + name + ' must be positive')
    if not 0 <= result.angle_tolerance_deg < 90:
        raise ValueError('orientation angle tolerance must be in [0,90)')
    if result.score_mode == 'positive' and result.angle_tolerance_deg + result.angle_falloff_deg > 90:
        raise ValueError('orientation tolerance + falloff must be <= 90 degrees')
    if result.score_mode == 'signed' and not result.angle_tolerance_deg < result.angle_zero_deg < result.angle_penalty_full_deg <= 90:
        raise ValueError('orientation requires tolerance < zero angle < full penalty angle <= 90')
    if not (result.full_credit_before_s < result.preparation_s and
            result.full_credit_after_s < result.late_tolerance_s):
        raise ValueError('orientation full-credit margins must be smaller than preparation/late margins')
    if result.forward_min_m >= result.forward_max_m or result.min_height_m >= result.max_height_m:
        raise ValueError('orientation reachability bounds must be increasing')
    return result


def smoothstep(value):
    value = value.clamp(0, 1)
    return value.square() * (3 - 2 * value)


class RacketOrientationReward:
    """One weight, half its absolute budget for preparation and half for impact.

    player_reach selects and locks a reachable interval of the incoming preview;
    smooth time ramps surround a plateau spanning that interval. Reachability
    uses the player, never the racket head. The original triangular time/racket
    distance gate and positive score remain defaults for archived YAMLs.
    Integrate against the time window's area, with a .5 absolute preparation cap.
    First accepted impact contributes in [-.5,.5] in signed mode. Misses receive
    preparation feedback too, independently of clean contact.
    """
    def __init__(self, count, device, settings, geometry, ball_radius, dt):
        self.settings = resolve_orientation(settings)
        self.center = torch.tensor(geometry.center, device=device, dtype=torch.float32)
        self.normal = torch.tensor(geometry.normal, device=device, dtype=torch.float32)
        self.ball_radius, self.dt = ball_radius, dt
        self.preparation_paid = torch.zeros(count, device=device)
        self.contact_seen = torch.zeros(count, device=device, dtype=torch.bool)
        self.contact_score = torch.zeros(count, device=device)
        self.contact_now = torch.zeros(count, device=device, dtype=torch.bool)
        self.eta = torch.zeros(count, device=device)
        self.valid = self.contact_seen.clone()
        self.position = torch.zeros(count, 3, device=device)
        self.target = torch.zeros_like(self.position)
        self.root = torch.zeros_like(self.position)
        self.opportunity_valid = self.valid.clone()
        self.locked = self.valid.clone()
        self.start = torch.zeros(count, device=device)
        self.end = torch.zeros(count, device=device)
        self.now = torch.zeros(count, device=device)
        self.times = torch.arange(100, device=device) * dt
        self.preparation_total = torch.zeros(count, device=device)
        self.impact_total = torch.zeros(count, device=device)
        self.impact_angle = torch.zeros(count, device=device)
        self.active_steps = torch.zeros(count, device=device)
        self.last_time_gate = torch.zeros(count, device=device)
        self.last_reach_gate = torch.zeros(count, device=device)

    def reset(self, ids):
        self.preparation_paid[ids] = self.contact_score[ids] = 0
        self.contact_seen[ids] = self.contact_now[ids] = self.valid[ids] = False
        self.eta[ids] = 0
        self.position[ids] = self.target[ids] = 0
        self.root[ids] = 0
        self.opportunity_valid[ids] = self.locked[ids] = False
        for buffer in (self.start, self.end, self.now, self.preparation_total,
                       self.impact_total, self.impact_angle, self.active_steps,
                       self.last_time_gate, self.last_reach_gate):
            buffer[ids] = 0

    def prepare(self, eta, valid, position, target, attempting, contact, *, root=None,
                root_velocity=None, trajectory=None, clock=None, ball_position=None, loaded=None):
        """Cache pre-action prediction; keep the actual post-step pose for scoring."""
        self.eta.copy_(eta)
        self.valid.copy_(valid & attempting & ~contact & ~self.contact_seen)
        if self.settings.gate_mode == 'racket_distance':
            self.position.copy_(position)
            self.opportunity_valid |= self.valid
        self.target.copy_(target)
        if self.settings.gate_mode == 'player_reach':
            if any(value is None for value in (root, root_velocity, trajectory, clock, ball_position, loaded)):
                raise ValueError('Player-reach orientation requires root state and the paired incoming preview')
            self.root.copy_(root)
            self._prepare_opportunity(trajectory, clock, root_velocity, ball_position,
                                      loaded & attempting & ~contact & ~self.contact_seen)

    def _prepare_opportunity(self, trajectory, clock, root_velocity, ball_position, live):
        """Select the first contiguous reachable interval, then lock before its entry."""
        if trajectory.shape[1:] != (100, 3):
            raise ValueError('Orientation requires a paired 100-frame XYZ preview')
        s = self.settings
        self.now.copy_(clock * self.dt)
        index = clock.clamp(0, 99).long()
        preview_now = trajectory.gather(1, index[:, None, None].expand(-1, 1, 3))[:, 0]
        live = (live & (clock < 100) & torch.isfinite(self.root).all(-1)
                & torch.isfinite(root_velocity).all(-1) & torch.isfinite(ball_position).all(-1)
                & ((preview_now - ball_position).norm(dim=-1) <= s.max_preview_error_m))
        remaining = self.times[None] - self.now[:, None]
        future_root = self.root[:, None, :2] + root_velocity[:, None, :2] * remaining.clamp(0, .25)[..., None]
        offset = trajectory[..., :2] - future_root
        candidates = ((remaining >= 0) & torch.isfinite(trajectory).all(-1)
                      & (trajectory[..., 2] >= s.min_height_m) & (trajectory[..., 2] <= s.max_height_m)
                      & (offset[..., 1] >= s.forward_min_m) & (offset[..., 1] <= s.forward_max_m)
                      & (offset[..., 0].abs() <= s.lateral_reach_m + s.movement_speed_mps * remaining.clamp_min(0)))
        indices = torch.arange(100, device=trajectory.device)[None]
        first = torch.where(candidates, indices, 100).min(dim=1).values
        stop = torch.where((indices > first[:, None]) & ~candidates, indices, 100).min(dim=1).values
        last = (stop - 1).clamp(0, 99)
        first_safe = first.clamp(0, 99)
        update = ~self.locked & live
        self.opportunity_valid.copy_(torch.where(update, first < 100, self.opportunity_valid))
        self.start.copy_(torch.where(update, self.times[first_safe], self.start))
        self.end.copy_(torch.where(update, self.times[last], self.end))
        middle = ((first_safe + last) // 2)[:, None, None].expand(-1, 1, 3)
        selected = trajectory.gather(1, middle)[:, 0]
        # Keep the locked reference point as well as its time window stable.
        self.position.copy_(torch.where((update & (first < 100))[:, None], selected, self.position))
        self.locked |= live & self.opportunity_valid & (self.now >= self.start - s.preparation_s)
        self.valid.copy_(live & self.opportunity_valid)

    def reachability_gate(self):
        s = self.settings
        offset = self.position[:, :2] - self.root[:, :2]
        excess = torch.stack(((offset[:, 0].abs() - s.lateral_reach_m).clamp_min(0),
            (s.forward_min_m - offset[:, 1]).clamp_min(0) + (offset[:, 1] - s.forward_max_m).clamp_min(0),
            (s.min_height_m - self.position[:, 2]).clamp_min(0)
            + (self.position[:, 2] - s.max_height_m).clamp_min(0)), dim=-1).norm(dim=-1)
        return 1 - smoothstep(excess / s.reach_falloff_m)

    def begin_step(self):
        self.contact_score.zero_()
        self.contact_now.zero_()

    def angle_error(self, racket, ball, intercept):
        s = self.settings
        target = self.target.clone()
        target[:, 2] += self.ball_radius
        outgoing = (target - intercept) / s.flight_time_s
        outgoing[:, 2] += .5 * 9.81 * s.flight_time_s
        impulse = outgoing - ball[:, 7:10]
        speed = impulse.norm(dim=-1)
        normal = rotate_vector_xyzw(racket[:, 3:7], self.normal)
        cosine = ((normal * impulse).sum(-1).abs() / speed.clamp_min(1e-8)).clamp(0, 1)
        angle = torch.acos(cosine) * (180 / math.pi)
        finite = (torch.isfinite(racket).all(-1) & torch.isfinite(ball).all(-1)
                  & torch.isfinite(intercept).all(-1) & torch.isfinite(target).all(-1)
                  & (speed > 1e-6) & (outgoing[:, 1] > 0))
        return angle, finite

    def score(self, racket, ball, intercept):
        angle, finite = self.angle_error(racket, ball, intercept)
        s = self.settings
        if s.score_mode == 'signed':
            positive = 1 - smoothstep((angle - s.angle_tolerance_deg) / (s.angle_zero_deg - s.angle_tolerance_deg))
            negative = smoothstep((angle - s.angle_zero_deg) / (s.angle_penalty_full_deg - s.angle_zero_deg))
            result = positive - negative
        else:
            result = (1 - (angle - s.angle_tolerance_deg).clamp_min(0) / s.angle_falloff_deg).clamp(0, 1)
        return torch.where(finite, torch.nan_to_num(result), torch.zeros_like(result))

    def record_contact(self, accepted, ball_pre, racket_pre):
        first = accepted & ~self.contact_seen
        score = self.score(racket_pre, ball_pre, ball_pre[:, :3])
        self.contact_score.copy_(torch.where(first, score, self.contact_score))
        self.contact_now |= first
        self.contact_seen |= first
        angle, finite = self.angle_error(racket_pre, ball_pre, ball_pre[:, :3])
        self.impact_angle.copy_(torch.where(first & finite, angle, self.impact_angle))

    def step(self, racket, ball):
        s = self.settings
        # Use end-of-control-step states for preparation. A contacted step uses
        # the captured incoming/physical pre-impact pose for its impact bonus;
        # never score a rebounding ball as an incoming trajectory.
        if s.gate_mode == 'player_reach':
            now = self.now + self.dt
            time_gate = (smoothstep((now - self.start + s.preparation_s)
                                   / (s.preparation_s - s.full_credit_before_s))
                         * (1 - smoothstep((now - self.end - s.full_credit_after_s)
                                           / (s.late_tolerance_s - s.full_credit_after_s))))
            distance_gate = self.reachability_gate()
            area = ((self.end - self.start).clamp_min(0)
                    + .5 * (s.preparation_s + s.late_tolerance_s
                            + s.full_credit_before_s + s.full_credit_after_s))
        else:
            eta = self.eta - self.dt
            time_gate = torch.where(eta >= 0, 1 - eta / s.preparation_s,
                                    1 + eta / s.late_tolerance_s).clamp(0, 1)
            center = racket[:, :3] + rotate_vector_xyzw(racket[:, 3:7], self.center)
            distance_gate = (1 - (center - self.position).norm(dim=-1) / s.max_distance_m).clamp(0, 1)
            area = .5 * (s.preparation_s + s.late_tolerance_s)
        self.last_time_gate.copy_(torch.nan_to_num(time_gate))
        self.last_reach_gate.copy_(torch.nan_to_num(distance_gate))
        preparation = (.5 * self.score(racket, ball, self.position) * time_gate * distance_gate
                       * (self.dt / area))
        preparation = torch.where(self.valid & ~self.contact_seen, preparation, 0.)
        preparation = torch.nan_to_num(preparation, nan=0., posinf=0., neginf=0.)
        preparation = preparation.sign() * torch.minimum(preparation.abs(), (.5 - self.preparation_paid).clamp_min(0))
        # Spend an absolute budget: alternating signs cannot replenish it.
        self.preparation_paid += preparation.abs()
        self.preparation_total += preparation
        self.active_steps += (self.valid & ~self.contact_seen & (time_gate > 0) & (distance_gate > 0)).float()
        impact = .5 * self.contact_score * self.contact_now
        self.impact_total += impact
        # Consuming the event here also makes repeated reward reads harmless.
        self.contact_now.zero_()
        return preparation + impact

    def stats(self, completed):
        contacts = completed & self.contact_seen
        return {'return_orientation_preparation_sum': self.preparation_total[completed].sum(),
                'return_orientation_impact_sum': self.impact_total[completed].sum(),
                'return_orientation_contact_score_sum': (2 * self.impact_total[contacts]).sum(),
                'return_orientation_contact_angle_sum': self.impact_angle[contacts].sum(),
                'return_orientation_contacts': contacts.sum(),
                'return_orientation_active_steps': self.active_steps[completed].sum(),
                'return_orientation_opportunities': (completed & self.opportunity_valid).sum()}
