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
    angle_tolerance_deg: float = 30.
    angle_falloff_deg: float = 60.
    preparation_s: float = .20
    late_tolerance_s: float = .10
    max_distance_m: float = .60
    flight_time_s: float = 1.40


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
    for name, value in vars(result).items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('orientation.' + name + ' must be finite and numeric')
        if value <= 0 and name != 'angle_tolerance_deg':
            raise ValueError('orientation.' + name + ' must be positive')
    if not 0 <= result.angle_tolerance_deg < 90:
        raise ValueError('orientation angle tolerance must be in [0,90)')
    if result.angle_tolerance_deg + result.angle_falloff_deg > 90:
        raise ValueError('orientation tolerance + falloff must be <= 90 degrees')
    return result


class RacketOrientationReward:
    """One weight, half its budget for preparation and half for first impact.

    Preparation is a triangular time window, smoothly distance-gated around the
    predicted intercept. Its integral is normalized by the triangle's area and
    capped at .5 per launch. First accepted impact can pay at most another .5.
    A failed interception can receive preparation feedback without clean contact.
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

    def reset(self, ids):
        self.preparation_paid[ids] = self.contact_score[ids] = 0
        self.contact_seen[ids] = self.contact_now[ids] = self.valid[ids] = False
        self.eta[ids] = 0
        self.position[ids] = self.target[ids] = 0

    def prepare(self, eta, valid, position, target, attempting, contact):
        """Cache pre-action prediction; keep the actual post-step pose for scoring."""
        self.eta.copy_(eta)
        self.valid.copy_(valid & attempting & ~contact & ~self.contact_seen)
        self.position.copy_(position)
        self.target.copy_(target)

    def begin_step(self):
        self.contact_score.zero_()
        self.contact_now.zero_()

    def score(self, racket, ball, intercept):
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
        result = (1 - (angle - s.angle_tolerance_deg).clamp_min(0) / s.angle_falloff_deg).clamp(0, 1)
        finite = (torch.isfinite(racket).all(-1) & torch.isfinite(ball).all(-1)
                  & torch.isfinite(intercept).all(-1) & torch.isfinite(target).all(-1)
                  & (speed > 1e-6) & (outgoing[:, 1] > 0))
        return torch.where(finite, torch.nan_to_num(result), torch.zeros_like(result))

    def record_contact(self, accepted, ball_pre, racket_pre):
        first = accepted & ~self.contact_seen
        score = self.score(racket_pre, ball_pre, ball_pre[:, :3])
        self.contact_score.copy_(torch.where(first, score, self.contact_score))
        self.contact_now |= first
        self.contact_seen |= first

    def step(self, racket, ball):
        s = self.settings
        # Use end-of-control-step states for preparation. A contacted step uses
        # the captured incoming/physical pre-impact pose for its impact bonus;
        # never score a rebounding ball as an incoming trajectory.
        eta = self.eta - self.dt
        time_gate = torch.where(eta >= 0, 1 - eta / s.preparation_s,
                                1 + eta / s.late_tolerance_s).clamp(0, 1)
        center = racket[:, :3] + rotate_vector_xyzw(racket[:, 3:7], self.center)
        distance_gate = (1 - (center - self.position).norm(dim=-1) / s.max_distance_m).clamp(0, 1)
        area = .5 * (s.preparation_s + s.late_tolerance_s)
        preparation = (.5 * self.score(racket, ball, self.position) * time_gate * distance_gate
                       * (self.dt / area))
        preparation = torch.where(self.valid & ~self.contact_seen, preparation, 0.)
        preparation = torch.nan_to_num(preparation, nan=0., posinf=0., neginf=0.)
        preparation = torch.minimum(preparation, (.5 - self.preparation_paid).clamp_min(0))
        self.preparation_paid += preparation
        impact = .5 * self.contact_score * self.contact_now
        # Consuming the event here also makes repeated reward reads harmless.
        self.contact_now.zero_()
        return preparation + impact
