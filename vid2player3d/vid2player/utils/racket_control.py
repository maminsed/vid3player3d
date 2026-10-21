"""Optional racket presets and batched interception from the paired launch preview.

The preview is privileged controller information, never an actor observation.
No extra simulator, CPU transfers, or per-environment Python loops are used.
"""
from dataclasses import dataclass, field
import math
import torch
from vid2player.utils.konia_transform import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis


@dataclass(frozen=True)
class InterceptionSettings:
    enabled: bool = False
    update_interval: int = 3
    preparation_s: float = .45
    late_tolerance_s: float = .10
    forward_m: float = .60
    forward_tolerance_m: float = .60
    lateral_reach_m: float = 1.20
    movement_speed_mps: float = 4.
    min_height_m: float = .45
    max_height_m: float = 2.50
    max_preview_error_m: float = 1.


@dataclass(frozen=True)
class RacketControlSettings:
    mode: str = 'legacy'
    blend_s: float = .10
    prediction: InterceptionSettings = field(default_factory=InterceptionSettings)


def resolve_racket_control(config=None):
    config = dict(config or {})
    prediction = InterceptionSettings(**config.pop('prediction', {}))
    settings = RacketControlSettings(prediction=prediction, **config)
    if settings.mode not in ('legacy', 'interception_window', 'none'):
        raise ValueError('racket_phase_correction.mode must be legacy, interception_window or none')
    if not isinstance(prediction.enabled, bool):
        raise ValueError('prediction.enabled must be boolean')
    if type(prediction.update_interval) is not int or prediction.update_interval < 1:
        raise ValueError('prediction.update_interval must be a positive integer')
    for name, value in vars(prediction).items():
        if name in ('enabled', 'update_interval'):
            continue
        if not math.isfinite(value) or value <= 0:
            raise ValueError('prediction.' + name + ' must be finite and positive')
    if prediction.min_height_m >= prediction.max_height_m:
        raise ValueError('prediction height range must be increasing')
    if not math.isfinite(settings.blend_s) or settings.blend_s <= 0:
        raise ValueError('racket_phase_correction.blend_s must be finite and positive')
    if settings.mode == 'interception_window' and not prediction.enabled:
        raise ValueError('interception_window requires prediction.enabled: true')
    return settings


def corrected_racket_rotations(decoded, residual, phase, swing, mode, weight=None, previous_delta=None, alpha=1.):
    """Add policy offsets in both experiments; smoothly introduce Djokovic presets.

    Filter the relative preset rotation on SO(3), not the whole arm pose, so
    genuine decoded motion and policy adjustments retain their responsiveness.
    """
    pose = rotation_matrix_to_angle_axis(decoded)
    offsets = residual.mul(.25).clamp(-.25, .25) * math.pi
    additive = pose.clone()
    additive[:, 0, 0] += offsets[:, 0]
    additive[:, 1, 1] += offsets[:, 1]
    additive[:, 1, 2] += offsets[:, 2]
    if mode == 'none':
        return additive, previous_delta
    fh = (phase > 2.) & (phase < 3.2) & (swing == 1)
    bh = (phase > 2.) & (phase < 3.2) & (swing == 2)
    target = pose.clone()
    target[:, 0, 0] = torch.where(fh, -.75 * math.pi, torch.where(bh, -.25 * math.pi, 0.)) + offsets[:, 0]
    target[:, 1, 0] = 0
    target[:, 1, 1] = offsets[:, 1]
    base = torch.where(fh, torch.where(phase < 3.1, -.25 * math.pi, .25 * math.pi),
                       torch.where(bh & (phase < 3.), .1 * math.pi, 0.))
    target[:, 1, 2] = base + offsets[:, 2]
    additive_matrix = angle_axis_to_rotation_matrix(additive)
    delta = angle_axis_to_rotation_matrix(target) @ additive_matrix.transpose(-1, -2)
    # No presets outside the decoded swing gates, including after phase 3.2.
    gate = weight * (fh | bh).to(weight.dtype)
    desired = angle_axis_to_rotation_matrix(rotation_matrix_to_angle_axis(delta) * gate[:, None, None])
    change = desired @ previous_delta.transpose(-1, -2)
    filtered = angle_axis_to_rotation_matrix(rotation_matrix_to_angle_axis(change) * alpha) @ previous_delta
    return rotation_matrix_to_angle_axis(filtered @ additive_matrix), filtered


class InterceptionWindow:
    def __init__(self, count, device, dt, settings, blend_s):
        self.settings, self.dt, self.blend_s = settings, dt, blend_s
        self.trajectory = torch.zeros(count, 100, 3, device=device)
        self.times = torch.arange(100, device=device) * dt
        self.deadline = torch.zeros(count, device=device)
        self.position = torch.zeros(count, 3, device=device)
        self.valid = torch.zeros(count, dtype=torch.bool, device=device)
        self.locked = self.valid.clone()
        self.loaded = self.valid.clone()
        self.active = self.valid.clone()
        self.side = torch.full((count,), -1, dtype=torch.long, device=device)
        self.eta = torch.zeros(count, device=device)
        self.weight = torch.zeros(count, device=device)
        self.preview_error = torch.zeros(count, device=device)
        self.tick = 0

    def reset(self, ids, trajectory):
        if trajectory.shape[1:] != (100, 3):
            raise ValueError('Interception requires a paired 100-frame XYZ preview')
        self.trajectory[ids] = trajectory.to(self.trajectory.device)
        self.valid[ids] = self.locked[ids] = self.active[ids] = False
        self.loaded[ids] = True
        self.weight[ids] = self.deadline[ids] = self.eta[ids] = 0
        self.position[ids] = 0
        self.side[ids] = -1

    @torch.no_grad()
    def step(self, clock, root, root_velocity, ball, attempting, contact, righthand=True):
        s = self.settings
        now = clock * self.dt
        index = clock.clamp(0, 99).long()
        predicted_now = self.trajectory.gather(1, index[:, None, None].expand(-1, 1, 3))[:, 0]
        self.preview_error = (predicted_now - ball).norm(dim=-1)
        live = (self.loaded & attempting & ~contact & (clock < 100)
                & torch.isfinite(ball).all(-1) & (ball[:, 1] >= root[:, 1] - 1.)
                & (self.preview_error <= s.max_preview_error_m))
        if self.tick % s.update_interval == 0:
            remaining = self.times[None] - now[:, None]
            # Short extrapolation of current body motion; reachable lateral travel
            # is a separate envelope, not a claim to solve full-body interception.
            future_root = root[:, None, :2] + root_velocity[:, None, :2] * remaining.clamp(0, .25)[..., None]
            offset = self.trajectory[..., :2] - future_root
            reachable = (s.lateral_reach_m + s.movement_speed_mps * remaining.clamp_min(0))
            candidates = ((remaining >= 0) & (self.trajectory[..., 2] >= s.min_height_m)
                          & (self.trajectory[..., 2] <= s.max_height_m)
                          & ((offset[..., 1] - s.forward_m).abs() <= s.forward_tolerance_m)
                          & (offset[..., 0].abs() <= reachable)
                          & torch.isfinite(self.trajectory).all(-1))
            score = ((offset[..., 1] - s.forward_m).square()
                     + .2 * (offset[..., 0].abs() - .65).square()
                     + .02 * remaining.clamp_min(0))
            score = score.masked_fill(~candidates, float('inf'))
            best_score, best = score.min(dim=1)
            update = ~self.locked & live
            self.valid = torch.where(update, torch.isfinite(best_score), self.valid)
            self.deadline = torch.where(update, self.times[best], self.deadline)
            position = self.trajectory.gather(1, best[:, None, None].expand(-1, 1, 3))[:, 0]
            self.position = torch.where(update[:, None], position, self.position)
            side = torch.where(position[:, 0] >= root[:, 0], 1 if righthand else 2, 2 if righthand else 1)
            self.side = torch.where(update, side, self.side)
        self.tick += 1
        self.valid &= live
        self.eta = self.deadline - now
        self.active = self.valid & (self.eta <= s.preparation_s) & (self.eta >= -s.late_tolerance_s)
        self.locked |= self.active
        # Do not chase a later candidate after the selected interception is past.
        target = self.active.to(self.weight.dtype)
        self.weight += (target - self.weight).clamp(-self.dt / self.blend_s, self.dt / self.blend_s)
        return self.weight
