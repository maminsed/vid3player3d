"""Versioned ball/court/string-bed contact baseline, shared by live and generation.

PhysX averages these material coefficients at an impact. This rigid approximation
is a repeatable generic hard-court baseline, not a fit to a particular US Open
court or string setup. Legacy mode leaves all old config fallbacks intact.
"""
from dataclasses import dataclass, fields
from collections.abc import Mapping
import math
import torch


@dataclass(frozen=True)
class TennisContacts:
    model: str = 'legacy'
    ball_restitution: float = .9
    ball_friction: float = .4
    court_restitution: float = .64
    court_friction: float = 1.
    racket_restitution: float = .8
    racket_friction: float = .6
    ball_contact_offset: float = .02
    racket_contact_offset: float = .05

    @property
    def enabled(self):
        return self.model in ('hardcourt_v1', 'hardcourt_v2')


def resolve_tennis_contacts(config=None):
    if isinstance(config, TennisContacts):
        result = config
    else:
        config = {} if config is None else config
        if not isinstance(config, Mapping):
            raise ValueError('tennis_contacts must be a mapping')
        unknown = set(config) - {f.name for f in fields(TennisContacts)}
        if unknown:
            raise ValueError(f'Unknown tennis_contacts options: {sorted(unknown)}')
        if config.get('model') == 'hardcourt_v2':
            config = dict(config)
            config.setdefault('ball_contact_offset', 1e-6)
            config.setdefault('racket_contact_offset', 1e-6)
        result = TennisContacts(**config)
    if result.model not in ('legacy', 'hardcourt_v1', 'hardcourt_v2'):
        raise ValueError(f'Unknown tennis contact model: {result.model}')
    for f in fields(result):
        if f.name == 'model':
            continue
        value = getattr(result, f.name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'tennis_contacts.{f.name} must be finite and nonnegative')
        if 'restitution' in f.name and value > 1:
            raise ValueError(f'tennis_contacts.{f.name} must be <= 1')
        if 'offset' in f.name and value <= 0:
            raise ValueError(f'tennis_contacts.{f.name} must be > 0 (rest offset is zero)')
    return result


def contact_materials(cfg):
    """Return enabled profile or the exact historic live-task fallback values."""
    c = resolve_tennis_contacts(cfg.get('tennis_contacts'))
    if c.enabled:
        return c
    return TennisContacts(
        ball_restitution=cfg.get('ball_restitution', cfg.get('restitution', 1.)),
        ball_friction=cfg.get('ball_friction', .8),
        racket_restitution=cfg.get('racket_restitution', cfg.get('restitution', 1.)),
        racket_friction=cfg.get('racket_friction', .8))


def ground_impact(previous, current, radius, dt):
    """Detect an observed ground rebound, estimate its center at z=radius.

    Uses one pre/post physics-step state pair. The event needs a downward/upward
    velocity reversal and reachability of the floor in that interval. Position
    is extrapolated from the pre-impact horizontal velocity to the ballistic
    floor-crossing time, not the old 4R/6R proximity point. Timing is approximate
    at the outer timestep, especially for multiple contacts within one step.
    Pure batched tensor operations; no device/host synchronization.
    """
    height = (previous[:, 2] - radius).clamp_min(0.)
    down = -previous[:, 9]
    discriminant = (down.square() + 2 * 9.81 * height).sqrt()
    time = 2 * height / (discriminant + down).clamp_min(1e-8)
    # Permit the speculative contact margin and integration uncertainty while
    # excluding airborne reversals (e.g. a normal racket strike).
    event = ((previous[:, 9] < 0) & (current[:, 9] > 0)
             & (time <= dt + .002)
             & (current[:, 2] <= radius + current[:, 9].clamp_min(0.) * dt + .02))
    time = time.clamp(0., dt)
    position = previous[:, :3] + previous[:, 7:10] * time[:, None]
    position[:, 2] = radius
    return event, position


def apply_solver_config(sim_params, physics):
    """Apply the same explicit high-level solver overrides during generation."""
    for name in ('dt', 'substeps'):
        if name in physics:
            setattr(sim_params, name, physics[name])
    for name in ('solver_type', 'num_position_iterations', 'num_velocity_iterations',
                 'contact_offset', 'rest_offset', 'bounce_threshold_velocity',
                 'max_depenetration_velocity', 'friction_offset_threshold',
                 'friction_correlation_distance'):
        if name in physics:
            setattr(sim_params.physx, name, physics[name])
    return sim_params


def racket_impact(previous, current, racket_previous, racket_current,
                  ball_contact_force, racket_contact_force, expected_velocity_change,
                  radius, contact_margin, dt):
    """Racket-region event from impulses and a swept proximity check.

    GPU contact tensors can omit short impacts even with all-substep collection.
    A velocity impulse (>2 m/s beyond smooth flight) is a fallback. Both paths
    must pass a swept racket-center check; this does not identify exact mesh
    contact pairs or distinguish a handle strike at the edge of the head.
    """
    start = previous[:, :3] - racket_previous
    # Extrapolate the incoming path: the endpoints of an actual bounce can
    # both be far in front of the head and their chord would miss the impact.
    end = previous[:, :3] + previous[:, 7:10] * dt - racket_current
    segment = end - start
    fraction = -(start * segment).sum(-1) / segment.square().sum(-1).clamp_min(1e-12)
    closest = start + fraction.clamp(0., 1.)[:, None] * segment
    near = closest.square().sum(-1) <= (.15 + radius + contact_margin)**2
    sensors = ((ball_contact_force.square().sum(-1) > 1e-8)
               & (racket_contact_force.square().sum(-1) > 1e-8))
    impulse = current[:, 7:10] - previous[:, 7:10] - expected_velocity_change
    return near & (sensors | (impulse.square().sum(-1) > 4.))


def preview_timing(sim_params):
    """Keep the 30 Hz observation/recording contract at finer physical timesteps."""
    dt = float(sim_params.dt)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('Physics dt must be finite and positive')
    count = round(1 / (30 * dt))
    if count < 1 or not math.isclose(count * dt, 1 / 30, rel_tol=1e-6):
        raise ValueError('Physics dt must divide the 30 Hz preview/controller interval')
    return dt, count
