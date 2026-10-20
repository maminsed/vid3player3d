"""Shared tennis-ball aerodynamics and versioned inertial assets; no Isaac Gym dependency.

Velocities are m/s and angular velocities are rad/s. ``aerodynamic_v1`` uses
the transverse surface speed R|omega x v_hat| in the empirical lift formula
of Brody/Cross/Lindsey, Ball Trajectories, Eq. 42.3. It is a baseline model,
not a calibration of a particular ball, court, or string bed.
"""
from dataclasses import dataclass
import math
from collections.abc import Mapping
from pathlib import Path
import xml.etree.ElementTree as ET

import torch

BALL_MASS_KG = 0.057
AIR_DENSITY_KG_M3 = 1.21
BASE_CD = 0.55


@dataclass(frozen=True)
class BallPhysics:
    model: str = 'legacy'
    # Used only by aerodynamic_v1; legacy retains Isaac Gym's asset defaults.
    angular_damping: float = 0.0
    max_angular_velocity: float = 1000.0

    @property
    def corrected(self):
        return self.model in ('aerodynamic_v1', 'aerodynamic_v2')

    @property
    def radius(self):
        return 0.0335 if self.corrected else 0.032

    @property
    def force_factor(self):
        return AIR_DENSITY_KG_M3 * math.pi * self.radius**2 / 2

    @property
    def asset_file(self):
        return ('tennis_ball_aerodynamic_v2.urdf' if self.model == 'aerodynamic_v2'
                else 'tennis_ball_aerodynamic_v1.urdf' if self.corrected else 'tennis_ball.urdf')


def resolve_ball_physics(config=None):
    """Read the env.vid2player.ball_physics mapping (omission means legacy)."""
    if isinstance(config, BallPhysics):
        settings = config
    else:
        if config is None:
            config = {}
        if not isinstance(config, Mapping):
            raise ValueError('ball_physics must be a mapping')
        unknown = set(config) - {'model', 'angular_damping', 'max_angular_velocity'}
        if unknown:
            raise ValueError(f'Unknown ball_physics options: {sorted(unknown)}')
        if config.get('model') == 'aerodynamic_v2' and 'max_angular_velocity' not in config:
            config = dict(config, max_angular_velocity=10000.)
        settings = BallPhysics(**config)
    if settings.model not in ('legacy', 'aerodynamic_v1', 'aerodynamic_v2'):
        raise ValueError(f'Unknown ball physics model: {settings.model}')
    if not math.isfinite(settings.angular_damping) or settings.angular_damping < 0:
        raise ValueError('ball_physics.angular_damping must be finite and >= 0')
    if not math.isfinite(settings.max_angular_velocity) or settings.max_angular_velocity <= 0:
        raise ValueError('ball_physics.max_angular_velocity must be finite and > 0')
    return settings


def configure_ball_options(options, settings):
    """Apply explicit spin options before loading the corrected ball asset."""
    settings = resolve_ball_physics(settings)
    if settings.corrected:
        options.angular_damping = settings.angular_damping
        options.max_angular_velocity = settings.max_angular_velocity
    return options


def validate_ball_asset(asset_root, asset_file, settings):
    """Reject corrected-mode custom assets that disagree with aerodynamic size/mass."""
    if not settings.corrected:
        return
    root = ET.parse(Path(asset_root) / asset_file).getroot()
    spheres = root.findall('.//collision/geometry/sphere')
    masses = root.findall('.//inertial/mass')
    if (len(spheres) != 1 or len(root.findall('.//collision')) != 1 or len(masses) != 1
            or not math.isclose(float(spheres[0].attrib['radius']), settings.radius, abs_tol=1e-8)
            or not math.isclose(float(masses[0].attrib['value']), BALL_MASS_KG, abs_tol=1e-8)):
        raise ValueError('Corrected ball models require a single spherical 0.0335 m, 0.057 kg asset')
    if settings.model == 'aerodynamic_v2':
        inertia = root.find('.//inertial/inertia')
        expected = .55 * BALL_MASS_KG * settings.radius**2
        if inertia is None or any(not math.isclose(float(inertia.get(k, 'nan')),
                expected if k in ('ixx', 'iyy', 'izz') else 0., abs_tol=1e-11)
                for k in ('ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz')):
            raise ValueError('aerodynamic_v2 requires isotropic finite-wall inertia 0.55*m*R^2')


def resolve_ball_asset(asset_root, settings, asset_file=None):
    """Use the versioned code asset for v1; retain legacy/custom asset locations."""
    if settings.corrected and asset_file is None:
        asset_root = str(Path(__file__).resolve().parents[1] / 'assets')
    asset_file = asset_file or settings.asset_file
    validate_ball_asset(asset_root, asset_file, settings)
    return asset_root, asset_file


def get_cl(v, vs):
    """Legacy scalar formula; vs must be surface speed for physical units."""
    return 1 / (2 + abs(v / (vs + 1e-6)))


def get_cd(v, vs):
    return BASE_CD


def ball_aerodynamic_force(velocity, angular_velocity, settings=None,
                           spin_scale=5.0, legacy_spin_sign=None):
    """Batched forces in N, in the same frame as velocity and angular velocity.

    Legacy generators carry a signed launch scalar separately from PhysX's
    angular velocity. Only that branch uses legacy_spin_sign and spin_scale.
    The corrected branch uses actual transverse angular velocity, including
    after contact; no forced post-bounce spin reversal is applied.
    """
    settings = resolve_ball_physics(settings)
    speed = velocity.norm(dim=-1, keepdim=True)
    safe_speed = torch.where(speed == 0, torch.ones_like(speed), speed)
    direction = velocity / safe_speed
    drag = -settings.force_factor * BASE_CD * safe_speed * velocity
    if not settings.corrected:
        spin = angular_velocity.norm(dim=-1, keepdim=True) / (2 * math.pi)
        if legacy_spin_sign is not None:
            spin = spin * legacy_spin_sign.reshape(spin.shape)
        cl = get_cl(safe_speed, spin * spin_scale)
        cl = cl * torch.where(spin > 0, -torch.ones_like(cl), torch.ones_like(cl))
        down = torch.zeros_like(velocity)
        down[..., 2] = -1
        tangent = torch.cross(direction, down, dim=-1)
        return drag - settings.force_factor * cl * safe_speed**2 * torch.cross(tangent, direction, dim=-1)

    # omega x v_hat supplies both Magnus direction and transverse spin rate.
    # Parallel (rifle) spin does not contribute to lift.
    transverse = torch.cross(angular_velocity, direction, dim=-1)
    transverse_rate = transverse.norm(dim=-1, keepdim=True)
    surface_speed = settings.radius * transverse_rate
    denominator = speed + 2 * surface_speed
    cl = surface_speed / denominator.clamp_min(torch.finfo(velocity.dtype).tiny)
    lift_direction = transverse / transverse_rate.clamp_min(torch.finfo(velocity.dtype).tiny)
    return drag + settings.force_factor * cl * speed**2 * lift_direction
