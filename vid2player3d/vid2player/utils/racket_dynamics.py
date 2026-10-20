"""Explicit racket inertial properties instead of collision-mesh densities.

The reference mass and transverse inertia are from Cross (2003), section
Experimental Method: 270 g and I_cm=0.0123 kg m^2. The 0.18 m contact-to-COM
separation reproduces that paper's approximately 0.4 normal apparent COR for
a hand-held racket when the material COR is 0.9. The longitudinal/twist moment
is a geometric estimate, not a measured Djokovic racket specification.
"""
from dataclasses import dataclass
import math
from collections.abc import Mapping


@dataclass(frozen=True)
class RacketDynamics:
    model: str = 'legacy'
    max_angular_velocity: float = 1000.
    mass: float = .270
    com_x: float = .180
    long_axis_inertia: float = .0012
    transverse_inertia: float = .0123
    normal_inertia: float = .01349

    @property
    def enabled(self):
        return self.model == 'published_reference_v1'


def resolve_racket_dynamics(config=None):
    if isinstance(config, RacketDynamics):
        settings = config
    else:
        config = {} if config is None else config
        if not isinstance(config, Mapping):
            raise ValueError('racket_dynamics must be a mapping')
        try:
            settings = RacketDynamics(**config)
        except TypeError as error:
            raise ValueError(f'Invalid racket_dynamics options: {error}') from error
    if settings.model not in ('legacy', 'published_reference_v1'):
        raise ValueError(f'Unknown racket dynamics model: {settings.model}')
    moments = [settings.long_axis_inertia, settings.transverse_inertia, settings.normal_inertia]
    if any(not math.isfinite(v) or v <= 0 for v in [settings.mass, settings.max_angular_velocity] + moments):
        raise ValueError('Racket mass and principal inertias must be finite and positive')
    if not math.isfinite(settings.com_x) or not 0 <= settings.com_x <= .5:
        raise ValueError('Racket COM must lie between the head and hand (0..0.5 m)')
    if max(moments) > sum(moments) - max(moments):
        raise ValueError('Racket principal inertias violate the triangle inequality')
    return settings


def configure_racket_body(gym, env, actor, body_id, settings):
    settings = resolve_racket_dynamics(settings)
    if not settings.enabled:
        return
    properties = gym.get_actor_rigid_body_properties(env, actor)
    prop = properties[body_id]
    prop.mass = settings.mass
    prop.com.x, prop.com.y, prop.com.z = settings.com_x, 0., 0.
    # The physical string normal is (0,1,1)/sqrt(2) in the body frame.
    yz = (settings.transverse_inertia + settings.normal_inertia) / 2
    cross = (settings.normal_inertia - settings.transverse_inertia) / 2
    prop.inertia.x.x = settings.long_axis_inertia
    prop.inertia.y.y = prop.inertia.z.z = yz
    prop.inertia.x.y = prop.inertia.x.z = prop.inertia.y.x = prop.inertia.z.x = 0.
    prop.inertia.y.z = prop.inertia.z.y = cross
    gym.set_actor_rigid_body_properties(env, actor, properties, False)
