"""Opt-in first-impact reward geometry; does not classify or change collisions.

The swept incoming path estimates a finite cylinder-face contact from outer
physics-step states. This is a geometric proxy, not a measured PhysX contact
point/pair or an individual internal-substep sample.
"""
from dataclasses import dataclass
from collections.abc import Mapping
import math
import xml.etree.ElementTree as ET

import torch

from vid2player.utils.racket_diagnostics import rotate_vector_xyzw


QUALITY_COMPONENTS = ('clean_contact', 'contact_angle')


@dataclass(frozen=True)
class ContactQualitySettings:
    full_credit_radius_fraction: float = .5
    zero_credit_radius_fraction: float = 1.
    angle_tolerance_deg: float = 30.
    angle_falloff_deg: float = 60.


def resolve_contact_quality(config=None):
    if isinstance(config, ContactQualitySettings):
        settings = config
    else:
        config = {} if config is None else config
        if not isinstance(config, Mapping):
            raise ValueError('return_curriculum.contact_quality must be a mapping')
        try:
            settings = ContactQualitySettings(**config)
        except TypeError as error:
            raise ValueError('Invalid contact_quality options: ' + str(error)) from error
    values = vars(settings)
    if any(not isinstance(v, (float, int)) or isinstance(v, bool) or not math.isfinite(v)
           for v in values.values()):
        raise ValueError('Contact quality settings must be finite numbers')
    if not 0 <= settings.full_credit_radius_fraction < settings.zero_credit_radius_fraction <= 1:
        raise ValueError('Contact quality requires 0 <= full radius < zero radius <= 1')
    if not 0 <= settings.angle_tolerance_deg < 90 or not 0 < settings.angle_falloff_deg <= 90:
        raise ValueError('Contact angle requires tolerance in [0,90) and falloff in (0,90] degrees')
    if settings.angle_tolerance_deg + settings.angle_falloff_deg > 90 + 1e-8:
        raise ValueError('Contact angle tolerance + falloff must be <= 90 degrees')
    return settings


@dataclass(frozen=True)
class StringBedGeometry:
    center: tuple
    normal: tuple
    radius: float
    half_thickness: float


def read_string_bed_geometry(asset_file):
    """Read the actual head cylinder in the Racket body frame, not a grip guess."""
    body = ET.parse(asset_file).getroot().find(".//body[@name='Racket']")
    geoms = [] if body is None else body.findall('geom')
    if len(geoms) != 2 or geoms[1].get('type') != 'cylinder' or 'fromto' not in geoms[1].attrib:
        raise ValueError('Contact quality requires a Racket body with handle and fromto cylinder head')
    head = geoms[1]
    ends = [float(x) for x in head.get('fromto').split()]
    size = [float(x) for x in head.get('size', '').split()]
    if len(ends) != 6 or len(size) != 1 or any(not math.isfinite(v) for v in ends + size):
        raise ValueError('Invalid racket head cylinder geometry')
    axis = [b - a for a, b in zip(ends[:3], ends[3:])]
    length = math.sqrt(sum(x*x for x in axis))
    if length <= 0 or size[0] <= 0:
        raise ValueError('Racket head cylinder must have positive radius and thickness')
    return StringBedGeometry(tuple((a+b)/2 for a,b in zip(ends[:3], ends[3:])),
                             tuple(x/length for x in axis), size[0], length/2)


def inverse_rotate(quaternion, vector):
    conjugate = torch.cat((-quaternion[..., :3], quaternion[..., 3:]), dim=-1)
    return rotate_vector_xyzw(conjugate, vector)


class RacketContactQuality:
    def __init__(self, count, device, settings, geometry, com_local, ball_radius, contact_margin):
        self.settings = resolve_contact_quality(settings)
        self.center = torch.tensor(geometry.center, device=device, dtype=torch.float32)
        self.normal = torch.tensor(geometry.normal, device=device, dtype=torch.float32)
        self.com = torch.tensor(com_local, device=device, dtype=torch.float32)
        self.radius, self.half_thickness = geometry.radius, geometry.half_thickness
        self.ball_radius, self.contact_margin = ball_radius, contact_margin
        if not (math.isfinite(ball_radius) and ball_radius > 0 and
                math.isfinite(contact_margin) and contact_margin >= 0):
            raise ValueError('Contact quality needs positive ball radius and nonnegative margin')
        self.scores = {key: torch.zeros(count, device=device) for key in QUALITY_COMPONENTS}

    def reset(self, ids=None):
        for value in self.scores.values():
            if ids is None:
                value.zero_()
            else:
                value[ids] = 0

    def record(self, accepted, ball_pre, racket_pre, racket_post, dt):
        """Latch only accepted first impacts, before later intervals overwrite state.

        Linear interpolation in the moving racket frame approximates translation
        and rotation. Extrapolate the incoming ball (not the rebounding endpoint)
        to avoid the chord across a bounce missing the head. Angular velocity
        uses the actual COM and the estimated point on the contacting face.
        """
        q0, q1 = racket_pre[:, 3:7], racket_post[:, 3:7]
        start = inverse_rotate(q0, ball_pre[:, :3] - racket_pre[:, :3]) - self.center
        end = inverse_rotate(q1, ball_pre[:, :3] + ball_pre[:, 7:10]*dt
                             - racket_post[:, :3]) - self.center
        h0, h1 = (start*self.normal).sum(-1), (end*self.normal).sum(-1)
        side = torch.where(h0 < 0, -torch.ones_like(h0), torch.ones_like(h0))
        surface = self.half_thickness + self.ball_radius
        delta = h0 - h1
        safe_delta = torch.where(delta.abs() > 1e-8, delta, torch.ones_like(delta))
        fraction = torch.where(delta.abs() > 1e-8,
                               (h0-side*surface)/safe_delta, torch.zeros_like(h0)).clamp(0, 1)
        point = start + fraction[:, None]*(end-start)
        height = (point*self.normal).sum(-1)
        planar = point - height[:, None]*self.normal
        radial = planar.norm(dim=-1)
        full = self.settings.full_credit_radius_fraction*self.radius
        zero = self.settings.zero_credit_radius_fraction*self.radius
        radial_score = ((zero-radial)/(zero-full)).clamp(0, 1)
        # Both the physical face and its configured speculative-contact region
        # are accepted; decay outside, never grant credit for remote sensor hits.
        excess = (height.abs()-surface).clamp_min(0)
        if self.contact_margin > 0:
            plane_score = (1-excess/self.contact_margin).clamp(0, 1)
        else:
            plane_score = (excess <= 1e-6).float()
        clean = radial_score*plane_score

        face_point = self.center + planar + (side*self.half_thickness)[:, None]*self.normal
        lever = rotate_vector_xyzw(q0, face_point-self.com)
        surface_velocity = racket_pre[:, 7:10] + torch.cross(racket_pre[:, 10:13], lever, dim=-1)
        relative = ball_pre[:, 7:10] - surface_velocity
        speed = relative.norm(dim=-1)
        normal_world = rotate_vector_xyzw(q0, self.normal)
        cosine = ((relative*normal_world).sum(-1).abs()/speed.clamp_min(1e-8)).clamp(0, 1)
        angle = torch.acos(cosine)*(180/math.pi)
        angle_score = (1-(angle-self.settings.angle_tolerance_deg).clamp_min(0)
                       /self.settings.angle_falloff_deg).clamp(0, 1)
        angle_score = torch.where(speed > 1e-6, angle_score, torch.zeros_like(angle_score))
        finite = (torch.isfinite(ball_pre).all(-1) & torch.isfinite(racket_pre).all(-1)
                  & torch.isfinite(racket_post).all(-1))
        for key, score in [('clean_contact', clean), ('contact_angle', clean*angle_score)]:
            score = torch.where(finite, torch.nan_to_num(score, nan=0., posinf=0., neginf=0.), 0.)
            self.scores[key].copy_(torch.where(accepted, score, self.scores[key]))
