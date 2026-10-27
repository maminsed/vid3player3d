"""Fixed SMPL asset contracts, independent of IsaacGym for offline validation."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


BODY_NAMES = [
    'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee',
    'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax',
    'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder',
    'R_Elbow', 'R_Wrist', 'R_Hand',
]
BODY_PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10,
                11, 12, 11, 14, 15, 16, 17, 11, 19, 20, 21, 22]


def read_fixed_asset(path):
    """Accept the 69-DOF SMPL layout and an optional fixed right-wrist racket."""
    root = ET.parse(Path(path)).getroot()
    names, parents, offsets, joints = [], [], [], []

    def visit(body, parent):
        index = len(names)
        names.append(body.get('name'))
        parents.append(parent)
        offsets.append(np.fromstring(body.get('pos', '0 0 0'), sep=' '))
        q = np.fromstring(body.get('quat', '1 0 0 0'), sep=' ')
        if (any(k in body.attrib for k in ('euler', 'axisangle', 'xyaxes', 'zaxis'))
                or q.shape != (4,) or not np.allclose(abs(q[0]), 1)
                or not np.allclose(q[1:], 0)):
            raise ValueError('Fixed SMPL assets require identity body rest rotations')
        joints.extend(j.get('name') for j in body.findall('joint') if j.get('type') != 'free')
        for child in body.findall('body'):
            visit(child, index)

    visit(root.find('worldbody/body'), -1)
    expected_joints = [f'{name}_{axis}' for name in BODY_NAMES[1:] for axis in ('x', 'y', 'z')]
    if names not in (BODY_NAMES, BODY_NAMES + ['Racket']) or joints != expected_joints:
        raise ValueError('Expected 24 SMPL bodies / 69 ordered DOFs and optional final Racket')
    if parents[:24] != BODY_PARENTS:
        raise ValueError('Fixed asset must preserve the canonical SMPL hierarchy')
    if len(names) == 25 and parents[-1] != BODY_NAMES.index('R_Wrist'):
        raise ValueError('Racket must be a fixed attachment to R_Wrist')
    offsets = np.asarray(offsets, dtype=np.float32)
    if offsets.shape != (len(names), 3) or not np.isfinite(offsets).all():
        raise ValueError('Invalid asset body offsets')
    rest = np.zeros((24, 3), dtype=np.float32)
    for i in range(1, 24):
        rest[i] = rest[parents[i]] + offsets[i]
    return names, np.asarray(parents), offsets, rest


def validate_motion_asset(motion_lib, names, parents, offsets, conditioning, scale):
    """Reject body/shape mismatches before silently training inconsistent targets."""
    import torch

    expected = torch.as_tensor(conditioning, device=motion_lib._motion_bodies.device,
                               dtype=motion_lib._motion_bodies.dtype)
    if (motion_lib._motion_bodies.shape[-1] != len(conditioning)
            or not torch.allclose(motion_lib._motion_bodies, expected.expand_as(motion_lib._motion_bodies), atol=1e-5, rtol=0)):
        raise ValueError('Motion conditioning differs from env.asset.gender_beta; retarget the library')
    if not torch.allclose(motion_lib._motion_body_scales, torch.full_like(motion_lib._motion_body_scales, scale), atol=1e-5, rtol=0):
        raise ValueError('Motion scale differs from env.asset.body_scale')
    # Cleaned libraries omit skeleton trees. Check FK against stored global states
    # at the start, middle and end of every sequence, on the library's device.
    starts = motion_lib.length_starts
    lengths = motion_lib._motion_num_frames
    indices = torch.cat((starts, starts + lengths // 2, starts + lengths - 1)).long()
    pos, rot = motion_lib.gts[indices], motion_lib.grs[indices]
    if pos.shape[1:] != (24, 3) or rot.shape[1:] != (24, 4):
        raise ValueError('Motion library must contain 24 humanoid bodies')
    parent = torch.as_tensor(parents[1:24], device=pos.device)
    q = rot[:, parent]
    v = torch.as_tensor(offsets[1:24], device=pos.device).expand(len(indices), -1, -1)
    uv = torch.cross(q[..., :3], v, dim=-1)
    rotated = v + 2 * (q[..., 3:] * uv + torch.cross(q[..., :3], uv, dim=-1))
    if not torch.allclose(pos[:, 1:] - pos[:, parent], rotated, atol=2e-5, rtol=0):
        raise ValueError('Motion skeleton differs from fixed asset; retarget the library')
