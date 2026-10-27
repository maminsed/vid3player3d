"""Per-environment tracking errors, kept on the input tensor's CPU/CUDA device."""
import math

import torch

from embodied_pose.utils.imitation_asset import BODY_PARENTS


def conjugate(q):
    return torch.cat((-q[..., :3], q[..., 3:]), dim=-1)


def multiply(a, b):
    av, bv = a[..., :3], b[..., :3]
    return torch.cat((a[..., 3:] * bv + b[..., 3:] * av + torch.cross(av, bv, dim=-1),
                      a[..., 3:] * b[..., 3:] - (av * bv).sum(-1, keepdim=True)), -1)


def rotate(q, v):
    v = v.expand_as(q[..., :3])
    uv = torch.cross(q[..., :3], v, dim=-1)
    return v + 2 * (q[..., 3:] * uv + torch.cross(q[..., :3], uv, dim=-1))


def body_tracking_errors(actual_pos, actual_rot, target_pos, target_rot):
    delta = actual_pos - target_pos
    # Canonical 24-body SMPL order, shared with fixed-asset validation.
    parents = BODY_PARENTS[1:]
    actual_local = multiply(conjugate(actual_rot[:, parents]), actual_rot[:, 1:])
    target_local = multiply(conjugate(target_rot[:, parents]), target_rot[:, 1:])
    relative = multiply(conjugate(target_local), actual_local)
    angles = 2 * torch.atan2(relative[..., :3].norm(dim=-1), relative[..., 3].abs())
    return {
        'root_error_m': delta[:, 0].norm(dim=-1),
        'root_xy_error_m': delta[:, 0, :2].norm(dim=-1),
        'body_position_error_m': delta.norm(dim=-1).mean(-1),
        'root_aligned_body_error_m': (delta - delta[:, :1]).norm(dim=-1).mean(-1),
        'joint_rotation_error_deg': angles.mean(-1) * (180 / math.pi),
    }


def racket_tracking_errors(actual_pos, actual_rot, target_pos, target_rot, offset, normal):
    actual_normal, target_normal = rotate(actual_rot, normal), rotate(target_rot, normal)
    cross = torch.cross(actual_normal, target_normal, dim=-1).norm(dim=-1)
    dot = (actual_normal * target_normal).sum(-1)
    actual_center = actual_pos + rotate(actual_rot, offset)
    target_center = target_pos + rotate(target_rot, offset)
    return {
        'racket_face_error_deg': torch.atan2(cross, dot) * (180 / math.pi),
        'racket_center_error_m': (actual_center - target_center).norm(dim=-1),
    }
