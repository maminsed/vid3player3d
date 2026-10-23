"""Evaluation-only snapshots; no simulator imports or changes to collision handling."""
import torch


def racket_ball_snapshot(physics):
    """Copy world-space simulator states before their shared buffers change.

    State layout: position XYZ, quaternion XYZW, linear velocity XYZ (m/s),
    angular velocity XYZ (rad/s). Use the actual body index, including lefties.
    """
    indices = (physics._contact_env_ids, physics._contact_racket_ids)
    racket = torch.cat([field[indices] for field in (
        physics._rigid_body_pos_origin, physics._rigid_body_rot_origin,
        physics._rigid_body_vel_origin, physics._rigid_body_ang_vel_origin)], dim=-1)
    body = torch.cat((physics._rigid_body_pos_origin, physics._rigid_body_rot_origin,
                      physics._rigid_body_vel_origin, physics._rigid_body_ang_vel_origin), dim=-1)
    return {'ball_state': physics._ball_root_states.clone(), 'racket_state': racket,
            'body_state': body}


def rotate_vector_xyzw(quaternion, vector):
    """Rotate a local vector into world space using simulator XYZW quaternions."""
    vector = torch.as_tensor(vector, dtype=quaternion.dtype, device=quaternion.device)
    vector = vector.expand_as(quaternion[..., :3])
    cross = 2 * torch.cross(quaternion[..., :3], vector, dim=-1)
    return vector + quaternion[..., 3:] * cross + torch.cross(quaternion[..., :3], cross, dim=-1)


def racket_geometry(snapshot, com_local, normal_local):
    """Face normal and surface-velocity proxy at the ball center, not contact point."""
    racket, ball = snapshot['racket_state'], snapshot['ball_state']
    com = racket[..., :3] + rotate_vector_xyzw(racket[..., 3:7], com_local)
    velocity = racket[..., 7:10] + torch.cross(
        racket[..., 10:13], ball[..., :3] - com, dim=-1)
    return {'racket_face_normal': rotate_vector_xyzw(racket[..., 3:7], normal_local),
            'racket_com_xyz': com, 'racket_velocity_at_ball_center_mps': velocity}
