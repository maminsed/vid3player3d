"""Align a decoded reset pose to the floor without changing its motion features."""
import torch


@torch.no_grad()
def reset_height_offset(smpl, root_pos, joint_rotmat, betas, clearance=0.002, batch_size=64):
    """Return the upward translation needed to clear the posed proxy mesh.

    Joint centres alone miss the sole surface. Evaluate the skinned mesh only at
    reset, in bounded chunks, and retain one translation for the ensuing motion.
    An already airborne pose is left at its original height.
    """
    if batch_size < 1 or clearance < 0:
        raise ValueError('Ground alignment requires a positive batch size and nonnegative clearance')
    offsets = root_pos.new_zeros(root_pos.shape[0])
    for start in range(0, len(root_pos), batch_size):
        end = min(start + batch_size, len(root_pos))
        rotations = joint_rotmat[start:end]
        output = smpl(
            global_orient=rotations[:, :1], body_pose=rotations[:, 1:],
            betas=betas[start:end], pose2rot=False,
            root_trans=root_pos[start:end], orig_joints=True)
        minimum = output.vertices[:, :, 2].amin(dim=1)
        if not torch.isfinite(minimum).all():
            raise ValueError('Nonfinite reset mesh height')
        offsets[start:end] = (clearance - minimum).clamp_min(0)
    return offsets
