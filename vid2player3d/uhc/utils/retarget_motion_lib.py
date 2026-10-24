"""Retarget corrected MotionLib folders onto a fixed deployed SMPL asset.

Example (from any working directory, in the vid2player3d conda environment)::

    python vid2player3d/uhc/utils/retarget_motion_lib.py \
        --input_dir vid2player3d/data/motion_lib_ik/SCENE \
        --out_dir /tmp/retargeted-SCENE --cfg djokovic_published_v1 --device cpu

CUDA accelerates batched FK, collision-mesh transforms, and SMPL skinning.
Contact IK, SciPy derivatives, plotting and MotionLib serialization use CPU.
No simulation or training runs here. Only load trusted project .pth/.pkl files.
"""

import argparse
import ast
import copy
from dataclasses import asdict
from contextlib import ExitStack
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

PROJECT = Path(__file__).resolve().parents[2]
for directory in (PROJECT, PROJECT.parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

RETARGET_CONFIGS = {
    'djokovic_published_v1': {
        'controller_cfg': 'vid2player/cfg/controller/usopen_train_stage_1/high_level_6.yaml',
        'player': 'djokovic',
        'gender': 'male',
        'grip': 'semi_western',
        'righthand': True,
        # Matches PhysicsMVAEController.create_sim; checked against that source.
        'beta': [-0.9807, 1.4050, -0.4144, 1.4028, -1.3299,
                 2.0045, -1.3108, 0.7475, -0.0924, -0.3262],
        'body_scale': 1.0,
        # Deliberate override even while the controller YAML still says legacy.
        'racket_dynamics': {'model': 'published_reference_v1'},
        'inferred_contact_speed_mps': 0.15,
        'inferred_contact_height_m': 0.08,
    },
}

BODY_NAMES = [
    'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee',
    'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax',
    'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder',
    'R_Elbow', 'R_Wrist', 'R_Hand',
]
FOOT_IDS = [3, 4, 7, 8]


def fingerprint(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'size_bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def resolve_settings(name):
    """Snapshot the source config plus the deliberate racket override."""
    from vid2player.utils.racket_dynamics import resolve_racket_dynamics
    from vid2player.utils.tennis_contacts import contact_materials

    result = copy.deepcopy(RETARGET_CONFIGS[name])
    controller_path = PROJECT / result['controller_cfg']
    controller = yaml.safe_load(controller_path.read_text())
    physics = controller['env']['physics']
    v2p = controller['env']['vid2player']
    for key in ('player', 'grip', 'righthand'):
        if v2p.get(key) != result[key]:
            raise ValueError(f'Preset {key} differs from controller; define a matching preset')
    # Avoid silently drifting away from the controller's hard-coded body shape.
    body_source = PROJECT / 'vid2player/env/tasks/physics_mvae_controller.py'
    shapes = {}
    for node in ast.walk(ast.parse(body_source.read_text())):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                    and target.value.id == 'smpl_beta_dict'):
                key = target.slice.value if isinstance(target.slice, ast.Index) else target.slice
                shapes[ast.literal_eval(key)] = ast.literal_eval(node.value)
    if shapes.get(result['player']) != result['beta']:
        raise ValueError('Controller body shape changed; update the retargeting preset')
    base_path = PROJECT / 'vid2player/cfg/im' / (physics['config'] + '.yaml')
    base = yaml.safe_load(base_path.read_text())
    asset = dict(base['env']['asset'])
    for key in ('assetRoot', 'assetFileName', 'angular_damping', 'max_angular_velocity'):
        if key in physics:
            asset[key] = physics[key]
    result['asset_path'] = str((PROJECT / asset['assetRoot'] / asset['assetFileName']).resolve())
    result['source_racket_dynamics'] = asdict(resolve_racket_dynamics(v2p.get('racket_dynamics')))
    # Keep custom numeric values in the source YAML, selecting the requested model.
    racket = dict(v2p.get('racket_dynamics', {}))
    racket.update(result['racket_dynamics'])
    result['racket_dynamics'] = asdict(resolve_racket_dynamics(racket))
    result['tennis_contacts'] = asdict(contact_materials(v2p))
    effective_env = copy.deepcopy(base['env'])
    for key in ('controlFrequencyInv', 'residual_force_scale', 'residual_torque_scale',
                'has_self_collision', 'has_racket_collision', 'kp_scale', 'kd_scale',
                'default_humanoid_mass', 'pd_tar_lim'):
        if key in physics:
            effective_env[key] = physics[key]
    effective_env['plane']['restitution'] = physics.get('plane_restitution', effective_env['plane']['restitution'])
    materials = result['tennis_contacts']
    if materials['model'] != 'legacy':
        effective_env['plane'].update(staticFriction=materials['court_friction'],
                                      dynamicFriction=materials['court_friction'],
                                      restitution=materials['court_restitution'])
    sim = copy.deepcopy(base['sim'])
    sim['dt'] = physics.get('dt', sim.get('dt', 1. / 60.))
    sim['substeps'] = physics.get('substeps', sim.get('substeps', 2))
    for key in ('solver_type', 'num_position_iterations', 'num_velocity_iterations',
                'contact_offset', 'rest_offset', 'bounce_threshold_velocity',
                'max_depenetration_velocity', 'friction_offset_threshold', 'friction_correlation_distance'):
        if key in physics:
            sim['physx'][key] = physics[key]
    result['effective_physics'] = {
        'sim': sim, 'plane': effective_env['plane'],
        'control_frequency_inv': effective_env['controlFrequencyInv'],
        'control_dt_s': sim['dt'] * effective_env['controlFrequencyInv'],
        'residual_force_scale': effective_env.get('residual_force_scale', 0),
        'kp_scale': effective_env.get('kp_scale', 1.),
        'kd_scale': effective_env.get('kd_scale', effective_env.get('kp_scale', 1.)),
        'default_humanoid_mass_kg': effective_env.get('default_humanoid_mass', 90.),
        'drive_scaling': 'XML stiffness/damping * actual humanoid mass / default_humanoid_mass * kp_scale/kd_scale',
        'has_self_collision': effective_env.get('has_self_collision', False),
        'has_racket_collision': effective_env.get('has_racket_collision', False),
        'asset_angular_damping': asset.get('angular_damping', .01),
        'asset_max_angular_velocity_radps': (result['racket_dynamics']['max_angular_velocity']
                                            if result['racket_dynamics']['model'] != 'legacy'
                                            else asset.get('max_angular_velocity', 100.)),
        'note': 'Recorded training target; no simulator properties were measured or modified',
    }
    result['physics_overrides'] = physics
    result['base_physics_config'] = base
    result['controller_config'] = controller
    result['source_files'] = [fingerprint(p) for p in (
        controller_path, base_path, body_source,
        PROJECT / 'vid2player/utils/racket_dynamics.py',
        PROJECT / 'vid2player/utils/tennis_contacts.py')]
    result['grounding'] = 'Per-frame lowest SMPL sole matches the original, including source penetration; no ground clamp'
    result['height_loader_contract'] = (
        'Source sole heights preserved: use get_motion_state(adjust_height=False). '
        'min_verts_h retains the original mean minimum SMPL skin height semantics; '
        'adjust_height=True applies an additional translation and can undo source sole alignment.')
    return result


def load_asset(path, device):
    """Read identity-rest-frame humanoid offsets and the fixed racket attachment."""
    import torch
    import trimesh
    from vid2player3d.poselib.poselib.skeleton.skeleton3d import SkeletonTree

    path = Path(path)
    doc = ET.parse(path).getroot()
    names, parents, offsets, bodies = [], [], [], {}

    def visit(body, parent):
        name = body.attrib['name']
        # Poselib's XML reader ignores rest rotations; reject unsupported assets.
        if any(key in body.attrib for key in ('euler', 'axisangle', 'xyaxes', 'zaxis')):
            raise ValueError(f'Unsupported rest orientation on {name}')
        q = np.fromstring(body.get('quat', '1 0 0 0'), sep=' ')
        if q.shape != (4,) or not np.allclose(abs(q[0]), 1) or not np.allclose(q[1:], 0):
            raise ValueError(f'Nonidentity rest orientation on {name}')
        index = len(names)
        names.append(name)
        parents.append(parent)
        offsets.append(np.fromstring(body.get('pos', '0 0 0'), sep=' '))
        bodies[name] = body
        for child in body.findall('body'):
            visit(child, index)

    visit(doc.find('worldbody/body'), -1)
    if names != BODY_NAMES + ['Racket'] or parents[-1] != BODY_NAMES.index('R_Wrist'):
        raise ValueError('Expected the 24-body right-handed SMPL asset with a fixed wrist racket')
    if bodies['Racket'].find('joint') is not None:
        raise ValueError('Racket must be a fixed attachment')
    tree = SkeletonTree(BODY_NAMES, torch.tensor(parents[:24], dtype=torch.int32),
                        torch.tensor(np.array(offsets[:24]), dtype=torch.float32))
    mesh_assets = {m.get('name'): m for m in doc.findall('asset/mesh')}
    files, feet = [fingerprint(path)], []
    mesh_dir = doc.find('compiler').get('meshdir', '')
    # Fingerprint all physical meshes, not only the ones used for grounding.
    for mesh in mesh_assets.values():
        files.append(fingerprint(path.parent / mesh_dir / mesh.get('file')))
    for index in FOOT_IDS:
        points = []
        for geom in bodies[BODY_NAMES[index]].findall('geom'):
            if geom.get('type') != 'mesh':
                raise ValueError('Foot grounding currently requires mesh collision geometry')
            mesh = mesh_assets[geom.get('mesh')]
            loaded = trimesh.load(path.parent / mesh_dir / mesh.get('file'), force='mesh', process=False)
            vertices = np.asarray(loaded.vertices) * np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
            if any(key in geom.attrib for key in ('euler', 'axisangle', 'xyaxes', 'zaxis', 'fromto')):
                raise ValueError('Unsupported foot geom transform')
            q = np.fromstring(geom.get('quat', '1 0 0 0'), sep=' ')
            vertices = Rotation.from_quat(q[[1, 2, 3, 0]]).apply(vertices)
            vertices += np.fromstring(geom.get('pos', '0 0 0'), sep=' ')
            points.append(vertices)
        feet.append(torch.tensor(np.concatenate(points), dtype=torch.float32, device=device))
    return tree, feet, np.array(offsets[-1]), files


def fk(local, root, offsets, parents):
    """Batched torch FK, with XYZW quaternions in the saved MotionLib order."""
    import torch
    from vid2player3d.poselib.poselib.core.rotation3d import quat_mul, quat_rotate

    positions, rotations = [root], [local[:, 0]]
    for joint in range(1, 24):
        parent = int(parents[joint])
        positions.append(positions[parent] + quat_rotate(
            rotations[parent], offsets[joint].expand(len(local), 3)))
        rotations.append(quat_mul(rotations[parent], local[:, joint]))
    return torch.stack(positions, 1), torch.stack(rotations, 1)


def foot_heights(positions, rotations, feet):
    import torch
    from vid2player3d.poselib.poselib.core.rotation3d import quat_rotate

    heights = []
    for index, vertices in zip(FOOT_IDS, feet):
        rotated = quat_rotate(rotations[:, index, None].expand(-1, len(vertices), -1),
                              vertices[None].expand(len(positions), -1, -1))
        heights.append((rotated[..., 2] + positions[:, index, None, 2]).amin(1))
    return torch.stack(heights, 1).reshape(-1, 2, 2).amin(2)


def continuous_quaternions(quaternions):
    """Normalize and choose consecutive quaternion hemispheres per joint."""
    q = np.asarray(quaternions, dtype=np.float64).copy()
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.isfinite(q).all() or np.any(norm < 1e-8):
        raise ValueError('Invalid source quaternions')
    q /= norm
    signs = np.where((q[1:] * q[:-1]).sum(-1) < 0, -1., 1.)
    q[1:] *= np.cumprod(signs, axis=0)[..., None]
    return q.astype(np.float32)


def contact_probabilities(source, name, positions, fps, settings):
    saved = source.get('sequences', {}).get(name, {}).get('ik', {})
    if 'stationary_probability' in saved:
        probabilities = np.asarray(saved['stationary_probability'])
        label = 'saved GVHMR stationary probabilities (predicted, not measured)'
    else:
        feet = positions[:, FOOT_IDS]
        speed = np.linalg.norm(np.gradient(feet, axis=0), axis=-1) * fps
        low = feet[..., 2] <= feet[..., 2].min() + settings['inferred_contact_height_m']
        probabilities = (low & (speed < settings['inferred_contact_speed_mps'])).astype(float)
        label = 'inferred from source ankle/toe speed and relative height (not measured)'
    if (probabilities.shape != (len(positions), 4) or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))):
        raise ValueError(f'Invalid saved contact probabilities for {name}')
    return probabilities, label


def rotation_steps(q):
    q = np.asarray(q, dtype=float)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    dots = np.abs((q[1:] * q[:-1]).sum(-1)).clip(0, 1)
    return np.concatenate([np.full_like(dots[:1], np.nan), np.degrees(2 * np.arccos(dots))])


def summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return ({'count': int(values.size), 'mean': float(values.mean()),
             'p50': float(np.percentile(values, 50)), 'p95': float(np.percentile(values, 95)),
             'max': float(values.max()), 'min': float(values.min())} if values.size else {'count': 0})


class SkinMetrics:
    """Chunked GPU/CPU skinning, using model translation consistent with pelvis."""
    def __init__(self, settings, device, batch_size):
        import torch
        from vid2player3d.uhc.smpllib.smpl_parser import SMPL_Parser, SMPL_BONE_ORDER_NAMES

        self.model = SMPL_Parser(str(PROJECT / 'data/smpl'), gender=settings['gender']).to(device)
        self.beta = torch.tensor([settings['beta']], dtype=torch.float32, device=device)
        self.batch_size = batch_size
        self.device = device
        self.source_models = {settings['gender']: self.model}
        self.to_smpl = [BODY_NAMES.index(n) for n in SMPL_BONE_ORDER_NAMES]
        with torch.no_grad():
            _, joints = self.model.get_joints_verts(torch.zeros(1, 72, device=device), self.beta)
        self.pelvis = joints[0, 0].detach().cpu().numpy()

    def measure(self, local, root, xml_positions):
        import torch

        pose = Rotation.from_quat(local[:, self.to_smpl].reshape(-1, 4)).as_rotvec().reshape(-1, 72)
        minimum, feet, gaps = [], [], []
        weights = self.model.lbs_weights
        masks = [weights[:, [a, b]].sum(1) > .5 for a, b in ((7, 10), (8, 11))]
        with torch.no_grad():
            for start in range(0, len(root), self.batch_size):
                end = min(len(root), start + self.batch_size)
                vertices, joints = self.model.get_joints_verts(
                    torch.tensor(pose[start:end], dtype=torch.float32, device=self.device), self.beta,
                    torch.tensor(root[start:end] - self.pelvis, dtype=torch.float32, device=self.device))
                minimum.append(vertices[..., 2].amin(1).cpu().numpy())
                feet.append(torch.stack([vertices[:, mask, 2].amin(1) for mask in masks], 1).cpu().numpy())
                gaps.append(np.linalg.norm(joints.cpu().numpy() - xml_positions[start:end, self.to_smpl], axis=-1))
        return pose, np.concatenate(minimum), np.concatenate(feet), np.concatenate(gaps)

    def measure_source_feet(self, motion_lib, index):
        """Original skinned soles at saved positions, without loader height adjustment.

        Original collision meshes are not retained in a cleaned MotionLib.
        Use its actual gender/beta/scale, matching the blue video ghost, and
        label these as SMPL skin measurements rather than collision geometry.
        """
        import torch
        from vid2player3d.uhc.smpllib.smpl_parser import SMPL_Parser

        body = motion_lib._motion_bodies[index].cpu().numpy()
        gender = {0: 'neutral', 1: 'male', 2: 'female'}[int(body[0])]
        if gender not in self.source_models:
            self.source_models[gender] = SMPL_Parser(str(PROJECT / 'data/smpl'), gender=gender).to(self.device)
        model = self.source_models[gender]
        begin = int(motion_lib.length_starts[index])
        count = int(motion_lib._motion_num_frames[index])
        local = motion_lib.lrs[begin:begin+count].cpu().numpy()
        pose = Rotation.from_quat(local[:, self.to_smpl].reshape(-1, 4)).as_rotvec().reshape(count, 72)
        root = motion_lib.gts[begin:begin+count, 0].cpu().numpy()
        masks = [model.lbs_weights[:, [a, b]].sum(1) > .5 for a, b in ((7, 10), (8, 11))]
        heights = []
        with torch.no_grad():
            for start in range(0, count, self.batch_size):
                stop = min(count, start + self.batch_size)
                vertices, _ = model.get_joints_verts(
                    torch.tensor(pose[start:stop], dtype=torch.float32, device=self.device),
                    torch.tensor(body[None, 1:11], dtype=torch.float32, device=self.device),
                    root_trans=torch.tensor(root[start:stop], dtype=torch.float32, device=self.device),
                    root_scale=torch.full((stop-start,), float(motion_lib._motion_body_scales[index]), device=self.device))
                heights.append(torch.stack([vertices[:, mask, 2].amin(1) for mask in masks], 1).cpu().numpy())
        return np.concatenate(heights)


def retarget_sequence(m, index, tree, feet, attachment, contacts, settings, device, skin, disable_ik):
    import torch
    from vid2player3d.uhc.utils.contact_ik import correct_contact_poses
    from vid2player3d.poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
    from vid2player3d.poselib.poselib.core.rotation3d import quat_rotate

    start, count = int(m.length_starts[index]), int(m._motion_num_frames[index])
    fps = float(m._motion_fps[index])
    if count < 2 or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Each motion must have at least two frames and positive FPS')
    name = str(m._motion_seq_names[index])
    original_q = m.lrs[start:start + count].cpu().numpy()
    if original_q.shape != (count, 24, 4):
        raise ValueError('Input library must use the original 24-body SMPL ordering')
    q = continuous_quaternions(original_q)
    source_positions = m.gts[start:start + count].cpu().numpy()
    source_root = source_positions[:, 0].copy()
    if not np.isfinite(source_positions).all():
        raise ValueError('Nonfinite source positions')
    root = source_root.copy()
    offsets, parents = tree.local_translation.to(device), tree.parent_indices

    def geometry(rotations, translation):
        chunks = []
        for begin in range(0, len(translation), skin.batch_size):
            end = begin + skin.batch_size
            pos, rot = fk(torch.tensor(rotations[begin:end], device=device),
                          torch.tensor(translation[begin:end], device=device), offsets, parents)
            chunks.append((pos, rot, foot_heights(pos, rot, feet)))
        return tuple(torch.cat([chunk[i] for chunk in chunks]) for i in range(3))

    positions, rotations, heights = geometry(q, root)
    raw_height = heights.cpu().numpy().copy()
    original_skin_feet = skin.measure_source_feet(m, index)
    _, _, raw_skin_feet, _ = skin.measure(q, root, positions.cpu().numpy())
    shift = original_skin_feet.min(1) - raw_skin_feet.min(1)
    root[:, 2] += shift
    # Translation does not change rotations or relative geometry. Reuse FK and
    # collision heights rather than transforming every foot vertex again.
    positions[:, :, 2] += torch.as_tensor(shift, device=device)[:, None]
    heights += torch.as_tensor(shift, device=device)[:, None]
    before_pos, before_rot = positions.cpu().numpy(), rotations.cpu().numpy()
    probabilities, contact_source = contact_probabilities(contacts, name, source_positions, fps, settings)

    # The existing leg solver is in SMPL order. Feed XML rest joints in that order.
    to_smpl = skin.to_smpl
    from_smpl = np.argsort(to_smpl)
    rest = np.zeros((24, 3))
    for joint in range(1, 24):
        rest[joint] = rest[int(parents[joint])] + tree.local_translation[joint].numpy()
    smpl_parents = np.array([-1 if int(parents[j]) < 0 else int(from_smpl[int(parents[j])]) for j in to_smpl])
    aa = Rotation.from_quat(q[:, to_smpl].reshape(-1, 4)).as_rotvec().reshape(count, 72)
    corrected, ik = correct_contact_poses(aa, root, rest[to_smpl], smpl_parents, probabilities,
                                        heights.cpu().numpy(), fps=fps, enabled=not disable_ik)
    corrected_q = Rotation.from_rotvec(corrected.reshape(-1, 3)).as_quat().reshape(count, 24, 4)[:, from_smpl]
    # Keep unmodified joints bit-exact, including root, collar and all upper-body joints.
    q_out = q.copy()
    leg_ids = [1, 2, 3, 5, 6, 7]
    q_out[:, leg_ids] = corrected_q[:, leg_ids]
    q_out = continuous_quaternions(q_out)
    positions, rotations, heights = geometry(q_out, root)
    pose, minimum, skin_feet, skin_gap = skin.measure(q_out, root, positions.cpu().numpy())
    # Restore the original sole-height envelope after IK as well. Negative
    # source heights are intentional here: upstream grounding is handled later.
    final_lift = original_skin_feet.min(1) - skin_feet.min(1)
    root[:, 2] += final_lift
    minimum += final_lift
    skin_feet += final_lift[:, None]
    positions[:, :, 2] += torch.as_tensor(final_lift, device=device)[:, None]
    heights += torch.as_tensor(final_lift, device=device)[:, None]
    pos, rot, height = positions.cpu().numpy(), rotations.cpu().numpy(), heights.cpu().numpy()
    state = SkeletonState.from_rotation_and_root_translation(tree, torch.from_numpy(q_out),
                                                            torch.from_numpy(root), is_local=True)
    motion = SkeletonMotion.from_skeleton_state(state, fps=fps).to_dict()
    motion.update(seq_name=name, seq_idx=int(m._motion_seq_ids[index]), pose_aa=pose,
                  trans=root - skin.pelvis, root_trans=root, beta=np.array(settings['beta']),
                  beta_idx=0, gender=settings['gender'], body_scale=settings['body_scale'],
                  min_verts_h=float(minimum.mean()), __name__='SkeletonMotion')

    racket_pos = (positions[:, 22] + quat_rotate(rotations[:, 22],
                  torch.tensor(attachment, dtype=torch.float32, device=device).expand(count, 3))).cpu().numpy()
    racket_rot = rot[:, 22]
    normal = quat_rotate(rotations[:, 22], torch.tensor(
        [0., 2 ** -.5, 2 ** -.5], device=device).expand(count, 3)).cpu().numpy()
    chest = Rotation.from_quat(rot[:, 11])
    relative_pos = chest.inv().apply(racket_pos - pos[:, 11])
    relative_rot = (chest.inv() * Rotation.from_quat(racket_rot)).as_quat()
    relative_normal = chest.inv().apply(normal)
    correction = np.degrees((Rotation.from_quat(q.reshape(-1, 4)).inv() *
                             Rotation.from_quat(q_out.reshape(-1, 4))).magnitude()).reshape(count, 24)
    source_foot = source_positions[:, FOOT_IDS].reshape(count, 2, 2, 3)
    target_foot = pos[:, FOOT_IDS].reshape(count, 2, 2, 3)
    before_foot = before_pos[:, FOOT_IDS].reshape(count, 2, 2, 3)
    # Report ankle and toe speeds separately; contact-conditioned summaries select support points.
    def speed(points):
        return np.r_[np.full((1,) + points.shape[1:-1], np.nan),
                     np.linalg.norm(np.diff(points[..., :2], axis=0), axis=-1) * fps]

    data = {
        'time_s': np.arange(count) / fps,
        'source_root_m': source_root, 'target_root_m': root,
        'root_z_change_m': root[:, 2] - source_root[:, 2],
        'initial_root_z_correction_m': shift,
        'post_ik_root_z_correction_m': final_lift,
        'lowest_sole_height_error_m': skin_feet.min(1) - original_skin_feet.min(1),
        'joint_correction_deg': correction,
        'source_collar_step_deg': rotation_steps(q[:, 19]),
        'target_collar_step_deg': rotation_steps(q_out[:, 19]),
        'target_joint_step_deg': rotation_steps(q_out),
        'target_joint_speed_radps': np.radians(rotation_steps(q_out)) * fps,
        'raw_collision_foot_height_m': raw_height,
        'collision_foot_height_m': height,
        'source_skin_foot_height_m': original_skin_feet,
        'skin_foot_height_change_m': skin_feet - original_skin_feet,
        'skin_foot_height_m': skin_feet, 'skin_min_height_m': minimum,
        'skin_xml_joint_gap_m': skin_gap,
        'stationary_probability': probabilities, 'contact_mode': ik['contact_mode'],
        'ik_limited': ik['limited'], 'ik_anchor_residual_m': ik['anchor_residual_m'],
        'source_foot_speed_mps': speed(source_foot),
        'pre_ik_foot_speed_mps': speed(before_foot), 'target_foot_speed_mps': speed(target_foot),
        'racket_center_m': racket_pos, 'racket_rotation_xyzw': racket_rot,
        'racket_normal': normal, 'racket_chest_relative_center_m': relative_pos,
        'racket_chest_relative_normal': relative_normal,
        'racket_speed_mps': np.r_[np.nan, np.linalg.norm(np.diff(racket_pos, axis=0), axis=-1) * fps],
        'racket_step_deg': rotation_steps(racket_rot),
        'racket_chest_relative_step_deg': rotation_steps(relative_rot),
        'racket_chest_relative_normal_step_deg': np.r_[np.nan, np.degrees(np.arccos(
            (relative_normal[1:] * relative_normal[:-1]).sum(-1).clip(-1, 1)))],
        'fk_error_m': np.linalg.norm(state.global_translation.numpy() - pos, axis=-1),
        'quaternion_norm_error': abs(np.linalg.norm(q_out, axis=-1) - 1),
        'bone_length_error_m': abs(np.linalg.norm(pos[:, 1:] - pos[:, parents[1:]], axis=-1)
                                  - np.linalg.norm(tree.local_translation.numpy()[1:], axis=-1)),
    }
    modes = ik['contact_mode']
    stable = (modes[1:] != 0) & (modes[1:] == modes[:-1])
    chosen = (modes[1:] == 2).astype(int)
    selected_before = np.take_along_axis(data['pre_ik_foot_speed_mps'][1:], chosen[..., None], axis=2)[..., 0]
    selected_after = np.take_along_axis(data['target_foot_speed_mps'][1:], chosen[..., None], axis=2)[..., 0]
    result = {
        'name': name, 'seq_idx': int(m._motion_seq_ids[index]), 'frames': count, 'fps': fps,
        'duration_s': (count - 1) / fps, 'contact_source': contact_source,
        'source_body': m._motion_bodies[index].cpu().tolist(),
        'source_body_scale': float(m._motion_body_scales[index]),
        'ground_offset_m': float((shift + final_lift).mean()),
        'initial_ground_offset_m': float(shift.mean()), 'post_ik_lift_m': float(final_lift.mean()),
        'offset_summary_convention': 'Legacy scalar offset fields are sequence means; per-frame corrections are saved in NPZ',
        'mean_skin_min_height_m': float(minimum.mean()),
        'loader_adjust_height_true_z_shift_m': -float(minimum.mean()),
        'ik': ik['summary'],
        'contact_speed_pre_ik_mps': summary(selected_before[stable]),
        'contact_speed_target_mps': summary(selected_after[stable]),
        'metrics': {key: summary(value) for key, value in data.items()},
        'warnings': [],
    }
    predicted_support = probabilities.reshape(count, 2, 2).max(2) >= .8
    result['predicted_support_clearance_m'] = summary(height[predicted_support])
    result['ik']['contact_source'] = contact_source + '; target XML collision foot heights'
    if data['fk_error_m'].max() > 2e-5 or data['bone_length_error_m'].max() > 2e-5:
        raise ValueError('Export FK or bone-length consistency check failed')
    if abs(data['lowest_sole_height_error_m']).max() > 2e-5:
        raise ValueError('Original sole-height preservation check failed')
    if abs(final_lift).max() > .02:
        result['warnings'].append('Post-IK sole alignment exceeds 2 cm on some frames; review contact anchors')
    if np.any(ik['limited']):
        result['warnings'].append('Some IK corrections hit reach or rotation limits; inspect residuals')
    return motion, data, result


METRIC_DEFINITIONS = {
    'conventions': 'Meters, seconds, radians/s, degrees; Z up; XYZW quaternions. Joint order is body_names. Foot sides L/R; point order ankle/toe.',
    'steps': 'Shortest orientation change from preceding frame; first-frame derivatives are NaN. Collar is R_Thorax.',
    'racket': 'Physical XML Racket body origin at head center; fixed to R_Wrist. Face normal is (0,1,1)/sqrt(2) in its body frame.',
    'grounding': 'Per-frame root Z aligns the lowest target SMPL sole to the lowest original SMPL sole, before and after IK. No floor clamp: original penetration and airborne clearance are retained. Collision heights are diagnostics only.',
    'skin': 'SMPL skin reconstructed from male/beta pose and pelvis translation; skin/XML gap measures geometry rounding/model differences.',
    'source_skin_foot_height_m': 'Original L/R sole minima over SMPL vertices weighted >0.5 to ankle/toe. Uses original gender, beta, scale and saved pelvis positions, without adjust_height. Same body as blue ghost; not original collision geometry.',
    'skin_foot_height_change_m': 'Target minus original L/R SMPL sole height in meters; positive means the target sole is higher. Includes changed proportions, grounding and IK.',
    'contact': 'Modes 0=swing, 1=ankle, 2=toe, 3=flat. Summary sliding uses support point XY speed only inside unchanged nonzero modes.',
    'ik_anchor_residual_m': 'Measured before the final per-frame sole alignment; post_ik_root_z_correction_m records that extra translation.',
    'lowest_sole_height_error_m': 'Lowest target minus lowest original SMPL sole height each frame. Must be zero within numerical precision; individual left/right heights need not both match with one root translation.',
    'missing': 'NaN denotes undefined derivatives or absent contact anchors, not invalid motion values. JSON summaries exclude NaNs.',
    'scope': 'Kinematic reference diagnostics only; no physical tracking, balance, or ball contact measurements.',
}


def plot_metrics(arrays, records, output):
    # The default home cache may be read-only on compute hosts. Reusing a
    # writable cache avoids scanning all installed fonts for every invocation.
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / f'vid3player-matplotlib-{os.getuid()}'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)
    for record in records:
        prefix = record['array_prefix']
        def get(key):
            return arrays[prefix + key]
        t = get('time_s')
        label = record['name'] if len(records) > 1 else ''
        axes[0, 0].plot(t, get('root_z_change_m') * 100, label=label or 'Root Z change')
        for side, name in enumerate(('L', 'R')):
            color = ('tab:blue', 'tab:orange')[side]
            axes[0, 1].plot(t, get('collision_foot_height_m')[:, side] * 100,
                            color=color, label=f'{label} {name} result collision')
            axes[0, 1].plot(t, get('source_skin_foot_height_m')[:, side] * 100, '--',
                            color=color, alpha=.8, label=f'{label} {name} original SMPL')
            axes[0, 1].plot(t, get('skin_foot_height_m')[:, side] * 100, ':',
                            color=color, alpha=.8, label=f'{label} {name} result SMPL')
            modes = get('contact_mode')[:, side]
            chosen = (modes == 2).astype(int)
            stable = np.r_[False, (modes[1:] != 0) & (modes[1:] == modes[:-1])]
            for key, style in (('pre_ik_foot_speed_mps', '--'), ('target_foot_speed_mps', '-')):
                velocity = get(key)[np.arange(len(t)), side, chosen]
                axes[1, 0].plot(t, np.where(stable, velocity, np.nan), style,
                                label=f'{label} {name} ' + ('before' if style == '--' else 'after'))
        axes[1, 1].plot(t, get('target_collar_step_deg'), label=label or 'Right collar')
        axes[1, 1].plot(t, get('joint_correction_deg').max(1), '--', label=label or 'Max joint correction')
        axes[2, 0].plot(t, get('racket_speed_mps'), label=label or 'Racket head')
        axes[2, 1].plot(t, get('racket_chest_relative_step_deg'), label=label or 'Full orientation')
        axes[2, 1].plot(t, get('racket_chest_relative_normal_step_deg'), '--', label=label or 'Face normal')
    axes[0, 1].axhline(0, color='gray', linewidth=.8)
    titles = [('Root Z correction for source sole alignment', 'Z change (cm)'), ('Foot clearance: original vs result', 'Sole height (cm)'),
              ('Predicted-contact support point sliding', 'XY speed (m/s)'), ('Collar motion / retarget correction', 'Degrees'),
              ('Racket-center speed', 'Speed (m/s)'), ('Chest-relative racket rotation', 'Degrees/frame')]
    for ax, (title, ylabel) in zip(axes.flat, titles):
        ax.set(title=title, xlabel='Sequence time (s)', ylabel=ylabel)
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    # Avoid magnifying float32 subtraction noise in the constant Z translation.
    shifts = np.concatenate([arrays[r['array_prefix'] + 'root_z_change_m'] * 100 for r in records])
    axes[0, 0].set_ylim(min(0., shifts.min()) - .5, max(0., shifts.max()) + .5)
    fig.suptitle('Retargeted reference metrics — kinematic, contacts are predictions')
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main(argv=None):
    started = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_dir', '--input-dir', type=Path, required=True)
    parser.add_argument('--out_dir', '--out-dir', type=Path, required=True)
    parser.add_argument('--cfg', choices=sorted(RETARGET_CONFIGS), default='djokovic_published_v1')
    parser.add_argument('--device', default='auto', help='auto (CUDA if available), cpu, cuda, or cuda:N')
    parser.add_argument('--batch-size', type=int, default=128, help='Maximum frames per FK/skinning batch')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--disable-ik', action='store_true', help='Retain source joint rotations; still retarget geometry and ground')
    parser.add_argument('--verbose', action='store_true',
                        help='Save comparison.mp4: faded blue source, orange result, and frame numbers (kinematic playback)')
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.cpu_threads < 1:
        parser.error('Batch size and CPU threads must be positive')
    source, output = args.input_dir.resolve(), args.out_dir.resolve()
    parts = sorted(source.glob('mlib_part_*.pth'))
    if not parts:
        parser.error('Input folder contains no mlib_part_*.pth files')
    if source == output or source in output.parents or output in source.parents:
        parser.error('Input and output directories must be separate, non-nested folders')
    if output.exists() and any(output.iterdir()):
        parser.error('Output folder must be empty; choose a new path')
    # Isaac Gym must precede torch, including unpickling the saved MotionLib.
    from isaacgym import gymapi  # noqa: F401
    import torch
    import joblib
    from vid2player3d.embodied_pose.utils.motion_lib import MotionLib

    torch.set_num_threads(args.cpu_threads)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda'):
        parser.error('Only CPU and CUDA devices are supported')
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable; use --device cpu')
    settings = resolve_settings(args.cfg)
    tree, feet, attachment, asset_files = load_asset(settings['asset_path'], device)
    skin = SkinMetrics(settings, device, args.batch_size)
    contact_path = source / 'correction_diagnostics.pkl'
    contacts = joblib.load(contact_path) if contact_path.exists() else {}
    provenance = [fingerprint(p) for p in parts]
    for filename in ('args.yml', 'args.yaml', 'correction_diagnostics.pkl', 'shape_data.pkl', 'seq_name_splits.pkl'):
        if (source / filename).is_file():
            provenance.append(fingerprint(source / filename))
    settings['source_files'] += asset_files + [fingerprint(PROJECT / 'data/smpl/SMPL_MALE.pkl'),
        fingerprint(Path(__file__)), fingerprint(PROJECT / 'uhc/utils/contact_ik.py'),
        fingerprint(PROJECT / 'embodied_pose/utils/motion_lib.py'),
        fingerprint(PROJECT / 'poselib/poselib/skeleton/skeleton3d.py'),
        fingerprint(PROJECT / 'poselib/poselib/core/rotation3d.py')]
    manifest = {'schema_version': 1, 'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'device': str(device), 'torch_version': str(torch.__version__), 'numpy_version': np.__version__,
                'package_versions': {name: version(name) for name in ('scipy', 'smplx', 'trimesh', 'matplotlib', 'joblib', 'PyYAML')},
                'settings': settings, 'inputs': provenance,
                'body_names': BODY_NAMES, 'quaternion_order': 'xyzw', 'units': 'meters; seconds',
                'training_pipeline_updated': False}
    timings = {'setup_s': perf_counter() - started, 'retarget_s': 0., 'video_s': 0.,
               'library_write_and_validation_s': 0.}
    manifest['timings'] = timings
    records, arrays, splits, names_seen = [], {}, {}, set()
    # Stage everything beside the output, so failures do not leave a usable-looking partial library.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.retarget-', dir=output.parent) as temporary, ExitStack() as resources:
        staging = Path(temporary)
        video = None
        if args.verbose:
            from vid2player3d.uhc.utils.retarget_motion_video import ComparisonVideo
            video = resources.enter_context(ComparisonVideo(staging / 'comparison.mp4', device, args.batch_size,
                                                           models=skin.source_models))
            settings['source_files'].append(fingerprint(Path(__file__).with_name('retarget_motion_video.py')))
        for part_index, path in enumerate(parts):
            m = torch.load(path, map_location='cpu')
            if m._num_dof != 69 or list(m._dof_body_ids) != list(range(1, 24)):
                raise ValueError('Expected the original SMPL library with 69 joint DOFs')
            if list(m._dof_offsets) != list(range(0, 70, 3)):
                raise ValueError('Unsupported source DOF mapping')
            if int(m._motion_num_frames.sum()) != len(m.gts):
                raise ValueError('Source frame boundaries do not cover the library')
            starts = m._motion_num_frames.cumsum(0) - m._motion_num_frames
            if not torch.equal(m.length_starts, starts) or len(m.lrs) != len(m.gts):
                raise ValueError('Inconsistent source sequence boundaries')
            motions = {}
            part_records = []
            for index in range(m.num_motions()):
                phase_started = perf_counter()
                motion, data, record = retarget_sequence(m, index, tree, feet, attachment, contacts,
                                                       settings, device, skin, args.disable_ik)
                timings['retarget_s'] += perf_counter() - phase_started
                if record['name'] in names_seen:
                    raise ValueError('Duplicate sequence name across library parts')
                names_seen.add(record['name'])
                prefix = f'seq_{len(records):05d}__'
                record.update(array_prefix=prefix, output_part=path.name)
                if video is not None:
                    phase_started = perf_counter()
                    record['video'] = video.add_sequence(m, index, motion, data, settings)
                    timings['video_s'] += perf_counter() - phase_started
                arrays.update({prefix + key: value for key, value in data.items()})
                motions[record['name']] = motion
                records.append(record)
                part_records.append(record)
                print(f"Retargeted {record['name']}: {record['frames']} frames, mean Z correction {record['ground_offset_m']:.4f} m", flush=True)
            phase_started = perf_counter()
            result = MotionLib(motions, m._dof_body_ids, m._dof_offsets,
                               m._key_body_ids.cpu().tolist(), device='cpu', clean_up=True)
            result._motion_weights = m._motion_weights.cpu().clone()
            result.retarget_metadata = {'cfg': args.cfg, 'grounded': False, 'source_sole_heights_preserved': True,
                                        'racket_dynamics': settings['racket_dynamics'],
                                        'height_loader_contract': settings['height_loader_contract']}
            torch.save(result, staging / path.name)
            validate_saved(staging / path.name, m, records=part_records)
            timings['library_write_and_validation_s'] += perf_counter() - phase_started
            splits[int(path.stem.rsplit('_', 1)[1])] = [name[2:] if name.startswith('0-') else name for name in motions]
        beta_key = ','.join(f'{x:.6f}' for x in settings['beta'])
        joblib.dump({'beta_arr': [np.array(settings['beta'])], 'beta_mapping': {beta_key: 0}}, staging / 'shape_data.pkl')
        # Preserve the producer's sequence-label convention when available.
        source_splits = source / 'seq_name_splits.pkl'
        joblib.dump(joblib.load(source_splits) if source_splits.exists() else splits,
                    staging / 'seq_name_splits.pkl')
        np.savez_compressed(staging / 'diagnostics.npz', **arrays)
        report = {'schema_version': 1, 'definitions': METRIC_DEFINITIONS, 'body_names': BODY_NAMES,
                  'settings': settings, 'sequences': records, 'timings': timings}
        phase_started = perf_counter()
        plot_metrics(arrays, records, staging / 'metrics.png')
        timings['plots_s'] = perf_counter() - phase_started
        # Flush the encoder before moving the completed output directory.
        resources.close()
        timings['total_s'] = perf_counter() - started
        (staging / 'diagnostics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        (staging / 'args.yaml').write_text(yaml.safe_dump(manifest, sort_keys=False))
        if output.exists():
            output.rmdir()  # Only an empty user-selected directory is permitted.
        staging.rename(output)
    print(f'Saved and validated {len(records)} sequences in {output}', flush=True)
    print(settings['height_loader_contract'])
    print('Timing: ' + ', '.join(f'{key}={value:.2f}' for key, value in timings.items()), flush=True)


def validate_saved(path, source, records):
    """Exercise actual loader interpolation and independently verify saved velocities."""
    import torch

    result = torch.load(path, map_location='cpu')
    for key in ('gts', 'grs', 'lrs', 'grvs', 'gravs', 'dvs'):
        if not torch.isfinite(getattr(result, key)).all():
            raise ValueError(f'Nonfinite saved {key}')
    for key in ('_motion_num_frames', '_motion_fps', '_motion_seq_ids', '_motion_lengths'):
        torch.testing.assert_close(getattr(result, key), getattr(source, key))
    if result._motion_seq_names != source._motion_seq_names:
        raise ValueError('Sequence identities changed')
    for index, record in enumerate(records):
        start, count = int(result.length_starts[index]), int(result._motion_num_frames[index])
        local = result.lrs[start:start + count, 1:]
        r0 = Rotation.from_quat(local[:-1].reshape(-1, 4).numpy())
        r1 = Rotation.from_quat(local[1:].reshape(-1, 4).numpy())
        expected = (r0.inv() * r1).as_rotvec().reshape(-1, 69) * float(result._motion_fps[index])
        error = float(abs(expected - result.dvs[start:start + count - 1].numpy()).max())
        record['saved_dof_velocity_max_error_radps'] = error
        # Legacy Poselib computes angles with float32 acos(w). Near identity,
        # its angular resolution is ~1e-3 rad/frame; SciPy uses stable float64.
        tolerance = .001 * float(result._motion_fps[index])
        record['saved_dof_velocity_tolerance_radps'] = tolerance
        if error > tolerance:
            raise ValueError(f'Saved joint angular velocities disagree with rotations: {error:.6g} rad/s')
        ids = torch.tensor([index, index, index])
        times = torch.tensor([0., .5 * float(result._motion_lengths[index]), float(result._motion_lengths[index])])
        samples = result.get_motion_state(ids, times, return_rigid_body=True, adjust_height=False)
        if not all(torch.isfinite(value).all() for value in samples if isinstance(value, torch.Tensor)):
            raise ValueError('MotionLib interpolation returned nonfinite state')
    torch.testing.assert_close(result.gts[..., 0, :2], source.gts[..., 0, :2], rtol=0, atol=0)


if __name__ == '__main__':
    main()
