"""Isolate ball-floor restitution without a player, policy or aerodynamic forces.

Run from vid2player3d: PYTHONPATH=. python -m vid2player.utils.check_ball_bounce
Add --gpu for GPU PhysX (CPU state reads, no gymtorch extension required).
"""
import argparse
import json
from pathlib import Path

from isaacgym import gymapi
import numpy as np

from vid2player.utils.tennis_materials import configure_ball_asset


def drop_ball(substeps, gpu=False, actor_only=False, plane_restitution=0.5):
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = 1 / 60
    params.substeps = substeps
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0, 0, -9.81)
    params.physx.use_gpu = gpu
    params.use_gpu_pipeline = False
    params.physx.solver_type = 1
    params.physx.num_position_iterations = 4 if substeps == 2 else 2
    params.physx.num_velocity_iterations = 0
    params.physx.contact_offset = 0.02
    params.physx.rest_offset = 0.0
    params.physx.bounce_threshold_velocity = 0.2
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)
    if sim is None:
        raise RuntimeError('Could not create PhysX simulator')
    try:
        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0, 0, 1)
        plane.static_friction = plane.dynamic_friction = 1.0
        plane.restitution = plane_restitution
        gym.add_ground(sim, plane)
        asset_dir = Path(__file__).resolve().parents[1] / 'data/assets'
        asset = gym.load_asset(sim, str(asset_dir), 'tennis_ball.urdf', gymapi.AssetOptions())
        if not actor_only:
            configure_ball_asset(gym, asset)
        env = gym.create_env(sim, gymapi.Vec3(-2, -2, 0), gymapi.Vec3(2, 2, 2), 1)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, 1.5)
        actor = gym.create_actor(env, asset, pose, 'ball', 0, 0)
        if actor_only:
            props = gym.get_actor_rigid_shape_properties(env, actor)
            props[0].restitution, props[0].friction, props[0].compliance = 0.9, 0.2, 0.5
            gym.set_actor_rigid_shape_properties(env, actor, props)
        gym.prepare_sim(sim)
        heights, velocities = [], []
        for _ in range(120):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            state = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)
            heights.append(float(state['pose']['p']['z'][0]))
            velocities.append(float(state['vel']['linear']['z'][0]))
        bounce = next((i for i in range(1, len(velocities))
                       if velocities[i-1] < 0 < velocities[i]), None)
        apex = max(heights[bounce:]) if bounce is not None else None
        return {'substeps': substeps, 'gpu_physx': gpu,
                'material_initialization': 'actor_only' if actor_only else 'asset',
                'plane_restitution': plane_restitution,
                'bounce_step_60hz': bounce, 'rebound_apex_m': apex,
                'passed': apex is not None and apex > 0.3}
    finally:
        gym.destroy_sim(sim)


def audit_evaluation(directory):
    with np.load(Path(directory) / 'rollout.npz') as data:
        actual = data['ball_xyz']
        preview = data['launch_row'][7:].reshape(-1, 3)
        # Preview includes the launch frame; evaluation records after stepping.
        count = min(len(actual), len(preview)-1)
        error = np.linalg.norm(actual[:count] - preview[1:count+1], axis=-1)
        impact = next((i for i, z in enumerate(actual[:, 2]) if z < 0.06), None)
        return {'directory': str(directory), 'first_near_floor_frame': impact,
                'preimpact_max_preview_error_m': float(error[:impact].max()) if impact else None,
                'postimpact_apex_m': float(actual[impact:, 2].max()) if impact is not None else None,
                'max_preview_error_m': float(error.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--compare-actor-only', action='store_true')
    parser.add_argument('--plane-restitution', type=float, default=0.5)
    parser.add_argument('--evaluation', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.gpu:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('--gpu requires a working CUDA device; refusing a CPU fallback')
    drops = [drop_ball(substeps, args.gpu, actor_only, args.plane_restitution)
             for substeps in (2, 6)
             for actor_only in ((False, True) if args.compare_actor_only else (False,))]
    report = {'drops': drops, 'evaluations': [audit_evaluation(p) for p in args.evaluation]}
    result = json.dumps(report, indent=2)
    print(result)
    if args.output:
        # Preserve earlier diagnostic evidence on accidental reruns.
        with args.output.open('x') as output:
            output.write(result + '\n')
    if any(not row['passed'] for row in drops if row['material_initialization'] == 'asset'):
        raise SystemExit('Ball-floor rebound check failed')


if __name__ == '__main__':
    main()
