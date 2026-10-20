"""Bounded-memory generation of the existing scalar-spin outgoing lookup format.

Imports Isaac Gym only when simulating; grid/interpolation contracts are CPU testable.
"""
from pathlib import Path
import math

import numpy as np
import torch
import torch.nn.functional as F

from vid2player.utils.ball_physics import ball_aerodynamic_force


def table_axes(params):
    return [np.arange(*getattr(params, name), dtype=np.float64) for name in
            ('VEL_X_RANGE', 'VEL_Y_RANGE', 'VSPIN_RANGE', 'TRAJ_X_RANGE', 'TRAJ_Y_RANGE')]


def decode_rows(indices, params):
    """Runtime ordering: horizontal speed, vertical speed, then spin (fastest)."""
    speed, vertical, spin, _, _ = table_axes(params)
    indices = np.asarray(indices, dtype=np.int64)
    if ((indices < 0) | (indices >= len(speed)*len(vertical)*len(spin))).any():
        raise ValueError('Outgoing row index outside lookup grid')
    return np.column_stack((speed[indices // (len(vertical)*len(spin))],
                            vertical[(indices // len(spin)) % len(vertical)],
                            spin[indices % len(spin)])).astype(np.float32)


def interpolate_tables(flight, dt, params, allow_landed_tail=False):
    """flight=(N,T,2) horizontal distance / relative center height, from t=0.

    Resolve descending height crossings, including the return to launch height.
    Reject unbracketed heights rather than extrapolating beyond simulated flight.
    Optional landed tails mark unreachable distance bins invalid for net clearance.
    """
    _, _, _, distances, drops = table_axes(params)
    distance = flight[:, :, 0].contiguous()
    height = flight[:, :, 1]
    if not torch.isfinite(flight).all() or not (distance.diff(dim=1) > 0).all():
        raise ValueError('Outgoing flight must be finite and advance horizontally')
    # Cumulative maximum skips the ascending branch without a per-row loop.
    descending_drop = (-height).cummax(dim=1).values.contiguous()

    def brackets(curve, values, right=False, physical_curve=None, allow_missing=False):
        query = torch.as_tensor(values, dtype=flight.dtype, device=flight.device)[None].expand(len(flight), -1).contiguous()
        index = torch.searchsorted(curve, query, right=right)
        missing = index >= flight.shape[1]
        if missing.any() and not allow_missing:
            raise ValueError('Outgoing flight horizon does not bracket every lookup query')
        index = index.clamp(min=1, max=flight.shape[1]-1)
        physical_curve = curve if physical_curve is None else physical_curve
        lower, upper = physical_curve.gather(1, index-1), physical_curve.gather(1, index)
        weight = (query-lower) / (upper-lower)
        return index, weight, missing

    if allow_landed_tail and not (descending_drop[:, -1] > drops[-1]).all():
        raise ValueError('Cannot mark distance bins invalid before every supported launch height has landed')
    index, weight, missing = brackets(distance, distances, allow_missing=allow_landed_tail)
    table_x = height.gather(1, index-1).lerp(height.gather(1, index), weight)
    # All supported launch heights have landed before these distance bins.
    # A finite below-ground sentinel makes the existing net-clearance test fail.
    # Slow balls can never reach 30m even in infinite free fall: do not extrapolate.
    table_x[missing] = -1e6
    index, weight, _ = brackets(descending_drop, drops, right=True, physical_curve=-height)
    landing_distance = distance.gather(1, index-1).lerp(distance.gather(1, index), weight)
    landing_time = (index.to(flight.dtype)-1+weight) * dt
    table_y = torch.stack((landing_distance, landing_time), dim=-1)
    if not torch.isfinite(table_x).all() or not torch.isfinite(table_y).all():
        raise ValueError('Nonfinite outgoing lookup values')
    return table_x.cpu().numpy(), table_y.cpu().numpy()


def simulate_tables(generator, launches, params, max_seconds=8.):
    """Record free flight at the configured outer physics dt, batching all balls.

    High launch altitude removes floor contact; gravity/aerodynamics are unchanged.
    Stop once all supported launch heights have landed. No substep change.
    """
    from isaacgym import gymapi, gymtorch
    gym, sim, device = generator.gym, generator.sim, generator.device
    count = len(launches)
    if count != generator.num_env:
        raise ValueError('Pad outgoing launch batches to the simulator environment count')
    _, _, _, _, drops = table_axes(params)
    dt = generator.physics_dt
    max_steps = math.ceil(max_seconds / dt)
    # Even a no-drag downward launch cannot reach the floor within this horizon.
    altitude = 10 + .5 * 9.81 * max_seconds**2 + max(0., -float(launches[:, 1].min())) * max_seconds
    state = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.refresh_actor_root_state_tensor(sim)
    state.zero_()
    state[:, 2], state[:, 6] = altitude, 1.
    state[:, 8] = torch.as_tensor(launches[:, 0], device=device)
    state[:, 9] = torch.as_tensor(launches[:, 1], device=device)
    spin = torch.as_tensor(launches[:, 2], device=device)
    gravity_axis = torch.zeros_like(state[:, 7:10]); gravity_axis[:, 2] = -1
    state[:, 10:13] = F.normalize(torch.cross(state[:, 7:10], gravity_axis, dim=-1), dim=-1) * (spin[:, None]*2*math.pi)
    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(state))
    flight = torch.empty((count, max_steps+1, 2), device=device)
    flight[:, 0] = 0
    reached = False
    for step in range(1, max_steps+1):
        forces = ball_aerodynamic_force(state[:, 7:10], state[:, 10:13], generator.ball_physics,
            spin_scale=generator.spin_scale, legacy_spin_sign=torch.where(spin > 0, 1., -1.))
        gym.apply_rigid_body_force_tensors(sim, gymtorch.unwrap_tensor(forces), None, gymapi.ENV_SPACE)
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.refresh_actor_root_state_tensor(sim)
        flight[:, step, 0] = state[:, 1]
        flight[:, step, 1] = state[:, 2]-altitude
        # Synchronize once per eight calls instead of once per lookup bin and step.
        if step % 8 == 0 or step == max_steps:
            reached = bool((state[:, 2]-altitude < -drops[-1]).all())
            if reached:
                break
    if not reached:
        raise ValueError(f'Outgoing flight did not cover the grid in {max_seconds}s; increase --outgoing-max-seconds')
    return interpolate_tables(flight[:, :step+1], dt, params, allow_landed_tail=True)


def generate_tables(generator, output_dir, params, max_seconds=8., progress=None):
    """Stream the complete runtime grid; include the final partial batch, padded.

    Custom params support focused validation; the CLI always uses the runtime grid.
    """
    speed, vertical, spin, distances, drops = table_axes(params)
    total = len(speed)*len(vertical)*len(spin)
    output_dir = Path(output_dir)
    paths = [output_dir / f'ball_traj_out_{axis}.incomplete.npy' for axis in ('x', 'y')]
    destinations = [output_dir / f'ball_traj_out_{axis}.npy' for axis in ('x', 'y')]
    if any(p.exists() for p in paths+destinations):
        raise FileExistsError('Outgoing table output already exists')
    x = np.lib.format.open_memmap(paths[0], mode='w+', dtype='float32', shape=(total, len(distances)))
    y = np.lib.format.open_memmap(paths[1], mode='w+', dtype='float32', shape=(total, len(drops), 2))
    for start in range(0, total, generator.num_env):
        stop = min(start+generator.num_env, total)
        # Repeat the final valid row for unused environments, then discard padding.
        indices = np.minimum(np.arange(start, start+generator.num_env), total-1)
        tx, ty = simulate_tables(generator, decode_rows(indices, params), params, max_seconds)
        x[start:stop], y[start:stop] = tx[:stop-start], ty[:stop-start]
        if progress:
            progress(stop, total)
    x.flush(); y.flush()
    del x, y
    for source, destination in zip(paths, destinations):
        source.rename(destination)
    return dict(rows=total, x_shape=[total, len(distances)], y_shape=[total, len(drops), 2],
                grid={name: list(getattr(params, name)) for name in ('VEL_X_RANGE', 'VEL_Y_RANGE', 'VSPIN_RANGE', 'TRAJ_X_RANGE', 'TRAJ_Y_RANGE')},
                spin_representation='signed transverse spin, rev/s; scalar lookup, not a vector predictor',
                flight_sample_dt=generator.physics_dt, max_flight_seconds=max_seconds,
                unsupported_distance_height_sentinel=-1e6,
                floor_target='center height equals floor at z=0, matching existing lookup consumer')
