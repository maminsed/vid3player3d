"""Versioned incoming previews and outgoing reward tables from a controller YAML."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

SCRIPT = Path(__file__).resolve()
PROJECT = SCRIPT.parents[2]
sys.path.insert(0, str(PROJECT))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value, base=PROJECT):
    path = Path(value).expanduser()
    if path.is_absolute() or path.exists():
        return path.resolve()
    return (base / path).resolve()


def load_config(value):
    import yaml
    path = resolve_path(value)
    if not path.is_file():
        path = PROJECT / 'vid2player/cfg/controller' / (Path(value).stem + '.yaml')
    with path.open() as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg.get('env', {}).get('physics'), dict):
        raise ValueError('Supply a full controller YAML with env.physics and env.vid2player')
    if not isinstance(cfg['env'].get('vid2player'), dict):
        raise ValueError('Missing env.vid2player')
    return path.resolve(), cfg


def validate_pool(pool):
    if pool.ndim != 2 or pool.shape[1] != 307 or len(pool) == 0:
        raise ValueError('Expected a nonempty incoming shot pool with shape (N, 307)')


def reserve_output(path):
    # mkdir's exclusive creation is also the overwrite guard against concurrent runs.
    path.mkdir(parents=True, exist_ok=False)
    return path


def render_video(path, trajectory, launch, seed, row, radius):
    """A fixed-scale court and height view; each simulated 1/30 s is one movie frame."""
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/vid2player-preview-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    import numpy as np

    if not FFMpegWriter.isAvailable():
        raise RuntimeError('ffmpeg is required for --test-only; install it and retry')
    fig = plt.figure(figsize=(12, 6), layout='constrained')
    court, height = fig.subplots(1, 2)
    court.set(xlabel='x (m)', ylabel='y (m)', title='Court, top view')
    for width in (8.23, 10.97):
        court.plot([-width/2, width/2, width/2, -width/2, -width/2],
                   [-11.885, -11.885, 11.885, 11.885, -11.885], color='seagreen')
    for y in (-6.4, 6.4):
        court.plot([-4.115, 4.115], [y, y], color='seagreen')
    court.plot([0, 0], [-6.4, 6.4], color='seagreen')
    court.plot([-5.485, 5.485], [0, 0], color='black', label='net')
    court.set_xlim(min(-6.5, float(trajectory[:, 0].min())-1), max(6.5, float(trajectory[:, 0].max())+1))
    ymin, ymax = min(-14, float(trajectory[:, 1].min())-1), max(14, float(trajectory[:, 1].max())+1)
    court.set_ylim(ymin, ymax)
    court.set_aspect('equal')
    height.set(xlabel='y (m)', ylabel='ball center z (m)', title='Height profile (separate x/z scales)',
               xlim=(ymin, ymax), ylim=(0, max(4, float(trajectory[:, 2].max())+0.5)))
    height.axhline(radius, color='seagreen', linewidth=1)
    height.plot([0, 0], [0, 1.07], color='black', label='net reference 1.07 m')
    height.legend(loc='upper right', fontsize=8)
    trails = [court.plot([], [], color='orange', alpha=.6)[0], height.plot([], [], color='orange', alpha=.6)[0]]
    dots = [court.plot([], [], 'o', color='darkorange')[0], height.plot([], [], 'o', color='darkorange')[0]]
    speed = float(np.linalg.norm(launch[3:6]))
    subtitle = fig.suptitle('')
    writer = FFMpegWriter(fps=30, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
    with writer.saving(fig, str(path), dpi=120):
        for frame in range(len(trajectory)):
            tail = trajectory[max(0, frame-15):frame+1]
            for trail, dot, a, b in zip(trails, dots, (0, 1), (1, 2)):
                trail.set_data(tail[:, a], tail[:, b])
                dot.set_data([trajectory[frame, a]], [trajectory[frame, b]])
            subtitle.set_text(f't={frame/30:.2f} s | launch {speed:.1f} m/s, spin {launch[6]:.1f} rev/s | seed {seed}, shot {row}\n'
                              'Real-time 30 fps; ball marker enlarged; reference net has no collider')
            writer.grab_frame()
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg', required=True, help='Controller name or YAML path')
    parser.add_argument('--artifacts', choices=('all', 'incoming', 'outgoing'), default='all',
                        help='Generate incoming previews, outgoing x/y tables, or both (default: all)')
    parser.add_argument('--output-dir', help='NEW directory; never append to or replace an existing one')
    parser.add_argument('--shots', help='Override launch-state source; defaults to cfg ball_traj_file')
    parser.add_argument('--test-only', action='store_true', help='Replay one random shot into MP4, without writing a pool')
    parser.add_argument('--resample', action='store_true', help='Sample new launches using cfg ranges and court/rebound acceptance')
    parser.add_argument('--num-shots', type=int, help='Replay a seeded subset, or number of accepted resampled shots')
    parser.add_argument('--batch-size', type=int, default=2048, help='Batched GPU ball environments (default: 2048)')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', default='cuda:0', help='cuda:N (default) or cpu')
    parser.add_argument('--max-batches', type=int, default=1000, help='Resampling safety bound')
    parser.add_argument('--outgoing-max-seconds', type=float, default=8., help='Maximum free-flight horizon for outgoing queries')
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.max_batches < 1 or (args.num_shots is not None and args.num_shots < 1):
        parser.error('batch-size, num-shots and max-batches must be positive')
    if args.resample and args.num_shots is None and not args.test_only:
        parser.error('--resample requires --num-shots')
    if args.resample and args.shots:
        parser.error('--shots and --resample are mutually exclusive')
    if args.artifacts == 'outgoing' and (args.resample or args.shots or args.num_shots is not None):
        parser.error('--resample, --shots and --num-shots apply only to incoming artifacts')
    if not 0 < args.outgoing_max_seconds <= 30:
        parser.error('--outgoing-max-seconds must be in (0, 30]')
    if args.device != 'cpu' and not (args.device.startswith('cuda:') and args.device[5:].isdigit()):
        parser.error('--device must be cpu or cuda:N')
    return args


def main(argv=None):
    args = parse_args(argv)
    cfg_path, cfg = load_config(args.cfg)
    # Import Isaac Gym before torch; do not expose this CLI to gymutil's parser.
    from isaacgym import gymapi
    import numpy as np
    import torch
    import yaml
    from vid2player.utils.tennis_ball import TennisBallGeneratorIsaac, simulate
    from vid2player.utils.ball_physics import resolve_ball_asset

    incoming = args.artifacts != 'outgoing'
    outgoing = args.artifacts != 'incoming' and not args.test_only
    source = None if args.resample or not incoming else resolve_path(args.shots or cfg['env']['vid2player']['ball_traj_file'])
    pool = None if source is None else np.load(source, mmap_mode='r', allow_pickle=False)
    if pool is not None:
        validate_pool(pool)
        if args.num_shots is not None and args.num_shots > len(pool) and not args.test_only:
            raise ValueError('--num-shots exceeds the source pool; use --resample for new shots')
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.test_only:
        indices = [int(rng.integers(len(pool)))] if pool is not None else None
        total = 1
    elif pool is not None:
        total = min(args.num_shots or len(pool), len(pool))
        indices = np.arange(len(pool)) if total == len(pool) else np.sort(rng.choice(len(pool), total, replace=False))
        # Offline start-position selection relies on ascending launch x.
        indices = indices[np.argsort(pool[indices, 0], kind='stable')]
    elif incoming:
        total, indices = args.num_shots, None
    else:
        total, indices = 0, None
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = resolve_path(args.output_dir) if args.output_dir else PROJECT / 'data/ball_traj/generated' / f'{cfg_path.stem}_{stamp}_{uuid.uuid4().hex[:8]}'
    reserve_output(out)
    manifest = dict(status='running', created_utc=stamp, arguments=vars(args), cfg_path=str(cfg_path),
                    cfg_sha256=sha256(cfg_path), mode='outgoing' if not incoming else ('resample' if args.resample else 'replay'),
                    source_path=str(source) if source else None, source_sha256=sha256(source) if source else None,
                    units=dict(position='m', velocity='m/s', signed_spin='rev/s'),
                    layout='float32 [launch_xyz(3), launch_velocity_xyz(3), signed_spin(1), preview_xyz(100*3)]',
                    preview_fps=30, preview_frame_zero='launch state, before the first simulation step',
                    outgoing_reward_tables_regenerated=False)
    def save_manifest():
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out / 'controller.source.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    save_manifest()
    gen = None
    try:
        # Isaac Gym's legacy asset paths are relative to the project directory.
        os.chdir(PROJECT)
        gpu = args.device != 'cpu'
        sim_args = SimpleNamespace(physics_engine=gymapi.SIM_PHYSX, use_gpu_pipeline=gpu,
                                   compute_device_id=int(args.device[5:]) if gpu else 0, graphics_device_id=-1)
        num_envs = args.batch_size if outgoing or args.resample else min(args.batch_size, max(total, 1))
        gen = TennisBallGeneratorIsaac(cfg, need_reset=False, num_envs=num_envs,
                                      sim_args=sim_args)
        asset_root, asset_file = resolve_ball_asset('vid2player/data/assets', gen.ball_physics)
        sources = [SCRIPT, PROJECT / 'vid2player/env/tasks/humanoid_smpl_im_mvae.py']
        sources += [PROJECT / 'vid2player/utils' / name for name in
                    ('tennis_ball.py', 'ball_physics.py', 'tennis_contacts.py', 'tennis_materials.py',
                     'ball_outgoing_generation.py', 'tennis_ball_out_estimator.py')]
        sources += [resolve_path(Path(asset_root) / asset_file)]
        manifest['source_hashes'] = {str(p): sha256(p) for p in sources}
        manifest['effective_physics'] = dict(dt=gen.sim_params.dt, substeps=gen.substeps,
            physics_calls_per_preview_frame=gen.control_freq_inv, solver_type=gen.sim_params.physx.solver_type,
            position_iterations=gen.sim_params.physx.num_position_iterations, velocity_iterations=gen.sim_params.physx.num_velocity_iterations,
            contact_offset=gen.sim_params.physx.contact_offset, device=args.device,
            ball_physics=vars(gen.ball_physics), tennis_contacts=vars(gen.contacts))
        target = None if args.test_only or not incoming else np.lib.format.open_memmap(out / 'pool.incomplete.npy', mode='w+', dtype='float32', shape=(total, 307))
        written, attempts, accepted = 0, 0, 0
        stats = dict(net_clear_count=0, first_bounce_count=0, bounce_in_sampling_zone_count=0,
                     bounce_xy_min=None, bounce_xy_max=None)
        while written < total:
            attempts += 1
            if attempts > args.max_batches and args.resample:
                raise RuntimeError('Resampling exhausted --max-batches before reaching --num-shots')
            if args.test_only and not incoming:
                # An illustrative outgoing shot on the receiving side of the court.
                launch = np.array([[0., -12., 1., rng.uniform(-3, 3), rng.uniform(18, 35),
                                    rng.uniform(1, 6), rng.uniform(-10, 10)]], dtype=np.float32)
                trajectory, _, _, _ = simulate(gen.gym, gen.sim,
                    torch.tensor(launch[:, :3], device=gen.device), torch.tensor(launch[:, 3:6], device=gen.device),
                    torch.tensor(launch[:, 6], device=gen.device), substeps=gen.substeps, spin_scale=gen.spin_scale,
                    device=gen.device, ball_physics=gen.ball_physics, tennis_contacts=gen.contacts)
                data = np.concatenate((launch, trajectory.cpu().numpy().reshape(1, 300)), axis=1)
            elif args.resample:
                # Existing sampler uses cfg ranges and filters with the selected physics.
                # Keep all GPU generation batched; no Python loop per trajectory.
                try:
                    gen.reset()
                except AssertionError:
                    print('No accepted shots this batch; retrying', flush=True)
                    continue
                traj, pos, vel, spin = gen.generate_all()
                data = torch.cat((pos, vel, spin[:, None], traj.reshape(-1, 300)), dim=1).numpy()
                accepted += len(data)
                if args.test_only:
                    data = data[rng.integers(len(data)):][:1]
                data = data[:total-written]
            else:
                chosen = indices[written:written+gen.num_env]
                launch = np.array(pool[chosen, :7], dtype=np.float32, copy=True)
                if not np.isfinite(launch).all():
                    raise ValueError('Launch states contain NaN or infinity')
                pos = torch.as_tensor(launch[:, :3], device=gen.device)
                vel = torch.as_tensor(launch[:, 3:6], device=gen.device)
                spin = torch.as_tensor(launch[:, 6], device=gen.device)
                traj, bounce, bounce_idx, net = simulate(gen.gym, gen.sim, pos, vel, spin,
                    num_frames=100, substeps=gen.substeps, spin_scale=gen.spin_scale, device=gen.device,
                    ball_physics=gen.ball_physics, tennis_contacts=gen.contacts)
                data = np.concatenate((launch, traj.cpu().numpy().reshape(-1, 300)), axis=1)
                stats['net_clear_count'] += int(net.sum())
                bounced = bounce[bounce.abs().sum(dim=1) > 0, :2].cpu().numpy()
                stats['first_bounce_count'] += len(bounced)
                if len(bounced):
                    lower, upper = gen.bounce_min[:2].numpy(), gen.bounce_max[:2].numpy()
                    stats['bounce_in_sampling_zone_count'] += int(((bounced > lower) & (bounced < upper)).all(axis=1).sum())
                    lo, hi = bounced.min(axis=0), bounced.max(axis=0)
                    stats['bounce_xy_min'] = np.minimum(stats['bounce_xy_min'], lo).tolist() if stats['bounce_xy_min'] is not None else lo.tolist()
                    stats['bounce_xy_max'] = np.maximum(stats['bounce_xy_max'], hi).tolist() if stats['bounce_xy_max'] is not None else hi.tolist()
            if not np.isfinite(data).all():
                raise RuntimeError('Simulation produced NaN or infinity; no completed pool will be published')
            if args.test_only:
                render_video(out / 'random_shot.mp4', data[0, 7:].reshape(100, 3), data[0, :7], args.seed,
                             int(indices[0]) if indices is not None else ('resampled' if incoming else 'outgoing'), gen.ball_physics.radius)
                manifest['test_shot'] = dict(source_row=int(indices[0]) if indices is not None else None, launch=data[0, :7].tolist())
            else:
                target[written:written+len(data)] = data
            written += len(data)
            manifest['num_shots_written'] = written
            print(f'{written}/{total} previews; output {out}', flush=True)
        if target is not None:
            target.flush()
            if args.resample:
                # Stream the final sorted pool instead of holding/sorting a multi-GB pool in RAM.
                order = np.argsort(target[:, 0], kind='stable')
                sorted_pool = np.lib.format.open_memmap(out / 'pool.sorted.incomplete.npy', mode='w+', dtype='float32', shape=target.shape)
                for start in range(0, total, args.batch_size):
                    sorted_pool[start:start+args.batch_size] = target[order[start:start+args.batch_size]]
                sorted_pool.flush()
                del sorted_pool, target
                (out / 'pool.incomplete.npy').unlink()
                (out / 'pool.sorted.incomplete.npy').rename(out / 'ball_traj_in.npy')
            else:
                del target
                (out / 'pool.incomplete.npy').rename(out / 'ball_traj_in.npy')
            # Snapshot for explicit adoption, without changing the original controller.
            cfg['env']['vid2player']['ball_traj_file'] = str(out / 'ball_traj_in.npy')
        if outgoing:
            from vid2player.utils.ball_outgoing_generation import generate_tables
            from vid2player.utils.tennis_ball_out_estimator import traj_out_params
            def progress(done, count):
                manifest['outgoing_rows_written'] = done
                save_manifest()
                print(f'{done}/{count} outgoing rows; output {out}', flush=True)
            manifest['outgoing'] = generate_tables(gen, out, traj_out_params,
                max_seconds=args.outgoing_max_seconds, progress=progress)
            manifest['outgoing_reward_tables_regenerated'] = True
            cfg['env']['vid2player']['ball_traj_out_x_file'] = str(out / 'ball_traj_out_x.npy')
            cfg['env']['vid2player']['ball_traj_out_y_file'] = str(out / 'ball_traj_out_y.npy')
        if not args.test_only:
            (out / 'controller.generated.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        manifest.update(status='complete', num_shots=written, batches=attempts,
                        replay_curriculum_statistics=stats if incoming and not args.resample else None,
                        resampling_statistics=dict(candidates=attempts*gen.num_env, accepted=accepted) if args.resample else None,
                        artifacts={p.name: sha256(p) for p in out.iterdir() if p.suffix in ('.npy', '.mp4')})
        save_manifest()
        if incoming and not args.resample:
            print(f"Bounces within cfg sampling zone: {stats['bounce_in_sampling_zone_count']}/{written}; replay does not filter or retarget launches", flush=True)
        print(f'Complete: {out}', flush=True)
    except BaseException as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        save_manifest()
        raise
    finally:
        if gen is not None:
            gen.gym.destroy_sim(gen.sim)


if __name__ == '__main__':
    main()
