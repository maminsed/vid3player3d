"""Record reset and per-step physics state during a normal CUDA evaluation.

Run like vid2player/run.py with --eval_controller and --diagnostics-output FILE.
Optional ablations: --diagnostic-floor 0, --diagnostic-disable-alignment,
--diagnostic-freeze-reference, and --diagnostic-zero-low-action.
Every ablation is local to this process; no checkpoint is trained or rewritten.
"""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--diagnostics-output', type=Path, required=True)
    parser.add_argument('--diagnostic-floor', type=float)
    parser.add_argument('--diagnostic-disable-alignment', action='store_true')
    parser.add_argument('--diagnostic-freeze-reference', action='store_true')
    parser.add_argument('--diagnostic-zero-low-action', action='store_true')
    options, remaining = parser.parse_known_args()
    if '--eval_controller' not in remaining:
        parser.error('This diagnostic requires --eval_controller; training is not supported')
    if options.diagnostics_output.exists():
        raise FileExistsError(options.diagnostics_output)
    sys.argv = [sys.argv[0]] + remaining

    import vid2player.run as run
    from vid2player.env.tasks.humanoid_smpl_im_mvae import HumanoidSMPLIMMVAE
    from vid2player.players.mvae_player import MVAEPlayer
    import torch

    load_cfg = run.load_cfg

    def diagnostic_cfg(args):
        cfg, train = load_cfg(args)
        if options.diagnostic_floor is not None:
            cfg['env']['physics']['plane_restitution'] = options.diagnostic_floor
        return cfg, train

    run.load_cfg = diagnostic_cfg
    initialize = HumanoidSMPLIMMVAE.__init__

    def diagnostic_init(self, cfg, *args, **kwargs):
        if options.diagnostic_disable_alignment:
            cfg['env']['align_reset_to_ground'] = False
        initialize(self, cfg, *args, **kwargs)

    HumanoidSMPLIMMVAE.__init__ = diagnostic_init
    rows = []

    def snapshot(task, label):
        root = task._humanoid_root_states[0].detach().cpu().tolist()
        rows.append({
            'event': label, 'root_xyz_m': root[:3], 'root_velocity_mps': root[7:10],
            'target_root_xyz_m': task._target_root_pos[0].detach().cpu().tolist(),
            'decoded_root_xyz_m': task._mvae_player._root_pos[0].detach().cpu().tolist(),
            'lowest_joint_z_m': task._rigid_body_pos[0, :24, 2].min().item(),
            'reset_height_offset_m': task._reset_height_offset[0].item(),
            'floor_restitution': task.plane_restitution,
            'alignment_enabled': task.cfg['env'].get('align_reset_to_ground', False),
        })

    reset = HumanoidSMPLIMMVAE._reset_envs

    def diagnostic_reset(self, *args, **kwargs):
        result = reset(self, *args, **kwargs)
        snapshot(self, 'reset')
        return result

    HumanoidSMPLIMMVAE._reset_envs = diagnostic_reset
    step = HumanoidSMPLIMMVAE._physics_step

    def diagnostic_step(self):
        result = step(self)
        snapshot(self, 'step')
        return result

    HumanoidSMPLIMMVAE._physics_step = diagnostic_step
    if options.diagnostic_freeze_reference:
        MVAEPlayer.step = lambda *args, **kwargs: None
    if options.diagnostic_zero_low_action:
        pre_step = HumanoidSMPLIMMVAE.pre_physics_step

        def zero_action_step(self, actions):
            return pre_step(self, torch.zeros_like(actions))

        HumanoidSMPLIMMVAE.pre_physics_step = zero_action_step

    completed = False
    try:
        run.main()
        completed = True
    finally:
        steps = [row for row in rows if row['event'] == 'step']
        report = {
            'completed': completed, 'options': vars(options), 'rows': rows,
            'summary': {
                'first_step_vertical_speed_mps': steps[0]['root_velocity_mps'][2] if steps else None,
                'maximum_root_height_m': max((r['root_xyz_m'][2] for r in steps), default=None),
                'maximum_root_height_first_second_m': max((r['root_xyz_m'][2] for r in steps[:30]), default=None),
            },
        }
        with options.diagnostics_output.open('x') as output:
            json.dump(report, output, indent=2, default=str)
            output.write('\n')


if __name__ == '__main__':
    main()
