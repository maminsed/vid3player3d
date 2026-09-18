#!/usr/bin/env python3
"""Plot court trajectories and foot heights using the converter's pipeline."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import vid2player3d.uhc.utils.convert_amass_isaac_correct_ground as ground

correct_smpl_sequence = ground.correct_smpl_sequence
get_foot_vertices = ground.get_foot_vertices
detect_ground_contacts_with_feet = ground.detect_ground_contacts_with_feet
smpl_robot_context = ground.smpl_robot_context


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Compare original and corrected court trajectories and foot heights')
    ground.add_input_arguments(parser)
    parser.add_argument('--sequence_name', help='Specific sequence to analyze (substring match)')
    parser.add_argument('--num_seq', type=int, default=None)
    parser.add_argument('--out_dir', type=Path, default=Path('.'))
    parser.add_argument('--no_show', action='store_true', help='Save plots without opening windows')
    return parser


def plot_trajectory_and_ground_contacts(verts, root_trans, corrected_root_trans, fps=30.0,
                                        sequence_name='', corrected_verts=None, court_enabled=False):
    """Compare roots in the same court frame and surface minimum foot heights.

    With court correction enabled, original arrays have the fixed first-frame
    alignment supplied by correct_smpl_sequence. No trajectory smoothing occurs.
    """
    if corrected_verts is None:
        if court_enabled:
            raise ValueError('Court-aligned plotting requires the reconstructed corrected mesh')
        corrected_verts = verts + (corrected_root_trans - root_trans)[:, None, :]
    original_feet = get_foot_vertices(verts)
    corrected_feet = get_foot_vertices(corrected_verts)
    time = np.arange(len(root_trans)) / fps
    fig = plt.figure(figsize=(17, 15))
    grid = fig.add_gridspec(3, 2)
    ax_xy = fig.add_subplot(grid[0, 0])
    ax_raw = fig.add_subplot(grid[0, 1])
    ax_3d = fig.add_subplot(grid[1, 0], projection='3d')
    ax_heights = fig.add_subplot(grid[1, 1])
    ax_x, ax_y = fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[2, 1])
    ground.draw_xy_comparison(ax_xy, ax_x, ax_y, root_trans, corrected_root_trans, fps, court_enabled)
    for name, color, before, after in zip(('Left', 'Right'), ('tab:blue', 'tab:orange'),
                                        original_feet, corrected_feet):
        original_height, corrected_height = before[:, :, 2].min(axis=1), after[:, :, 2].min(axis=1)
        ax_raw.plot(time, original_height, color=color, label=f'{name} foot')
        ax_heights.plot(time, original_height, '--', color=color, alpha=0.6, label=f'{name} original')
        ax_heights.plot(time, corrected_height, color=color, label=f'{name} corrected')
        foot_center = after.mean(axis=1)
        ax_3d.plot(foot_center[:, 0], foot_center[:, 1], corrected_height, color=color, label=f'{name} foot')
        print(f'{name} foot minimum surface Z: original {original_height.min():.3f}..{original_height.max():.3f} m; '
              f'corrected {corrected_height.min():.3f}..{corrected_height.max():.3f} m; '
              f'{np.count_nonzero(corrected_height <= 0)}/{len(time)} frames at/below ground')
    for ax, title in ((ax_raw, 'Original foot heights'), (ax_heights, 'Original vs corrected foot heights')):
        ax.axhline(0, color='black', linewidth=1)
        ax.set(xlabel='Time (s)', ylabel='Minimum surface Z (m)', title=title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    ax_3d.plot(*corrected_root_trans.T, color='tab:green', label='Corrected pelvis root')
    ax_3d.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', title='Corrected 3D trajectory')
    ax_3d.legend(fontsize=8)
    fig.suptitle(str(sequence_name))
    fig.tight_layout()
    return fig


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.num_seq is not None and args.num_seq < 1:
        raise ValueError('num_seq must be positive')
    data, context, provenance = ground.load_correction_inputs(
        args.gvhmr_dir, args.tennisproject_data, args.disable_xy_correction)
    names = [name for name in data if not args.sequence_name or args.sequence_name in str(name)]
    if args.num_seq is not None:
        names = names[:args.num_seq]
    if not names:
        raise ValueError('No matching sequence')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    completed = {}
    with smpl_robot_context() as robot:
        for name in names:
            sequence = correct_smpl_sequence(data[name], robot, fps=provenance['fps'],
                                             plot_results=False, sequence_name=str(name), xy_context=context)
            fig = plot_trajectory_and_ground_contacts(
                sequence['comparison_verts'].numpy(), sequence['comparison_root_trans'],
                sequence['corrected_root_trans'], sequence['fps'], str(name),
                corrected_verts=sequence['corrected_verts'].numpy(), court_enabled=context['enabled'])
            path = args.out_dir / f"trajectory_and_ground_contacts_{str(name).replace('/', '_')}.png"
            fig.savefig(path, dpi=180, bbox_inches='tight')
            print(f'Saved plot: {path}')
            completed[name] = {'alignment_metadata': sequence['alignment_metadata']}
            ground.save_correction_metadata(args.out_dir, args, provenance, completed)
            if not args.no_show:
                plt.show()
            plt.close(fig)


if __name__ == '__main__':
    main()
