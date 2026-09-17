#!/usr/bin/env python3
"""Visualize original and corrected foot heights using the converter's pipeline."""

import argparse
from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Allow both direct script execution and python -m uhc.utils....
from vid2player3d.uhc.utils.convert_amass_isaac_correct_ground import (
    correct_smpl_sequence,
    detect_ground_contacts_with_feet,
    get_foot_vertices,
    smpl_robot_context,
)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Plot original and corrected foot heights")
    parser.add_argument('--amass_data', type=str, default="data/amass/amass_copycat_take5_5.pkl")
    parser.add_argument('--sequence_name', type=str, default=None,
                        help="Specific sequence to analyze (substring match)")
    parser.add_argument('--num_seq', type=int, default=None)
    parser.add_argument('--out_dir', type=str, default=".")
    parser.add_argument('--no_show', action='store_true',
                        help='Save plots without opening windows or prompting between sequences')
    return parser


def plot_trajectory_and_ground_contacts(verts, root_trans, corrected_root_trans, fps=30.0, sequence_name="",
                                        corrected_verts=None):
    """
    Create a 2x2 subplot showing:
    1. Horizontal XY trajectory of root and feet
    2. Original foot heights over time
    3. 3D trajectory of root and feet (using corrected heights)
    4. Comparison of original vs corrected foot heights
    
    Args:
        verts: (T, V, 3) vertex positions
        root_trans: (T, 3) original root translations
        corrected_root_trans: (T, 3) corrected root translations
        fps: Frames per second
        sequence_name: Name of the sequence for plot title
        corrected_verts: Mesh reconstructed by the shared correction pipeline.
            If omitted, translate the original mesh by the root displacement.
    """
    T = verts.shape[0]
    time_coords = np.arange(T) / fps
    
    # Get foot vertices
    left_foot_verts, right_foot_verts = get_foot_vertices(verts)
    
    # Get minimum Z-coordinate for each foot (lowest point of foot surface)
    left_foot_z = left_foot_verts[:, :, 2].min(axis=1)
    right_foot_z = right_foot_verts[:, :, 2].min(axis=1)
    
    # Get foot positions in XY plane (center of foot)
    left_foot_x = left_foot_verts[:, :, 0].mean(axis=1)  # Average X position
    left_foot_y = left_foot_verts[:, :, 1].mean(axis=1)  # Average Y position
    right_foot_x = right_foot_verts[:, :, 0].mean(axis=1)  # Average X position
    right_foot_y = right_foot_verts[:, :, 1].mean(axis=1)  # Average Y position
    
    # Check if ANY part of the foot surface is on or below ground
    # This is more accurate than just checking the minimum point
    left_foot_on_ground = left_foot_verts[:, :, 2] <= 0  # Check all foot vertices
    right_foot_on_ground = right_foot_verts[:, :, 2] <= 0  # Check all foot vertices
    
    # A foot is considered "on ground" if ANY of its vertices are at or below ground
    left_foot_ground_mask = np.any(left_foot_on_ground, axis=1)
    right_foot_ground_mask = np.any(right_foot_on_ground, axis=1)
    
    # Detect ground contact points
    valley_indices, foot_contact_info = detect_ground_contacts_with_feet(verts)
    
    # Create figure with subplots (3D plot for bottom left)
    fig = plt.figure(figsize=(16, 12))
    ax1 = plt.subplot(2, 2, 1)  # XY trajectory
    ax2 = plt.subplot(2, 2, 2)  # Original Z heights
    ax3 = plt.subplot(2, 2, 3, projection='3d')  # 3D trajectory
    ax4 = plt.subplot(2, 2, 4)  # Comparison
    
    # Plot 1: Horizontal XY trajectory (Root + Left Foot + Right Foot)
    root_x_coords = root_trans[:, 0]
    root_y_coords = root_trans[:, 1]
    
    # Color trajectory by time
    colors = plt.cm.viridis(np.linspace(0, 1, len(root_x_coords)))
    
    # Plot root trajectory with color gradient
    for i in range(len(root_x_coords) - 1):
        ax1.plot(root_x_coords[i:i+2], root_y_coords[i:i+2], color=colors[i], linewidth=3, alpha=0.8, label='Root' if i == 0 else "")
    
    # Get corrected foot vertices and heights for ground detection
    if corrected_verts is None:
        corrected_verts = verts + (corrected_root_trans - root_trans)[:, None, :]
    
    # Get corrected foot vertices
    corrected_left_foot_verts = get_foot_vertices(corrected_verts)[0]
    corrected_right_foot_verts = get_foot_vertices(corrected_verts)[1]
    
    # Get corrected foot heights
    corrected_left_foot_z = corrected_left_foot_verts[:, :, 2].min(axis=1)
    corrected_right_foot_z = corrected_right_foot_verts[:, :, 2].min(axis=1)
    
    # Check if ANY part of the corrected foot surface is on or below ground
    corrected_left_foot_on_ground = corrected_left_foot_verts[:, :, 2] <= 0
    corrected_right_foot_on_ground = corrected_right_foot_verts[:, :, 2] <= 0
    
    # A foot is considered "on ground" if ANY of its corrected vertices are at or below ground
    left_foot_ground_mask = np.any(corrected_left_foot_on_ground, axis=1)
    right_foot_ground_mask = np.any(corrected_right_foot_on_ground, axis=1)
    
    # Debug: Print some information about corrected foot heights and ground detection
    print(f"Corrected left foot minimum height range: {corrected_left_foot_z.min():.3f} to {corrected_left_foot_z.max():.3f}")
    print(f"Corrected right foot minimum height range: {corrected_right_foot_z.min():.3f} to {corrected_right_foot_z.max():.3f}")
    print(f"Corrected left foot on ground: {np.sum(left_foot_ground_mask)} out of {T}")
    print(f"Corrected right foot on ground: {np.sum(right_foot_ground_mask)} out of {T}")
    
    # Plot left foot trajectory points (both on ground and off ground)
    left_foot_off_ground_mask = ~left_foot_ground_mask
    
    if np.any(left_foot_ground_mask):
        ax1.scatter(left_foot_x[left_foot_ground_mask], left_foot_y[left_foot_ground_mask], 
                   c='blue', s=20, alpha=0.7, label='Left Foot (on ground)', zorder=4)
    
    if np.any(left_foot_off_ground_mask):
        ax1.scatter(left_foot_x[left_foot_off_ground_mask], left_foot_y[left_foot_off_ground_mask], 
                   c='lightblue', s=15, alpha=0.5, label='Left Foot (off ground)', zorder=3)
    
    # Plot right foot trajectory points (both on ground and off ground)
    right_foot_off_ground_mask = ~right_foot_ground_mask
    
    if np.any(right_foot_ground_mask):
        ax1.scatter(right_foot_x[right_foot_ground_mask], right_foot_y[right_foot_ground_mask], 
                   c='red', s=20, alpha=0.7, label='Right Foot (on ground)', zorder=4)
    
    if np.any(right_foot_off_ground_mask):
        ax1.scatter(right_foot_x[right_foot_off_ground_mask], right_foot_y[right_foot_off_ground_mask], 
                   c='lightcoral', s=15, alpha=0.5, label='Right Foot (off ground)', zorder=3)
    
    # Mark start and end points for root
    ax1.scatter(root_x_coords[0], root_y_coords[0], c='green', s=100, marker='o', label='Root Start', zorder=5)
    ax1.scatter(root_x_coords[-1], root_y_coords[-1], c='red', s=100, marker='s', label='Root End', zorder=5)
    
    # Mark start and end points for feet (always show them)
    ax1.scatter(left_foot_x[0], left_foot_y[0], c='blue', s=80, marker='^', label='Left Foot Start', zorder=5)
    ax1.scatter(left_foot_x[-1], left_foot_y[-1], c='blue', s=80, marker='v', label='Left Foot End', zorder=5)
    ax1.scatter(right_foot_x[0], right_foot_y[0], c='orange', s=80, marker='^', label='Right Foot Start', zorder=5)
    ax1.scatter(right_foot_x[-1], right_foot_y[-1], c='orange', s=80, marker='v', label='Right Foot End', zorder=5)
    
    # Plot ground contact points on XY trajectory (only when feet are on ground according to corrected heights)
    for frame_idx, left_z, right_z, use_both_feet in foot_contact_info:
        if use_both_feet:
            # Use both feet - only plot if they're on ground according to corrected heights
            if left_foot_ground_mask[frame_idx]:
                left_foot_pos = left_foot_verts[frame_idx, :, :2].mean(axis=0)
                ax1.scatter(left_foot_pos[0], left_foot_pos[1], c='blue', s=50, marker='o', alpha=0.8, zorder=6)
            if right_foot_ground_mask[frame_idx]:
                right_foot_pos = right_foot_verts[frame_idx, :, :2].mean(axis=0)
                ax1.scatter(right_foot_pos[0], right_foot_pos[1], c='orange', s=50, marker='o', alpha=0.8, zorder=6)
        else:
            # Use single foot - only plot if it's on ground according to corrected heights
            if left_foot_ground_mask[frame_idx] or right_foot_ground_mask[frame_idx]:
                min_z_idx = np.argmin(verts[frame_idx, :, 2])
                foot_pos = verts[frame_idx, min_z_idx, :2]
                ax1.scatter(foot_pos[0], foot_pos[1], c='purple', s=50, marker='x', alpha=0.8, zorder=6)
    
    # Add legend for ground contacts (these will be added automatically by the scatter plots above)
    # The legend will now include all foot trajectory categories
    
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title(f'Horizontal XY Trajectories - {sequence_name}')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Original foot heights over time
    ax2.plot(time_coords, left_foot_z, 'b-', linewidth=2, label='Left Foot', alpha=0.8)
    ax2.plot(time_coords, right_foot_z, 'r-', linewidth=2, label='Right Foot', alpha=0.8)
    
    # Mark ground contact points
    contact_times = []
    contact_types = []
    
    for frame_idx, left_z, right_z, use_both_feet in foot_contact_info:
        contact_times.append(time_coords[frame_idx])
        contact_types.append('both' if use_both_feet else 'single')
    
    # Plot ground contacts with different markers
    if contact_times:
        both_feet_times = [t for t, typ in zip(contact_times, contact_types) if typ == 'both']
        both_feet_left_heights = [left_foot_z[int(t * fps)] for t in both_feet_times]
        both_feet_right_heights = [right_foot_z[int(t * fps)] for t in both_feet_times]
        single_foot_times = [t for t, typ in zip(contact_times, contact_types) if typ == 'single']
        single_foot_heights = [min(left_foot_z[int(t * fps)], right_foot_z[int(t * fps)]) for t in single_foot_times]
        
        if both_feet_times:
            ax2.scatter(both_feet_times, both_feet_left_heights, c='blue', s=100, marker='o', 
                      label=f'Left Foot Contact ({len(both_feet_times)})', zorder=5, alpha=0.8)
            ax2.scatter(both_feet_times, both_feet_right_heights, c='orange', s=100, marker='o', 
                      label=f'Right Foot Contact ({len(both_feet_times)})', zorder=5, alpha=0.8)
        
        if single_foot_times:
            ax2.scatter(single_foot_times, single_foot_heights, c='purple', s=80, marker='s', 
                      label=f'Single Foot Contact ({len(single_foot_times)})', zorder=5, alpha=0.8)
    
    # Add horizontal line at y=0 for reference
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Ground Level (Z=0)')
    
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Z-coordinate (height)')
    ax2.set_title('Original Foot Heights Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: 3D trajectory of root and feet (using corrected heights)
    # Get corrected foot vertices and positions
    if corrected_verts is None:
        corrected_verts = verts + (corrected_root_trans - root_trans)[:, None, :]
    
    # Get corrected foot vertices
    corrected_left_foot_verts = get_foot_vertices(corrected_verts)[0]
    corrected_right_foot_verts = get_foot_vertices(corrected_verts)[1]
    
    # Get corrected foot positions in 3D
    corrected_left_foot_x = corrected_left_foot_verts[:, :, 0].mean(axis=1)
    corrected_left_foot_y = corrected_left_foot_verts[:, :, 1].mean(axis=1)
    corrected_left_foot_z = corrected_left_foot_verts[:, :, 2].min(axis=1)
    
    corrected_right_foot_x = corrected_right_foot_verts[:, :, 0].mean(axis=1)
    corrected_right_foot_y = corrected_right_foot_verts[:, :, 1].mean(axis=1)
    corrected_right_foot_z = corrected_right_foot_verts[:, :, 2].min(axis=1)
    
    # Get corrected root positions
    corrected_root_x = corrected_root_trans[:, 0]
    corrected_root_y = corrected_root_trans[:, 1]
    corrected_root_z = corrected_root_trans[:, 2]
    
    # Color trajectories by time
    colors = plt.cm.viridis(np.linspace(0, 1, len(corrected_root_x)))
    
    # Plot root trajectory in 3D with color gradient
    for i in range(len(corrected_root_x) - 1):
        ax3.plot(corrected_root_x[i:i+2], corrected_root_y[i:i+2], corrected_root_z[i:i+2], 
                color=colors[i], linewidth=3, alpha=0.8)
    
    # Plot left foot trajectory in 3D
    ax3.plot(corrected_left_foot_x, corrected_left_foot_y, corrected_left_foot_z, 
            'b-', linewidth=2, alpha=0.7, label='Left Foot')
    
    # Plot right foot trajectory in 3D
    ax3.plot(corrected_right_foot_x, corrected_right_foot_y, corrected_right_foot_z, 
            'r-', linewidth=2, alpha=0.7, label='Right Foot')
    
    # Mark start and end points
    ax3.scatter(corrected_root_x[0], corrected_root_y[0], corrected_root_z[0], 
               c='green', s=100, marker='o', label='Root Start', zorder=5)
    ax3.scatter(corrected_root_x[-1], corrected_root_y[-1], corrected_root_z[-1], 
               c='red', s=100, marker='s', label='Root End', zorder=5)
    
    ax3.scatter(corrected_left_foot_x[0], corrected_left_foot_y[0], corrected_left_foot_z[0], 
               c='blue', s=80, marker='^', label='Left Foot Start', zorder=5)
    ax3.scatter(corrected_left_foot_x[-1], corrected_left_foot_y[-1], corrected_left_foot_z[-1], 
               c='blue', s=80, marker='v', label='Left Foot End', zorder=5)
    
    ax3.scatter(corrected_right_foot_x[0], corrected_right_foot_y[0], corrected_right_foot_z[0], 
               c='orange', s=80, marker='^', label='Right Foot Start', zorder=5)
    ax3.scatter(corrected_right_foot_x[-1], corrected_right_foot_y[-1], corrected_right_foot_z[-1], 
               c='orange', s=80, marker='v', label='Right Foot End', zorder=5)
    
    # Mark ground contact points in 3D
    for frame_idx, left_z, right_z, use_both_feet in foot_contact_info:
        if use_both_feet:
            # Use both feet
            left_foot_pos = corrected_left_foot_verts[frame_idx, :, :].mean(axis=0)
            right_foot_pos = corrected_right_foot_verts[frame_idx, :, :].mean(axis=0)
            ax3.scatter(left_foot_pos[0], left_foot_pos[1], left_foot_pos[2], 
                       c='blue', s=50, marker='o', alpha=0.8, zorder=6)
            ax3.scatter(right_foot_pos[0], right_foot_pos[1], right_foot_pos[2], 
                       c='orange', s=50, marker='o', alpha=0.8, zorder=6)
        else:
            # Use single foot
            min_z_idx = np.argmin(corrected_verts[frame_idx, :, 2])
            foot_pos = corrected_verts[frame_idx, min_z_idx, :]
            ax3.scatter(foot_pos[0], foot_pos[1], foot_pos[2], 
                       c='purple', s=50, marker='x', alpha=0.8, zorder=6)
    
    # Add ground plane reference
    x_min, x_max = corrected_root_x.min(), corrected_root_x.max()
    y_min, y_max = corrected_root_y.min(), corrected_root_y.max()
    xx, yy = np.meshgrid([x_min, x_max], [y_min, y_max])
    zz = np.zeros_like(xx)
    ax3.plot_surface(xx, yy, zz, alpha=0.1, color='gray')
    
    ax3.set_xlabel('X Position')
    ax3.set_ylabel('Y Position')
    ax3.set_zlabel('Z Position (height)')
    ax3.set_title('3D Trajectories (Corrected Heights)')
    
    # Create custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Left Foot'),
        Line2D([0], [0], color='red', lw=2, label='Right Foot'),
        Line2D([0], [0], color='green', lw=3, label='Root (time gradient)'),
        Patch(color='gray', alpha=0.1, label='Ground Plane')
    ]
    ax3.legend(handles=legend_elements)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Comparison of original vs corrected foot heights
    ax4.plot(time_coords, left_foot_z, 'b-', linewidth=2, label='Original Left Foot', alpha=0.6)
    ax4.plot(time_coords, right_foot_z, 'r-', linewidth=2, label='Original Right Foot', alpha=0.6)
    ax4.plot(time_coords, corrected_left_foot_z, 'b-', linewidth=2, label='Corrected Left Foot', alpha=0.8)
    ax4.plot(time_coords, corrected_right_foot_z, 'r-', linewidth=2, label='Corrected Right Foot', alpha=0.8)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Ground Level (Z=0)')
    
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Z-coordinate (height)')
    ax4.set_title('Original vs Corrected Foot Heights')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add statistics text
    left_foot_ground_time = np.sum(left_foot_ground_mask) / fps
    right_foot_ground_time = np.sum(right_foot_ground_mask) / fps
    total_ground_time = time_coords[-1]
    
    stats_text = f"""
    Duration: {time_coords[-1]:.2f} seconds
    Ground Contacts: {len(contact_times)}
    - Dual Foot: {sum(1 for typ in contact_types if typ == 'both')}
    - Single Foot: {sum(1 for typ in contact_types if typ == 'single')}
    
    Ground Time:
    - Left Foot: {left_foot_ground_time:.2f}s ({100*left_foot_ground_time/total_ground_time:.1f}%)
    - Right Foot: {right_foot_ground_time:.2f}s ({100*right_foot_ground_time/total_ground_time:.1f}%)
    
    Foot Height Ranges:
    - Left Foot: {left_foot_z.min():.3f} to {left_foot_z.max():.3f}
    - Right Foot: {right_foot_z.min():.3f} to {right_foot_z.max():.3f}
    
    Corrected Foot Height Ranges:
    - Left Foot: {corrected_left_foot_z.min():.3f} to {corrected_left_foot_z.max():.3f}
    - Right Foot: {corrected_right_foot_z.min():.3f} to {corrected_right_foot_z.max():.3f}
    """
    
    # Add text box with statistics
    fig.text(0.02, 0.02, stats_text, fontsize=10, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    return fig

def main(argv=None):
    """Load data and run the same SMPL/ground correction as the converter."""
    args = build_arg_parser().parse_args(argv)
    amass_data = joblib.load(args.amass_data)
    sequences = list(amass_data.keys())
    if args.sequence_name:
        sequences = [seq for seq in sequences if args.sequence_name in str(seq)]
    elif args.num_seq:
        sequences = sequences[:args.num_seq]
    else:
        sequences = sequences[:3]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Analyzing {len(sequences)} sequences for ground contacts with correction...")
    with smpl_robot_context() as smpl_local_robot:
        for i, key_name in enumerate(tqdm(sequences)):
            print(f"\nAnalyzing sequence: {key_name}")
            sequence = correct_smpl_sequence(
                amass_data[key_name], smpl_local_robot,
                plot_results=False, sequence_name=str(key_name),
            )
            fig = plot_trajectory_and_ground_contacts(
                sequence['verts'].numpy(),
                sequence['root_trans'],
                sequence['corrected_root_trans'],
                fps=sequence['fps'],
                sequence_name=str(key_name),
                corrected_verts=sequence['corrected_verts'].numpy(),
            )
            safe_name = str(key_name).replace('/', '_')
            plot_filename = out_dir / f"trajectory_and_ground_contacts_{safe_name}.png"
            fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {plot_filename}")
            if not args.no_show:
                plt.show()
            plt.close(fig)
            if not args.no_show and i < len(sequences) - 1:
                response = input("\nPress Enter to continue to next sequence, or 'q' to quit: ")
                if response.lower() == 'q':
                    break
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
