import joblib
import numpy as np
import os
import sys
import argparse
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.patches as patches

sys.path.append(os.getcwd())
from embodied_pose.utils.motion_lib import MotionLib
import torch
from scipy.spatial.transform import Rotation as sRot
import yaml
from tqdm import tqdm

from uhc.smpllib.smpl_parser import SMPL_BONE_ORDER_NAMES as joint_names
from uhc.smpllib.smpl_local_robot import Robot as LocalRobot

from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState

parser = argparse.ArgumentParser()
parser.add_argument('--amass_data', type=str, default="data/amass/amass_copycat_take5_5.pkl")
parser.add_argument('--out_dir', type=str, default="data/motion_lib/amass")
parser.add_argument('--num_seq', type=int, default=None)
parser.add_argument('--sequence_name', type=str, default=None, help="Specific sequence to analyze")
args = parser.parse_args()

# Reuse the ground contact detection functions from the original script
def get_foot_vertices(verts):
    """
    Extract left and right foot vertices from SMPL mesh.
    
    Args:
        verts: (T, V, 3) vertex positions
    
    Returns:
        left_foot_verts: (T, N, 3) left foot vertices
        right_foot_verts: (T, N, 3) right foot vertices
    """
    # SMPL foot vertex indices (approximate - may need adjustment based on your SMPL model)
    left_foot_indices = [
        3216, 3217, 3218, 3219, 3220, 3221, 3222, 3223, 3224, 3225,  # Left foot bottom
        3226, 3227, 3228, 3229, 3230, 3231, 3232, 3233, 3234, 3235,
        3236, 3237, 3238, 3239, 3240, 3241, 3242, 3243, 3244, 3245,
        3246, 3247, 3248, 3249, 3250, 3251, 3252, 3253, 3254, 3255
    ]
    
    right_foot_indices = [
        6746, 6747, 6748, 6749, 6750, 6751, 6752, 6753, 6754, 6755,  # Right foot bottom
        6756, 6757, 6758, 6759, 6760, 6761, 6762, 6763, 6764, 6765,
        6766, 6767, 6768, 6769, 6770, 6771, 6772, 6773, 6774, 6775,
        6776, 6777, 6778, 6779, 6780, 6781, 6782, 6783, 6784, 6785
    ]
    
    left_foot_verts = verts[:, left_foot_indices, :]
    right_foot_verts = verts[:, right_foot_indices, :]
    
    return left_foot_verts, right_foot_verts

def detect_ground_contacts_with_feet(verts, window_size=3, prominence=0.02, foot_threshold=0.05):
    """
    Detect ground contact points using both feet information.
    
    Args:
        verts: (T, V, 3) vertex positions
        window_size: Size of smoothing window
        prominence: Minimum prominence for peak detection
        foot_threshold: Maximum height difference between feet to consider as ground contact
    
    Returns:
        valley_indices: Indices of detected ground contact frames
        foot_contact_info: List of (frame_idx, left_foot_z, right_foot_z, use_both_feet) tuples
    """
    # Get foot vertices
    left_foot_verts, right_foot_verts = get_foot_vertices(verts)
    
    # Get minimum Z-coordinate for each frame (overall feet position)
    min_verts_z = verts[:, :, 2].min(axis=1)
    
    # Get minimum Z-coordinate for each foot
    left_foot_z = left_foot_verts[:, :, 2].min(axis=1)
    right_foot_z = right_foot_verts[:, :, 2].min(axis=1)
    
    # Smooth the signal to reduce noise
    smoothed = np.convolve(min_verts_z, np.ones(window_size)/window_size, mode='same')
    
    # Find valleys (negative peaks)
    valley_indices, _ = find_peaks(-smoothed, prominence=prominence)
    
    # Check foot contact information for each valley
    foot_contact_info = []
    for idx in valley_indices:
        left_z = left_foot_z[idx]
        right_z = right_foot_z[idx]
        
        # Check if both feet are close to the ground
        feet_height_diff = abs(left_z - right_z)
        use_both_feet = feet_height_diff < foot_threshold
        
        foot_contact_info.append((idx, left_z, right_z, use_both_feet))
    
    return valley_indices, foot_contact_info

def analyze_horizontal_trajectory(root_trans, verts, fps=30.0, sequence_name=""):
    """
    Analyze horizontal trajectory and detect potential drifting.
    
    Args:
        root_trans: (T, 3) root translations
        verts: (T, V, 3) vertex positions
        fps: Frames per second
        sequence_name: Name of the sequence for plot titles
    
    Returns:
        trajectory_data: Dictionary containing analysis results
    """
    T = root_trans.shape[0]
    time_coords = np.arange(T) / fps
    
    # Extract horizontal coordinates (X, Y)
    x_coords = root_trans[:, 0]
    y_coords = root_trans[:, 1]
    
    # Detect ground contact points
    valley_indices, foot_contact_info = detect_ground_contacts_with_feet(verts)
    
    # Get foot positions for ground contacts
    left_foot_verts, right_foot_verts = get_foot_vertices(verts)
    
    ground_contact_positions = []
    for frame_idx, left_z, right_z, use_both_feet in foot_contact_info:
        if use_both_feet:
            # Use both feet
            left_foot_pos = left_foot_verts[frame_idx, :, :2].mean(axis=0)  # Average X, Y
            right_foot_pos = right_foot_verts[frame_idx, :, :2].mean(axis=0)  # Average X, Y
            ground_contact_positions.append({
                'frame': frame_idx,
                'time': time_coords[frame_idx],
                'left_foot': left_foot_pos,
                'right_foot': right_foot_pos,
                'root_pos': root_trans[frame_idx, :2],
                'use_both_feet': True
            })
        else:
            # Use single foot (minimum Z-coordinate)
            min_z_idx = np.argmin(verts[frame_idx, :, 2])
            foot_pos = verts[frame_idx, min_z_idx, :2]
            ground_contact_positions.append({
                'frame': frame_idx,
                'time': time_coords[frame_idx],
                'left_foot': foot_pos,
                'right_foot': foot_pos,
                'root_pos': root_trans[frame_idx, :2],
                'use_both_feet': False
            })
    
    # Calculate trajectory statistics
    total_distance = np.sum(np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2))
    straight_line_distance = np.sqrt((x_coords[-1] - x_coords[0])**2 + (y_coords[-1] - y_coords[0])**2)
    efficiency = straight_line_distance / total_distance if total_distance > 0 else 0
    
    # Detect potential drifting by analyzing velocity changes
    velocities = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2) / (1/fps)
    velocity_smooth = np.convolve(velocities, np.ones(5)/5, mode='same')
    
    # Find periods of low velocity (potential stationary periods)
    low_velocity_threshold = np.percentile(velocity_smooth, 25)
    stationary_periods = velocity_smooth < low_velocity_threshold
    
    return {
        'time_coords': time_coords,
        'x_coords': x_coords,
        'y_coords': y_coords,
        'velocities': velocities,
        'velocity_smooth': velocity_smooth,
        'ground_contacts': ground_contact_positions,
        'stationary_periods': stationary_periods,
        'total_distance': total_distance,
        'straight_line_distance': straight_line_distance,
        'efficiency': efficiency,
        'sequence_name': sequence_name
    }

def create_interactive_trajectory_plot(trajectory_data):
    """
    Create an interactive matplotlib plot for analyzing horizontal trajectories.
    
    Args:
        trajectory_data: Dictionary containing trajectory analysis results
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Create subplots
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], width_ratios=[1, 1])
    
    # Main trajectory plot
    ax_traj = fig.add_subplot(gs[0, :])
    
    # Velocity plot
    ax_vel = fig.add_subplot(gs[1, :])
    
    # Statistics panel
    ax_stats = fig.add_subplot(gs[2, 0])
    ax_contacts = fig.add_subplot(gs[2, 1])
    
    # Plot main trajectory
    x_coords = trajectory_data['x_coords']
    y_coords = trajectory_data['y_coords']
    time_coords = trajectory_data['time_coords']
    
    # Color trajectory by time
    colors = plt.cm.viridis(np.linspace(0, 1, len(x_coords)))
    
    # Plot trajectory with color gradient
    for i in range(len(x_coords) - 1):
        ax_traj.plot(x_coords[i:i+2], y_coords[i:i+2], color=colors[i], linewidth=2, alpha=0.8)
    
    # Mark start and end points
    ax_traj.scatter(x_coords[0], y_coords[0], c='green', s=100, marker='o', label='Start', zorder=5)
    ax_traj.scatter(x_coords[-1], y_coords[-1], c='red', s=100, marker='s', label='End', zorder=5)
    
    # Plot ground contact points
    for contact in trajectory_data['ground_contacts']:
        if contact['use_both_feet']:
            # Plot both feet
            ax_traj.scatter(contact['left_foot'][0], contact['left_foot'][1], 
                          c='blue', s=50, marker='^', alpha=0.7, zorder=4)
            ax_traj.scatter(contact['right_foot'][0], contact['right_foot'][1], 
                          c='orange', s=50, marker='v', alpha=0.7, zorder=4)
        else:
            # Plot single foot
            ax_traj.scatter(contact['left_foot'][0], contact['left_foot'][1], 
                          c='purple', s=50, marker='x', alpha=0.7, zorder=4)
    
    # Add legend for ground contacts
    ax_traj.scatter([], [], c='blue', s=50, marker='^', label='Left Foot Contact')
    ax_traj.scatter([], [], c='orange', s=50, marker='v', label='Right Foot Contact')
    ax_traj.scatter([], [], c='purple', s=50, marker='x', label='Single Foot Contact')
    
    ax_traj.set_xlabel('X Position')
    ax_traj.set_ylabel('Y Position')
    ax_traj.set_title(f'Horizontal Trajectory - {trajectory_data["sequence_name"]}')
    ax_traj.legend()
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_aspect('equal')
    
    # Plot velocity over time
    velocities = trajectory_data['velocities']
    velocity_smooth = trajectory_data['velocity_smooth']
    stationary_periods = trajectory_data['stationary_periods']
    
    ax_vel.plot(time_coords[1:], velocities, 'b-', alpha=0.5, label='Instantaneous Velocity')
    ax_vel.plot(time_coords[1:], velocity_smooth, 'r-', linewidth=2, label='Smoothed Velocity')
    
    # Highlight stationary periods
    stationary_times = time_coords[1:][stationary_periods]
    stationary_vels = velocity_smooth[stationary_periods]
    ax_vel.scatter(stationary_times, stationary_vels, c='green', s=30, alpha=0.7, label='Stationary Periods')
    
    # Mark ground contact times
    contact_times = [contact['time'] for contact in trajectory_data['ground_contacts']]
    contact_vels = [velocity_smooth[int(contact['time'] * 30)] if int(contact['time'] * 30) < len(velocity_smooth) else 0 
                   for contact in trajectory_data['ground_contacts']]
    ax_vel.scatter(contact_times, contact_vels, c='red', s=50, marker='o', label='Ground Contacts', zorder=5)
    
    ax_vel.set_xlabel('Time (seconds)')
    ax_vel.set_ylabel('Velocity (units/frame)')
    ax_vel.set_title('Velocity Profile')
    ax_vel.legend()
    ax_vel.grid(True, alpha=0.3)
    
    # Statistics panel
    ax_stats.axis('off')
    stats_text = f"""
    Trajectory Statistics:
    
    Total Distance: {trajectory_data['total_distance']:.2f}
    Straight Line Distance: {trajectory_data['straight_line_distance']:.2f}
    Efficiency: {trajectory_data['efficiency']:.3f}
    
    Ground Contacts: {len(trajectory_data['ground_contacts'])}
    Dual Foot Contacts: {sum(1 for c in trajectory_data['ground_contacts'] if c['use_both_feet'])}
    Single Foot Contacts: {sum(1 for c in trajectory_data['ground_contacts'] if not c['use_both_feet'])}
    
    Sequence Length: {len(time_coords)} frames
    Duration: {time_coords[-1]:.2f} seconds
    """
    ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes, fontsize=10, 
                 verticalalignment='top', fontfamily='monospace')
    
    # Ground contact details
    ax_contacts.axis('off')
    if trajectory_data['ground_contacts']:
        contact_text = "Ground Contact Details:\n\n"
        for i, contact in enumerate(trajectory_data['ground_contacts'][:10]):  # Show first 10
            contact_text += f"Frame {contact['frame']}: "
            if contact['use_both_feet']:
                contact_text += "Dual foot\n"
            else:
                contact_text += "Single foot\n"
        
        if len(trajectory_data['ground_contacts']) > 10:
            contact_text += f"\n... and {len(trajectory_data['ground_contacts']) - 10} more"
    else:
        contact_text = "No ground contacts detected"
    
    ax_contacts.text(0.1, 0.9, contact_text, transform=ax_contacts.transAxes, fontsize=9, 
                    verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Add interactive elements
    def on_hover(event):
        if event.inaxes == ax_traj:
            # Find closest point
            distances = np.sqrt((x_coords - event.xdata)**2 + (y_coords - event.ydata)**2)
            closest_idx = np.argmin(distances)
            
            # Update title with frame info
            ax_traj.set_title(f'Horizontal Trajectory - {trajectory_data["sequence_name"]} '
                            f'(Frame {closest_idx}, Time {time_coords[closest_idx]:.2f}s)')
            fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    
    return fig

def main():
    """Main function to analyze horizontal trajectory drifting."""
    
    # Load AMASS data
    amass_data = joblib.load(args.amass_data)
    info = joblib.load('data/misc/smpl_body_info.pkl')
    
    # Setup SMPL robot
    robot_cfg = {
        "mesh": True,
        "model": "smpl",
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
    }
    
    model_xml_path = f"./embodied_pose/data/mjcf/smpl_mesh_humanoid_v1_convert.xml"
    
    smpl_local_robot = LocalRobot(
        robot_cfg,
        data_dir="data/smpl",
        model_xml_path=model_xml_path
    )
    
    # SMPL to MuJoCo joint mapping
    mujoco_joint_names = [
        'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee',
        'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax',
        'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder',
        'R_Elbow', 'R_Wrist', 'R_Hand'
    ]
    smpl_2_mujoco = [
        joint_names.index(q) for q in mujoco_joint_names
        if q in joint_names
    ]
    
    # Select sequences to analyze
    sequences = list(amass_data.keys())
    if args.sequence_name:
        sequences = [seq for seq in sequences if args.sequence_name in str(seq)]
    elif args.num_seq:
        sequences = sequences[:args.num_seq]
    else:
        sequences = sequences[:5]  # Default to first 5 sequences
    
    print(f"Analyzing {len(sequences)} sequences for horizontal trajectory drifting...")
    
    for i, key_name in enumerate(tqdm(sequences)):
        key_name = key_name.item() if hasattr(key_name, 'item') else key_name
        
        print(f"\nAnalyzing sequence: {key_name}")
        
        # Load sequence data
        smpl_data_entry = amass_data[key_name]
        seq_len = smpl_data_entry['pose_aa'].shape[0]
        
        pose_aa = smpl_data_entry['pose_aa'].copy()
        trans = smpl_data_entry['trans_orig'].copy()
        beta = smpl_data_entry['beta'][:10].copy()
        gender = smpl_data_entry['gender']
        fps = 30.0
        
        if isinstance(gender, np.ndarray):
            gender = gender.item()
        if isinstance(gender, bytes):
            gender = gender.decode("utf-8")
        
        if gender == "neutral":
            gender_number = [0]
            smpl_parser = smpl_local_robot.smpl_parser_n
        elif gender == "male":
            gender_number = [1]
            smpl_parser = smpl_local_robot.smpl_parser_m
        elif gender == "female":
            gender_number = [2]
            smpl_parser = smpl_local_robot.smpl_parser_f
        else:
            print(f"Warning: Unknown gender {gender}, using neutral")
            gender_number = [0]
            smpl_parser = smpl_local_robot.smpl_parser_n
        
        # Process pose data
        batch_size = pose_aa.shape[0]
        pose_aa = np.concatenate([pose_aa[:, :66], np.zeros((batch_size, 6))], axis=1)
        pose_quat = sRot.from_rotvec(pose_aa.reshape(-1, 3)).as_quat().reshape(batch_size, 24, 4)[..., smpl_2_mujoco, :]
        
        smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None, ]), gender=gender_number)
        smpl_local_robot.write_xml()
        skeleton_tree = SkeletonTree.from_mjcf(model_xml_path)
        root_trans = trans + skeleton_tree.local_translation[0].numpy()
        
        # Get vertices for ground contact detection
        verts, joints = smpl_parser.get_joints_verts(
            pose=torch.from_numpy(pose_aa),
            th_betas=torch.from_numpy(beta[None, ]),
            th_trans=torch.from_numpy(trans)
        )
        
        # Analyze horizontal trajectory
        trajectory_data = analyze_horizontal_trajectory(
            root_trans, 
            verts.numpy(), 
            fps=fps,
            sequence_name=str(key_name)
        )
        
        # Create interactive plot
        fig = create_interactive_trajectory_plot(trajectory_data)
        
        # Save plot
        plot_filename = f"horizontal_trajectory_{str(key_name).replace('/', '_')}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {plot_filename}")
        
        # Show plot (interactive)
        plt.show()
        
        # Ask user if they want to continue to next sequence
        if i < len(sequences) - 1:
            response = input(f"\nPress Enter to continue to next sequence, or 'q' to quit: ")
            if response.lower() == 'q':
                break
    
    smpl_local_robot.clean_up()
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 
