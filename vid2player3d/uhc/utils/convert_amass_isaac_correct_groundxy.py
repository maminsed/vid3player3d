import joblib
import numpy as np
import os
import sys
import argparse
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from embodied_pose.utils.motion_lib import MotionLib
import torch
from scipy.spatial.transform import Rotation as sRot
import yaml
from tqdm import tqdm

from uhc.smpllib.smpl_parser import SMPL_BONE_ORDER_NAMES as joint_names
from uhc.smpllib.smpl_local_robot import Robot as LocalRobot

from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState

parser = argparse.ArgumentParser()
parser.add_argument('--amass_data', type=str, default="data/amass/amass_copycat_take5_5.pkl")
parser.add_argument('--out_dir', type=str, default="data/motion_lib/amass")
parser.add_argument('--num_seq', type=int, default=None)
parser.add_argument('--num_motion_libs', type=int, default=14)
args = parser.parse_args()

num_seq = args.num_seq
num_motion_libs = args.num_motion_libs

os.makedirs(args.out_dir, exist_ok=True)
meta_data = {
    "amass_data": args.amass_data,
    "num_seq": num_seq,
    "num_motion_libs": num_motion_libs
}
yaml.safe_dump(meta_data, open(f'{args.out_dir}/args.yml', 'w'))

amass_data = joblib.load(args.amass_data)
info = joblib.load('data/misc/smpl_body_info.pkl')


# body_shapes
all_beta = [x['beta'][:10] for x in amass_data.values()]
_, index = np.unique([",".join([f"{x:.6f}" for x in beta]) for beta in all_beta], return_index=True)
index.sort()
beta_arr = [all_beta[i] for i in index]
beta_mapping = dict()
for i, beta in enumerate(beta_arr):
    key = ",".join([f"{x:.6f}" for x in beta])
    beta_mapping[key] = i
print(f'AMASS data has {len(beta_mapping)} unique body shapes!')
joblib.dump({'beta_arr': beta_arr, 'beta_mapping': beta_mapping}, f'{args.out_dir}/shape_data.pkl')

robot_cfg = {
    "mesh": True,
    "model": "smpl",
    "body_params": {},
    "joint_params": {},
    "geom_params": {},
    "actuator_params": {},
}

model_xml_path = f"/tmp/smpl/smpl_mesh_humanoid_v1_convert.xml"

smpl_local_robot = LocalRobot(
    robot_cfg,
    data_dir= "data/smpl",
    model_xml_path=model_xml_path
)


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

amass_full_motion_dict = {}
sequences = np.array(list(amass_data.keys()))
if num_seq is not None:
    sequences = sequences[:num_seq]

seq_mapping = {seq_name.item(): seq_idx for seq_idx, seq_name in enumerate(sequences)}
motion_lib_seq_arr = np.array_split(sequences, num_motion_libs)

seq_name_splits = {}
for i, seq_arr in enumerate(motion_lib_seq_arr):
    seq_name_splits[i] = [seq_name.item()[2:] for seq_name in seq_arr]
joblib.dump(seq_name_splits, f'{args.out_dir}/seq_name_splits.pkl')


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
    # These are typical foot vertex indices in SMPL, but you may need to verify for your specific model
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

def fit_ground_plane_ransac_with_feet(verts, foot_contact_info, time_coords):
    """
    Fit a plane to the ground contact points using both feet when available.
    
    Args:
        verts: (T, V, 3) vertex positions
        foot_contact_info: List of (frame_idx, left_foot_z, right_foot_z, use_both_feet) tuples
        time_coords: (T,) time coordinates for each frame
    
    Returns:
        ground_plane: RANSAC model that takes (x, y, t) and returns z
    """
    if len(foot_contact_info) < 3:
        # If not enough valleys, use all frames with single foot
        foot_contact_info = [(i, verts[i, :, 2].min(), verts[i, :, 2].min(), False) 
                            for i in range(len(verts))]
    
    # Get foot vertices
    left_foot_verts, right_foot_verts = get_foot_vertices(verts)
    
    # Extract ground contact points
    ground_points = []
    for frame_idx, left_z, right_z, use_both_feet in foot_contact_info:
        if use_both_feet:
            # Use both feet for ground plane estimation
            # Get foot vertex positions (X, Y coordinates)
            left_foot_pos = left_foot_verts[frame_idx, :, :2].mean(axis=0)  # Average X, Y
            right_foot_pos = right_foot_verts[frame_idx, :, :2].mean(axis=0)  # Average X, Y
            
            # Add both feet as ground points
            ground_points.append([
                left_foot_pos[0],   # X
                left_foot_pos[1],   # Y
                time_coords[frame_idx],  # T
                left_z              # Z
            ])
            ground_points.append([
                right_foot_pos[0],  # X
                right_foot_pos[1],  # Y
                time_coords[frame_idx],  # T
                right_z             # Z
            ])
        else:
            # Use single foot (minimum Z-coordinate)
            min_z_idx = np.argmin(verts[frame_idx, :, 2])
            ground_points.append([
                verts[frame_idx, min_z_idx, 0],  # X
                verts[frame_idx, min_z_idx, 1],  # Y
                time_coords[frame_idx],          # T
                verts[frame_idx, min_z_idx, 2]   # Z
            ])
    
    ground_points = np.array(ground_points)
    
    # Fit plane using RANSAC: Z = a*X + b*Y + c*T + d
    X = ground_points[:, :3]  # X, Y, T
    Z = ground_points[:, 3]   # Z
    
    # Use RANSAC for robust fitting
    model = RANSACRegressor(
        estimator=LinearRegression(),
        min_samples=3,
        max_trials=100,
        residual_threshold=0.1,
        random_state=42
    )
    model.fit(X, Z)
    
    return model

def get_multi_frequency_windows(T, base_window=15, frequencies=[1, 2]):
    """
    Generate overlapping windows at multiple frequencies for robust estimation.
    
    Args:
        T: Total number of frames
        base_window: Base window size
        frequencies: List of frequency multipliers
    
    Returns:
        windows: List of (start_frame, end_frame) tuples
    """
    windows = []
    
    for freq in frequencies:
        window_size = base_window * freq
        step_size = base_window // 2  # 50% overlap
        
        for start_frame in range(0, T, step_size):
            end_frame = min(start_frame + window_size, T)
            if end_frame - start_frame >= base_window:  # Ensure minimum window size
                windows.append((start_frame, end_frame))
    
    return windows

def correct_ground_height(verts, root_trans, fps=30.0, base_window=15, plot_results=True, sequence_name=""):
    """
    Correct the ground height using multi-frequency sampling and RANSAC with foot information.
    First applies local plane corrections, then shifts to align with z=0.
    
    Args:
        verts: (T, V, 3) vertex positions
        root_trans: (T, 3) root translations
        fps: Frames per second
        base_window: Base window size for multi-frequency sampling
        plot_results: Whether to create plots showing the correction
        sequence_name: Name of the sequence for plot titles
    
    Returns:
        corrected_root_trans: (T, 3) corrected root translations
    """
    T = verts.shape[0]
    time_coords = np.arange(T) / fps
    
    # Get minimum Z-coordinate for each frame (feet position)
    min_verts_z = verts[:, :, 2].min(axis=1)
    
    # Store data for plotting
    original_heights = min_verts_z.copy()
    corrected_heights = min_verts_z.copy()
    valley_points = []
    window_boundaries = []
    
    # Generate multi-frequency windows
    windows = get_multi_frequency_windows(T, base_window)
    print(f"Generated {len(windows)} overlapping windows for multi-frequency sampling")
    
    # Calculate corrected ground height for each frame
    corrected_root_trans = root_trans.copy()
    
    # For each frame, collect predictions from all relevant windows
    frame_predictions = [[] for _ in range(T)]
    frame_weights = [[] for _ in range(T)]
    frame_plane_offsets = [[] for _ in range(T)]  # Store plane offsets from z=0
    
    # Process each window
    for window_idx, (start_frame, end_frame) in enumerate(windows):
        window_frames = np.arange(start_frame, end_frame)
        
        if window_idx % 10 == 0:  # Print progress every 10 windows
            print(f"Processing window {window_idx+1}/{len(windows)}: frames {start_frame}-{end_frame-1}")
        
        # Get data for this window
        window_verts = verts[window_frames]
        window_time_coords = time_coords[window_frames]
        window_min_verts_z = min_verts_z[window_frames]
        
        # Detect ground contact points using foot information
        valley_indices, foot_contact_info = detect_ground_contacts_with_feet(window_verts)
        
        # Convert valley indices to global frame indices
        global_valley_indices = window_frames[valley_indices]
        
        # Store valley points for plotting
        for idx in global_valley_indices:
            valley_points.append((idx, min_verts_z[idx]))
        
        # Store window boundaries for plotting
        window_boundaries.append((start_frame, end_frame))
        
        # Fit ground plane using RANSAC with foot information for this window
        ground_model = fit_ground_plane_ransac_with_feet(window_verts, foot_contact_info, window_time_coords)
        
        # Calculate the offset of this local plane from z=0
        # Use the center of the window to estimate the plane offset
        window_center_idx = (start_frame + end_frame) // 2
        window_center_x = root_trans[window_center_idx, 0]
        window_center_y = root_trans[window_center_idx, 1]
        window_center_time = time_coords[window_center_idx]
        
        # Predict the local plane height at window center
        local_plane_height = ground_model.predict([[window_center_x, window_center_y, window_center_time]])[0]
        plane_offset_from_z0 = local_plane_height  # Distance from local plane to z=0
        
        # Generate predictions for all frames in this window
        for local_idx, global_idx in enumerate(window_frames):
            # Get current root position
            current_root_x = root_trans[global_idx, 0]
            current_root_y = root_trans[global_idx, 1]
            current_time = window_time_coords[local_idx]
            
            # Predict ground height using this window's model (local plane)
            predicted_ground_z = ground_model.predict([[current_root_x, current_root_y, current_time]])[0]
            
            # Calculate weight based on distance from window center
            window_center = (start_frame + end_frame) / 2
            distance_from_center = abs(global_idx - window_center)
            weight = np.exp(-distance_from_center / (base_window / 2))  # Gaussian weight
            
            # Store prediction, weight, and plane offset
            frame_predictions[global_idx].append(predicted_ground_z)
            frame_weights[global_idx].append(weight)
            frame_plane_offsets[global_idx].append(plane_offset_from_z0)
    
    # Combine predictions using weighted average and apply z=0 alignment
    print("Combining predictions and aligning to z=0...")
    for t in range(T):
        if frame_predictions[t]:
            # Weighted average of all predictions for this frame
            predictions = np.array(frame_predictions[t])
            weights = np.array(frame_weights[t])
            plane_offsets = np.array(frame_plane_offsets[t])
            
            # Normalize weights
            weights = weights / weights.sum()
            
            # Calculate weighted average prediction (local plane)
            predicted_local_ground_z = np.sum(predictions * weights)
            
            # Calculate weighted average plane offset from z=0
            weighted_plane_offset = np.sum(plane_offsets * weights)
            
            # Get current minimum Z-coordinate
            current_min_z = min_verts_z[t]
            
            # Apply two-step correction:
            # 1. First, correct to local plane (maintain relative relationships)
            local_height_diff = predicted_local_ground_z - current_min_z
            
            # 2. Then, shift everything so local plane aligns with z=0
            global_height_diff = local_height_diff - weighted_plane_offset
            
            # Apply correction to root translation
            corrected_root_trans[t, 2] += global_height_diff
            
            # Update corrected heights for plotting
            corrected_heights[t] = current_min_z + global_height_diff
    
    # Create plots if requested
    if plot_results:
        create_ground_correction_plots(
            time_coords, original_heights, corrected_heights, 
            valley_points, window_boundaries, sequence_name
        )
    
    return corrected_root_trans

def create_ground_correction_plots(time_coords, original_heights, corrected_heights, 
                                  valley_points, window_boundaries, sequence_name):
    """
    Create plots showing the ground correction process.
    
    Args:
        time_coords: Time coordinates for each frame
        original_heights: Original minimum Z-coordinates
        corrected_heights: Corrected minimum Z-coordinates
        valley_points: List of (frame_idx, height) tuples for detected valleys
        window_boundaries: List of (start_frame, end_frame) tuples for windows
        sequence_name: Name of the sequence for plot titles
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Original vs Corrected Heights
    ax1.plot(time_coords, original_heights, 'b-', label='Original Heights', alpha=0.7)
    ax1.plot(time_coords, corrected_heights, 'r-', label='Corrected Heights', alpha=0.7)
    
    # Mark valley points
    if valley_points:
        valley_times = [time_coords[idx] for idx, _ in valley_points]
        valley_heights = [height for _, height in valley_points]
        ax1.scatter(valley_times, valley_heights, c='green', s=50, label='Ground Contact Points', zorder=5)
    
    # Mark window boundaries
    for start_frame, end_frame in window_boundaries:
        ax1.axvline(x=time_coords[start_frame], color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(x=time_coords[end_frame-1], color='gray', linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Height (Z-coordinate)')
    ax1.set_title(f'Ground Correction Results - {sequence_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Height Difference
    height_diff = corrected_heights - original_heights
    ax2.plot(time_coords, height_diff, 'purple', label='Height Correction')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Mark window boundaries
    for start_frame, end_frame in window_boundaries:
        ax2.axvline(x=time_coords[start_frame], color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(x=time_coords[end_frame-1], color='gray', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Height Difference')
    ax2.set_title('Height Correction Applied')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f"ground_correction_{sequence_name}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Ground correction plot saved as: {plot_filename}")
    
    # Show the plot
    plt.show()


for i, motion_lib_seqs in enumerate(tqdm(motion_lib_seq_arr)):
    
    motion_lib_input_dict = dict()

    for key_name in motion_lib_seqs:
        key_name = key_name.item()
        smpl_data_entry = amass_data[key_name]
        file_name = f"data/amass/singles/{key_name}.npy"
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
            import ipdb
            ipdb.set_trace()
            raise Exception("Gender Not Supported!!")
        
        batch_size = pose_aa.shape[0]
        pose_aa = np.concatenate([pose_aa[:, :66], np.zeros((batch_size, 6))], axis=1)  # TODO: need to extract correct handle rotations instead of zero
        pose_quat = sRot.from_rotvec(pose_aa.reshape(-1, 3)).as_quat().reshape(batch_size, 24, 4)[..., smpl_2_mujoco, :]
        smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None, ]), gender=gender_number)
        smpl_local_robot.write_xml()
        skeleton_tree = SkeletonTree.from_mjcf(model_xml_path)
        root_trans = trans + skeleton_tree.local_translation[0].numpy()
        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat),
            torch.from_numpy(root_trans),
            is_local=True)

        verts, joints = smpl_parser.get_joints_verts(
            pose=torch.from_numpy(pose_aa),
            th_betas=torch.from_numpy(beta[None, ]),
            th_trans=torch.from_numpy(trans)
        )

        # Apply ground correction to keep character properly grounded
        print(f"Applying ground correction for sequence {key_name}")
        corrected_root_trans = correct_ground_height(
            verts.numpy(), 
            root_trans, 
            fps=fps,
            sequence_name=key_name
        )
        
        # Update the skeleton state with corrected root translation
        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat),
            torch.from_numpy(corrected_root_trans),
            is_local=True)

        # Recalculate vertices with corrected root translation
        verts, joints = smpl_parser.get_joints_verts(
            pose=torch.from_numpy(pose_aa),
            th_betas=torch.from_numpy(beta[None, ]),
            th_trans=torch.from_numpy(corrected_root_trans)
        )

        # min_verts_h = verts[..., 2].min().item()

        min_verts_h = verts[..., 2].min(dim=-1)[0][0]

        beta_key = ",".join([f"{x:.6f}" for x in beta])

        new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=fps)
        new_motion_out = new_motion.to_dict()
        new_motion_out['seq_name'] = key_name
        new_motion_out['seq_idx'] = seq_mapping[key_name]
        new_motion_out['trans'] = trans
        new_motion_out['root_trans'] = corrected_root_trans  # Use corrected root translation
        new_motion_out['pose_aa'] = pose_aa
        new_motion_out['beta'] = beta
        new_motion_out['beta_idx'] = beta_mapping[beta_key]
        new_motion_out['gender'] = gender
        new_motion_out['min_verts_h'] = min_verts_h
        new_motion_out['body_scale'] = 1.0
        new_motion_out['__name__'] = "SkeletonMotion"
        motion_lib_input_dict[key_name] = new_motion_out

    motion_lib = MotionLib(motion_file=motion_lib_input_dict,
        dof_body_ids=info['dof_body_ids'],
        dof_offsets=info['dof_offsets'],
        key_body_ids=info['key_body_ids'],
        device='cpu',
        clean_up=True
    )

    torch.save(motion_lib, f"{args.out_dir}/mlib_part_{i:05d}.pth")

    del motion_lib_input_dict
    del motion_lib


smpl_local_robot.clean_up()
