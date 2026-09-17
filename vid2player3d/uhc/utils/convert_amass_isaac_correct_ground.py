"""Ground correction helpers and AMASS-to-Isaac motion library conversion.

Importing this module does not parse arguments, load motion data, or initialize
SMPL/MuJoCo. Simulation dependencies are loaded only when converting a sequence.
"""

import argparse
import ast
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import tempfile

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression, RANSACRegressor


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Convert AMASS motions with ground correction")
    parser.add_argument('--amass_data', type=str, default="data/amass/amass_copycat_take5_5.pkl")
    parser.add_argument('--tennisproject_data', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default="data/motion_lib/amass")
    parser.add_argument('--num_seq', type=int, default=None)
    parser.add_argument('--num_motion_libs', type=int, default=14)
    parser.add_argument('--disable_xy_correction', action='store_true')
    parser.add_argument('--tp_xy_units', type=str, default='pixels', choices=['meters', 'pixels'])
    parser.add_argument('--tp_pixel_to_meters', type=float, default=0.01003)
    parser.add_argument('--tp_frame_window', type=int, default=5)
    parser.add_argument('--tp_frame_offset', type=int, default=0)
    parser.add_argument('--tp_flip_y', action='store_true')
    parser.add_argument('--tp_xy_smoothing', type=int, default=0)
    parser.add_argument('--tp_net_x1', type=float, default=286.0)
    parser.add_argument('--tp_net_y1', type=float, default=1748.0)
    parser.add_argument('--tp_net_x2', type=float, default=1379.0)
    parser.add_argument('--tp_net_y2', type=float, default=1748.0)
    parser.add_argument('--trim_frames', type=int, default=0)
    parser.add_argument('--no_show', action='store_true', help='Save plots without opening windows')
    return parser


def safe_literal_eval(value):
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None
    return value


def parse_xy_point(value):
    parsed = safe_literal_eval(value)
    if parsed is None:
        return None
    try:
        arr = np.asarray(parsed, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    arr = np.squeeze(arr)
    if arr.ndim != 1 or arr.shape[0] < 2:
        return None
    return arr[:2]


def parse_person_bottom_point(value, point_column):
    parsed = safe_literal_eval(value)
    if parsed is None:
        return None

    if point_column == 'person_points':
        if not isinstance(parsed, (list, tuple, np.ndarray)) or len(parsed) < 2:
            return None
        return parse_xy_point(parsed[1])

    return parse_xy_point(parsed)


def prepare_xy_context(args, output_video_name):
    xy_context = {
        "enabled": False,
        "xy_smoothing": max(0, int(args.tp_xy_smoothing)),
    }

    if args.disable_xy_correction:
        print("XY correction disabled via --disable_xy_correction")
        return xy_context

    if args.tennisproject_data is None:
        print("Warning: --tennisproject_data is not provided. XY correction will be skipped.")
        return xy_context

    if args.tp_xy_units == 'meters' and args.tp_pixel_to_meters <= 0:
        raise ValueError("--tp_pixel_to_meters must be > 0 when --tp_xy_units=meters")

    if args.tp_xy_units == 'pixels':
        print("Warning: XY units are set to pixels. This keeps video scale but may not match metric world units.")

    tp_data = pd.read_csv(args.tennisproject_data)
    if 'output_video_name' in tp_data.columns:
        tp_data = tp_data[tp_data['output_video_name'] == output_video_name]
    else:
        print("Warning: 'output_video_name' column missing in tennisproject CSV. Using all rows.")

    if tp_data.empty:
        print(f"Warning: no tennisproject rows found for '{output_video_name}'. XY correction will be skipped.")
        return xy_context

    frame_column = 'local_frame' if 'local_frame' in tp_data.columns else ('frame' if 'frame' in tp_data.columns else None)
    if frame_column is None:
        print("Warning: tennisproject CSV must contain 'local_frame' or 'frame'. XY correction will be skipped.")
        return xy_context

    if 'person_point_bottom' in tp_data.columns:
        point_column = 'person_point_bottom'
    elif 'person_points' in tp_data.columns:
        point_column = 'person_points'
    else:
        print("Warning: tennisproject CSV missing person point columns. XY correction will be skipped.")
        return xy_context

    net_center = np.array([
        (args.tp_net_x1 + args.tp_net_x2) / 2.0,
        (args.tp_net_y1 + args.tp_net_y2) / 2.0,
    ], dtype=np.float64)

    print(f"XY correction enabled. net_center={net_center.tolist()}, units={args.tp_xy_units}")

    xy_context.update({
        "enabled": True,
        "tp_data": tp_data,
        "frame_column": frame_column,
        "point_column": point_column,
        "frame_window": max(0, int(args.tp_frame_window)),
        "frame_offset": int(args.tp_frame_offset),
        "xy_units": args.tp_xy_units,
        "pixel_to_meters": float(args.tp_pixel_to_meters),
        "flip_y": bool(args.tp_flip_y),
        "xy_smoothing": max(0, int(args.tp_xy_smoothing)),
        "net_center": net_center,
    })
    return xy_context


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

def get_multi_frequency_windows(T, base_window=15, frequencies=(1, 2)):
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

def correct_ground_height(verts, root_trans, fps=30.0, base_window=15,
                          plot_results=True, sequence_name="", xy_context=None,
                          out_dir=".", show_plots=True):
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
        xy_context: Optional court tracking configuration from prepare_xy_context
        out_dir: Directory for diagnostic plots
        show_plots: Whether to display diagnostic plots interactively
    
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

    if xy_context is None:
        xy_context = {"enabled": False}
    
    # Process each window
    for window_idx, (start_frame, end_frame) in enumerate(windows):
        window_frames = np.arange(start_frame, end_frame)
        
        if window_idx % 10 == 0:  # Print progress every 10 windows
            print(f"Processing window {window_idx+1}/{len(windows)}: frames {start_frame}-{end_frame-1}")
        
        # Get data for this window
        window_verts = verts[window_frames]
        window_time_coords = time_coords[window_frames]
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

        if xy_context["enabled"]:
            aligned_t = t + xy_context["frame_offset"]
            frame_window = xy_context["frame_window"]
            frame_col = xy_context["frame_column"]
            point_col = xy_context["point_column"]
            frame_mask = (
                (xy_context["tp_data"][frame_col] >= aligned_t - frame_window)
                & (xy_context["tp_data"][frame_col] <= aligned_t + frame_window)
            )
            data_list = xy_context["tp_data"].loc[frame_mask, point_col].tolist()
            parsed_data = [parse_person_bottom_point(entry, point_col) for entry in data_list]
            parsed_data = [entry for entry in parsed_data if entry is not None]

            if parsed_data:
                tp_bottom_person_xy = np.mean(np.stack(parsed_data, axis=0), axis=0)
                xy_new = tp_bottom_person_xy - xy_context["net_center"]

                if xy_context["xy_units"] == 'meters':
                    xy_new = xy_new * xy_context["pixel_to_meters"]

                if xy_context["flip_y"]:
                    xy_new[1] *= -1.0

                corrected_root_trans[t, 0] = float(xy_new[0])
                corrected_root_trans[t, 1] = float(xy_new[1])
            elif t > 0:
                print("prev x,y")
                corrected_root_trans[t, 0] = corrected_root_trans[t - 1, 0]
                corrected_root_trans[t, 1] = corrected_root_trans[t - 1, 1]

    if xy_context.get("xy_smoothing", 0) > 1:
        kernel_size = int(xy_context["xy_smoothing"])
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones(kernel_size, dtype=np.float64) / kernel_size
        for axis in (0, 1):
            corrected_root_trans[:, axis] = np.convolve(corrected_root_trans[:, axis], kernel, mode='same')
    
    # Create plots if requested
    if plot_results:
        create_ground_correction_plots(
            time_coords, original_heights, corrected_heights, 
            valley_points, window_boundaries, sequence_name,
            out_dir=out_dir, show=show_plots,
        )
    
    return corrected_root_trans

def create_ground_correction_plots(time_coords, original_heights, corrected_heights, 
                                  valley_points, window_boundaries, sequence_name,
                                  out_dir=".", show=True):
    """
    Create plots showing the ground correction process.
    
    Args:
        time_coords: Time coordinates for each frame
        original_heights: Original minimum Z-coordinates
        corrected_heights: Corrected minimum Z-coordinates
        valley_points: List of (frame_idx, height) tuples for detected valleys
        window_boundaries: List of (start_frame, end_frame) tuples for windows
        sequence_name: Name of the sequence for plot titles
        out_dir: Directory in which to save the plot
        show: Whether to display the plot interactively
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
    os.makedirs(out_dir, exist_ok=True)
    safe_name = str(sequence_name).replace("/", "_")
    plot_filename = os.path.join(out_dir, f"ground_correction_{safe_name}.png")
    fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Ground correction plot saved as: {plot_filename}")
    
    # Show the plot
    if show:
        plt.show()
    plt.close(fig)


@contextmanager
def smpl_robot_context(data_dir="data/smpl"):
    """Own temporary XML/geometry files and clean them up even on failure."""
    from vid2player3d.uhc.smpllib.smpl_local_robot import Robot as LocalRobot

    robot_cfg = {
        "mesh": True,
        "model": "smpl",
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
    }
    with tempfile.TemporaryDirectory(prefix="vid3player-smpl-") as model_dir:
        robot = LocalRobot(
            robot_cfg,
            data_dir=data_dir,
            model_xml_path=os.path.join(model_dir, "smpl_mesh_humanoid_v1_convert.xml"),
        )
        try:
            yield robot
        finally:
            robot.clean_up()


def correct_smpl_sequence(smpl_data_entry, smpl_local_robot, fps=30.0,
                          base_window=15, plot_results=True, sequence_name="",
                          xy_context=None, out_dir=".", show_plots=True):
    """Reconstruct and correct a sequence for both conversion and visualization.

    ``trans`` is SMPL model translation; ``root_trans`` is the pelvis world
    position. Returned original/corrected vertices use the corresponding SMPL
    translations, while skeleton states must use ``corrected_root_trans``.
    """
    import torch
    from scipy.spatial.transform import Rotation as sRot
    from uhc.smpllib.smpl_parser import SMPL_BONE_ORDER_NAMES as joint_names
    from vid2player3d.poselib.poselib.skeleton.skeleton3d import SkeletonTree

    pose_aa = smpl_data_entry['pose_aa'].copy()
    trans = smpl_data_entry['trans_orig'].copy()
    beta = smpl_data_entry['beta'][:10].copy()
    gender = smpl_data_entry['gender']
    if isinstance(gender, np.ndarray):
        gender = gender.item()
    if isinstance(gender, bytes):
        gender = gender.decode("utf-8")
    if gender not in ("neutral", "male", "female"):
        raise ValueError(f"Unsupported gender: {gender!r}")
    gender_number, parser_attribute = {
        "neutral": (0, "smpl_parser_n"),
        "male": (1, "smpl_parser_m"),
        "female": (2, "smpl_parser_f"),
    }[gender]
    smpl_parser = getattr(smpl_local_robot, parser_attribute)

    mujoco_joint_names = [
        'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee',
        'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax',
        'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder',
        'R_Elbow', 'R_Wrist', 'R_Hand',
    ]
    smpl_2_mujoco = [joint_names.index(name) for name in mujoco_joint_names]
    batch_size = pose_aa.shape[0]
    pose_aa = np.concatenate([pose_aa[:, :66], np.zeros((batch_size, 6))], axis=1)
    pose_quat = sRot.from_rotvec(pose_aa.reshape(-1, 3)).as_quat().reshape(
        batch_size, 24, 4
    )[..., smpl_2_mujoco, :]

    with torch.no_grad():
        smpl_local_robot.load_from_skeleton(
            betas=torch.from_numpy(beta[None, :]), gender=[gender_number]
        )
        smpl_local_robot.write_xml()
        skeleton_tree = SkeletonTree.from_mjcf(smpl_local_robot.model_xml_path)
        pelvis_offset = skeleton_tree.local_translation[0].numpy()
        root_trans = trans + pelvis_offset
        verts, _ = smpl_parser.get_joints_verts(
            pose=torch.from_numpy(pose_aa),
            th_betas=torch.from_numpy(beta[None, :]),
            th_trans=torch.from_numpy(trans),
        )
        corrected_root_trans = correct_ground_height(
            verts.numpy(), root_trans, fps=fps, base_window=base_window,
            plot_results=plot_results, sequence_name=sequence_name,
            xy_context=xy_context, out_dir=out_dir, show_plots=show_plots,
        )
        # Convert pelvis world position back to SMPL model translation.
        corrected_trans = corrected_root_trans - pelvis_offset
        corrected_verts, _ = smpl_parser.get_joints_verts(
            pose=torch.from_numpy(pose_aa),
            th_betas=torch.from_numpy(beta[None, :]),
            th_trans=torch.from_numpy(corrected_trans),
        )

    return {
        "pose_aa": pose_aa,
        "pose_quat": pose_quat,
        "beta": beta,
        "gender": gender,
        "fps": fps,
        "skeleton_tree": skeleton_tree,
        "trans": trans,
        "root_trans": root_trans,
        "verts": verts,
        "corrected_trans": corrected_trans,
        "corrected_root_trans": corrected_root_trans,
        "corrected_verts": corrected_verts,
    }


def build_motion_output(sequence, seq_name, seq_idx, beta_idx, trim_frames=0):
    """Build simulation and render outputs using consistent corrected positions."""
    import torch
    from vid2player3d.poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState

    frame_count = len(sequence['pose_aa'])
    if trim_frames < 0 or frame_count - 2 * trim_frames < 2:
        raise ValueError("trim_frames must be nonnegative and leave at least two frames for velocities")
    frames = slice(trim_frames, -trim_frames if trim_frames else None)
    corrected_trans = sequence['corrected_trans'][frames]
    corrected_root_trans = sequence['corrected_root_trans'][frames]
    verts = sequence['corrected_verts'][frames]
    state = SkeletonState.from_rotation_and_root_translation(
        sequence['skeleton_tree'],
        torch.from_numpy(sequence['pose_quat'][frames]),
        torch.from_numpy(corrected_root_trans),
        is_local=True,
    )
    motion_out = SkeletonMotion.from_skeleton_state(state, fps=sequence['fps']).to_dict()
    motion_out.update({
        'seq_name': seq_name,
        'seq_idx': seq_idx,
        'trans': corrected_trans,
        'root_trans': corrected_root_trans,
        'pose_aa': sequence['pose_aa'][frames],
        'beta': sequence['beta'],
        'beta_idx': beta_idx,
        'gender': sequence['gender'],
        'min_verts_h': verts[..., 2].min(dim=-1)[0].mean().item(),
        'body_scale': 1.0,
        '__name__': "SkeletonMotion",
    })
    render_data = {'verts': verts.clone(), 'fps': sequence['fps']}
    return motion_out, render_data


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.num_motion_libs < 1:
        raise ValueError("num_motion_libs must be positive")
    # MotionLib imports Isaac Gym; keep this before importing torch directly.
    from embodied_pose.utils.motion_lib import MotionLib
    import torch
    import yaml
    from tqdm import tqdm

    num_seq = args.num_seq
    num_motion_libs = args.num_motion_libs

    os.makedirs(args.out_dir, exist_ok=True)
    meta_data = {
        "amass_data": args.amass_data,
        "tennisproject_data": args.tennisproject_data,
        "num_seq": num_seq,
        "num_motion_libs": num_motion_libs,
        "disable_xy_correction": args.disable_xy_correction,
        "tp_xy_units": args.tp_xy_units,
        "tp_pixel_to_meters": args.tp_pixel_to_meters,
        "tp_frame_window": args.tp_frame_window,
        "tp_frame_offset": args.tp_frame_offset,
        "tp_flip_y": args.tp_flip_y,
        "tp_xy_smoothing": args.tp_xy_smoothing,
        "tp_net": [[args.tp_net_x1, args.tp_net_y1], [args.tp_net_x2, args.tp_net_y2]],
        "trim_frames": args.trim_frames,
    }
    with open(os.path.join(args.out_dir, 'args.yml'), 'w') as metadata_file:
        yaml.safe_dump(meta_data, metadata_file)
    amass_data = joblib.load(args.amass_data)
    amass_file_name = os.path.basename(args.amass_data)
    if amass_file_name.endswith("_amass.pkl"):
        output_video_name = amass_file_name.replace("_amass.pkl", ".mp4")
    elif amass_file_name.endswith(".pkl"):
        output_video_name = amass_file_name.replace(".pkl", ".mp4")
    else:
        output_video_name = f"{amass_file_name}.mp4"
    print(f"Output video name: {output_video_name}")
    xy_context = prepare_xy_context(args, output_video_name)
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

    sequences = np.array(list(amass_data.keys()))
    if num_seq is not None:
        sequences = sequences[:num_seq]

    seq_mapping = {seq_name.item(): seq_idx for seq_idx, seq_name in enumerate(sequences)}
    motion_lib_seq_arr = np.array_split(sequences, num_motion_libs)

    seq_name_splits = {}
    for i, seq_arr in enumerate(motion_lib_seq_arr):
        seq_name_splits[i] = [seq_name.item()[2:] for seq_name in seq_arr]
    joblib.dump(seq_name_splits, f'{args.out_dir}/seq_name_splits.pkl')
    with smpl_robot_context() as smpl_local_robot:
        for i, motion_lib_seqs in enumerate(tqdm(motion_lib_seq_arr)):
            motion_lib_input_dict = {}
            render_data_dict = {}
            for key_name in motion_lib_seqs:
                key_name = key_name.item()
                print(f"Applying ground correction for sequence {key_name}")
                sequence = correct_smpl_sequence(
                    amass_data[key_name], smpl_local_robot, sequence_name=key_name,
                    xy_context=xy_context, out_dir=args.out_dir,
                    show_plots=not args.no_show,
                )
                beta_key = ",".join(f"{x:.6f}" for x in sequence['beta'])
                motion_out, render_data = build_motion_output(
                    sequence, key_name, seq_mapping[key_name], beta_mapping[beta_key],
                    trim_frames=args.trim_frames,
                )
                motion_lib_input_dict[key_name] = motion_out
                render_data_dict[key_name] = render_data

            if not motion_lib_input_dict:
                continue
            motion_lib = MotionLib(
                motion_file=motion_lib_input_dict,
                dof_body_ids=info['dof_body_ids'],
                dof_offsets=info['dof_offsets'],
                key_body_ids=info['key_body_ids'],
                device='cpu',
                clean_up=True,
            )
            torch.save(motion_lib, f"{args.out_dir}/mlib_part_{i:05d}.pth")
            render_pkl_path = f"{args.out_dir}/mlib_part_{i:05d}_render.pkl"
            joblib.dump(render_data_dict, render_pkl_path)
            print(f"Saved render data to {render_pkl_path}")
            del motion_lib


if __name__ == "__main__":
    main()
