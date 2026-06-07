#!/usr/bin/env python3
"""
Simplified ground contact visualization with horizontal trajectories and corrected ground height.
Plots both XY trajectories and the lowest Z-coordinate over time (before and after correction).
"""

import joblib
import numpy as np
import os
import sys
import argparse
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression, RANSACRegressor
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
parser.add_argument('--sequence_name', type=str, default=None, help="Specific sequence to analyze")
parser.add_argument('--num_seq', type=int, default=None)
args = parser.parse_args()

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

def plot_trajectory_and_ground_contacts(verts, root_trans, corrected_root_trans, fps=30.0, sequence_name=""):
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
    corrected_verts = verts.copy()
    corrected_verts[:, :, 2] += (corrected_root_trans[:, 2] - root_trans[:, 2])[:, None]
    
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
    corrected_verts = verts.copy()
    corrected_verts[:, :, 2] += (corrected_root_trans[:, 2] - root_trans[:, 2])[:, None]
    
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

def main():
    """Main function to analyze and plot ground contacts with correction."""
    
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
    
    model_xml_path = f"/tmp/smpl/smpl_mesh_humanoid_v1_convert.xml"
    
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
        sequences = sequences[:3]  # Default to first 3 sequences
    
    print(f"Analyzing {len(sequences)} sequences for ground contacts with correction...")
    
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
        
        # Apply ground correction
        print("Applying ground correction...")
        corrected_root_trans = correct_ground_height(
            verts.numpy(), 
            root_trans, 
            fps=fps,
            plot_results=False,  # We'll create our own plots
            sequence_name=str(key_name)
        )
        
        # Create the plot
        fig = plot_trajectory_and_ground_contacts(
            verts.numpy(), 
            root_trans,
            corrected_root_trans,
            fps=fps,
            sequence_name=str(key_name)
        )
        
        # Save plot
        plot_filename = f"trajectory_and_ground_contacts_{str(key_name).replace('/', '_')}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {plot_filename}")
        
        # Show plot
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
