"""Ground correction helpers and AMASS-to-Isaac motion library conversion.

Importing this module does not parse arguments, load motion data, or initialize
SMPL/MuJoCo. Simulation dependencies are loaded only when converting a sequence.
"""

import argparse
import ast
import json
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
from vid2player3d.embodied_pose.utils.motion_lib import MotionLib
import torch
from tqdm import tqdm

# TennisProject/court_reference.py and main.py. These are reference-image
# pixels, not input-video pixels. Match the downstream renderer baselines at
# +/-11.89 m and the controller's near-player side at negative Y.
TP_IMAGE_SIZE = (640, 360)
COURT_WIDTH = 10.97
COURT_LENGTH = 23.78
TP_NET_CENTER = np.array([832.5, 1748.0])
TP_METERS_PER_PIXEL = np.array([COURT_WIDTH / 1093, -COURT_LENGTH / 2374])
TP_COURT_POINTS = np.array([
    (286, 561), (1379, 561), (286, 2935), (1379, 2935),
    (423, 561), (423, 2935), (1242, 561), (1242, 2935),
    (423, 1110), (1242, 1110), (423, 2386), (1242, 2386),
    (832, 1110), (832, 2386),
], dtype=np.float64)
HIP_INDICES = (11, 12)  # GVHMR/hmr4d/configs/demo.yaml: COCO17.
HIP_CONFIDENCE_THRESHOLD = 0.4
MAX_COURT_REPROJECTION_PX = 8.0  # At 640x360, including estimated-K error.
MAX_COURT_KEYPOINT_ERROR_PX = 2.0


def add_input_arguments(parser):
    parser.add_argument('--gvhmr_dir', type=Path, required=True,
                        help='One GVHMR output folder containing AMASS, video, JSON and preprocess/')
    parser.add_argument('--tennisproject_data', '--csv', dest='tennisproject_data',
                        type=Path, required=True, help='Matching TennisProject CSV')
    parser.add_argument('--disable_xy_correction', action='store_true')


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Convert AMASS motions with ground correction")
    add_input_arguments(parser)
    parser.add_argument('--out_dir', type=str, default="data/motion_lib/amass")
    parser.add_argument('--num_seq', type=int, default=None)
    parser.add_argument('--num_motion_libs', type=int, default=1)
    parser.add_argument('--no_show', action='store_true', help='Save plots without opening windows')
    return parser


def normalized_video_name(value):
    name = Path(str(value)).name
    if name.endswith(('.mp4', '.csv', '.pkl')):
        name = name.rsplit('.', 1)[0]
    return name[2:] if name.startswith('0-') else name


def court_points_meters():
    return np.column_stack(((TP_COURT_POINTS - TP_NET_CENTER) * TP_METERS_PER_PIXEL,
                            np.zeros(len(TP_COURT_POINTS))))


def estimate_court_camera(inv_matrix, intrinsic, court_kps=None):
    """Fit court->OpenCV camera extrinsics; coordinates are 640x360 pixels.

    H supplies subpixel court correspondences; rounded CSV court_kps validate
    the coordinate scaling. The saved K is an estimate, so expose fit residuals.
    """
    import cv2

    h = np.asarray(inv_matrix, dtype=np.float64)
    k = np.asarray(intrinsic, dtype=np.float64)
    if h.shape != (3, 3) or not np.isfinite(h).all() or np.linalg.matrix_rank(h) != 3:
        raise ValueError('Missing or singular image-to-court homography')
    if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError('Invalid camera intrinsics')
    image_points = cv2.perspectiveTransform(TP_COURT_POINTS[:, None], np.linalg.inv(h))[:, 0]
    if not np.isfinite(image_points).all():
        raise ValueError('Homography projects court points to infinity')
    if court_kps is not None:
        if court_kps.shape != (14, 2) or not np.isfinite(court_kps).all():
            raise ValueError('Expected 14 finite court keypoints')
        disagreement = np.linalg.norm(court_kps - image_points, axis=1).max()
        if disagreement > MAX_COURT_KEYPOINT_ERROR_PX:
            raise ValueError(f'Court keypoints/homography disagree by {disagreement:.2f}px; check resolution/source')
    world_points = court_points_meters()
    ok, rvec, tvec = cv2.solvePnP(world_points, image_points, k, None,
                                 flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError('Court camera pose estimation failed')
    rotation = cv2.Rodrigues(rvec)[0]
    translation = tvec[:, 0]
    center = -rotation.T @ translation
    depth = (world_points @ rotation.T + translation)[:, 2]
    projected = cv2.projectPoints(world_points, rvec, tvec, k, None)[0][:, 0]
    rms = float(np.sqrt(np.mean(np.sum((projected - image_points) ** 2, axis=1))))
    if not np.isfinite(rms) or rms > MAX_COURT_REPROJECTION_PX or center[2] <= 0 or np.any(depth <= 0):
        raise ValueError(f'Invalid court camera fit (RMS={rms:.2f}px, camera height={center[2]:.2f}m)')
    return rotation, translation, rms


def intersect_rays_at_height(image_points, intrinsic, rotation, translation, heights):
    """Intersect each camera ray with its known court Z plane (all batched)."""
    pixels = np.column_stack((image_points, np.ones(len(image_points))))
    rays_camera = np.linalg.solve(intrinsic, pixels[..., None])[..., 0]
    camera_to_court = rotation.transpose(0, 2, 1)
    rays = np.einsum('nij,nj->ni', camera_to_court, rays_camera)
    centers = -np.einsum('nij,nj->ni', camera_to_court, translation)
    if np.any(np.abs(rays[:, 2]) < 1e-6):
        raise ValueError('Hip ray is parallel to the requested height plane')
    distance = (heights - centers[:, 2]) / rays[:, 2]
    if not np.isfinite(distance).all() or np.any(distance <= 0):
        raise ValueError('Hip height intersects the viewing ray behind the camera')
    return centers + distance[:, None] * rays


def court_heading_rotations(camera_rotation, incam_root_rotation, amass_root_rotation):
    """Project the inferred world->court rotation onto SO(2), preserving Z.

    Camera/world SMPL root rotations describe the same body. Their relative
    rotation supplies world->camera; court extrinsics complete the transform.
    Only yaw is applied, keeping the existing gravity and height estimates.
    """
    from scipy.spatial.transform import Rotation

    relative = camera_rotation.transpose(0, 2, 1) @ incam_root_rotation @ amass_root_rotation.transpose(0, 2, 1)
    sine = relative[:, 1, 0] - relative[:, 0, 1]
    cosine = relative[:, 0, 0] + relative[:, 1, 1]
    if np.any(np.hypot(sine, cosine) < 1e-6):
        raise ValueError('Cannot determine court yaw from the camera/body rotations')
    yaw = np.arctan2(sine, cosine)
    rotations = Rotation.from_euler('z', yaw).as_matrix()
    residual = Rotation.from_matrix(rotations.transpose(0, 2, 1) @ relative).magnitude()
    return rotations, yaw, residual


def load_correction_inputs(gvhmr_dir, csv_path, disable_xy=False):
    """Load one clip, validate artifact identity, and join exact source frames.

    Missing/ambiguous frames and unusable calibration fail explicitly instead
    of freezing the player or silently applying a different clip's trajectory.
    """
    import cv2
    import torch
    from scipy.spatial.transform import Rotation

    directory, csv_path = Path(gvhmr_dir).resolve(), Path(csv_path).resolve()
    amass_paths = sorted(directory.glob('*_amass.pkl'))
    if len(amass_paths) != 1:
        raise ValueError(f'Expected exactly one *_amass.pkl in {directory}, found {len(amass_paths)}')
    data = joblib.load(amass_paths[0])
    if len(data) != 1:
        raise ValueError('A GVHMR clip folder must contain exactly one AMASS sequence')
    sequence_name, entry = next(iter(data.items()))
    metadata = json.loads((directory / '0_input_video.json').read_text())
    source = metadata['source']
    start, end = int(source['start_frame']), int(source['end_frame'])
    count = len(entry['pose_aa'])
    if count < 2 or end - start != count or len(entry['trans_orig']) != count:
        raise ValueError('JSON frame range and AMASS lengths disagree (or fewer than two frames)')
    fps = float(source['fps'])
    if not np.isclose(fps, 30) or not np.isclose(float(entry.get('fps', fps)), fps):
        raise ValueError('Expected matching 30 FPS GVHMR source and AMASS data')
    video = cv2.VideoCapture(str(directory / '0_input_video.mp4'))
    try:
        width, height = video.get(cv2.CAP_PROP_FRAME_WIDTH), video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        video_count, video_fps = video.get(cv2.CAP_PROP_FRAME_COUNT), video.get(cv2.CAP_PROP_FPS)
    finally:
        video.release()
    if width <= 0 or height <= 0 or int(video_count) != count or not np.isclose(video_fps, fps):
        raise ValueError('Input video dimensions/frame count/FPS disagree with metadata')
    context = {'enabled': not disable_xy}
    provenance = {
        'gvhmr_dir': str(directory), 'tennisproject_data': str(csv_path),
        'amass_data': str(amass_paths[0]), 'source': source,
        'sequence_name': str(sequence_name), 'frame_count': count, 'fps': fps,
        'frame_join': 'CSV global_frame = JSON source.start_frame + motion frame',
        'input_video_size': [int(width), int(height)],
        'xy_enabled': not disable_xy,
        'coordinate_system': {'units': 'meters', 'origin': 'net center on ground',
                              'x': 'reference court right', 'y': 'toward far baseline', 'z': 'up',
                              'near_baseline_y': -COURT_LENGTH / 2,
                              'quaternion_order': 'xyzw', 'handedness': 'right'},
        'constants': {'tp_image_size': list(TP_IMAGE_SIZE), 'tp_net_center': TP_NET_CENTER.tolist(),
                      'tp_meters_per_pixel': TP_METERS_PER_PIXEL.tolist(),
                      'hip_indices': list(HIP_INDICES), 'hip_confidence_threshold': HIP_CONFIDENCE_THRESHOLD,
                      'max_camera_reprojection_px': MAX_COURT_REPROJECTION_PX,
                      'max_court_keypoint_error_px': MAX_COURT_KEYPOINT_ERROR_PX},
        'xy_temporal_smoothing': False,
        'z_correction': 'existing multi-window foot/RANSAC correction, before court alignment',
        'downstream_height_adjustment': 'MotionLib may additionally subtract min_verts_h - ground_tolerance',
    }
    if disable_xy:
        provenance['coordinate_system'] = {'units': 'meters', 'z': 'up', 'xy': 'original GVHMR frame'}
        return data, context, provenance
    name = normalized_video_name(source['path'])
    if normalized_video_name(amass_paths[0].stem[:-6]) != name:
        raise ValueError('AMASS filename and JSON source video disagree')
    df = pd.read_csv(csv_path)
    if 'output_video_name' in df:
        df = df[df.output_video_name.map(normalized_video_name) == name]
    elif normalized_video_name(csv_path) != name:
        raise ValueError('CSV filename does not match the source video')
    if 'global_frame' not in df or 'inv_matrix' not in df or 'court_kps' not in df:
        raise ValueError('CSV requires global_frame, inv_matrix and court_kps columns')
    source_frames = np.arange(start, end)
    selected = df[df.global_frame.isin(source_frames)]
    if selected.global_frame.duplicated().any() or len(selected) != count:
        raise ValueError('CSV must contain exactly one row for each source frame in the clip')
    selected = selected.set_index('global_frame').loc[source_frames]
    poses = torch.load(str(directory / 'preprocess/vitpose.pt'), map_location='cpu').numpy()
    results = torch.load(str(directory / 'hmr4d_results.pt'), map_location='cpu')
    k = results['K_fullimg'].numpy().astype(np.float64)
    incam = results['smpl_params_incam']['global_orient'].numpy()
    global_params = results['smpl_params_global']
    global_orient = global_params['global_orient'].numpy()
    if poses.shape != (count, 17, 3) or k.shape != (count, 3, 3) or incam.shape != (count, 3) or global_orient.shape != (count, 3):
        raise ValueError('ViTPose, camera and AMASS frame counts/shapes disagree')
    export_rotation = Rotation.from_euler('x', np.pi / 2).as_matrix()
    amass_rotation = Rotation.from_rotvec(entry['pose_aa'][:, :3]).as_matrix()
    expected_rotation = export_rotation @ Rotation.from_rotvec(global_orient).as_matrix()
    expected_trans = global_params['transl'].numpy() @ export_rotation.T
    expected_trans[:, 2] -= expected_trans[0, 2] - 0.92
    if (not np.allclose(amass_rotation, expected_rotation, atol=1e-5)
            or not np.allclose(entry['pose_aa'][:, 3:66], global_params['body_pose'].numpy(), atol=1e-5)
            or not np.allclose(entry['trans_orig'], expected_trans, atol=1e-5)
            or not np.allclose(entry['beta'][:10], global_params['betas'][0].numpy()[:10], atol=1e-5)):
        raise ValueError('AMASS and hmr4d_results do not describe the same exported motion')
    hips = poses[:, HIP_INDICES, :]
    valid = np.isfinite(hips).all(axis=(1, 2)) & (hips[:, :, 2] > HIP_CONFIDENCE_THRESHOLD).all(axis=1)
    if not valid.all():
        raise ValueError(f'Unusable processed hip detections at motion frames {np.flatnonzero(~valid)[:10].tolist()}')
    scale = np.array(TP_IMAGE_SIZE) / [width, height]
    k[:, :2, :] *= scale[None, :, None]
    rotations, translations, errors = [], [], []
    for frame, row in selected.iterrows():
        try:
            # NumPy's CSV matrix string has whitespace/newlines, not commas.
            values = np.fromstring(str(row.inv_matrix).replace('[', ' ').replace(']', ' '), sep=' ')
            h = values.reshape(3, 3)
            court_kps = np.asarray(ast.literal_eval(row.court_kps), dtype=np.float64) * scale
            rotation, translation, error = estimate_court_camera(h, k[len(rotations)], court_kps)
        except (ValueError, SyntaxError, TypeError) as exc:
            raise ValueError(f'CSV global_frame {frame}: {exc}') from exc
        rotations.append(rotation)
        translations.append(translation)
        errors.append(error)
    rotations, translations = np.stack(rotations), np.stack(translations)
    yaw_rotation, yaw, residual = court_heading_rotations(
        rotations, Rotation.from_rotvec(incam).as_matrix(), amass_rotation)
    context.update(intrinsic=k, camera_rotation=rotations, camera_translation=translations,
                   hip_pixels=hips[:, :, :2].mean(axis=1) * scale, yaw_rotation=yaw_rotation)
    provenance['camera'] = {
        'method': 'solvePnP with saved K and homography court correspondences; zero distortion assumed',
        'intrinsics_source': 'hmr4d_results.pt/K_fullimg (GVHMR normally estimates focal length)',
        'rotation_convention': 'p_camera = R_court_to_camera @ p_court + t_court_to_camera',
        'source_frames': source_frames.tolist(), 'intrinsics_640x360': k.tolist(),
        'court_to_camera_rotation': rotations.tolist(), 'court_to_camera_translation': translations.tolist(),
        'reprojection_rms_640x360_px': errors,
        'reprojection_rms_median_px': float(np.median(errors)),
        'reprojection_rms_max_px': float(np.max(errors)),
        'world_to_court_yaw_radians': yaw.tolist(),
        'discarded_tilt_radians': residual.tolist(),
        'hip_confidence_note': 'Processed ViTPose scores include validity=1 for upstream interpolated joints',
    }
    print(f'Court calibration: median/max RMS {np.median(errors):.2f}/{max(errors):.2f}px at 640x360')
    return data, context, provenance


def apply_camera_xy(root_trans, grounded_root, joints, pose_aa, context):
    """Recover root XY from hip rays and pose-dependent hip/root offsets."""
    from scipy.spatial.transform import Rotation

    yaw = context['yaw_rotation']
    # SMPL rotates about its true rest pelvis; XML rounds that pivot slightly.
    hip_relative = np.einsum('nij,nj->ni', yaw, joints[:, [1, 2]].mean(axis=1) - joints[:, 0])
    hip_relative += joints[:, 0] - root_trans
    hip_height = grounded_root[:, 2] + hip_relative[:, 2]
    hip_world = intersect_rays_at_height(context['hip_pixels'], context['intrinsic'],
                                        context['camera_rotation'], context['camera_translation'], hip_height)
    corrected_root = grounded_root.copy()
    corrected_root[:, :2] = hip_world[:, :2] - hip_relative[:, :2]
    corrected_pose = pose_aa.copy()
    corrected_pose[:, :3] = Rotation.from_matrix(
        yaw @ Rotation.from_rotvec(pose_aa[:, :3]).as_matrix()).as_rotvec()
    # A fixed first-frame registration lets plots compare trajectory changes
    # without confusing a different world origin/heading with drift correction.
    reference_root = (root_trans - root_trans[0]) @ yaw[0].T
    reference_root[:, :2] += corrected_root[0, :2]
    reference_root[:, 2] = root_trans[:, 2]
    return corrected_root, corrected_pose, reference_root, hip_world


def save_correction_metadata(out_dir, args, provenance, sequences):
    import yaml

    metadata = dict(provenance)
    metadata['arguments'] = {key: str(value) if isinstance(value, Path) else value
                             for key, value in vars(args).items()}
    metadata['sequences'] = {str(name): sequence['alignment_metadata'] for name, sequence in sequences.items()}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with (Path(out_dir) / 'args.yml').open('w') as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False)


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
                          plot_results=True, sequence_name="",
                          out_dir=".", show_plots=True, diagnostics=None):
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

    if diagnostics is not None:
        diagnostics.update(time_coords=time_coords, original_heights=original_heights,
                           corrected_heights=corrected_heights, valley_points=valley_points,
                           window_boundaries=window_boundaries)
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
                                  out_dir=".", show=True, original_root=None,
                                  corrected_root=None, fps=30.0, court_enabled=False):
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
    if original_root is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    else:
        fig = plt.figure(figsize=(16, 13))
        grid = fig.add_gridspec(3, 2)
        ax1, ax2 = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
        draw_xy_comparison(fig.add_subplot(grid[1:, 0]), fig.add_subplot(grid[1, 1]),
                           fig.add_subplot(grid[2, 1]), original_root, corrected_root,
                           fps, court_enabled)
    
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


def draw_xy_comparison(ax_xy, ax_x, ax_y, original, corrected, fps, court_enabled=True):
    """Shared court trajectory and coordinate/time comparisons for both CLIs."""
    label = 'Original GVHMR (first-frame court alignment)' if court_enabled else 'Original GVHMR'
    ax_xy.plot(original[:, 0], original[:, 1], '--', label=label, alpha=0.8)
    ax_xy.plot(corrected[:, 0], corrected[:, 1], label='Corrected')
    ax_xy.scatter(*corrected[0, :2], c='green', marker='o', label='Start')
    ax_xy.scatter(*corrected[-1, :2], c='red', marker='x', label='End')
    if court_enabled:
        w, l = COURT_WIDTH / 2, COURT_LENGTH / 2
        ax_xy.plot([-w, w, w, -w, -w], [-l, -l, l, l, -l], color='gray', alpha=0.6)
        ax_xy.plot([-w, w], [0, 0], color='black', label='Net')
        for x in (-8.23 / 2, 8.23 / 2):
            ax_xy.plot([x, x], [-l, l], color='gray', alpha=0.4)
        for y in (-6.4, 6.4):
            ax_xy.plot([-8.23 / 2, 8.23 / 2], [y, y], color='gray', alpha=0.4)
    ax_xy.set(xlabel='X (m)', ylabel='Y (m)', title='Root XY trajectory')
    ax_xy.set_aspect('equal', adjustable='datalim')
    times = np.arange(len(original)) / fps
    for axis, ax, name in ((0, ax_x, 'X'), (1, ax_y, 'Y')):
        ax.plot(times, original[:, axis], '--', label=label)
        ax.plot(times, corrected[:, axis], label='Corrected')
        ax.set(xlabel='Time (s)', ylabel=f'{name} (m)', title=f'Root {name}: original vs corrected')
    for ax in (ax_xy, ax_x, ax_y):
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)


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
    original_pose_aa = pose_aa.copy()
    diagnostics = {}
    xy_enabled = xy_context is not None and xy_context.get('enabled', False)
    alignment_metadata = {'xy_enabled': xy_enabled}

    with torch.no_grad():
        smpl_local_robot.load_from_skeleton(
            betas=torch.from_numpy(beta[None, :]), gender=[gender_number]
        )
        smpl_local_robot.write_xml()
        skeleton_tree = SkeletonTree.from_mjcf(smpl_local_robot.model_xml_path)
        pelvis_offset = skeleton_tree.local_translation[0].numpy()
        root_trans = trans + pelvis_offset
        verts, joints = smpl_parser.get_joints_verts(
            pose=torch.from_numpy(pose_aa),
            th_betas=torch.from_numpy(beta[None, :]),
            th_trans=torch.from_numpy(trans),
        )
        corrected_root_trans = correct_ground_height(
            verts.numpy(), root_trans, fps=fps, base_window=base_window,
            plot_results=False, diagnostics=diagnostics,
        )
        comparison_root = root_trans.copy()
        comparison_verts = verts
        if xy_enabled:
            corrected_root_trans, pose_aa, comparison_root, hip_world = apply_camera_xy(
                root_trans, corrected_root_trans, joints.numpy(), pose_aa, xy_context)
            yaw0 = xy_context['yaw_rotation'][0]
            comparison_offset = comparison_root[0] - root_trans[0] @ yaw0.T
            comparison_verts = torch.from_numpy(verts.numpy() @ yaw0.T + comparison_offset)
            alignment_metadata.update(
                comparison_world_to_court_rotation=yaw0.tolist(),
                comparison_world_to_court_translation=comparison_offset.tolist(),
                comparison_note='Fixed first-frame yaw and XY registration, original Z retained',
                reconstructed_hip_midpoint_m=hip_world.tolist(),
                original_root_m=root_trans.tolist(),
                corrected_root_m=corrected_root_trans.tolist(),
            )
        # Convert pelvis world position back to SMPL model translation.
        corrected_trans = corrected_root_trans - pelvis_offset
        corrected_verts, _ = smpl_parser.get_joints_verts(
            pose=torch.from_numpy(pose_aa),
            th_betas=torch.from_numpy(beta[None, :]),
            th_trans=torch.from_numpy(corrected_trans),
        )

    pose_quat = sRot.from_rotvec(pose_aa.reshape(-1, 3)).as_quat().reshape(
        batch_size, 24, 4
    )[..., smpl_2_mujoco, :]
    if plot_results:
        create_ground_correction_plots(
            **diagnostics, sequence_name=sequence_name, out_dir=out_dir, show=show_plots,
            original_root=comparison_root, corrected_root=corrected_root_trans,
            fps=fps, court_enabled=xy_enabled,
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
        "original_pose_aa": original_pose_aa,
        "comparison_root_trans": comparison_root,
        "comparison_verts": comparison_verts,
        "alignment_metadata": alignment_metadata,
    }


def build_motion_output(sequence, seq_name, seq_idx, beta_idx):
    """Build simulation and render outputs using consistent corrected positions."""
    import torch
    from vid2player3d.poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState

    frame_count = len(sequence['pose_aa'])
    if frame_count < 2:
        raise ValueError("At least two frames are required for velocities")
    frames = slice(None)
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
    if args.num_seq is not None and args.num_seq < 1:
        raise ValueError("num_seq must be positive")
    # MotionLib imports Isaac Gym; keep this before importing torch directly.

    num_seq = args.num_seq
    num_motion_libs = args.num_motion_libs

    os.makedirs(args.out_dir, exist_ok=True)
    amass_data, xy_context, provenance = load_correction_inputs(
        args.gvhmr_dir, args.tennisproject_data, args.disable_xy_correction)
    completed = {}
    save_correction_metadata(args.out_dir, args, provenance, completed)
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
                    amass_data[key_name], smpl_local_robot, sequence_name=key_name, fps=provenance["fps"],
                    xy_context=xy_context, out_dir=args.out_dir,
                    show_plots=not args.no_show,
                )
                completed[key_name] = {'alignment_metadata': sequence['alignment_metadata']}
                save_correction_metadata(args.out_dir, args, provenance, completed)
                beta_key = ",".join(f"{x:.6f}" for x in sequence['beta'])
                motion_out, render_data = build_motion_output(
                    sequence, key_name, seq_mapping[key_name], beta_mapping[beta_key],
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
