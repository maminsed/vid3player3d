import cv2
import torch
import pytorch_lightning as pl
import numpy as np
import argparse
from vid3player.GVHMR.hmr4d.utils.pylogger import Log
import hydra
from hydra import initialize_config_module, compose
from pathlib import Path
from pytorch3d.transforms import quaternion_to_matrix
import joblib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib
    MATPLOTLIB_AVAILABLE = True
    
    # Configure matplotlib for better interactive experience
    import os
    if 'DISPLAY' in os.environ and os.environ['DISPLAY']:
        matplotlib.use('TkAgg')  # Use TkAgg backend for better interactive support
    else:
        matplotlib.use('Agg')  # Use non-interactive backend if no display
        Log.warning("No display detected. Interactive mode will fall back to saving files.")
    plt.rcParams['figure.max_open_warning'] = 0  # Suppress warning about too many open figures
    
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    Log.warning("Matplotlib not available. Trajectory plotting will be disabled.")

from vid3player.GVHMR.hmr4d.configs import register_store_gvhmr
from vid3player.GVHMR.hmr4d.utils.video_io_utils import (
    get_video_lwh,
    read_video_np,
    save_video,
    merge_videos_horizontal,
    get_writer,
    get_video_reader,
)
from vid3player.GVHMR.hmr4d.utils.vis.cv2_utils import draw_bbx_xyxy_on_image_batch, draw_coco17_skeleton_batch

from vid3player.GVHMR.hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor, SimpleVO

from vid3player.GVHMR.hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K, convert_K_to_K4, create_camera_sensor
from vid3player.GVHMR.hmr4d.utils.geo_transform import compute_cam_angvel
from vid3player.GVHMR.hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from vid3player.GVHMR.hmr4d.utils.net_utils import detach_to_cpu, to_cuda
from vid3player.GVHMR.hmr4d.utils.smplx_utils import make_smplx
from vid3player.GVHMR.hmr4d.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points
from tqdm import tqdm
from vid3player.GVHMR.hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from einops import einsum, rearrange


CRF = 23  # 17 is lossless, every +6 halves the mp4 size


def parse_args_to_cfg():
    # Put all args to cfg
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="inputs/demo/dance_3.mp4")
    parser.add_argument("--output_root", type=str, default=None, help="by default to outputs/demo")
    parser.add_argument("-s", "--static_cam", action="store_true", help="If true, skip DPVO")
    parser.add_argument("--use_dpvo", action="store_true", help="If true, use DPVO. By default not using DPVO.")
    parser.add_argument(
        "--f_mm",
        type=int,
        default=None,
        help="Focal length of fullframe camera in mm. Leave it as None to use default values."
        "For iPhone 15p, the [0.5x, 1x, 2x, 3x] lens have typical values [13, 24, 48, 77]."
        "If the camera zoom in a lot, you can try 135, 200 or even larger values.",
    )
    parser.add_argument("--verbose", action="store_true", help="If true, draw intermediate results")
    parser.add_argument("--save_amass", action="store_true", help="If true, save AMASS format data for convert_amass_isaac.py")
    parser.add_argument("--amass_output", type=str, default=None, help="Output path for AMASS format data")
    parser.add_argument("--plot_trajectory", action="store_true", help="If true, generate trajectory and ground contact plots")
    parser.add_argument("--interactive", action="store_true", help="If true, show plots interactively instead of just saving them")
    args = parser.parse_args()

    # Input
    video_path = Path(args.video)
    assert video_path.exists(), f"Video not found at {video_path}"
    length, width, height = get_video_lwh(video_path)
    Log.info(f"[Input]: {video_path}")
    Log.info(f"(L, W, H) = ({length}, {width}, {height})")
    # Cfg
    with initialize_config_module(version_base="1.3", config_module=f"hmr4d.configs"):
        overrides = [
            f"video_name={video_path.stem}",
            f"static_cam={args.static_cam}",
            f"verbose={args.verbose}",
            f"use_dpvo={args.use_dpvo}",
        ]
        if args.f_mm is not None:
            overrides.append(f"f_mm={args.f_mm}")

        # Allow to change output root
        if args.output_root is not None:
            overrides.append(f"output_root={args.output_root}")
        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=overrides)

    # Output
    Log.info(f"[Output Dir]: {cfg.output_dir}")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

    # Copy raw-input-video to video_path
    Log.info(f"[Copy Video] {video_path} -> {cfg.video_path}")
    if not Path(cfg.video_path).exists() or get_video_lwh(video_path)[0] != get_video_lwh(cfg.video_path)[0]:
        reader = get_video_reader(video_path)
        writer = get_writer(cfg.video_path, fps=30, crf=CRF)
        for img in tqdm(reader, total=get_video_lwh(video_path)[0], desc=f"Copy"):
            writer.write_frame(img)
        writer.close()
        reader.close()

    # Store AMASS settings separately (not in cfg to avoid OmegaConf issues)
    amass_settings = {
        'save_amass': args.save_amass,
        'amass_output': None,
        'plot_trajectory': args.plot_trajectory,
        'interactive': args.interactive
    }
    
    if args.save_amass:
        if args.amass_output is None:
            amass_settings['amass_output'] = Path(cfg.output_dir) / f"{video_path.stem}_amass.pkl"
        else:
            amass_settings['amass_output'] = Path(args.amass_output)

    return cfg, amass_settings


@torch.no_grad()
def run_preprocess(cfg):
    Log.info(f"[Preprocess] Start!")
    tic = Log.time()
    video_path = cfg.video_path
    paths = cfg.paths
    static_cam = cfg.static_cam
    verbose = cfg.verbose

    # Get bbx tracking result
    if not Path(paths.bbx).exists():
        tracker = Tracker()
        bbx_xyxy = tracker.get_one_track(video_path).float()  # (L, 4)
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()  # (L, 3) apply aspect ratio and enlarge
        torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
        del tracker
    else:
        bbx_xys = torch.load(paths.bbx)["bbx_xys"]
        Log.info(f"[Preprocess] bbx (xyxy, xys) from {paths.bbx}")
    if verbose:
        video = read_video_np(video_path)
        bbx_xyxy = torch.load(paths.bbx)["bbx_xyxy"]
        video_overlay = draw_bbx_xyxy_on_image_batch(bbx_xyxy, video)
        save_video(video_overlay, cfg.paths.bbx_xyxy_video_overlay)

    # Get VitPose
    if not Path(paths.vitpose).exists():
        vitpose_extractor = VitPoseExtractor()
        vitpose = vitpose_extractor.extract(video_path, bbx_xys)
        torch.save(vitpose, paths.vitpose)
        del vitpose_extractor
    else:
        vitpose = torch.load(paths.vitpose)
        Log.info(f"[Preprocess] vitpose from {paths.vitpose}")
    if verbose:
        video = read_video_np(video_path)
        video_overlay = draw_coco17_skeleton_batch(video, vitpose, 0.5)
        save_video(video_overlay, paths.vitpose_video_overlay)

    # Get vit features
    if not Path(paths.vit_features).exists():
        extractor = Extractor()
        vit_features = extractor.extract_video_features(video_path, bbx_xys)
        torch.save(vit_features, paths.vit_features)
        del extractor
    else:
        Log.info(f"[Preprocess] vit_features from {paths.vit_features}")

    # Get visual odometry results
    if not static_cam:  # use slam to get cam rotation
        if not Path(paths.slam).exists():
            if not cfg.use_dpvo:
                simple_vo = SimpleVO(cfg.video_path, scale=0.5, step=8, method="sift", f_mm=cfg.f_mm)
                vo_results = simple_vo.compute()  # (L, 4, 4), numpy
                torch.save(vo_results, paths.slam)
            else:  # DPVO
                from vid3player.GVHMR.hmr4d.utils.preproc.slam import SLAMModel

                length, width, height = get_video_lwh(cfg.video_path)
                K_fullimg = estimate_K(width, height)
                intrinsics = convert_K_to_K4(K_fullimg)
                slam = SLAMModel(video_path, width, height, intrinsics, buffer=4000, resize=0.5)
                bar = tqdm(total=length, desc="DPVO")
                while True:
                    ret = slam.track()
                    if ret:
                        bar.update()
                    else:
                        break
                slam_results = slam.process()  # (L, 7), numpy
                torch.save(slam_results, paths.slam)
        else:
            Log.info(f"[Preprocess] slam results from {paths.slam}")

    Log.info(f"[Preprocess] End. Time elapsed: {Log.time()-tic:.2f}s")


def load_data_dict(cfg):
    paths = cfg.paths
    length, width, height = get_video_lwh(cfg.video_path)
    if cfg.static_cam:
        R_w2c = torch.eye(3).repeat(length, 1, 1)
    else:
        traj = torch.load(cfg.paths.slam)
        if cfg.use_dpvo:  # DPVO
            traj_quat = torch.from_numpy(traj[:, [6, 3, 4, 5]])
            R_w2c = quaternion_to_matrix(traj_quat).mT
        else:  # SimpleVO
            R_w2c = torch.from_numpy(traj[:, :3, :3])
    if cfg.f_mm is not None:
        K_fullimg = create_camera_sensor(width, height, cfg.f_mm)[2].repeat(length, 1, 1)
    else:
        K_fullimg = estimate_K(width, height).repeat(length, 1, 1)

    data = {
        "length": torch.tensor(length),
        "bbx_xys": torch.load(paths.bbx)["bbx_xys"],
        "kp2d": torch.load(paths.vitpose),
        "K_fullimg": K_fullimg,
        "cam_angvel": compute_cam_angvel(R_w2c),
        "f_imgseq": torch.load(paths.vit_features),
    }
    return data


def render_incam(cfg):
    incam_video_path = Path(cfg.paths.incam_video)
    if incam_video_path.exists():
        Log.info(f"[Render Incam] Video already exists at {incam_video_path}")
        return

    pred = torch.load(cfg.paths.hmr4d_results)
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    faces_smpl = make_smplx("smpl").faces

    # smpl
    smplx_out = smplx(**to_cuda(pred["smpl_params_incam"]))
    pred_c_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])

    # -- rendering code -- #
    video_path = cfg.video_path
    length, width, height = get_video_lwh(video_path)
    K = pred["K_fullimg"][0]

    # renderer
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)
    reader = get_video_reader(video_path)  # (F, H, W, 3), uint8, numpy
    bbx_xys_render = torch.load(cfg.paths.bbx)["bbx_xys"]

    # -- render mesh -- #
    verts_incam = pred_c_verts
    writer = get_writer(incam_video_path, fps=30, crf=CRF)
    for i, img_raw in tqdm(enumerate(reader), total=get_video_lwh(video_path)[0], desc=f"Rendering Incam"):
        img = renderer.render_mesh(verts_incam[i].cuda(), img_raw, [0.8, 0.8, 0.8])

        # # bbx
        # bbx_xys_ = bbx_xys_render[i].cpu().numpy()
        # lu_point = (bbx_xys_[:2] - bbx_xys_[2:] / 2).astype(int)
        # rd_point = (bbx_xys_[:2] + bbx_xys_[2:] / 2).astype(int)
        # img = cv2.rectangle(img, lu_point, rd_point, (255, 178, 102), 2)

        writer.write_frame(img)
    writer.close()
    reader.close()


def render_global(cfg):
    global_video_path = Path(cfg.paths.global_video)
    if global_video_path.exists():
        Log.info(f"[Render Global] Video already exists at {global_video_path}")
        return

    debug_cam = False
    pred = torch.load(cfg.paths.hmr4d_results)
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    faces_smpl = make_smplx("smpl").faces
    J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()

    # smpl
    smplx_out = smplx(**to_cuda(pred["smpl_params_global"]))
    pred_ay_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])

    def move_to_start_point_face_z(verts):
        "XZ to origin, Start from the ground, Face-Z"
        # position
        verts = verts.clone()  # (L, V, 3)
        offset = einsum(J_regressor, verts[0], "j v, v i -> j i")[0]  # (3)
        offset[1] = verts[:, :, [1]].min()
        verts = verts - offset
        # face direction
        T_ay2ayfz = compute_T_ayfz2ay(einsum(J_regressor, verts[[0]], "j v, l v i -> l j i"), inverse=True)
        verts = apply_T_on_points(verts, T_ay2ayfz)
        return verts

    verts_glob = move_to_start_point_face_z(pred_ay_verts)
    joints_glob = einsum(J_regressor, verts_glob, "j v, l v i -> l j i")  # (L, J, 3)
    global_R, global_T, global_lights = get_global_cameras_static(
        verts_glob.cpu(),
        beta=2.0,
        cam_height_degree=20,
        target_center_height=1.0,
    )

    # -- rendering code -- #
    video_path = cfg.video_path
    length, width, height = get_video_lwh(video_path)
    _, _, K = create_camera_sensor(width, height, 24)  # render as 24mm lens

    # renderer
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)
    # renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K, bin_size=0)

    # -- render mesh -- #
    scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], verts_glob)
    renderer.set_ground(scale * 1.5, cx, cz)
    color = torch.ones(3).float().cuda() * 0.8

    render_length = length if not debug_cam else 8
    writer = get_writer(global_video_path, fps=30, crf=CRF)
    for i in tqdm(range(render_length), desc=f"Rendering Global"):
        cameras = renderer.create_camera(global_R[i], global_T[i])
        img = renderer.render_with_ground(verts_glob[[i]], color[None], cameras, global_lights)
        writer.write_frame(img)
    writer.close()


def save_amass_format(amass_output, pred):
    """Save HMR4D predictions in AMASS format for convert_amass_isaac.py"""
    if amass_output is None:
        return
    
    Log.info(f"[Save AMASS] Saving to {amass_output}")
    
    # Get SMPL parameters from predictions (use global parameters)
    smpl_params_global = pred["smpl_params_global"]
    
    # Debug: Print the actual structure of smpl_params_global
    Log.info(f"[Save AMASS] smpl_params_global keys: {list(smpl_params_global.keys())}")
    for key, value in smpl_params_global.items():
        if isinstance(value, torch.Tensor):
            Log.info(f"[Save AMASS] {key} shape: {value.shape}, dtype: {value.dtype}")
        else:
            Log.info(f"[Save AMASS] {key} type: {type(value)}")
    
    # Extract individual SMPL parameters
    global_orient = smpl_params_global["global_orient"].cpu().numpy()  # (L, 3, 3) - root rotation matrix
    body_pose = smpl_params_global["body_pose"].cpu().numpy()  # (L, 23, 3, 3) - body joint rotation matrices
    transl = smpl_params_global["transl"].cpu().numpy()  # (L, 3) - translation
    betas = smpl_params_global["betas"].cpu().numpy()  # (L, 10) - body shape parameters

    # Debug: Print the actual shapes
    Log.info(f"[Save AMASS] Actual global_orient shape: {global_orient.shape}")
    Log.info(f"[Save AMASS] Actual body_pose shape: {body_pose.shape}")
    Log.info(f"[Save AMASS] Actual transl shape: {transl.shape}")
    Log.info(f"[Save AMASS] Actual betas shape: {betas.shape}")
    
    from scipy.spatial.transform import Rotation as sRot
    
    # Check if the parameters are already in axis-angle format
    if global_orient.ndim == 2 and global_orient.shape[1] == 3:
        Log.info(f"[Save AMASS] Parameters appear to be already in axis-angle format")
        # Parameters are already in axis-angle format
        pose_aa = np.concatenate([global_orient, body_pose], axis=1)  # (L, 72)
    else:
        # Parameters are in rotation matrix format, need conversion
        Log.info(f"[Save AMASS] Parameters are in rotation matrix format, converting to axis-angle")
        
        def validate_and_fix_rotation_matrix(R):
            """Validate and fix rotation matrix if needed"""
            # Ensure R is 2D
            if R.ndim == 1:
                R = R.reshape(3, 3)
            
            # Check if determinant is close to 1 (valid rotation matrix)
            det = np.linalg.det(R)
            if abs(det - 1.0) > 1e-6:
                # If determinant is not 1, try to fix it
                if abs(det) > 1e-6:  # If not singular
                    # Normalize to make it a valid rotation matrix
                    U, S, Vt = np.linalg.svd(R)
                    S = np.clip(S, 0, 1)  # Ensure singular values are in valid range
                    R_fixed = U @ np.diag(S) @ Vt
                    # Re-normalize to ensure determinant is 1
                    det_fixed = np.linalg.det(R_fixed)
                    if det_fixed < 0:
                        Vt[-1, :] *= -1
                        R_fixed = U @ np.diag(S) @ Vt
                    return R_fixed
                else:
                    # If singular, return identity matrix
                    return np.eye(3)
            return R
        
        # Validate and fix global orientation matrices
        global_orient_fixed = np.zeros_like(global_orient)
        for i in range(global_orient.shape[0]):
            global_orient_fixed[i] = validate_and_fix_rotation_matrix(global_orient[i])
        
        # Validate and fix body pose matrices
        body_pose_fixed = np.zeros_like(body_pose)
        for i in range(body_pose.shape[0]):
            for j in range(body_pose.shape[1]):
                body_pose_fixed[i, j] = validate_and_fix_rotation_matrix(body_pose[i, j])
        
        # Convert global orientation (root) from rotation matrix to axis-angle
        global_orient_aa = sRot.from_matrix(global_orient_fixed.reshape(-1, 3, 3)).as_rotvec().reshape(global_orient.shape[0], 3)  # (L, 3)
        
        # Convert body pose from rotation matrices to axis-angle
        body_pose_aa = sRot.from_matrix(body_pose_fixed.reshape(-1, 3, 3)).as_rotvec().reshape(body_pose.shape[0], -1)  # (L, 69) - 23 joints × 3
        
        # Combine global_orient and body_pose to create pose_aa (72 parameters total)
        pose_aa = np.concatenate([global_orient_aa, body_pose_aa], axis=1)  # (L, 72)
    
    Log.info(f"[Save AMASS] Final pose_aa shape: {pose_aa.shape}")
    
    # Apply coordinate system transformation (similar to convert_data_mdm.py)
    Log.info(f"[Save AMASS] Applying coordinate system transformation")
    
    # Create transformation: 90-degree rotation around X-axis
    transform = sRot.from_euler('xyz', np.array([np.pi / 2, 0, 0]), degrees=False)
    
    # Transform root orientation (first 3 parameters)
    new_root = (transform * sRot.from_rotvec(pose_aa[:, :3])).as_rotvec()
    pose_aa[:, :3] = new_root
    
    # Transform translation
    trans_orig = transl.copy()
    trans_orig = trans_orig.dot(transform.as_matrix().T)
    # Adjust height relative to first frame (similar to convert_data_mdm.py)
    trans_orig[:, 2] = trans_orig[:, 2] - (trans_orig[0, 2] - 0.92)
    
    Log.info(f"[Save AMASS] Applied coordinate transformation")
    Log.info(f"[Save AMASS] Original translation range: [{transl.min():.3f}, {transl.max():.3f}]")
    Log.info(f"[Save AMASS] Transformed translation range: [{trans_orig.min():.3f}, {trans_orig.max():.3f}]")
    
    # Create AMASS format entry
    video_name = amass_output.stem.replace('_amass', '')  # Remove _amass suffix if present
    amass_entry = {
        'pose_aa': pose_aa,  # (L, 72) - SMPL pose parameters in axis-angle (transformed)
        'trans_orig': trans_orig,  # (L, 3) - translation (transformed)
        'beta': betas[0] if betas.ndim > 1 else betas,  # (10,) - body shape parameters (use first frame)
        'gender': 'neutral',  # Default gender, can be modified if needed
        'fps': 30.0,  # Default fps
    }
    
    # Create AMASS data dictionary with proper key format
    # Use string key - convert_amass_isaac.py will convert to numpy array when loading
    amass_data = {
        f"demo_{video_name}": amass_entry
    }
    
    # Save in joblib format
    joblib.dump(amass_data, amass_output)
    Log.info(f"[Save AMASS] Saved AMASS format data to {amass_output}")
    Log.info(f"[Save AMASS] Sequence length: {pose_aa.shape[0]}")
    Log.info(f"[Save AMASS] Pose parameters shape: {pose_aa.shape}")
    Log.info(f"[Save AMASS] Translation shape: {trans_orig.shape}")
    Log.info(f"[Save AMASS] Beta shape: {betas.shape}")
    Log.info(f"[Save AMASS] Gender: {amass_entry['gender']}")
    Log.info(f"[Save AMASS] Can be used with convert_amass_isaac.py")
    Log.info(f"[Save AMASS] AMASS data keys: {list(amass_data.keys())}")


def plot_trajectory_and_ground_contact(cfg, pred):
    """Plot horizontal trajectory and ground contact points."""
    transl = pred["smpl_params_global"]["transl"].cpu().numpy()
    
    # Get static confidence logits for ground contact detection
    static_conf_logits = pred["net_outputs"]["static_conf_logits"][0].cpu().numpy()  # (L, 6)
    static_conf = 1 / (1 + np.exp(-static_conf_logits))  # sigmoid to get probabilities
    
    # Simulate original trajectory before gravity correction
    # The gravity correction subtracts the minimum Y value to put the person on the ground
    # We can reverse this by adding back the minimum Y value
    transl_original = transl.copy()
    min_y = transl[:, 1].min()
    transl_original[:, 1] += min_y  # Reverse the gravity correction
    
    Log.info(f"[Gravity Analysis] Original min Y: {min_y:.3f}m")
    Log.info(f"[Gravity Analysis] Gravity correction applied: Y -= {min_y:.3f}m")
    Log.info(f"[Gravity Analysis] This prevents 'flying' poses by grounding the person")
    
    # Joint IDs for ground contact: [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]
    joint_names = ['L_Ankle', 'L_foot', 'R_Ankle', 'R_foot', 'L_wrist', 'R_wrist']
    foot_joint_ids = [0, 1, 2, 3]  # L_Ankle, L_foot, R_Ankle, R_foot
    
    # Plot horizontal trajectory (X-Z plane, Y is gravity axis)
    plt.figure(figsize=(12, 8))
    
    # Enable interactive mode if requested
    if amass_settings.get('interactive', False):
        plt.ion()  # Turn on interactive mode
    
    # Main trajectory - Before vs After Gravity Correction
    plt.subplot(2, 2, 1)
    plt.plot(transl_original[:, 0], transl_original[:, 2], 'r-', linewidth=2, alpha=0.7, label='Before Gravity Correction')
    plt.plot(transl[:, 0], transl[:, 2], 'b-', linewidth=2, label='After Gravity Correction')
    plt.scatter(transl[0, 0], transl[0, 2], c='red', s=100, marker='o', label='Start')
    plt.scatter(transl[-1, 0], transl[-1, 2], c='green', s=100, marker='s', label='End')
    plt.xlabel('X (meters)')
    plt.ylabel('Z (meters)')
    plt.title('Horizontal Trajectory: Before vs After Gravity Correction\nInteractive: Zoom, pan, and hover for details')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Add interactive features if enabled
    if amass_settings.get('interactive', False):
        # Add hover annotations
        for i, (x, z) in enumerate(zip(transl[:, 0], transl[:, 2])):
            plt.annotate(f'Frame {i}', (x, z), xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.7, visible=False)
    
    # Ground contact visualization
    plt.subplot(2, 2, 2)
    # Plot trajectory with ground contact points highlighted
    plt.plot(transl[:, 0], transl[:, 2], 'b-', linewidth=1, alpha=0.5, label='Trajectory')
    
    # Find frames with ground contact (high confidence for foot joints)
    foot_contact = static_conf[:, foot_joint_ids].max(axis=1) > 0.8  # threshold for ground contact
    contact_frames = np.where(foot_contact)[0]
    
    if len(contact_frames) > 0:
        plt.scatter(transl[contact_frames, 0], transl[contact_frames, 2], 
                   c='red', s=30, alpha=0.7, label=f'Ground Contact ({len(contact_frames)} frames)')
    
    plt.scatter(transl[0, 0], transl[0, 2], c='red', s=100, marker='o', label='Start')
    plt.scatter(transl[-1, 0], transl[-1, 2], c='green', s=100, marker='s', label='End')
    plt.xlabel('X (meters)')
    plt.ylabel('Z (meters)')
    plt.title('Trajectory with Ground Contact Points')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Ground contact confidence over time
    plt.subplot(2, 2, 3)
    time_frames = np.arange(len(static_conf))
    plt.plot(time_frames, static_conf[:, 0], 'b-', label='L_Ankle', alpha=0.7)
    plt.plot(time_frames, static_conf[:, 1], 'g-', label='L_foot', alpha=0.7)
    plt.plot(time_frames, static_conf[:, 2], 'r-', label='R_Ankle', alpha=0.7)
    plt.plot(time_frames, static_conf[:, 3], 'c-', label='R_foot', alpha=0.7)
    plt.axhline(y=0.8, color='black', linestyle='--', alpha=0.5, label='Contact Threshold')
    plt.xlabel('Frame')
    plt.ylabel('Ground Contact Confidence')
    plt.title('Ground Contact Confidence Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Height over time (Y-axis, gravity direction) - Before vs After Gravity Correction
    plt.subplot(2, 2, 4)
    plt.plot(time_frames, transl_original[:, 1], 'r-', linewidth=2, alpha=0.7, label='Before Gravity Correction')
    plt.plot(time_frames, transl[:, 1], 'b-', linewidth=2, label='After Gravity Correction')
    if len(contact_frames) > 0:
        plt.scatter(contact_frames, transl[contact_frames, 1], 
                   c='red', s=20, alpha=0.7, label='Ground Contact Frames')
    
    # Add ground level reference
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, label='Ground Level')
    plt.axhline(y=min_y, color='orange', linestyle='--', alpha=0.5, label=f'Original Ground Level ({min_y:.3f}m)')
    
    plt.xlabel('Frame')
    plt.ylabel('Height (meters)')
    plt.title('Height Over Time: Before vs After Gravity Correction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    trajectory_plot_path = Path(cfg.output_dir) / "trajectory_analysis.png"
    
    if amass_settings.get('interactive', False):
        Log.info(f"[Interactive Mode] Trajectory plots displayed interactively")
        Log.info(f"[Interactive Mode] Use mouse to zoom, pan, and explore the data")
        Log.info(f"[Interactive Mode] Close plot windows to continue processing")
        try:
            plt.show(block=True)  # Block until window is closed
        except Exception as e:
            Log.warning(f"[Interactive Mode] Error displaying plot: {e}")
            Log.info("[Interactive Mode] Falling back to saving plot to file")
            plt.savefig(trajectory_plot_path, dpi=300, bbox_inches='tight')
        finally:
            plt.close()  # Clean up after window is closed
    else:
        plt.savefig(trajectory_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        Log.info(f"[Trajectory Analysis] Saved to {trajectory_plot_path}")
    
    Log.info(f"[Trajectory Analysis] Total frames: {len(transl)}")
    Log.info(f"[Trajectory Analysis] Ground contact frames: {len(contact_frames)} ({len(contact_frames)/len(transl)*100:.1f}%)")
    Log.info(f"[Trajectory Analysis] Trajectory length: {np.linalg.norm(transl[-1] - transl[0]):.3f} meters")
    
    # Compare before and after gravity correction
    Log.info(f"[Gravity Analysis] === BEFORE vs AFTER GRAVITY CORRECTION ===")
    Log.info(f"[Gravity Analysis] Original height range: [{transl_original[:, 1].min():.3f}, {transl_original[:, 1].max():.3f}] meters")
    Log.info(f"[Gravity Analysis] Corrected height range: [{transl[:, 1].min():.3f}, {transl[:, 1].max():.3f}] meters")
    Log.info(f"[Gravity Analysis] Height difference: {transl_original[:, 1].max() - transl[:, 1].max():.3f} meters")
    Log.info(f"[Gravity Analysis] Gravity correction applied: Y -= {min_y:.3f}m")
    Log.info(f"[Gravity Analysis] This prevents 'flying' poses by grounding the person")
    Log.info(f"[Gravity Analysis] ================================================")
    
    # Analyze ground contact patterns
    if len(contact_frames) > 0:
        contact_intervals = []
        start_frame = contact_frames[0]
        for i in range(1, len(contact_frames)):
            if contact_frames[i] - contact_frames[i-1] > 1:
                contact_intervals.append((start_frame, contact_frames[i-1]))
                start_frame = contact_frames[i]
        contact_intervals.append((start_frame, contact_frames[-1]))
        
        Log.info(f"[Ground Contact] Found {len(contact_intervals)} contact intervals:")
        for i, (start, end) in enumerate(contact_intervals):
            duration = end - start + 1
            Log.info(f"[Ground Contact] Interval {i+1}: frames {start}-{end} (duration: {duration} frames)")
    
    # Create additional detailed analysis plot
    create_detailed_gravity_analysis(cfg, pred, static_conf, contact_frames, amass_settings)


def create_detailed_gravity_analysis(cfg, pred, static_conf, contact_frames, amass_settings):
    """Create detailed analysis of gravity correction and ground contact."""
    transl = pred["smpl_params_global"]["transl"].cpu().numpy()
    
    # Simulate original trajectory before gravity correction
    transl_original = transl.copy()
    min_y = transl[:, 1].min()
    transl_original[:, 1] += min_y  # Reverse the gravity correction
    
    # Create a more detailed analysis figure
    fig = plt.figure(figsize=(18, 12))
    
    # Enable interactive mode if requested
    if amass_settings.get('interactive', False):
        plt.ion()  # Turn on interactive mode
    
    # 1. 3D trajectory view - Before vs After Gravity Correction
    ax = fig.add_subplot(2, 3, 1, projection='3d')
    ax.plot(transl_original[:, 0], transl_original[:, 2], transl_original[:, 1], 'r-', linewidth=2, alpha=0.7, label='Before Gravity Correction')
    ax.plot(transl[:, 0], transl[:, 2], transl[:, 1], 'b-', linewidth=2, label='After Gravity Correction')
    if len(contact_frames) > 0:
        ax.scatter(transl[contact_frames, 0], transl[contact_frames, 2], transl[contact_frames, 1], 
                  c='red', alpha=0.7, label='Ground Contact')
    ax.scatter(transl[0, 0], transl[0, 2], transl[0, 1], c='red', marker='o', label='Start')
    ax.scatter(transl[-1, 0], transl[-1, 2], transl[-1, 1], c='green', marker='s', label='End')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Z (meters)')
    ax.set_zlabel('Y (meters) - Gravity')
    ax.set_title('3D Trajectory: Before vs After Gravity Correction')
    ax.legend()
    
    # 2. Ground contact heatmap
    ax = fig.add_subplot(2, 3, 2)
    joint_names = ['L_Ankle', 'L_foot', 'R_Ankle', 'R_foot', 'L_wrist', 'R_wrist']
    im = ax.imshow(static_conf.T, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=1)
    ax.set_yticks(range(len(joint_names)))
    ax.set_yticklabels(joint_names)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Joint')
    ax.set_title('Ground Contact Confidence Heatmap')
    plt.colorbar(im, ax=ax, label='Contact Confidence')
    
    # 3. Velocity analysis
    ax = fig.add_subplot(2, 3, 3)
    velocity = np.diff(transl, axis=0)
    velocity_magnitude = np.linalg.norm(velocity, axis=1)
    ax.plot(velocity_magnitude, 'b-', linewidth=2, label='Velocity Magnitude')
    if len(contact_frames) > 0:
        contact_velocities = velocity_magnitude[np.clip(contact_frames - 1, 0, len(velocity_magnitude) - 1)]
        ax.scatter(np.clip(contact_frames - 1, 0, len(velocity_magnitude) - 1), contact_velocities, 
                  c='red', s=20, alpha=0.7, label='Contact Frames')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Velocity (m/frame)')
    ax.set_title('Velocity Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Height distribution - Before vs After Gravity Correction
    ax = fig.add_subplot(2, 3, 4)
    ax.hist(transl_original[:, 1], bins=30, alpha=0.5, color='red', edgecolor='black', label='Before Gravity Correction')
    ax.hist(transl[:, 1], bins=30, alpha=0.7, color='blue', edgecolor='black', label='After Gravity Correction')
    ax.axvline(transl_original[:, 1].min(), color='red', linestyle='--', label=f'Original Min: {transl_original[:, 1].min():.3f}m')
    ax.axvline(transl_original[:, 1].max(), color='red', linestyle='--', label=f'Original Max: {transl_original[:, 1].max():.3f}m')
    ax.axvline(transl[:, 1].min(), color='blue', linestyle='--', label=f'Corrected Min: {transl[:, 1].min():.3f}m')
    ax.axvline(transl[:, 1].max(), color='blue', linestyle='--', label=f'Corrected Max: {transl[:, 1].max():.3f}m')
    ax.set_xlabel('Height (meters)')
    ax.set_ylabel('Frequency')
    ax.set_title('Height Distribution: Before vs After Gravity Correction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Contact duration analysis
    ax = fig.add_subplot(2, 3, 5)
    if len(contact_frames) > 0:
        # Calculate contact durations
        contact_durations = []
        current_duration = 1
        for i in range(1, len(contact_frames)):
            if contact_frames[i] - contact_frames[i-1] == 1:
                current_duration += 1
            else:
                contact_durations.append(current_duration)
                current_duration = 1
        contact_durations.append(current_duration)
        
        if contact_durations:
            ax.hist(contact_durations, bins=min(20, len(contact_durations)), alpha=0.7, color='green', edgecolor='black')
            ax.set_xlabel('Contact Duration (frames)')
            ax.set_ylabel('Frequency')
            ax.set_title('Ground Contact Duration Distribution')
            ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No ground contact detected', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Ground Contact Duration Distribution')
    
    # 6. Trajectory curvature
    ax = fig.add_subplot(2, 3, 6)
    # Calculate trajectory curvature (simplified)
    if len(transl) > 2:
        # Calculate angles between consecutive segments
        segments = np.diff(transl[:, [0, 2]], axis=0)  # Use X-Z plane
        angles = []
        for i in range(1, len(segments)):
            dot_product = np.dot(segments[i-1], segments[i])
            norms = np.linalg.norm(segments[i-1]) * np.linalg.norm(segments[i])
            if norms > 0:
                cos_angle = np.clip(dot_product / norms, -1, 1)
                angle = np.arccos(cos_angle)
                angles.append(angle)
        
        if angles:
            ax.plot(angles, 'b-', linewidth=2, label='Trajectory Curvature')
            ax.set_xlabel('Segment')
            ax.set_ylabel('Angle (radians)')
            ax.set_title('Trajectory Curvature Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient data for curvature analysis', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Trajectory Curvature Analysis')
    else:
        ax.text(0.5, 0.5, 'Insufficient data for curvature analysis', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Trajectory Curvature Analysis')
    
    plt.tight_layout()
    detailed_analysis_path = Path(cfg.output_dir) / "detailed_trajectory_analysis.png"
    
    if amass_settings.get('interactive', False):
        try:
            plt.show(block=True)  # Block until window is closed
        except Exception as e:
            Log.warning(f"[Interactive Mode] Error displaying detailed plot: {e}")
            Log.info("[Interactive Mode] Falling back to saving detailed plot to file")
            plt.savefig(detailed_analysis_path, dpi=300, bbox_inches='tight')
        finally:
            plt.close()  # Clean up after window is closed
    else:
        plt.savefig(detailed_analysis_path, dpi=300, bbox_inches='tight')
        plt.close()
        Log.info(f"[Detailed Analysis] Saved to {detailed_analysis_path}")


if __name__ == "__main__":
    cfg, amass_settings = parse_args_to_cfg()
    paths = cfg.paths
    Log.info(f"[GPU]: {torch.cuda.get_device_name()}")
    Log.info(f'[GPU]: {torch.cuda.get_device_properties("cuda")}')

    # ===== Preprocess and save to disk ===== #
    run_preprocess(cfg)
    data = load_data_dict(cfg)

    # ===== HMR4D ===== #
    if not Path(paths.hmr4d_results).exists():
        Log.info("[HMR4D] Predicting")
        model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(cfg.ckpt_path)
        model = model.eval().cuda()
        tic = Log.sync_time()
        pred = model.predict(data, static_cam=cfg.static_cam)
        pred = detach_to_cpu(pred)
        data_time = data["length"] / 30
        Log.info(f"[HMR4D] Elapsed: {Log.sync_time() - tic:.2f}s for data-length={data_time:.1f}s")
        torch.save(pred, paths.hmr4d_results)
    else:
        pred = torch.load(paths.hmr4d_results)

    # ===== Save AMASS format ===== #
    save_amass_format(amass_settings['amass_output'], pred)

    # ===== Render ===== #
    render_incam(cfg)
    render_global(cfg)
    if not Path(paths.incam_global_horiz_video).exists():
        Log.info("[Merge Videos]")
        merge_videos_horizontal([paths.incam_video, paths.global_video], paths.incam_global_horiz_video)

    # ===== Plot trajectory and ground contact ===== #
    if amass_settings['plot_trajectory']:
        if MATPLOTLIB_AVAILABLE:
            plot_trajectory_and_ground_contact(cfg, pred)
        else:
            Log.error("Cannot generate trajectory plots: matplotlib not available")
