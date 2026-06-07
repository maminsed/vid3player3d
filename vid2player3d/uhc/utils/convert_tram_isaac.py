import joblib
import numpy as np
import os
import sys
sys.path.append(os.getcwd())

import argparse
from embodied_pose.utils.motion_lib import MotionLib
import torch
from tqdm import tqdm
import yaml
from glob import glob
import contextlib

from scipy.spatial.transform import Rotation as sRot
from uhc.smpllib.smpl_parser import SMPL_BONE_ORDER_NAMES as joint_names
from uhc.smpllib.smpl_local_robot import Robot as LocalRobot
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState

# SMPL constants from TRAM
JOINT_NAMES = [
'OP Nose', 'OP Neck', 'OP RShoulder',           #0,1,2
'OP RElbow', 'OP RWrist', 'OP LShoulder',       #3,4,5
'OP LElbow', 'OP LWrist', 'OP MidHip',          #6, 7,8
'OP RHip', 'OP RKnee', 'OP RAnkle',             #9,10,11
'OP LHip', 'OP LKnee', 'OP LAnkle',             #12,13,14
'OP REye', 'OP LEye', 'OP REar',                #15,16,17
'OP LEar', 'OP LBigToe', 'OP LSmallToe',        #18,19,20
'OP LHeel', 'OP RBigToe', 'OP RSmallToe', 'OP RHeel',  #21, 22, 23, 24  ##Total 25 joints  for openpose
'Right Ankle', 'Right Knee', 'Right Hip',               #0,1,2
'Left Hip', 'Left Knee', 'Left Ankle',                  #3, 4, 5
'Right Wrist', 'Right Elbow', 'Right Shoulder',     #6
'Left Shoulder', 'Left Elbow', 'Left Wrist',            #9
'Neck (LSP)', 'Top of Head (LSP)',                      #12, 13
'Pelvis (MPII)', 'Thorax (MPII)',                       #14, 15
'Spine (H36M)', 'Jaw (H36M)',                           #16, 17
'Head (H36M)', 'Nose', 'Left Eye',                      #18, 19, 20
'Right Eye', 'Left Ear', 'Right Ear'                    #21,22,23 (Total 24 joints)
]

# Dict containing the joints in numerical order
JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))}

# Map joints to SMPL joints
JOINT_MAP = {
'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
'Spine (H36M)': 51, 'Jaw (H36M)': 52,
'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}

# SMPL data path
SMPL_DATA_PATH = "data/smpl/"

SMPL_MODEL_PATH = os.path.join(SMPL_DATA_PATH, "SMPL_NEUTRAL.pkl")
SMPL_MEAN_PARAMS = os.path.join(SMPL_DATA_PATH, "smpl_mean_params.npz")
SMPL_KINTREE_PATH = os.path.join(SMPL_DATA_PATH, "kintree_table.pkl")
JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(SMPL_DATA_PATH, 'J_regressor_extra.npy')
JOINT_REGRESSOR_H36M = os.path.join(SMPL_DATA_PATH, 'J_regressor_h36m.npy')

# SMPL class from TRAM
from smplx import SMPL as _SMPL
from smplx import SMPLLayer as _SMPLLayer
from smplx.body_models import SMPLOutput
from smplx.lbs import vertices2joints

class SMPL(_SMPL):
    def __init__(self, create_default=False, *args, **kwargs):
        kwargs["model_path"] = "data/smpl"

        # remove the verbosity for the 10-shapes beta parameters
        with contextlib.redirect_stdout(None):
            super(SMPL, self).__init__(
                create_body_pose=create_default,
                create_betas=create_default,
                create_global_orient=create_default,
                create_transl=create_default,
                *args, 
                **kwargs
            )

        # SPIN 49(25 OP + 24) joints
        joints = [JOINT_MAP[i] for i in JOINT_NAMES]
        J_regressor_extra = np.load(JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)
        
    def forward(self, default_smpl=False, *args, **kwargs):
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        if default_smpl:
            return smpl_output

        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]

        output = SMPLOutput(vertices=smpl_output.vertices,
                            global_orient=smpl_output.global_orient,
                            body_pose=smpl_output.body_pose,
                            betas=smpl_output.betas,
                            full_pose=smpl_output.full_pose,
                            joints=joints)

        return output

    def query(self, hmr_output, default_smpl=False):
        pred_rotmat = hmr_output['pred_rotmat']
        pred_shape = hmr_output['pred_shape']

        smpl_out = self(global_orient=pred_rotmat[:, [0]],
                        body_pose = pred_rotmat[:, 1:],
                        betas = pred_shape,
                        default_smpl = default_smpl,
                        pose2rot=False)
        return smpl_out

parser = argparse.ArgumentParser()
parser.add_argument('--tram_dir', type=str, required=True, help='Path to TRAM results directory (should contain hps/)')
parser.add_argument('--out_dir', type=str, default="data/motion_lib/tram")
parser.add_argument('--num_motion_libs', type=int, default=14)
args = parser.parse_args()

num_motion_libs = args.num_motion_libs
tram_dir = args.tram_dir
hps_dir = os.path.join(tram_dir, 'hps')
hps_files = sorted(glob(os.path.join(hps_dir, '*.npy')))

os.makedirs(args.out_dir, exist_ok=True)
meta_data = {
    "tram_dir": args.tram_dir,
    "num_motion_libs": num_motion_libs
}
yaml.safe_dump(meta_data, open(f'{args.out_dir}/args.yml', 'w'))

info = joblib.load('data/misc/smpl_body_info.pkl')

# Collect all unique betas
all_beta = []
for hps_file in hps_files:
    d = np.load(hps_file, allow_pickle=True).item()
    pred_shape = d['pred_shape']
    mean_shape = pred_shape.mean(dim=0, keepdim=True)
    all_beta.append(mean_shape.squeeze().cpu().numpy())
_, index = np.unique([",".join([f"{x:.6f}" for x in beta]) for beta in all_beta], return_index=True)
index.sort()
beta_arr = [all_beta[i] for i in index]
beta_mapping = dict()
for i, beta in enumerate(beta_arr):
    key = ",".join([f"{x:.6f}" for x in beta])
    beta_mapping[key] = i
print(f'TRAM data has {len(beta_mapping)} unique body shapes!')
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

# Split files for motion libs
motion_lib_seq_arr = np.array_split(hps_files, num_motion_libs)
seq_name_splits = {}
for i, seq_arr in enumerate(motion_lib_seq_arr):
    seq_name_splits[i] = [os.path.basename(f) for f in seq_arr]
joblib.dump(seq_name_splits, f'{args.out_dir}/seq_name_splits.pkl')

for i, motion_lib_seqs in enumerate(tqdm(motion_lib_seq_arr)):
    motion_lib_input_dict = dict()
    for hps_file in motion_lib_seqs:
        d = np.load(hps_file, allow_pickle=True).item()
        batch_size = 500
        pred_rotmat = d['pred_rotmat'][:batch_size]  # [N, 24, 3, 3], torch.Tensor
        pred_shape = d['pred_shape'][:batch_size]    # [N, 10], torch.Tensor
        pred_trans = d['pred_trans'][:batch_size]    # [N, 1, 3], torch.Tensor
        frame = d['frame'][:batch_size]              # [N], torch.Tensor
        
        # Get root translation with offset (no coordinate transformation)
        root_trans = pred_trans[:, 0].cpu().numpy()  # Use raw root joint position
        
        first_frame_pos = root_trans[0:1].repeat(batch_size, axis=0)
        root_trans = first_frame_pos
        
 
        # Prepare pose_aa for MuJoCo (add zeros for hands if needed)
        pose_aa = torch.from_numpy(sRot.from_matrix(pred_rotmat.cpu().numpy().reshape(-1, 3, 3)).as_rotvec()).float().reshape(pred_rotmat.shape[0], 24*3)
        pose_aa = np.concatenate([pose_aa.numpy()[:, :66], np.zeros((batch_size, 6))], axis=1)
        pose_aa_mj = pose_aa.reshape(-1, 24, 3)[..., smpl_2_mujoco, :].copy()

        # Convert to quaternions (no coordinate transformation)
        pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(batch_size, 24, 4)
        
        # Gender: default to neutral
        gender_number = [0]
        smpl_parser = smpl_local_robot.smpl_parser_n
        smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None, ]), gender=gender_number)
        smpl_local_robot.write_xml()
        skeleton_tree = SkeletonTree.from_mjcf(model_xml_path)
        
        root_trans_offset = torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]

        # Create skeleton state
        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat),
            root_trans_offset,
            is_local=True)

        # Get vertices and calculate minimum height
        verts, joints = smpl_parser.get_joints_verts(
            pose=torch.from_numpy(pose_aa),
            th_betas=torch.from_numpy(beta[None, ]),
            th_trans=torch.from_numpy(root_trans)
        )
        min_verts_h = verts[..., 2].min(dim=-1)[0].mean().item()

        # Convert to global rotations (no coordinate transformation)
        pose_quat_global = new_sk_state.global_rotation.numpy().reshape(batch_size, -1, 4)

        # Create motion
        new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=30.0)
        
        # Get sequence name and beta key
        seq_name = os.path.splitext(os.path.basename(hps_file))[0]
        beta_key = ",".join([f"{x:.6f}" for x in beta])
        
        # Prepare output dictionary
        new_motion_out = new_motion.to_dict()
        new_motion_out['seq_name'] = seq_name
        new_motion_out['seq_idx'] = int(seq_name.split('_')[-1])
        new_motion_out['trans'] = root_trans
        new_motion_out['root_trans'] = root_trans_offset.numpy()
        new_motion_out['pose_aa'] = pose_aa
        new_motion_out['beta'] = beta
        new_motion_out['beta_idx'] = beta_mapping[beta_key]
        new_motion_out['gender'] = 'neutral'
        new_motion_out['min_verts_h'] = min_verts_h
        new_motion_out['body_scale'] = 1.0
        new_motion_out['__name__'] = "SkeletonMotion"
        
        print(f"Motion sequence length: {len(new_motion_out['pose_aa'])}")
        print(f"Motion sequence shape: {new_motion_out['pose_aa'].shape}")
        
        motion_lib_input_dict[seq_name] = new_motion_out

    if motion_lib_input_dict:
        # Save the motion data to a temporary file and pass its path
        temp_motion_file = os.path.join(args.out_dir, f"temp_motion_{i}.pkl")
        joblib.dump(motion_lib_input_dict, temp_motion_file)
        motion_lib = MotionLib(motion_file=temp_motion_file,
            dof_body_ids=info['dof_body_ids'],
            dof_offsets=info['dof_offsets'],
            key_body_ids=info['key_body_ids'],
            device='cpu',
            clean_up=True
        )
        print(f"\nMotion library part {i} created with {len(motion_lib_input_dict)} sequences")
        torch.save(motion_lib, f"{args.out_dir}/mlib_part_{i:05d}.pth")
    else:
        print(f"No motion data found for motion_lib_part_{i:05d}. Skipping.")
    del motion_lib_input_dict

smpl_local_robot.clean_up()
