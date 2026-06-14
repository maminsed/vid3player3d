#!/usr/bin/env python3
"""Convert low-level policy rollout exports into Video3DPoseDataset files.

This consumes the pickle files written by embodied_pose/run.py with
--export_dataset. Those pickles contain physically simulated SMPL pose/root
tracks. The output directory matches vid2player.motion_vae.dataset.Video3DPoseDataset.
"""

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
SMPL_VIS_ROOT = REPO_ROOT / "smpl_visualizer"
if str(SMPL_VIS_ROOT) not in sys.path:
    sys.path.append(str(SMPL_VIS_ROOT))


SMPL_JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Torso", "L_Knee", "R_Knee", "Spine",
    "L_Ankle", "R_Ankle", "Chest", "L_Toe", "R_Toe", "Neck", "L_Thorax",
    "R_Thorax", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
]

EXPORT_JOINT_NAMES = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee",
    "R_Ankle", "R_Toe", "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand", "R_Thorax",
    "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
]

EXPORT_TO_SMPL = [EXPORT_JOINT_NAMES.index(name) for name in SMPL_JOINT_NAMES]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build an MVAE tennis_dataset directory from --export_dataset "
            "rollout pickle files."
        )
    )
    parser.add_argument(
        "--input-yaml",
        required=True,
        help=(
            "YAML listing export pickle files. Accepts {'motions': [{'file': ...}]}, "
            "{'exports': [...]}, or a plain list."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="tennis_dataset_physics",
        help="Directory to write manifest.json and *.npy arrays.",
    )
    parser.add_argument("--side", choices=["fg", "bg"], default="fg")
    parser.add_argument("--background", default="usopen")
    parser.add_argument("--manifest-gender", default="mens")
    parser.add_argument("--player", default="Federer")
    parser.add_argument("--handness", default="right")
    parser.add_argument("--is-orig", action="store_true",
                        help="Mark manifest videos as original annotations. Leave unset for no-phase weak data.")
    parser.add_argument("--trim-start", type=int, default=0,
                        help="Drop this many frames from the start of every export.")
    parser.add_argument("--trim-end", type=int, default=0,
                        help="Drop this many frames from the end of every export.")
    parser.add_argument("--min-frames", type=int, default=12,
                        help="Skip sequences shorter than this after trimming.")
    parser.add_argument("--smpl-model-dir", default=None,
                        help="Override SMPL model directory.")
    parser.add_argument("--device", default="cpu",
                        help="Torch device for SMPL FK, e.g. cpu or cuda:0.")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Frames per SMPL FK batch.")
    return parser.parse_args()


def resolve_path(path, base_dir):
    path = Path(path)
    if path.is_absolute():
        return path
    base_path = (base_dir / path).resolve()
    if base_path.exists():
        return base_path
    return (Path.cwd() / path).resolve()


def iter_yaml_entries(yaml_path):
    with open(yaml_path, "r") as fp:
        data = yaml.safe_load(fp)

    if isinstance(data, dict):
        if "motions" in data:
            entries = data["motions"]
        elif "exports" in data:
            entries = data["exports"]
        elif "files" in data:
            entries = data["files"]
        else:
            entries = list(data.values())
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError(f"Unsupported YAML structure in {yaml_path}")

    base_dir = Path(yaml_path).resolve().parent
    for idx, entry in enumerate(entries):
        if isinstance(entry, str):
            yield {"file": resolve_path(entry, base_dir), "name": None}
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"Unsupported entry #{idx}: {entry!r}")
        file_value = entry.get("file") or entry.get("path")
        if file_value is None:
            raise ValueError(f"Entry #{idx} is missing 'file' or 'path': {entry!r}")
        yield {
            "file": resolve_path(file_value, base_dir),
            "name": entry.get("name"),
            "player": entry.get("player"),
            "handness": entry.get("handness"),
            "side": entry.get("side"),
        }


def load_export_sequences(entry):
    export_path = entry["file"]
    if not export_path.exists():
        raise FileNotFoundError(export_path)

    payload = joblib.load(export_path)
    if isinstance(payload, dict) and {"pose_aa", "trans_orig"}.issubset(payload):
        payload = {entry.get("name") or export_path.stem: payload}
    if not isinstance(payload, dict):
        raise ValueError(f"{export_path} must contain a dict export payload.")

    for seq_key, seq_data in payload.items():
        if not isinstance(seq_data, dict):
            continue
        if "pose_aa" not in seq_data or ("trans_orig" not in seq_data and "trans" not in seq_data):
            continue
        name = entry.get("name") or f"{export_path.stem}_{seq_key}"
        yield name, seq_data


def normalize_pose_array(pose_aa):
    pose_aa = np.asarray(pose_aa, dtype=np.float32)
    if pose_aa.ndim == 2:
        if pose_aa.shape[1] < 72:
            raise ValueError(f"pose_aa must have at least 72 columns, got {pose_aa.shape}")
        pose_aa = pose_aa[:, :72].reshape(-1, 24, 3)
    elif pose_aa.ndim == 3:
        if pose_aa.shape[1:] != (24, 3):
            raise ValueError(f"pose_aa must have shape (T, 24, 3), got {pose_aa.shape}")
    else:
        raise ValueError(f"pose_aa must be 2D or 3D, got {pose_aa.shape}")

    return pose_aa[:, EXPORT_TO_SMPL]


def normalize_trans_array(seq_data):
    trans = seq_data.get("trans_orig", seq_data.get("trans"))
    trans = np.asarray(trans, dtype=np.float32)
    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"trans_orig/trans must have shape (T, 3), got {trans.shape}")
    return trans


def normalize_beta(seq_data):
    beta = np.asarray(seq_data.get("beta", np.zeros(10)), dtype=np.float32)
    if beta.ndim > 1:
        beta = beta.reshape(-1, beta.shape[-1])[0]
    if beta.size < 10:
        beta = np.pad(beta, (0, 10 - beta.size))
    return beta[:10]


def normalize_gender(seq_data):
    gender = seq_data.get("gender", "neutral")
    if isinstance(gender, np.ndarray):
        gender = gender.item()
    if isinstance(gender, bytes):
        gender = gender.decode("utf-8")
    gender = str(gender).lower()
    if gender not in {"neutral", "male", "female"}:
        return "neutral"
    return gender


def trim_sequence(pose, trans, trim_start, trim_end):
    if trim_start < 0 or trim_end < 0:
        raise ValueError("--trim-start and --trim-end must be non-negative")
    end = len(pose) - trim_end if trim_end > 0 else len(pose)
    if trim_start >= end:
        return pose[:0], trans[:0]
    return pose[trim_start:end], trans[trim_start:end]


def compute_joint_arrays(pose_aa, trans, beta, gender, smpl_model_dir, device, batch_size):
    from smpl_visualizer.smpl import SMPL, SMPL_MODEL_DIR

    if smpl_model_dir is None:
        repo_smpl_dir = REPO_ROOT / "data" / "smpl"
        smpl_model_dir = str(repo_smpl_dir if repo_smpl_dir.exists() else SMPL_MODEL_DIR)

    rotmat = Rotation.from_rotvec(pose_aa.reshape(-1, 3)).as_matrix()
    rotmat = rotmat.reshape(len(pose_aa), 24, 3, 3).astype(np.float32)

    smpl = SMPL(
        smpl_model_dir,
        create_transl=False,
        gender=gender,
        betas=beta[None],
        batch_size=max(1, int(batch_size)),
        device=torch.device(device),
    ).to(device)

    joint_pos_chunks = []
    with torch.no_grad():
        for start in range(0, len(pose_aa), batch_size):
            end = min(start + batch_size, len(pose_aa))
            rotmat_chunk = torch.from_numpy(rotmat[start:end]).to(device=device, dtype=torch.float32)
            trans_chunk = torch.from_numpy(trans[start:end]).to(device=device, dtype=torch.float32)
            joints = smpl.get_joints_fast(pose=rotmat_chunk, root_trans=trans_chunk)
            joints = joints[:, :24].detach().cpu().numpy().astype(np.float32)
            root_pos = trans[start:end].astype(np.float32)
            rel_joint_pos = joints[:, 1:] - joints[:, :1]
            joint_pos_chunks.append(np.concatenate([root_pos, rel_joint_pos.reshape(end - start, -1)], axis=1))

    joint_pos = np.concatenate(joint_pos_chunks, axis=0)
    return joint_pos.astype(np.float32), pose_aa.astype(np.float32), rotmat


def sanitize_valid_frames(pose_aa, trans):
    valid = np.isfinite(pose_aa).all(axis=(1, 2)) & np.isfinite(trans).all(axis=1)
    pose_safe = pose_aa.copy()
    trans_safe = trans.copy()
    if not valid.all():
        pose_safe[~valid] = 0.0
        trans_safe[~valid] = 0.0
    return pose_safe, trans_safe, valid


def build_manifest_entry(name, seq_meta, base, length, beta, args, entry):
    side = entry.get("side") or args.side
    player = entry.get("player") or args.player
    handness = entry.get("handness") or args.handness
    seq = {
        "base": int(base),
        "start": 0,
        "length": int(length),
        "beta": beta.astype(float).tolist(),
        "player": player,
        "handness": handness,
        "source_export": seq_meta["source_export"],
        "source_key": seq_meta["source_key"],
        "fps": float(seq_meta.get("fps", 30.0)),
    }
    return {
        "name": name,
        "background": args.background,
        "gender": args.manifest_gender,
        "is_orig": bool(args.is_orig),
        "sequences": {
            "fg": [seq] if side == "fg" else [],
            "bg": [seq] if side == "bg" else [],
        },
    }


def main():
    args = parse_args()

    global joblib, np, torch, yaml, Rotation
    import joblib
    import numpy as np
    import torch
    import yaml
    from scipy.spatial.transform import Rotation

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    joint_pos_arr = []
    joint_rot_arr = []
    joint_rotmat_arr = []
    valid_arr = []
    base = 0
    skipped = []

    entries = list(iter_yaml_entries(args.input_yaml))
    if not entries:
        raise RuntimeError(f"No export files listed in {args.input_yaml}")

    for entry in entries:
        for seq_key, seq_data in load_export_sequences(entry):
            pose_aa = normalize_pose_array(seq_data["pose_aa"])
            trans = normalize_trans_array(seq_data)
            num_frames = min(len(pose_aa), len(trans))
            pose_aa, trans = pose_aa[:num_frames], trans[:num_frames]
            pose_aa, trans = trim_sequence(pose_aa, trans, args.trim_start, args.trim_end)

            if len(pose_aa) < args.min_frames:
                skipped.append((seq_key, len(pose_aa), "too_short"))
                continue

            beta = normalize_beta(seq_data)
            gender = normalize_gender(seq_data)
            pose_aa, trans, valid = sanitize_valid_frames(pose_aa, trans)
            joint_pos, joint_rot, joint_rotmat = compute_joint_arrays(
                pose_aa=pose_aa,
                trans=trans,
                beta=beta,
                gender=gender,
                smpl_model_dir=args.smpl_model_dir,
                device=args.device,
                batch_size=args.batch_size,
            )

            seq_meta = {
                "source_export": str(entry["file"]),
                "source_key": seq_key,
                "fps": seq_data.get("fps", 30.0),
            }
            manifest.append(build_manifest_entry(
                name=seq_key,
                seq_meta=seq_meta,
                base=base,
                length=len(pose_aa),
                beta=beta,
                args=args,
                entry=entry,
            ))

            joint_pos_arr.append(joint_pos)
            joint_rot_arr.append(joint_rot)
            joint_rotmat_arr.append(joint_rotmat)
            valid_arr.append(valid.astype(bool))
            base += len(pose_aa)
            print(f"Added {seq_key}: {len(pose_aa)} frames")

    if not joint_pos_arr:
        details = ", ".join([f"{name}:{length}:{reason}" for name, length, reason in skipped])
        raise RuntimeError(f"No valid sequences converted. Skipped: {details}")

    joint_pos_arr = np.concatenate(joint_pos_arr, axis=0)
    joint_rot_arr = np.concatenate(joint_rot_arr, axis=0)
    joint_rotmat_arr = np.concatenate(joint_rotmat_arr, axis=0)
    valid_arr = np.concatenate(valid_arr, axis=0)

    np.save(out_dir / "joint_pos.npy", joint_pos_arr)
    np.save(out_dir / "joint_rot.npy", joint_rot_arr)
    np.save(out_dir / "joint_rotmat.npy", joint_rotmat_arr)
    np.save(out_dir / "valid.npy", valid_arr)
    with open(out_dir / "manifest.json", "w") as fp:
        json.dump(manifest, fp, indent=2)

    summary = {
        "num_sequences": len(manifest),
        "num_frames": int(len(valid_arr)),
        "num_valid_frames": int(valid_arr.sum()),
        "predict_phase": False,
        "joint_pos_shape": list(joint_pos_arr.shape),
        "joint_rot_shape": list(joint_rot_arr.shape),
        "joint_rotmat_shape": list(joint_rotmat_arr.shape),
        "skipped": skipped,
    }
    with open(out_dir / "conversion_summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    print(f"Wrote MVAE dataset to {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
