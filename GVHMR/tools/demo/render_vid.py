"""
Use these command to copy your stuff to here:
- cp vid2player3d/data/motion_lib/updated_0-Amanda_Anisimova_vs._Beatriz_Haddad_Maia_reencoded-scene000-000/mlib_part_00000_render.pkl GVHMR/tmp/
- cp GVHMR/outputs/demo/0-Amanda_Anisimova_vs._Beatriz_Haddad_Maia_reencoded-scene000-000/1_incam.mp4 GVHMR/tmp/
"""
from pathlib import Path
import torch
from tqdm import tqdm
from einops import einsum

from GVHMR.hmr4d.utils.pylogger import Log
from GVHMR.hmr4d.utils.smplx_utils import make_smplx
from GVHMR.hmr4d.utils.net_utils import to_cuda
from GVHMR.hmr4d.utils.vis.renderer import (
    Renderer,
    get_global_cameras_static,
    get_ground_params_from_points,
)
from GVHMR.hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from GVHMR.hmr4d.utils.video_io_utils import get_video_lwh, get_writer
from GVHMR.hmr4d.utils.geo.hmr_cam import create_camera_sensor
import joblib

CRF = 23

def render_global(cfg):
    global_video_path = Path(cfg.paths.global_video)
    if global_video_path.exists():
        Log.info(f"[Render Global] Video already exists at {global_video_path}")
        return

    debug_cam = False
    faces_smpl = make_smplx("smpl").faces
    J_regressor = torch.load("/pub2/amin/vid3player/GVHMR/hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()

    pred_path = str(cfg.paths.pred_path)
    if pred_path.endswith('.pkl'):
        # Load from convert_amass render pkl
        pred = joblib.load(pred_path)
        seq_name = getattr(cfg, 'seq_name', None) or list(pred.keys())[0]
        seq_data = pred[seq_name]
        verts_raw = seq_data['verts']
        if not isinstance(verts_raw, torch.Tensor):
            verts_raw = torch.tensor(verts_raw)
        verts_raw = verts_raw.float()
        # Convert from Z-up to Y-up: (x, y, z) -> (x, z, -y)
        pred_ay_verts = torch.zeros_like(verts_raw)
        pred_ay_verts[:, :, 0] = verts_raw[:, :, 0]
        pred_ay_verts[:, :, 1] = verts_raw[:, :, 2]
        pred_ay_verts[:, :, 2] = -verts_raw[:, :, 1]
        pred_ay_verts = pred_ay_verts.cuda()
        fps = seq_data.get('fps', 30.0)
        Log.info(f"[Render Global] Loaded convert_amass data: seq={seq_name}, frames={pred_ay_verts.shape[0]}")
    else:
        # Original GVHMR format (.pth)
        pred = torch.load(pred_path)
        smplx_model = make_smplx("supermotion").cuda()
        smplx2smpl = torch.load("/pub2/amin/GVHMR/hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
        smplx_out = smplx_model(**to_cuda(pred["smpl_params_global"]))
        pred_ay_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])
        fps = 30.0

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
    if hasattr(cfg, 'video_path') and cfg.video_path and Path(str(cfg.video_path)).exists():
        _, width, height = get_video_lwh(cfg.video_path)
    else:
        width, height = 1920, 1080
    _, _, K = create_camera_sensor(width, height, 24)  # render as 24mm lens

    # renderer
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K, bin_size=0)

    # -- render mesh -- #
    scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], verts_glob)
    renderer.set_ground(scale * 1.5, cx, cz)
    color = torch.ones(3).float().cuda() * 0.8

    render_length = verts_glob.shape[0] if not debug_cam else 8
    writer = get_writer(global_video_path, fps=int(fps), crf=CRF)
    for i in tqdm(range(render_length), desc=f"Rendering Global"):
        cameras = renderer.create_camera(global_R[i], global_T[i])
        img = renderer.render_with_ground(verts_glob[[i]], color[None], cameras, global_lights)
        writer.write_frame(img)
    writer.close()


def main():
    from types import SimpleNamespace

    # Path to the render pkl saved by convert_amass_isaac_correct_ground.py
    PRED_PATH = "/pub2/amin/vid3player/GVHMR/tmp/mlib_part_00000_render.pkl"
    OUT_VIDEO = Path("/pub2/amin/vid3player/GVHMR/tmp/render_global.mp4")
    REF_VIDEO = Path("/pub2/amin/vid3player/GVHMR/tmp/1_incam.mp4")  # optional, for video dimensions

    paths = SimpleNamespace(
        global_video=OUT_VIDEO,
        pred_path=PRED_PATH,
    )
    cfg = SimpleNamespace(
        paths=paths,
        video_path=REF_VIDEO if REF_VIDEO.exists() else None,
        seq_name=None,  # defaults to first sequence in pkl
    )
    render_global(cfg)


if __name__ == "__main__":
    main()
