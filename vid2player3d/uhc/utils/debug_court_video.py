"""CUDA-only, display-free video of corrected SMPL vertices in court coordinates."""

from itertools import product
from pathlib import Path
import tempfile

import numpy as np


VIDEO_SIZE = (960, 540)
VERTICAL_FOV = 45.0
CAMERA_ELEVATION = 20.0


def require_cuda_renderer():
    """Fail before conversion if the requested GPU renderer cannot run."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            '--verbose requires CUDA for video rendering; there is no CPU fallback. '
            'Check GPU access (including sandbox restrictions) in this process.')
    from pytorch3d.renderer import MeshRasterizer, PerspectiveCameras, RasterizationSettings
    from pytorch3d.structures import Meshes

    # Importing the extension alone does not establish CUDA kernel compatibility.
    mesh = Meshes(
        verts=[torch.tensor([[-.5, -.5, 2.], [.5, -.5, 2.], [0., .5, 2.]], device='cuda')],
        faces=[torch.tensor([[0, 1, 2]], device='cuda')])
    rasterizer = MeshRasterizer(
        cameras=PerspectiveCameras(device='cuda'),
        raster_settings=RasterizationSettings(image_size=16, faces_per_pixel=1))
    with torch.no_grad():
        rasterizer(mesh)
    torch.cuda.synchronize()


def court_geometry(width, length):
    """Simple colored boxes: court surface at Z=0, lines, posts and an open net."""
    vertices, faces, colors = [], [], []
    triangles = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    corners = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]) / 2

    def box(center, size, color):
        faces.append(triangles + len(vertices) * 8)
        vertices.append(corners * size + center)
        colors.append(np.tile(color, (8, 1)))

    blue, green, white, dark = (.12, .28, .48), (.15, .32, .25), (.95, .95, .95), (.12, .14, .16)
    box((0, 0, -.07), (width + 16, length + 24, .1), green)
    box((0, 0, -.01), (width, length, .02), blue)
    for x in (-width / 2, -8.23 / 2, 8.23 / 2, width / 2):
        box((x, 0, .008), (.06, length, .008), white)
    for y in (-length / 2, length / 2):
        box((0, y, .008), (width, .06, .008), white)
    for y in (-6.4, 6.4):
        box((0, y, .008), (8.23, .06, .008), white)
    box((0, 0, .008), (.06, 12.8, .008), white)
    # Match the existing tennis viewer's simplified 1.07 m net and post spacing.
    net_width = width + 2 * .91
    for x in (-net_width / 2, net_width / 2):
        box((x, 0, .535), (.09, .09, 1.07), dark)
    box((0, 0, 1.05), (net_width, .045, .04), white)
    for z in np.arange(.1, 1.01, .15):
        box((0, 0, z), (net_width, .015, .015), dark)
    for x in np.arange(-net_width / 2, net_width / 2, .3):
        box((x, 0, .525), (.015, .015, 1.05), dark)
    return tuple(np.concatenate(items) for items in (vertices, faces, colors))


def fixed_rear_camera(bounds, width, image_size=VIDEO_SIZE):
    """Fit all motion bounds and the net, keeping one camera for the clip."""
    motion_corners = np.array(list(product(*zip(*bounds))))
    net_corners = np.array(list(product((-width / 2 - .91, width / 2 + .91),
                                        (0.,), (0., 1.1))))
    points = np.concatenate((motion_corners, net_corners))
    target = (points.min(0) + points.max(0)) / 2
    target[2] = .8
    elevation = np.deg2rad(CAMERA_ELEVATION)
    # Slight rear diagonal, always on the negative-Y (near-player) side.
    backward = np.array([.12 * np.cos(elevation), -np.cos(elevation), np.sin(elevation)])
    backward /= np.linalg.norm(backward)
    forward = -backward
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    relative = points - target
    tan_v = np.tan(np.deg2rad(VERTICAL_FOV / 2))
    tan_h = tan_v * image_size[0] / image_size[1]
    depth = relative @ forward
    distance = max(np.max(np.abs(relative @ right) / (.88 * tan_h) - depth),
                   np.max(np.abs(relative @ up) / (.88 * tan_v) - depth), 5.)
    return target + backward * distance, target


def save_corrected_court_video(verts, faces, fps, output_path, court_width, court_length):
    """Render exact saved vertices, without recentering or changing their height.

    Only one body frame is uploaded at a time. Rasterization and shading run on
    CUDA; completed RGB frames are streamed to FFmpeg for H.264 encoding.
    """
    import cv2
    import imageio.v2 as imageio
    import torch
    from pytorch3d.renderer import (
        MeshRasterizer, MeshRenderer, PerspectiveCameras, PointLights,
        RasterizationSettings, SoftPhongShader, TexturesVertex, look_at_view_transform)
    from pytorch3d.structures import Meshes
    from tqdm import tqdm

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for debug video; CPU rendering is disabled.')
    verts = torch.as_tensor(verts).detach()
    if verts.ndim != 3 or verts.shape[-1] != 3 or not len(verts):
        raise ValueError('Expected vertices with shape (frames, vertices, 3)')
    if not torch.isfinite(verts).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Video vertices and positive FPS must be finite')
    width, height = VIDEO_SIZE
    bounds = torch.stack((verts.amin(dim=(0, 1)), verts.amax(dim=(0, 1)))).cpu().numpy()
    eye, target = fixed_rear_camera(bounds, court_width)
    rotation, translation = look_at_view_transform(
        eye=[eye.tolist()], at=[target.tolist()], up=((0, 0, 1),), device='cuda')
    focal = height / (2 * np.tan(np.deg2rad(VERTICAL_FOV / 2)))
    camera = PerspectiveCameras(
        device='cuda', R=rotation, T=translation, focal_length=((focal, focal),),
        principal_point=((width / 2, height / 2),), image_size=((height, width),), in_ndc=False)
    light = PointLights(device='cuda', location=[eye.tolist()],
                        ambient_color=((.5, .5, .5),), diffuse_color=((.5, .5, .5),),
                        specular_color=((0., 0., 0.),))
    court_verts, court_faces, court_colors = court_geometry(court_width, court_length)
    court_verts = torch.as_tensor(court_verts, dtype=torch.float32, device='cuda')
    all_faces = torch.cat((torch.as_tensor(np.asarray(faces, dtype=np.int64), device='cuda'),
                          torch.as_tensor(court_faces, dtype=torch.int64, device='cuda') + verts.shape[1]))
    all_colors = torch.cat((torch.tensor([.85, .55, .25], device='cuda').expand(verts.shape[1], 3),
                           torch.as_tensor(court_colors, dtype=torch.float32, device='cuda')))
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=camera, raster_settings=RasterizationSettings(
            image_size=(height, width), faces_per_pixel=1, blur_radius=0,
            z_clip_value=.1, cull_to_frustum=True,
            max_faces_per_bin=len(all_faces))),
        shader=SoftPhongShader(device='cuda', cameras=camera, lights=light))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a complete, decodable video; preserve an earlier output on failure.
    with tempfile.TemporaryDirectory(prefix='.court-video-', dir=output_path.parent) as temp_dir:
        temporary_path = Path(temp_dir) / 'video.mp4'
        with imageio.get_writer(str(temporary_path), fps=float(fps), codec='libx264',
                               pixelformat='yuv420p', macro_block_size=None,
                               ffmpeg_params=['-crf', '26', '-preset', 'fast']) as writer:
            with torch.no_grad():
                for i in tqdm(range(len(verts)), desc='Rendering corrected court (CUDA)'):
                    frame_verts = torch.cat((verts[i].to(device='cuda', dtype=torch.float32), court_verts))
                    mesh = Meshes(verts=[frame_verts], faces=[all_faces],
                                  textures=TexturesVertex(verts_features=[all_colors]))
                    rgb = renderer(mesh)[0, ..., :3].clamp(0, 1)
                    frame = (rgb * 255).to(torch.uint8).cpu().numpy()
                    cv2.putText(frame, 'Corrected motion | frame %d | %.2f s' % (i, i / fps),
                                (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (25, 25, 25), 1, cv2.LINE_AA)
                    writer.append_data(frame)
        video = cv2.VideoCapture(str(temporary_path))
        try:
            if (not video.isOpened() or int(video.get(cv2.CAP_PROP_FRAME_COUNT)) != len(verts)
                    or not np.isclose(video.get(cv2.CAP_PROP_FPS), fps)
                    or int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) != width
                    or int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) != height):
                raise RuntimeError('Encoded debug video has incorrect frame count, FPS or dimensions')
            video.set(cv2.CAP_PROP_POS_FRAMES, len(verts) - 1)
            if not video.read()[0]:
                raise RuntimeError('Encoded debug video final frame could not be decoded')
        finally:
            video.release()
        temporary_path.replace(output_path)
    print('Saved corrected court video to %s' % output_path)
    return {
        'path': str(output_path.resolve()), 'renderer': 'pytorch3d', 'device': 'cuda',
        'gpu': torch.cuda.get_device_name(), 'resolution': [width, height],
        'fps': float(fps), 'frames': len(verts), 'codec': 'h264',
        'camera': {'type': 'fixed_rear', 'framing': 'All corrected motion and net; court may extend outside frame',
                   'eye_m': eye.tolist(), 'target_m': target.tolist(),
                   'up': [0, 0, 1], 'vertical_fov_degrees': VERTICAL_FOV},
        'court_width_m': court_width, 'court_length_m': court_length,
        'net_height_m': 1.07, 'vertices': 'Exact corrected mesh; no additional position or height changes',
    }
