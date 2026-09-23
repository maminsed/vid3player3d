"""CUDA-only, display-free comparison of SMPL motions in court coordinates."""

from itertools import product
from pathlib import Path
import tempfile

import numpy as np


VIDEO_SIZE = (960, 540)
VERTICAL_FOV = 45.0
CAMERA_ELEVATION = 20.0
CAMERA_FRAME_FILL = .80  # Leave 10% of the image free at each edge.
CORRECTED_COLOR = (.85, .55, .25)
ORIGINAL_COLOR = (.55, .85, 1.)
GHOST_OPACITY = .45
COMPARISON_HEADER_HEIGHT = 92


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


def fixed_rear_camera(bounds, width, image_size=VIDEO_SIZE, *, court_length=23.78):
    """Fit both motions and the full court from its centerline, with extra room."""
    motion_corners = np.array(list(product(*zip(*bounds))))
    net_corners = np.array(list(product((-width / 2 - .91, width / 2 + .91),
                                        (0.,), (0., 1.1))))
    court_corners = np.array(list(product((-width / 2, width / 2),
                                          (-court_length / 2, court_length / 2), (0.,))))
    points = np.concatenate((motion_corners, net_corners))
    # Aim along the court centerline even if the player stays on one side.
    target = np.array([0., (points[:, 1].min() + points[:, 1].max()) / 2, .8])
    points = np.concatenate((points, court_corners))
    elevation = np.deg2rad(CAMERA_ELEVATION)
    backward = np.array([0., -np.cos(elevation), np.sin(elevation)])
    forward = -backward
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    relative = points - target
    tan_v = np.tan(np.deg2rad(VERTICAL_FOV / 2))
    tan_h = tan_v * image_size[0] / image_size[1]
    depth = relative @ forward
    distance = max(np.max(np.abs(relative @ right) / (CAMERA_FRAME_FILL * tan_h) - depth),
                   np.max(np.abs(relative @ up) / (CAMERA_FRAME_FILL * tan_v) - depth), 5.)
    return target + backward * distance, target


def following_rear_cameras(frame_bounds, fps, image_size=VIDEO_SIZE):
    """Follow both bodies with one smooth translation and a constant zoom.

    The camera keeps its rear heading. Fit every frame around its smoothed
    target, reserving space for the shared header, without changing the meshes.
    """
    from scipy.ndimage import gaussian_filter1d

    bounds = np.asarray(frame_bounds)
    targets = gaussian_filter1d(bounds.mean(axis=1), sigma=max(fps * .12, .5), axis=0, mode='nearest')
    corners = np.stack([bounds[:, indices, np.arange(3)] for indices in product((0, 1), repeat=3)], axis=1)
    elevation = np.deg2rad(CAMERA_ELEVATION)
    backward = np.array([0., -np.cos(elevation), np.sin(elevation)])
    forward = -backward
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    relative = corners - targets[:, None]
    tan_v = np.tan(np.deg2rad(VERTICAL_FOV / 2))
    tan_h = tan_v * image_size[0] / image_size[1]
    usable_vertical = (image_size[1] - COMPARISON_HEADER_HEIGHT) / image_size[1]
    depth = relative @ forward
    distance = max(np.max(np.abs(relative @ right) / (CAMERA_FRAME_FILL * tan_h) - depth),
                   np.max(np.abs(relative @ up) / (CAMERA_FRAME_FILL * usable_vertical * tan_v) - depth), 3.)
    return targets + backward * distance, targets


def ik_overlay_lines(frame, fps, diagnostics=None, source_start_frame=0):
    """One compact overlay for both panels; frame-zero speed is unavailable."""
    lines = ['Frame %d (source %d) | %.2f s' % (frame, source_start_frame + frame, frame / fps)]
    if diagnostics is None:
        return lines + ['IK disabled | contact predictions unavailable']
    from uhc.utils.contact_ik import MODE_NAMES
    enabled = diagnostics['summary']['enabled']
    lines[0] += ' | IK %s | root XYZ / heading change: 0' % ('ON' if enabled else 'OFF')
    def number(value, scale=1):
        return 'n/a' if not np.isfinite(value) else '%.1f' % (value * scale)
    for side, name in enumerate(('L', 'R')):
        mode = MODE_NAMES[int(diagnostics['contact_mode'][frame, side])]
        probability = diagnostics['stationary_probability'][frame, side*2:side*2+2]
        lines.append(('%s: %s | P(ankle/toe) %.2f/%.2f | blend %.2f | '
                      'move %s cm | anchor error %s cm | XY speed %s -> %s cm/s%s') % (
            name, mode, *probability, diagnostics['blend_weight'][frame, side],
            number(diagnostics['foot_change_m'][frame, side], 100),
            number(diagnostics['anchor_residual_m'][frame, side], 100),
            number(diagnostics['speed_before_mps'][frame, side], 100),
            number(diagnostics['speed_after_mps'][frame, side], 100),
            ' | LIMITED' if diagnostics['limited'][frame, side] else ''))
    return lines


def save_corrected_court_video(verts, faces, fps, output_path, court_width, court_length,
                              comparison_verts=None, ik_diagnostics=None, source_start_frame=0,
                              comparison_mode='side_by_side'):
    """Render exact saved vertices, without recentering or changing their height.

    Side-by-side: comparison vertices are pre-IK, with matching close cameras.
    Ghost: comparison vertices are original GVHMR with fixed first-frame court
    registration; overlay them on the final motion with a full-court camera.
    """
    import cv2
    import imageio.v2 as imageio
    import torch
    from pytorch3d.renderer import (
        MeshRasterizer, MeshRenderer, PerspectiveCameras, PointLights,
        RasterizationSettings, SoftPhongShader, TexturesVertex, look_at_view_transform)
    from pytorch3d.structures import Meshes
    from tqdm import tqdm

    if comparison_mode not in ('side_by_side', 'ghost'):
        raise ValueError('Expected side_by_side or ghost comparison mode')
    side_by_side = comparison_mode == 'side_by_side'
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for debug video; CPU rendering is disabled.')
    verts = torch.as_tensor(verts).detach()
    if verts.ndim != 3 or verts.shape[-1] != 3 or not len(verts):
        raise ValueError('Expected vertices with shape (frames, vertices, 3)')
    if not torch.isfinite(verts).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Video vertices and positive FPS must be finite')
    if comparison_verts is not None:
        comparison_verts = torch.as_tensor(comparison_verts).detach()
        if comparison_verts.shape != verts.shape or not torch.isfinite(comparison_verts).all():
            raise ValueError('Comparison vertices must be finite and match corrected vertices shape')
    width, height = VIDEO_SIZE
    output_width = width * (2 if side_by_side and comparison_verts is not None else 1)
    frame_bounds = torch.stack((verts.amin(dim=1), verts.amax(dim=1)), dim=1).cpu().numpy()
    if comparison_verts is not None:
        original_bounds = torch.stack((comparison_verts.amin(dim=1),
                                       comparison_verts.amax(dim=1)), dim=1).cpu().numpy()
        frame_bounds[:, 0] = np.minimum(frame_bounds[:, 0], original_bounds[:, 0])
        frame_bounds[:, 1] = np.maximum(frame_bounds[:, 1], original_bounds[:, 1])
    if side_by_side:
        eyes, targets = following_rear_cameras(frame_bounds, fps)
    else:
        bounds = np.stack((frame_bounds[:, 0].min(axis=0), frame_bounds[:, 1].max(axis=0)))
        eye, target = fixed_rear_camera(bounds, court_width, court_length=court_length)
        eyes, targets = np.tile(eye, (len(verts), 1)), np.tile(target, (len(verts), 1))
    eye, target = eyes[0], targets[0]
    rotation, translation = look_at_view_transform(
        eye=[eye.tolist()], at=[target.tolist()], up=((0, 0, 1),), device='cuda')
    focal = height / (2 * np.tan(np.deg2rad(VERTICAL_FOV / 2)))
    camera = PerspectiveCameras(
        device='cuda', R=rotation, T=translation, focal_length=((focal, focal),),
        principal_point=((width / 2, (height + (COMPARISON_HEADER_HEIGHT if side_by_side else 0)) / 2),),
        image_size=((height, width),), in_ndc=False)
    light = PointLights(device='cuda', location=[eye.tolist()],
                        ambient_color=((.5, .5, .5),), diffuse_color=((.5, .5, .5),),
                        specular_color=((0., 0., 0.),))
    court_verts, court_faces, court_colors = court_geometry(court_width, court_length)
    court_verts = torch.as_tensor(court_verts, dtype=torch.float32, device='cuda')
    all_faces = torch.cat((torch.as_tensor(np.asarray(faces, dtype=np.int64), device='cuda'),
                          torch.as_tensor(court_faces, dtype=torch.int64, device='cuda') + verts.shape[1]))
    all_colors = torch.cat((torch.tensor(CORRECTED_COLOR, device='cuda').expand(verts.shape[1], 3),
                           torch.as_tensor(court_colors, dtype=torch.float32, device='cuda')))
    original_colors = all_colors.clone()
    original_colors[:verts.shape[1]] = torch.tensor(ORIGINAL_COLOR, device='cuda')
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
                for i in tqdm(range(len(verts)), desc='Rendering court comparison (CUDA)'):
                    if side_by_side:
                        camera.R, camera.T = look_at_view_transform(
                            eye=[eyes[i].tolist()], at=[targets[i].tolist()], up=((0, 0, 1),), device='cuda')
                        light.location = torch.as_tensor(eyes[i:i+1], dtype=torch.float32, device='cuda')
                    frame_verts = torch.cat((verts[i].to(device='cuda', dtype=torch.float32), court_verts))
                    mesh = Meshes(verts=[frame_verts], faces=[all_faces],
                                  textures=TexturesVertex(verts_features=[all_colors]))
                    fragments = renderer.rasterizer(mesh)
                    rgb = renderer.shader(fragments, mesh)[0, ..., :3].clamp(0, 1)
                    if comparison_verts is not None:
                        original_mesh = Meshes(
                            verts=[torch.cat((comparison_verts[i].to(device='cuda', dtype=torch.float32),
                                              court_verts))], faces=[all_faces],
                            textures=TexturesVertex(verts_features=[original_colors]))
                        original_fragments = renderer.rasterizer(original_mesh)
                        original_rgb = renderer.shader(original_fragments, original_mesh)[0, ..., :3].clamp(0, 1)
                        if not side_by_side:
                            # Blend only visible ghost-body pixels. Preserve
                            # opaque final-body and court/net depth occlusion.
                            ghost_face = original_fragments.pix_to_face[0, ..., 0]
                            visible = (ghost_face >= 0) & (ghost_face < len(faces))
                            visible &= ((fragments.pix_to_face[0, ..., 0] < 0)
                                        | (original_fragments.zbuf[0, ..., 0] <= fragments.zbuf[0, ..., 0]))
                            alpha = visible[..., None] * GHOST_OPACITY
                            rgb = rgb * (1 - alpha) + original_rgb * alpha
                    panels = [(rgb, 'POST IK - exported motion', CORRECTED_COLOR)]
                    if side_by_side and comparison_verts is not None:
                        panels.insert(0, (original_rgb, 'PRE IK - corrected court trajectory', ORIGINAL_COLOR))
                    frame = np.concatenate([(panel * 255).to(torch.uint8).cpu().numpy()
                                            for panel, _, _ in panels], axis=1).copy()
                    if side_by_side:
                        cv2.rectangle(frame, (0, 0), (output_width, COMPARISON_HEADER_HEIGHT), (25, 25, 25), -1)
                        for row, line in enumerate(ik_overlay_lines(i, fps, ik_diagnostics, source_start_frame)):
                            cv2.putText(frame, line, (12, 17 + row*18), cv2.FONT_HERSHEY_SIMPLEX,
                                        .4, (240, 240, 240), 1, cv2.LINE_AA)
                        for p, (_, label, color) in enumerate(panels):
                            cv2.putText(frame, label, (12 + p*width, 81), cv2.FONT_HERSHEY_SIMPLEX,
                                        .43, tuple(int(c*255) for c in color), 1, cv2.LINE_AA)
                    else:
                        cv2.rectangle(frame, (8, 8), (610, 64), (25, 25, 25), -1)
                        cv2.putText(frame, 'Frame %d (source %d) | %.2f s' % (i, source_start_frame+i, i/fps),
                                    (16, 24), cv2.FONT_HERSHEY_SIMPLEX, .4, (240, 240, 240), 1, cv2.LINE_AA)
                        cv2.putText(frame, 'Final result', (16, 46), cv2.FONT_HERSHEY_SIMPLEX,
                                    .43, tuple(int(c*255) for c in CORRECTED_COLOR), 1, cv2.LINE_AA)
                        if comparison_verts is not None:
                            cv2.putText(frame, 'Original GVHMR (aligned ghost)', (130, 46), cv2.FONT_HERSHEY_SIMPLEX,
                                        .43, tuple(int(c*255) for c in ORIGINAL_COLOR), 1, cv2.LINE_AA)
                    writer.append_data(frame)
        video = cv2.VideoCapture(str(temporary_path))
        try:
            if (not video.isOpened() or int(video.get(cv2.CAP_PROP_FRAME_COUNT)) != len(verts)
                    or not np.isclose(video.get(cv2.CAP_PROP_FPS), fps)
                    or int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) != output_width
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
        'gpu': torch.cuda.get_device_name(), 'resolution': [output_width, height],
        'fps': float(fps), 'frames': len(verts), 'codec': 'h264',
        'camera': {'type': 'following_rear' if side_by_side else 'fixed_rear',
                   'framing': ('Smooth shared body tracking, constant zoom, reserved header' if side_by_side else
                               'Both motions, full court and net with 10% edge margins'),
                   'eye_m': eye.tolist(), 'target_m': target.tolist(),
                   'positions_note': 'Initial frame only; camera follows smoothed mesh bounds' if side_by_side else 'Fixed for clip',
                   'up': [0, 0, 1], 'vertical_fov_degrees': VERTICAL_FOV},
        'court_width_m': court_width, 'court_length_m': court_length,
        'net_height_m': 1.07, 'vertices': 'Exact corrected mesh; no additional position or height changes',
        'corrected_color_rgb': list(CORRECTED_COLOR),
        'comparison': {
            'enabled': comparison_verts is not None, 'color_rgb': list(ORIGINAL_COLOR),
            'mode': comparison_mode,
            'layout': ('Pre IK left, post IK right; identical cameras and root trajectories' if side_by_side else
                       'Original GVHMR ghost over final corrected motion'),
            'alignment': 'Same corrected root' if side_by_side else 'Fixed first-frame yaw/XY registration, original Z retained',
            'opacity': 1. if side_by_side else GHOST_OPACITY,
        },
        'overlay': ('Shared compact header: predicted support mode; ankle/toe probability; blend; '
                    'max ankle/toe displacement; selected anchor residual; selected joint XY speed '
                    '(backward difference); solver limits' if side_by_side else
                    'Frame/source frame/time and final/original ghost legend'),
        'source_start_frame': int(source_start_frame),
    }
