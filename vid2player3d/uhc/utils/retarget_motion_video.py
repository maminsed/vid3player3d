"""Small headless SMPL comparison videos, with CPU rasterization and batched skinning."""

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation


class ComparisonVideo:
    """One 640x480 MP4; concatenate sequences in library order at 30 fps.

    Replay the exact reference poses, without a low-level policy or physics.
    Skinning uses the converter's selected device. OpenCV paints depth-sorted
    triangles on CPU, avoiding display/GL requirements and a GPU-only renderer.
    """

    size = (640, 480)
    fps = 30.
    original_color = np.array([235, 145, 65])  # BGR blue
    result_color = np.array([35, 140, 245])    # BGR orange

    def __init__(self, path, device, batch_size, models=None):
        self.path = Path(path)
        self.device, self.batch_size = device, batch_size
        self.models = models if models is not None else {}
        self.palettes = {}
        self.frame_count = 0

    def __enter__(self):
        import imageio.v2 as imageio
        self.writer = imageio.get_writer(str(self.path), fps=self.fps, codec='libx264',
                                        quality=4, macro_block_size=2, pixelformat='yuv420p')
        return self

    def __exit__(self, *exception):
        self.writer.close()

    def _model(self, gender):
        from vid2player3d.uhc.smpllib.smpl_parser import SMPL_Parser
        from vid2player3d.uhc.utils.retarget_motion_lib import PROJECT
        if gender not in self.models:
            self.models[gender] = SMPL_Parser(str(PROJECT / 'data/smpl'), gender=gender).to(self.device)
        return self.models[gender]

    def _vertices(self, model, pose, beta, root, scale=1.):
        import torch
        with torch.no_grad():
            vertices, joints = model.get_joints_verts(
                torch.as_tensor(pose, dtype=torch.float32, device=self.device),
                torch.as_tensor(beta[None], dtype=torch.float32, device=self.device))
            vertices = (vertices - joints[:, :1]) * scale
            vertices += torch.as_tensor(root[:, None], dtype=torch.float32, device=self.device)
        return vertices.cpu().numpy()

    def _mesh(self, image, vertices, faces, project, color):
        import cv2
        screen = np.rint(project(vertices)).astype(np.int32)
        triangles = vertices[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        # The closed SMPL surface has consistently outward-facing triangles.
        # Back faces cannot be visible with this orthographic camera.
        visible = np.einsum('ij,j->i', normals, self.forward) < 0
        faces, normals = faces[visible], normals[visible]
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        light = np.array([.3, -.5, .8])
        light /= np.linalg.norm(light)
        # Einsum avoids starting a BLAS thread pool for these small dot products.
        shades = np.rint((.55 + .45 * abs(np.einsum('ij,j->i', normals, light))) * 255).astype(np.uint8)
        depths = np.einsum('ij,j->i', vertices, self.forward)[faces].sum(1)
        order = np.argsort(depths)[::-1]
        key = tuple(color)
        if key not in self.palettes:
            self.palettes[key] = [tuple(int(c) for c in color * (shade / 255.)) for shade in range(256)]
        palette = self.palettes[key]
        # Materialize indexed polygons once instead of doing NumPy indexing and
        # creating three Python color integers for every triangle on every frame.
        for polygon, shade in zip(screen[faces[order]], shades[order]):
            cv2.fillConvexPoly(image, polygon, palette[shade])

    def add_sequence(self, source, index, target, diagnostics, settings):
        import cv2
        from vid2player3d.uhc.smpllib.smpl_parser import SMPL_BONE_ORDER_NAMES
        from vid2player3d.uhc.utils.retarget_motion_lib import BODY_NAMES
        from vid2player3d.poselib.poselib.skeleton.skeleton3d import SkeletonMotion

        start = int(source.length_starts[index])
        n, fps = int(source._motion_num_frames[index]), float(source._motion_fps[index])
        body = source._motion_bodies[index].cpu().numpy()
        gender = {0: 'neutral', 1: 'male', 2: 'female'}.get(int(body[0]))
        if gender is None:
            raise ValueError('Unsupported source SMPL gender code for video')
        original, result = self._model(gender), self._model(settings['gender'])
        to_smpl = [BODY_NAMES.index(name) for name in SMPL_BONE_ORDER_NAMES]
        source_pose = Rotation.from_quat(source.lrs[start:start+n, to_smpl].numpy().reshape(-1, 4))
        source_pose = source_pose.as_rotvec().reshape(n, 72)
        source_root = source.gts[start:start+n, 0].numpy()
        target_root = target['root_trans']
        target_positions = SkeletonMotion.from_dict(target).global_translation.numpy()
        centers = (source_root + target_root) / 2
        centers[:, 2] += .15
        centers = gaussian_filter1d(centers, sigma=max(.15 * fps, .5), axis=0, mode='nearest')
        self.forward = np.array([-3., 6., -2.6])
        self.forward /= np.linalg.norm(self.forward)
        right = np.cross(self.forward, [0., 0., 1.])
        right /= np.linalg.norm(right)
        up = np.cross(right, self.forward)
        basis = np.stack((right, up))
        extent = np.concatenate([source.gts[start:start+n].numpy() - centers[:, None],
                                 target_positions - centers[:, None]], axis=1) @ basis.T
        width, height = self.size
        scale = min((width - 70) / (2 * (abs(extent[..., 0]).max() + .4)),
                    (height - 100) / (2 * (abs(extent[..., 1]).max() + .4)))
        output_indices = np.minimum(np.floor(np.arange(max(1, round(n * self.fps / fps))) * fps / self.fps).astype(int), n-1)
        first_frame = self.frame_count
        for begin in range(0, n, self.batch_size):
            end = min(n, begin + self.batch_size)
            src = self._vertices(original, source_pose[begin:end], body[1:11], source_root[begin:end],
                                 float(source._motion_body_scales[index]))
            dst = self._vertices(result, target['pose_aa'][begin:end], np.array(settings['beta']),
                                 target_root[begin:end], settings['body_scale'])
            for frame in output_indices[(output_indices >= begin) & (output_indices < end)]:
                def project(points):
                    xy = (np.asarray(points) - centers[frame]) @ basis.T * scale
                    return xy * [1., -1.] + [width / 2, height / 2 + 20]
                image = np.full((height, width, 3), (32, 29, 25), dtype=np.uint8)
                center = centers[frame]
                for x in range(int(center[0]) - 5, int(center[0]) + 6):
                    line = np.rint(project([[x, center[1]-5, 0], [x, center[1]+5, 0]])).astype(int)
                    cv2.line(image, tuple(line[0]), tuple(line[1]), (60, 56, 50), 1)
                for y in range(int(center[1]) - 5, int(center[1]) + 6):
                    line = np.rint(project([[center[0]-5, y, 0], [center[0]+5, y, 0]])).astype(int)
                    cv2.line(image, tuple(line[0]), tuple(line[1]), (60, 56, 50), 1)
                ghost = image.copy()
                self._mesh(ghost, src[frame-begin], original.faces, project, self.original_color)
                image = cv2.addWeighted(ghost, .35, image, .65, 0)
                self._mesh(image, dst[frame-begin], result.faces, project, self.result_color)
                # The target's fixed wrist racket is part of the comparison.
                rotation = Rotation.from_quat(diagnostics['racket_rotation_xyzw'][frame])
                racket = diagnostics['racket_center_m'][frame]
                along = rotation.apply([-1., 0., 0.])
                across = np.cross(diagnostics['racket_normal'][frame], along)
                theta = np.linspace(0, 2*np.pi, 40)
                rim = racket + .15 * (np.cos(theta)[:, None] * along + np.sin(theta)[:, None] * across)
                cv2.polylines(image, [np.rint(project(rim)).astype(np.int32)], True, (35, 175, 255), 2)
                handle = np.rint(project([target_positions[frame, 22], racket + .15 * -along])).astype(int)
                cv2.line(image, tuple(handle[0]), tuple(handle[1]), (35, 175, 255), 2)
                cv2.rectangle(image, (0, 0), (width, 57), (20, 18, 16), -1)
                cv2.putText(image, f'Frame {frame} / {n-1}   {frame/fps:.2f} s', (12, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, .58, (240, 240, 240), 1, cv2.LINE_AA)
                cv2.putText(image, 'Original (ghost)', (12, 46), cv2.FONT_HERSHEY_SIMPLEX, .48,
                            tuple(int(c) for c in self.original_color), 1, cv2.LINE_AA)
                cv2.putText(image, 'Retargeted', (165, 46), cv2.FONT_HERSHEY_SIMPLEX, .48,
                            tuple(int(c) for c in self.result_color), 1, cv2.LINE_AA)
                cv2.putText(image, target['seq_name'][:85], (10, height - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, .35, (200, 200, 200), 1, cv2.LINE_AA)
                self.writer.append_data(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                self.frame_count += 1
            print(f'Video: {target["seq_name"]} {end}/{n} frames', flush=True)
        return {'file': self.path.name, 'first_video_frame': first_frame,
                'video_frame_count': self.frame_count - first_frame, 'fps': self.fps,
                'resolution': list(self.size), 'playback': 'kinematic references; no policy or physics',
                'source_color': 'faded blue, opacity 0.35', 'target_color': 'orange',
                'frame_number': 'zero-based frame index within the source sequence',
                'skinning_device': str(self.device), 'rasterizer': 'CPU depth-sorted mesh triangles'}
