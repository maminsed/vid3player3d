from vid2player.motion_vae.base import MotionVAEModel
from vid2player.motion_vae.dataset import Video3DPoseDataset, encode_action
from vid2player.utils.common import *
from vid2player.utils.racket import infer_racket_from_smpl

from smpl_visualizer.smpl_visualizer.vis_sport import SportVisualizer
from smpl_visualizer.smpl_visualizer.vis_scenepic import SportVisualizerHTML
from smpl_visualizer.smpl_visualizer.vis import vstack_videos
from smpl_visualizer.smpl_visualizer.smpl import SMPL, SMPL_MODEL_DIR

import torch
import copy
import os
import imageio
from tqdm import tqdm


SMPL_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int64,
)


def project_points(points, camera=(0, -25, 3), look_at=(0, 0, 0), up=(0, 0, 1)):
    points = np.asarray(points, dtype=np.float32)
    camera = np.asarray(camera, dtype=np.float32)
    look_at = np.asarray(look_at, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    forward = look_at - camera
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    true_up = np.cross(right, forward)
    rel = points - camera
    return np.stack([rel @ right, rel @ true_up], axis=-1)


def world_to_image(projected, bounds, width, height, margin=40):
    projected = np.asarray(projected, dtype=np.float32)
    min_xy, max_xy = bounds
    span = np.maximum(max_xy - min_xy, 1e-6)
    scale = min((width - 2 * margin) / span[0], (height - 2 * margin) / span[1])
    offset = np.array([
        margin - min_xy[0] * scale + (width - 2 * margin - span[0] * scale) * 0.5,
        margin - min_xy[1] * scale + (height - 2 * margin - span[1] * scale) * 0.5,
    ], dtype=np.float32)
    xy = projected * scale + offset
    xy[..., 1] = height - xy[..., 1]
    return xy


def draw_polyline(img, points, color, closed=False, thickness=2):
    import cv2
    points = np.asarray(points, dtype=np.int32)
    if len(points) < 2:
        return
    cv2.polylines(img, [points], closed, color, thickness, lineType=cv2.LINE_AA)


def save_motion_as_cv_video(video_path, joint_pos_all, trans_all, racket_all=None, fps=30,
    width=1280, height=720):
    """
    CPU-only debug renderer. It writes a projected skeleton/court MP4 without
    PyVista, OpenGL, Xvfb, or a browser.
    """
    import cv2

    joint_world = (joint_pos_all + trans_all[:, :, None, :]).detach().cpu().numpy()
    num_runner, nframes = joint_world.shape[:2]

    court_lines = []
    for x, length in zip([-10.97/2, -8.23/2, 0, 8.23/2, 10.97/2], [23.77, 23.77, 12.8, 23.77, 23.77]):
        court_lines.append(np.array([[x, -length/2, 0], [x, length/2, 0]], dtype=np.float32))
    for y, width_line in zip([-11.89, -6.4, 0, 6.4, 11.89], [10.97, 8.23, 10.97, 8.23, 10.97]):
        court_lines.append(np.array([[-width_line/2, y, 0], [width_line/2, y, 0]], dtype=np.float32))
    court_lines.append(np.array([[-6.4, 0, 1.07], [6.4, 0, 1.07]], dtype=np.float32))

    bounds_points = [project_points(joint_world.reshape(-1, 3))]
    bounds_points += [project_points(line) for line in court_lines]
    bounds_points = np.concatenate(bounds_points, axis=0)
    min_xy = bounds_points.min(axis=0)
    max_xy = bounds_points.max(axis=0)
    pad = np.maximum((max_xy - min_xy) * 0.08, 0.5)
    bounds = (min_xy - pad, max_xy + pad)

    colors = [
        (236, 101, 80),
        (86, 153, 214),
        (104, 182, 112),
        (213, 154, 54),
        (167, 111, 194),
        (92, 188, 178),
    ]

    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    writer = imageio.get_writer(video_path, fps=fps, quality=8, macro_block_size=None)
    try:
        for fr in tqdm(range(nframes)):
            img = np.full((height, width, 3), 245, dtype=np.uint8)

            court_fill = np.array([
                [-10.97/2, -11.89, 0],
                [10.97/2, -11.89, 0],
                [10.97/2, 11.89, 0],
                [-10.97/2, 11.89, 0],
            ], dtype=np.float32)
            court_xy = world_to_image(project_points(court_fill), bounds, width, height).astype(np.int32)
            cv2.fillConvexPoly(img, court_xy, (157, 96, 74), lineType=cv2.LINE_AA)

            for line in court_lines:
                line_xy = world_to_image(project_points(line), bounds, width, height)
                draw_polyline(img, line_xy, (255, 255, 255), thickness=2)

            for r in range(num_runner):
                color = colors[r % len(colors)]
                pts = world_to_image(project_points(joint_world[r, fr]), bounds, width, height)
                for child, parent in enumerate(SMPL_PARENTS):
                    if parent < 0:
                        continue
                    draw_polyline(img, [pts[parent], pts[child]], color, thickness=3)
                for p in pts:
                    cv2.circle(img, tuple(np.round(p).astype(int)), 4, color, -1, lineType=cv2.LINE_AA)

                if racket_all is not None and racket_all[r][fr] is not None:
                    racket = racket_all[r][fr]
                    root = racket.get('root', np.zeros(3, dtype=np.float32))
                    for key_a, key_b in [
                        ('handle_center', 'shaft_left_center'),
                        ('handle_center', 'shaft_right_center'),
                        ('shaft_left_center', 'head_center'),
                        ('shaft_right_center', 'head_center'),
                    ]:
                        if key_a in racket and key_b in racket:
                            line = np.stack([racket[key_a] + root, racket[key_b] + root])
                            line_xy = world_to_image(project_points(line), bounds, width, height)
                            draw_polyline(img, line_xy, (35, 35, 35), thickness=2)

            cv2.putText(img, f'frame {fr + 1}/{nframes}', (24, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)
            writer.append_data(img)
    finally:
        writer.close()
    print(f'Animation saved to {video_path}')


class BaseRunner(object):

    def __init__(self):
        self.root_cur = None
        self.joint_pos_cur = None
        self.root_history = []
        self.joint_pos_history = []


    def step(self):
        pass


class MotionVAERunner(BaseRunner):

    def __init__(self, opt):
        super().__init__()

        opt = copy.deepcopy(opt)
        opt.test_only = True
        self.motion_vae = MotionVAEModel(opt)

        self.base_action = torch.zeros(opt.latent_size).float()
        self.base_action.normal_(0, 1)
        self.latent = None
        self.action = None
        self.phase = torch.FloatTensor([0, 1])
        self.phase_rad = 0

    def init_state(self, dataset):
        opt = self.motion_vae.opt
        first_frame = dataset.sample_first_frame()
        self.root_cur = torch.from_numpy(first_frame['root_pos']).float()
        if opt.update_joint_pos:
            self.joint_pos_cur = torch.from_numpy(first_frame['joint_pos']).float()
        else:
            self.joint_rot_cur = torch.from_numpy(first_frame['joint_rot']).float()
        self.condition = torch.from_numpy(first_frame['condition']).float() # T x F
        
        # initialize root
        self.root_cur[:2] = torch.FloatTensor([0, -12])

        self.root_history = self.root_cur.unsqueeze(0).clone()
        if opt.update_joint_pos:
            self.joint_pos_history = self.joint_pos_cur.view(23, 3).unsqueeze(0).clone()
        else:
            self.joint_rot_history = self.joint_rot_cur.view(24, 3).unsqueeze(0).clone()


    def set_latent_random(self):
        latent = torch.zeros_like(self.base_action)
        latent.normal_(0, 1)
        self.latent = latent
    

    def set_action(self):
        action = torch.LongTensor([6])
        self.action = encode_action(action, self.motion_vae.opt.action_dim)


    def step(self):
        next_frame = self.motion_vae.infer_single(self.latent, self.condition, self.action)
        self.update_state(next_frame)
    

    def update_state(self, frame):
        opt = self.motion_vae.opt
        self.root_cur = self.root_cur + frame['root_velo']
        # bring player back to court
        court_bbox = torch.FloatTensor([-5, -15, 5, 0])
        if not test_point_in_bbox(self.root_cur[:2], court_bbox):
            self.root_cur[:2] = torch.FloatTensor([0, -13])

        self.root_history = torch.cat((self.root_history, self.root_cur.unsqueeze(0)))
        if opt.update_joint_pos:
            self.joint_pos_cur = frame['joint_pos']
            self.joint_pos_history = torch.cat((self.joint_pos_history, self.joint_pos_cur.view(23, 3).unsqueeze(0)))
        else:
            self.joint_rot_cur = frame['joint_rot']
            self.joint_rot_history = torch.cat((self.joint_rot_history, self.joint_rot_cur.view(24, 3).unsqueeze(0)))
        self.condition = self.condition.roll(-1, dims=0)
        self.condition[-1].copy_(frame['feature'])
        if 'root_pos' in opt.pose_feature:
            self.condition[-1, :3].copy_(self.root_cur)
        if opt.predict_phase:
            self.phase = frame['phase']
            self.phase_rad = frame['phase_rad']
        

def test_motion_vae_randomwalk(opt, num_test=5, num_runner=5, result_dir_suffix='',
    same_init_state=True, nframes=1000, interactive=False, record_html=False,
    record_cv=False,
    ):
    """
    random walk for motion vae model
    """
    result_dir = os.path.join(opt.result_dir, opt.model_ver + result_dir_suffix)
    print("Save video results to {}".format(result_dir))
    
    if record_html:
        visualizer = SportVisualizerHTML(gender='male', show_ball=False)
    elif record_cv:
        visualizer = None
    else:
        visualizer = SportVisualizer(
            verbose=False, 
            show_smpl=not opt.update_joint_pos,
            show_skeleton=False,
            show_racket=opt.infer_racket,
            correct_root_height=True,
            gender='male',
        )
    opt.batch_size = 1e9 # HACK for random sampling
    dataset = Video3DPoseDataset(opt)

    if opt.infer_racket or record_cv:
        smpl = SMPL(SMPL_MODEL_DIR, create_transl=False, gender='male')

    # render a video for each clip, start with the initial frame of the clip
    for tid in range(num_test):
        tid += 1
        result_sub_dir = os.path.join(result_dir, '{:03}'.format(tid))
        os.makedirs(result_sub_dir, exist_ok=True)
        print("Running test", tid)

        runner_dict = {}
        for r in range(num_runner):
            set_seed(tid if same_init_state else tid + r)
            runner_dict[r] = MotionVAERunner(opt)
            runner_dict[r].init_state(dataset)

        for idx in tqdm(range(nframes - 1)):
            for r in range(num_runner):
                runner_dict[r].set_latent_random()
                runner_dict[r].step()
        
        # render video
        joint_pos_all = torch.zeros((num_runner, nframes, 24, 3))
        joint_rot_all = torch.zeros((num_runner, nframes, 24, 3))
        trans_all = torch.empty((num_runner, nframes, 3))
        for r in range(num_runner):
            if opt.update_joint_pos:
                joint_pos_all[r, :, 1:, :] = runner_dict[r].joint_pos_history # N x 23 x 3
            else:
                joint_rot_all[r, :, :, :] = runner_dict[r].joint_rot_history # N x 24 x 3
            trans_all[r, ...] = runner_dict[r].root_history # N x 3
        joint_rot_all[..., -1, :] = torch.FloatTensor([0, 0, np.pi/2])
        if opt.infer_racket or record_cv:
            smpl_motion = smpl(
                global_orient=joint_rot_all[:, :, 0].reshape(-1, 3),
                body_pose=joint_rot_all[:, :, 1:].reshape(-1, 69),
                betas=torch.zeros(num_runner*nframes, 10).float(),
                root_trans = trans_all.reshape(-1, 3),
                return_full_pose=True,
                orig_joints=True
            )
            joint_pos_all = smpl_motion.joints.reshape(num_runner, nframes, 24, 3) - \
                trans_all.reshape(num_runner, nframes, 1, 3)
        if opt.infer_racket:
            racket_all = []
            for r in range(num_runner):
                racket_all.append([])
                for i in range(nframes):
                    racket_all[r] += [infer_racket_from_smpl(
                        joint_pos_all[r][i].numpy(), joint_rot_all[r][i].numpy(), 
                        sport=opt.sport, righthand=opt.player_name!=['Nadal'])]
        
        smpl_seq = {
            'trans': trans_all,
            'orient': None,
            'betas': torch.zeros((num_runner, 10)),
        }
        if opt.update_joint_pos:
            smpl_seq['joint_pos'] = joint_pos_all
        else:
            smpl_seq['joint_rot'] = joint_rot_all.view(num_runner, nframes, 24*3)
        
        init_args = {
            'smpl_seq': smpl_seq, 
            'num_actors': num_runner, 
            'sport': opt.sport,
            'camera': 'front',
            'racket_seq': racket_all if opt.infer_racket else None,
        }
        if record_cv:
            vid_path = os.path.join(result_sub_dir, 'random_front_cv.mp4')
            save_motion_as_cv_video(
                vid_path,
                joint_pos_all,
                trans_all,
                racket_all if opt.infer_racket else None,
                fps=30,
            )
        elif record_html:
            html_path = os.path.join(result_sub_dir, 'random_front.html')
            visualizer.save_animation_as_html(
                init_args=init_args,
                html_path=html_path,
            )
        elif interactive:
            visualizer.show_animation(
                init_args=init_args, 
                fps=30, 
                window_size=(1000, 1000), 
                enable_shadow=True
            )
        else:
            vid_path = os.path.join(result_sub_dir, 'random_front.mp4')
            visualizer.save_animation_as_video(
                vid_path,
                init_args=init_args, 
                fps=30, 
                window_size=(1000, 1000), 
                enable_shadow=True,
                cleanup=True
            )
