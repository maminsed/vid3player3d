"""Camera framing and explicit CUDA-only behavior; full video checked on the GPU."""

from itertools import product
import unittest
from unittest.mock import patch

import numpy as np

from uhc.utils import debug_court_video as video


class DebugCourtVideoTests(unittest.TestCase):
    def test_fixed_camera_keeps_extreme_motion_and_net_in_frame(self):
        from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform
        import torch

        for bounds in (np.array([[-2, -13, -.2], [2, -7, 3]]),
                       np.array([[-9, -20, -1], [10, 17, 5]]),
                       np.array([[7, -15, 0], [12, -10, 3]])):
            eye, target = video.fixed_rear_camera(bounds, 10.97)
            self.assertEqual(eye[0], 0)
            self.assertEqual(target[0], 0)
            self.assertLess(eye[1], bounds[0, 1])
            self.assertGreater(eye[2], 0)
            rotation, translation = look_at_view_transform(
                eye=[eye.tolist()], at=[target.tolist()], up=((0, 0, 1),))
            width, height = video.VIDEO_SIZE
            focal = height / (2 * np.tan(np.deg2rad(video.VERTICAL_FOV / 2)))
            camera = PerspectiveCameras(R=rotation, T=translation, in_ndc=False,
                                       focal_length=((focal, focal),),
                                       principal_point=((width / 2, height / 2),),
                                       image_size=((height, width),))
            points = list(product(*zip(*bounds))) + list(product((-6.395, 6.395), (0,), (0, 1.1)))
            points += list(product((-5.485, 5.485), (-11.89, 11.89), (0,)))
            pixels = camera.transform_points_screen(torch.tensor(points, dtype=torch.float32))
            margin = (1 - video.CAMERA_FRAME_FILL) / 2
            self.assertTrue((pixels[:, 0] >= margin * width - .01).all())
            self.assertTrue((pixels[:, 1] >= margin * height - .01).all())
            self.assertTrue((pixels[:, 0] <= (1 - margin) * width + .01).all())
            self.assertTrue((pixels[:, 1] <= (1 - margin) * height + .01).all())
            self.assertTrue((pixels[:, 2] > 0).all())

    def test_missing_cuda_fails_instead_of_using_cpu(self):
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'no CPU fallback'):
                video.require_cuda_renderer()

    def test_following_camera_preserves_framing_during_translation_and_pose_changes(self):
        from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform
        import torch

        centers = np.column_stack((np.linspace(-4, 4, 40), np.linspace(-13, -7, 40), np.ones(40)))
        half_size = np.tile([.5, .3, 1.], (40, 1))
        half_size[15:20] = [1.2, .5, 1.4]  # Extended pose/jump.
        bounds = np.stack((centers-half_size, centers+half_size), axis=1)
        original = bounds.copy()
        eyes, targets = video.following_rear_cameras(bounds, 30)
        np.testing.assert_array_equal(bounds, original)
        offsets = eyes-targets
        np.testing.assert_allclose(offsets, np.broadcast_to(offsets[0], offsets.shape))
        self.assertGreater(targets[-1, 0]-targets[0, 0], 6.)
        width, height = video.VIDEO_SIZE
        focal = height / (2*np.tan(np.deg2rad(video.VERTICAL_FOV/2)))
        for frame in (0, 15, 19, 39):
            rotation, translation = look_at_view_transform(
                eye=[eyes[frame].tolist()], at=[targets[frame].tolist()], up=((0, 0, 1),))
            camera = PerspectiveCameras(R=rotation, T=translation, in_ndc=False,
                                       focal_length=((focal, focal),),
                                       principal_point=((width/2, (height+video.COMPARISON_HEADER_HEIGHT)/2),),
                                       image_size=((height, width),))
            corners = torch.tensor(list(product(*zip(*bounds[frame]))), dtype=torch.float32)
            pixels = camera.transform_points_screen(corners)
            self.assertTrue((pixels[:, 0] > 0).all() and (pixels[:, 0] < width).all())
            self.assertTrue((pixels[:, 1] > video.COMPARISON_HEADER_HEIGHT).all())
            self.assertTrue((pixels[:, 1] < height).all())
            self.assertTrue((pixels[:, 2] > 0).all())


if __name__ == '__main__':
    unittest.main()
