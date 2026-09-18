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
                       np.array([[-9, -20, -1], [10, 17, 5]])):
            eye, target = video.fixed_rear_camera(bounds, 10.97)
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
            pixels = camera.transform_points_screen(torch.tensor(points, dtype=torch.float32))
            self.assertTrue((pixels[:, :2] > 0).all())
            self.assertTrue((pixels[:, 0] < width).all())
            self.assertTrue((pixels[:, 1] < height).all())
            self.assertTrue((pixels[:, 2] > 0).all())

    def test_missing_cuda_fails_instead_of_using_cpu(self):
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'no CPU fallback'):
                video.require_cuda_renderer()


if __name__ == '__main__':
    unittest.main()
