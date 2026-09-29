"""CPU checks for reconstruction alignment, resets, metrics, and headless recording."""
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import imageio
import numpy as np
import torch

from vid2player.motion_vae.config import MotionVAEOption
from vid2player.motion_vae.reconstruction import (
    CourtCamera, reconstruct, render_frame, save_report, select_sequence,
)


class FakeVAE:
    def __init__(self):
        self.conditions = []

    def encode(self, truth, condition):
        self.conditions.append(condition.clone())
        return torch.ones(1, 2), torch.zeros(1, 2)

    def sample(self, latent, condition):
        output = condition[:, -288:].clone() + 1
        output[:, 3:6] = torch.tensor([1., 0., 0.])
        return output


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        opt = MotionVAEOption()
        opt.load('usopen')
        self.model = SimpleNamespace(opt=opt, device=torch.device('cpu'), model=FakeVAE(),
                                     normalize=lambda x: x / 2, unnormalize=lambda x: x * 2)
        self.features = np.zeros((6, 288), dtype=np.float32)
        self.roots = np.zeros((6, 3), dtype=np.float32)
        self.roots[:, 0] = np.arange(6) * 10
        self.features[:, :3] = self.roots
        self.frames = np.arange(1, 7)

    def run_reconstruction(self, resample=None):
        return reconstruct(self.model, self.features, self.roots, self.frames, 30, resample)

    def test_no_reset_integrates_root_and_feeds_back_normalized_prediction(self):
        result = self.run_reconstruction()
        np.testing.assert_allclose(result['roots'][1, :, 0], [0, 2, 4, 6, 8, 10])
        self.assertIsNone(result['rows'][0]['recon'])
        self.assertEqual(sum(r['conditioned_from_gt'] for r in result['rows']), 1)
        self.assertEqual(result['rows'][1]['kl'], .5)
        self.assertEqual(result['rows'][1]['kl_weighted'], .25)
        # The decoder emitted local joint position 2, not a positional delta.
        np.testing.assert_allclose(result['joints'][1, 1, 1], [4, 2, 2])
        self.assertEqual(float(self.model.model.conditions[1][0, 0]), 1.)
        self.assertEqual(result['rows'][1]['frame'], 2)
        self.assertAlmostEqual(result['rows'][1]['time_s'], 2 / 30)

    def test_periodic_reset_uses_previous_ground_truth_without_erasing_errors(self):
        result = self.run_reconstruction(resample=2)
        np.testing.assert_allclose(result['roots'][1, :, 0], [0, 2, 4, 22, 24, 42])
        self.assertEqual([i for i, r in enumerate(result['rows']) if r['conditioned_from_gt']], [0, 3, 5])
        self.assertEqual(result['rows'][2]['root_error_m'], 16.)
        self.assertEqual(float(self.model.model.conditions[2][0, 0]), 10.)

    def test_resample_one_teacher_forces_every_condition(self):
        result = self.run_reconstruction(resample=1)
        np.testing.assert_allclose(result['roots'][1, 1:, 0], self.roots[:-1, 0] + 2)
        for i, condition in enumerate(self.model.model.conditions):
            np.testing.assert_allclose(condition.numpy()[0], self.features[i] / 2)

    def test_multiple_condition_frames_reset_whole_history(self):
        self.model.opt.num_condition_frames = 2
        self.run_reconstruction(resample=2)
        expected = (self.features[2:4] / 2).reshape(1, -1)
        np.testing.assert_allclose(self.model.model.conditions[2].numpy(), expected)

    def test_selection_rejects_ambiguity_and_filters(self):
        seq = dict(base=0, source_export='/data/clip.pk', source_key='key')
        dataset = SimpleNamespace(opt=SimpleNamespace(side='fg'), sequences=[seq],
                                  manifest=[dict(name='video', sequences={'fg': [seq]})])
        self.assertEqual(select_sequence(dataset, 'clip')[0], 'video:fg:0')
        dataset.manifest[0]['sequences']['fg'].append(copy.deepcopy(seq))
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            select_sequence(dataset, 'video')
        self.assertEqual(select_sequence(dataset, 'video:fg:1')[1]['base'], 0)
        dataset.sequences = []
        with self.assertRaisesRegex(ValueError, 'excluded'):
            select_sequence(dataset, 'video:fg:0')

    def test_invalid_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'positive'):
            self.run_reconstruction(resample=0)
        self.features[2, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            self.run_reconstruction()

    def test_headless_video_and_report_have_aligned_frame_counts(self):
        result = self.run_reconstruction(resample=2)
        camera = CourtCamera(width=384, height=360)
        camera_before = camera.eye.copy()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with imageio.get_writer(str(directory / 'test.mp4'), fps=30, codec='libx264',
                                    macro_block_size=1) as writer:
                for i in range(6):
                    writer.append_data(render_frame(result, i, camera, 'test', 2))
            with imageio.get_reader(str(directory / 'test.mp4')) as reader:
                self.assertEqual(reader.count_frames(), 6)
                self.assertEqual(reader.get_data(0).shape, (360, 384, 3))
            save_report(directory, result, dict(name='test', resample=2))
            self.assertTrue((directory / 'metrics.png').stat().st_size > 1000)
            self.assertEqual(len((directory / 'history.csv').read_text().splitlines()), 7)
        np.testing.assert_array_equal(camera.eye, camera_before)


if __name__ == '__main__':
    unittest.main()
