"""CPU checks for reconstruction alignment, resets, metrics, and headless recording."""
import copy
import csv
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import imageio
import numpy as np
import torch

from vid2player.motion_vae.tests.options import motion_options
from vid2player.motion_vae.reconstruction import (
    CourtCamera, reconstruct, render_frame, save_report, select_sequence,
    record_unit_actions, unit_action_sequences, save_experiments,
    record_reconstruction,
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
        opt = motion_options()
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

    def test_feature_weights_change_reported_objective_only(self):
        baseline = self.run_reconstruction()
        self.model.opt.recon_feature_weights = {'root_velo': 5.0}
        weighted = self.run_reconstruction()
        np.testing.assert_array_equal(weighted['roots'], baseline['roots'])
        np.testing.assert_array_equal(weighted['joints'], baseline['joints'])
        for old, new in zip(baseline['rows'][1:], weighted['rows'][1:]):
            self.assertEqual(old['recon'], new['recon'])
            self.assertEqual(old['kl_weighted'], new['kl_weighted'])
            # FakeVAE emits displacement [1, 0, 0]; normalized target is zero.
            self.assertAlmostEqual(new['recon_weighted'], old['recon_weighted'] + 4 / 288, delta=2e-6)

    def test_trajectory_report_uses_metres_and_restarts_at_resample_resets(self):
        self.features[:, 3] = 10  # Make target displacements consistent with roots.
        baseline = self.run_reconstruction()
        self.model.opt.accumulate_trajectory_loss = True
        self.model.opt.update(weights={'trajectory': 2})
        weighted = self.run_reconstruction()
        np.testing.assert_array_equal(weighted['roots'], baseline['roots'])
        for i, row in enumerate(weighted['rows'][1:], start=1):
            self.assertAlmostEqual(row['trajectory_mse_m2'], (8 * i) ** 2 / 3, places=4)
            self.assertEqual(row['trajectory_weighted'], 2 * row['trajectory_mse_m2'])
            self.assertEqual(row['recon_weighted'], baseline['rows'][i]['recon_weighted'])
        reset = self.run_reconstruction(resample=1)
        for row in reset['rows'][1:]:
            self.assertAlmostEqual(row['trajectory_mse_m2'], 64 / 3, places=5)
        with tempfile.TemporaryDirectory() as directory:
            metadata = {'name': 'trajectory-test', 'resample': None}
            save_report(Path(directory), weighted, metadata)
            self.assertIn('trajectory_mse_m2_mean', metadata)
            self.assertEqual(metadata['trajectory_weighted_mean'], 2 * metadata['trajectory_mse_m2_mean'])

    def test_trajectory_chunk_boundaries_do_not_reset_integrated_root(self):
        self.features[:, 3] = 10
        baseline = self.run_reconstruction()
        self.model.opt.update(accumulate_trajectory_loss=True, trajectory_window_frames=2,
                              weights={'trajectory': 1})
        result = self.run_reconstruction()
        np.testing.assert_array_equal(result['roots'], baseline['roots'])
        for row, prefix in zip(result['rows'][1:], (1, 2, 1, 2, 1)):
            self.assertAlmostEqual(row['trajectory_mse_m2'], (8 * prefix) ** 2 / 3, places=4)
        # Reset after three predictions is offset from two-frame chunk boundaries.
        reset = self.run_reconstruction(resample=3)
        for row, prefix in zip(reset['rows'][1:], (1, 2, 1, 1, 2)):
            self.assertAlmostEqual(row['trajectory_mse_m2'], (8 * prefix) ** 2 / 3, places=4)

    def test_periodic_reset_uses_previous_ground_truth_without_erasing_errors(self):
        result = self.run_reconstruction(resample=2)
        np.testing.assert_allclose(result['roots'][1, :, 0], [0, 2, 4, 22, 24, 42])
        self.assertEqual([i for i, r in enumerate(result['rows']) if r['conditioned_from_gt']], [0, 3, 5])
        self.assertEqual(result['rows'][2]['root_error_m'], 16.)
        self.assertEqual(float(self.model.model.conditions[2][0, 0]), 10.)
        # A reset uses the true preceding pose when deriving wrist speed;
        # the correction from the previously displayed prediction is not motion.
        self.assertAlmostEqual(result['rows'][3]['pred_pose_wrist_speed_mps'],
                               result['rows'][1]['pred_pose_wrist_speed_mps'])

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

    def test_zero_rollout_uses_own_history_without_encoder(self):
        class LatentVAE:
            def encode(self, truth, condition):
                return torch.ones(1, 32), torch.zeros(1, 32)

            def sample(self, latent, condition):
                output = condition[:, -288:].clone()
                output[:, 3:6] = 0
                output[:, 3] = condition[:, 0] + 1 + latent[:, 0]
                return output

        self.model.model = LatentVAE()
        posterior = self.run_reconstruction()
        self.model.model.encode = lambda *args: self.fail('Zero rollout must bypass encoder')
        zero = reconstruct(self.model, self.features, self.roots, self.frames, 30,
                           latent_mode='zero')
        # Independent integrated root feeds into each subsequent prediction.
        np.testing.assert_allclose(zero['roots'][1, :, 0], [0, 2, 6, 14, 30, 62])
        np.testing.assert_allclose(posterior['roots'][1, :, 0], [0, 4, 12, 28, 60, 124])
        self.assertTrue(all(r['kl'] is None and r['kl_weighted'] is None for r in zero['rows']))

    def test_zero_rollout_honors_reset_schedule_and_multiple_conditions(self):
        self.model.opt.num_condition_frames = 2
        self.model.model.encode = lambda *args: self.fail('Zero rollout must bypass encoder')
        result = reconstruct(self.model, self.features, self.roots, self.frames, 30,
                             resample=2, latent_mode='zero')
        np.testing.assert_allclose(result['roots'][1, :, 0], [0, 10, 12, 14, 32, 34])
        self.assertEqual([i for i, r in enumerate(result['rows']) if r['conditioned_from_gt']], [0, 1, 4])

    def test_explicit_actions_bypass_encoder_and_feed_back_their_own_root(self):
        class ActionVAE:
            def encode(self, *args):
                raise AssertionError('Action rollout must bypass encoder')

            def sample(self, latent, condition):
                output = condition[:, -288:].clone()
                output[:, 3:6] = 0
                output[:, 3] = condition[:, 0] + 1 + latent[:, 0]
                return output

        self.model.model = ActionVAE()
        actions = np.zeros((5, 32), dtype=np.float32)
        actions[:, 0] = [1, 0, 1, 0, 0]
        result = reconstruct(self.model, self.features, self.roots, self.frames, 30,
                             latent_mode='actions', actions=actions)
        np.testing.assert_allclose(result['roots'][1, :, 0], [0, 4, 10, 24, 50, 102])
        self.assertTrue(all(r['kl'] is None for r in result['rows']))
        self.assertEqual(sum(r['conditioned_from_gt'] for r in result['rows']), 1)
        with self.assertRaisesRegex(ValueError, 'shape'):
            reconstruct(self.model, self.features, self.roots, self.frames, 30,
                        latent_mode='actions', actions=actions[:-1])
        actions[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            reconstruct(self.model, self.features, self.roots, self.frames, 30,
                        latent_mode='actions', actions=actions)

    def test_unit_actions_are_reproducible_unit_gaussians_with_correct_holds(self):
        sequences = unit_action_sequences(23, 32, seed=7, hold_frames=10)
        repeated = unit_action_sequences(23, 32, seed=7, hold_frames=10)
        for name, actions in sequences.items():
            np.testing.assert_array_equal(actions, repeated[name])
            self.assertEqual(actions.shape, (23, 32))
            np.testing.assert_array_equal(actions[0], sequences['per_frame'][0])
        np.testing.assert_array_equal(sequences['fixed'], np.repeat(sequences['fixed'][:1], 23, axis=0))
        for i in range(23):
            np.testing.assert_array_equal(sequences['held'][i], sequences['per_frame'][(i // 10) * 10])
        # Catch accidental unit-norm normalization or clipping.
        draws = unit_action_sequences(10000, 32)['per_frame']
        self.assertAlmostEqual(float(draws.std()), 1, places=2)
        self.assertGreater(float(np.abs(draws).max()), 3)

    def test_unit_action_reports_are_separate_and_never_reset_from_parent(self):
        self.model.model.encode = lambda *args: self.fail('Probes must not encode targets')
        metadata = dict(name='test', resample=1, kl_mean=123, zero_latent_rollout={'kl_mean': 456})
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / 'unit_actions'
            zero = reconstruct(self.model, self.features, self.roots, self.frames, 30,
                               latent_diagnostics=False, latent_mode='zero')
            with patch('vid2player.motion_vae.reconstruction.reconstruct', wraps=reconstruct) as rollout:
                record_unit_actions(directory, self.model, self.features, self.roots,
                                    self.frames, 30, metadata, zero, seed=7, hold_frames=2)
                self.assertEqual(rollout.call_count, 3)
                self.assertTrue(all(call.kwargs['latent_mode'] == 'actions' for call in rollout.call_args_list))
            self.assertEqual(metadata['resample'], 1)
            self.assertFalse((directory / 'zero').exists())
            for schedule in ('fixed', 'held', 'per_frame'):
                output = directory / schedule
                summary = json.loads((output / 'summary.json').read_text())
                self.assertIsNone(summary['resample'])
                self.assertNotIn('kl_mean', summary)
                self.assertEqual(summary['root_endpoint_error_m'], 40)
                self.assertTrue((output / 'metrics.png').stat().st_size > 1000)
                with imageio.get_reader(str(output / 'rollout.mp4')) as reader:
                    self.assertEqual(reader.count_frames(), 6)
                self.assertEqual(np.load(output / 'actions.npy').shape, (5, 32))
                self.assertNotIn('zero_latent_rollout', summary)
                self.assertFalse((output / 'zero_latent_history.csv').exists())
                with (output / 'history.csv').open() as stream:
                    columns = csv.DictReader(stream).fieldnames
                self.assertNotIn('kl', columns)
                self.assertNotIn('recon_zero_latent', columns)
                self.assertNotIn('recon_joint_rotmat', columns)
                self.assertIn('local_joint_difference_from_zero_m', columns)
            comparison = json.loads((directory / 'summary.json').read_text())
            self.assertEqual(set(comparison['comparisons_to_zero']), {'fixed', 'held', 'per_frame'})

    def test_recording_runs_unit_actions_only_when_explicitly_enabled(self):
        dataset = SimpleNamespace(
            valid_arr=np.ones(7, dtype=bool), joint_pos_arr=np.zeros((7, 3), dtype=np.float32),
            get_window=lambda *args: {'feature': self.features})
        sequence = dict(base=0, start=0, length=7, fps=30)
        self.model.checkpoint_dir = '/tmp/checkpoints'
        self.model.loaded_checkpoint_path = '/tmp/checkpoints/epoch_50.tar'
        with tempfile.TemporaryDirectory() as temp:
            self.model.opt.result_dir = temp
            with patch('vid2player.motion_vae.reconstruction.Video3DPoseDataset', return_value=dataset), \
                 patch('vid2player.motion_vae.reconstruction.select_sequence', return_value=('test', sequence)), \
                 patch('vid2player.motion_vae.reconstruction.MotionVAEModel', return_value=self.model), \
                 patch('vid2player.motion_vae.reconstruction.write_rollout_video'), \
                 patch('vid2player.motion_vae.reconstruction.save_experiments'), \
                 patch('vid2player.motion_vae.reconstruction.reconstruct', wraps=reconstruct) as rollouts, \
                 patch('vid2player.motion_vae.reconstruction.record_unit_actions') as probes:
                # The mocked report still needs the existing print-summary values.
                def report(directory, result, metadata, **kwargs):
                    self.assertEqual(metadata['checkpoint'], '/tmp/checkpoints/epoch_50.tar')
                    metadata.update(recon_mean=0., kl_weighted_mean=0.,
                                    root_error_m_mean=0., local_mpjpe_m_mean=0.)
                with patch('vid2player.motion_vae.reconstruction.save_report', side_effect=report):
                    default_dir = record_reconstruction(self.model.opt, 'test')
                    probes.assert_not_called()
                    self.assertEqual(sum(call.kwargs.get('latent_mode') == 'zero'
                                         for call in rollouts.call_args_list), 1)
                    self.assertFalse((default_dir / 'unit_actions').exists())
                    explicit_dir = record_reconstruction(self.model.opt, 'test', unit_actions=True,
                                                         latent_seed=7, latent_hold_frames=3)
                    probes.assert_called_once()
                    self.assertEqual(sum(call.kwargs.get('latent_mode') == 'zero'
                                         for call in rollouts.call_args_list), 2)
                    self.assertEqual(probes.call_args.args[0], explicit_dir / 'unit_actions')
                    self.assertEqual(probes.call_args.args[-2:], (7, 3))

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

    def test_latent_statistics_include_per_dimension_moments_and_survive_report(self):
        result=self.run_reconstruction(resample=1)
        stats=result['latent_statistics']
        self.assertEqual(stats['frames'],5)
        self.assertEqual(stats['mu_rms'],1.)
        self.assertEqual(stats['mu_mean_per_dim'],[1.,1.])
        self.assertEqual(stats['mu_std_per_dim'],[0.,0.])
        self.assertEqual(stats['posterior_std_mean_per_dim'],[1.,1.])
        np.testing.assert_allclose(stats['z_rms_per_dim'],[np.sqrt(2),np.sqrt(2)])
        with tempfile.TemporaryDirectory() as temp:
            save_report(Path(temp),result,dict(name='latents',resample=1))
            summary=json.loads((Path(temp)/'summary.json').read_text())
            self.assertEqual(summary['latent_statistics'],stats)
            self.assertIn('latent_mu_rms',(Path(temp)/'history.csv').read_text().splitlines()[0])
        zero=reconstruct(self.model,self.features,self.roots,self.frames,30.,latent_mode='zero')
        self.assertNotIn('latent_statistics',zero)

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
            zero = reconstruct(self.model, self.features, self.roots, self.frames, 30,
                               resample=2, latent_mode='zero')
            save_report(directory, result, dict(name='test', resample=2))
            zero_dir = directory / 'zero'
            zero_dir.mkdir()
            save_report(zero_dir, zero, dict(name='test', resample=2, rollout_label='Always z=0'))
            save_experiments(directory, {'Posterior (resets every 2 frames)': result,
                                        'z=0 (resets every 2 frames)': zero}, 'test')
            summary = json.loads((directory / 'summary.json').read_text())
            self.assertNotIn('zero_latent_rollout', summary)
            self.assertFalse((directory / 'zero_latent_history.csv').exists())
            self.assertEqual(len((zero_dir / 'history.csv').read_text().splitlines()), 7)
            self.assertTrue((directory / 'metrics.png').stat().st_size > 1000)
            self.assertFalse((directory / 'diagnostics.png').exists())
            self.assertEqual({p.name for p in directory.glob('*.png')}, {'metrics.png', 'experiments.png'})
            self.assertEqual(len((directory / 'history.csv').read_text().splitlines()), 7)
        np.testing.assert_array_equal(camera.eye, camera_before)


if __name__ == '__main__':
    unittest.main()
