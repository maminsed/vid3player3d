"""CPU regression tests for held-out splits, physical metrics, and validation isolation."""
import copy
import contextlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from vid2player.motion_vae.base import MotionVAEModel
from vid2player.motion_vae.config import MotionVAEOption
from vid2player.motion_vae.dataset import Video3DPoseDataset
from vid2player.motion_vae.metrics import add_motion_metrics, summarize_rows
from vid2player.motion_vae.validation import MotionValidation


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        data = root / 'data'
        data.mkdir()
        self.length = 10
        count = 8 * self.length
        positions = np.zeros((count, 72), dtype=np.float32)
        manifest = []
        for video in range(4):
            for scene in range(2):
                base = (video * 2 + scene) * self.length
                positions[base:base + self.length, 0] = video * 1000 + scene * 100 + np.arange(self.length) ** 2 * .01
                positions[base:base + self.length, 3:] = np.arange(self.length)[:, None] * .01
                manifest.append(dict(name='video{}-scene{}'.format(video, scene), background='usopen',
                    gender='mens', is_orig=False, sequences={'bg': [], 'fg': [dict(
                        base=base, start=0, length=self.length, beta=[0.] * 10, player='Federer',
                        handness='right', fps=30, source_export='/source/video{}-scene{}.pk'.format(video, scene))]}))
        np.save(str(data / 'joint_pos.npy'), positions)
        np.save(str(data / 'joint_rot.npy'), np.zeros((count, 24, 3), dtype=np.float32))
        np.save(str(data / 'joint_rotmat.npy'), np.tile(np.eye(3, dtype=np.float32), (count, 24, 1, 1)))
        np.save(str(data / 'valid.npy'), np.ones(count, dtype=bool))
        (data / 'manifest.json').write_text(json.dumps(manifest))
        self.opt = MotionVAEOption()
        self.opt.load('federer_physics')
        self.opt.update(dataset_dir=str(data), checkpoint_dir=str(root), model_ver='model',
                        gpu_ids=[], hidden_size=8, latent_size=2, num_experts=2,
                        batch_size=2, nframes_seq=3, nseqs=4, num_threads=0, n_epochs=1,
                        n_epochs_decay=0, log_freq=4, no_log=True, validation_num_videos=1,
                        validation_window_frames=3, validation_windows_per_video=2,
                        validation_freq_steps=1, curriculum_schedule=None)

    def model(self, **changes):
        opt = copy.deepcopy(self.opt)
        opt.update(**changes)
        with contextlib.redirect_stdout(io.StringIO()):
            return MotionVAEModel(opt)

    def test_one_video_split_keeps_other_scenes_and_excludes_normalization_leakage(self):
        model = self.model()
        split = model.validation_split
        self.assertEqual(len(split['validation_videos']), 1)
        held_out = set(split['validation_videos'])
        self.assertTrue(all(s['_video_name'] not in held_out for s in model.dataset.sequences))
        self.assertEqual(len(model.validation.dataset.sequences), 1)
        self.assertEqual(len(model.dataset.sequences), 7)
        source = next(iter(held_out)).split('-scene')[0]
        self.assertTrue(any(s['_video_name'].startswith(source + '-scene') for s in model.dataset.sequences))
        self.assertTrue(all(s['_video_name'] in held_out for s in model.validation.dataset.sequences))
        train_bases = {s['base'] for s in model.dataset.sequences}
        val_bases = {s['base'] for s in model.validation.dataset.sequences}
        self.assertFalse(train_bases & val_bases)
        samples = np.concatenate([model.validation.dataset.get_window(s['base'], s['length'] - 1)['feature']
                                  for s in model.dataset.sequences])
        expected = samples.std(0)
        expected[expected == 0] = 1.
        np.testing.assert_allclose(model.dataset.avg, samples.mean(0), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(model.dataset.std, expected, rtol=1e-5, atol=1e-6)
        self.assertLess(model.dataset.std[3], .1)  # Huge scene jumps were excluded.
        self.assertIsNone(model.validation.dataset.std)  # Use model's training stats at evaluation.
        # Last possible training window is included.
        self.assertIn(model.dataset.sequences[0]['base'] + self.length - self.opt.nframes_seq - 1,
                      model.dataset.rollouts)

    def test_two_videos_remove_exactly_two_manifest_entries(self):
        model = self.model(validation_num_videos=2)
        self.assertEqual(len(model.validation_split['validation_videos']), 2)
        self.assertEqual(len(model.validation.dataset.sequences), 2)
        self.assertEqual(len(model.dataset.sequences), 6)

    def test_validation_persists_only_tensorboard_metrics(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        model = self.model(no_log=False)
        self.addCleanup(model.writer.close)
        model.validation.evaluate(model, 1, 2, .5)
        model.writer.flush()
        directory = Path(model.checkpoint_dir)
        self.assertFalse((directory / 'validation').exists())
        self.assertEqual(list(directory.rglob('*.json')), [directory / 'run_config.json'])
        self.assertFalse(list(directory.rglob('*.csv')))
        events = EventAccumulator(str(directory / 'logs'))
        events.Reload()
        tag = 'val_full/root_error_m'
        self.assertIn(tag, events.Tags()['scalars'])
        self.assertEqual(events.Scalars(tag)[0].step, 2)
        tags = events.Tags()['scalars']
        self.assertEqual(len([t for t in tags if not t.startswith('diagnostics/')]), 12)
        self.assertLessEqual(len(tags), 46)
        self.assertTrue(any(t.startswith('diagnostics/') for t in tags))
        self.assertFalse(any(t.startswith('validation/') or 'video' in t or 'gt_' in t or 'frames' in t for t in tags))
        metadata = json.loads((directory / 'run_config.json').read_text())
        self.assertEqual(metadata['config']['validation_num_videos'], 1)
        self.assertEqual(metadata['validation_reference']['window_3']['predicted_frames'], 6)
        self.assertEqual(metadata['validation_reference']['coverage']['selected_sequences'], 1)
        self.assertEqual(metadata['validation_split'], model.validation_split)
        self.assertEqual(metadata['training_data']['batches_per_epoch'], 2)

    def test_invalid_predecessors_excluded_from_normalization(self):
        opt = copy.deepcopy(self.opt)
        opt.pose_feature = ['root_velo']
        with contextlib.redirect_stdout(io.StringIO()):
            dataset = Video3DPoseDataset(opt)
        dataset.valid_arr[4] = False
        dataset.get_normalization_stats()
        expected = []
        for seq in dataset.sequences:
            for i in range(seq['base'] + 1, seq['base'] + seq['length']):
                if dataset.valid_arr[i] and dataset.valid_arr[i - 1]:
                    expected.append(dataset.joint_pos_arr[i, :3] - dataset.joint_pos_arr[i - 1, :3])
        np.testing.assert_allclose(dataset.avg, np.mean(expected, 0))
        std = np.std(expected, 0); std[std == 0] = 1
        np.testing.assert_allclose(dataset.std, std)

    def test_resume_preserves_split_steps_and_rejects_changes(self):
        original = self.model()
        original.train_steps = 17
        original.save_checkpoint('latest')
        resumed = self.model(continue_train=True)
        self.assertEqual(resumed.validation_split, original.validation_split)
        self.assertEqual(resumed.train_steps, 17)
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.model(continue_train=True, validation_seed=123)
        with self.assertRaisesRegex(ValueError, 'disable'):
            self.model(continue_train=True, validation_num_videos=0)
        checkpoint = Path(original.checkpoint_dir) / 'latest.tar'
        payload = torch.load(str(checkpoint))
        del payload['validation_split']
        torch.save(payload, str(checkpoint))
        with self.assertRaisesRegex(ValueError, 'no held-out split'):
            self.model(continue_train=True)

    def test_evaluation_is_repeatable_restores_rng_mode_and_does_not_update_weights(self):
        model = self.model()
        weights = {key: value.clone() for key, value in model.model.state_dict().items()}
        torch.manual_seed(12); np.random.seed(13); random.seed(14)
        torch_state = torch.get_rng_state().clone()
        np_state = np.random.get_state()
        python_state = random.getstate()
        first = model.validation.evaluate(model, 0, 0, 0)
        self.assertTrue(model.model.training)
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        np.testing.assert_array_equal(np_state[1], np.random.get_state()[1])
        self.assertEqual(python_state, random.getstate())
        self.assertFalse(model.optimizer.state)
        for key, value in model.model.state_dict().items():
            self.assertTrue(torch.equal(weights[key], value))
        second = model.validation.evaluate(model, 1, 2, .5)
        self.assertEqual(first['summary'], second['summary'])
        for key in ('window_3/teacher_forced', 'window_3/autoregressive'):
            self.assertEqual(first['summary'][key]['predicted_frames'], 6)
        self.assertEqual(first['summary']['whole_sequence/autoregressive']['predicted_frames'], 8)
        self.assertFalse((Path(model.checkpoint_dir) / 'validation').exists())
        self.assertEqual(len(first['rollouts']), 6)
        model.model.eval()
        model.validation.evaluate(model, 2, 4, 1)
        self.assertFalse(model.model.training)

    def test_periodic_evaluation_counts_batches_and_runs_final_partial_interval(self):
        model = self.model(validation_freq_steps=3)
        model.validation.evaluate = Mock()
        model.train_batch = Mock(return_value={'recon': 1.})
        model.train()
        self.assertEqual([call.args[1] for call in model.validation.evaluate.call_args_list], [0, 2])
        model = self.model(model_ver='scheduled', validation_freq_steps=1)
        model.validation.evaluate = Mock()
        model.train_batch = Mock(return_value={'recon': 1.})
        model.train()
        self.assertEqual([call.args[1] for call in model.validation.evaluate.call_args_list], [0, 1, 2])
        self.assertEqual(model.train_steps, 2)

    def test_invalid_gaps_are_not_stitched_into_a_full_sequence(self):
        model = self.model()
        data = model.validation.dataset
        seq = data.sequences[0]
        valid = np.load(str(Path(self.opt.dataset_dir) / 'valid.npy'))
        valid[seq['base'] + 5] = False
        np.save(str(Path(self.opt.dataset_dir) / 'valid.npy'), valid)
        with contextlib.redirect_stdout(io.StringIO()):
            validation = MotionValidation(self.opt, model.validation_split['validation_videos'])
        parts = [s for s in validation.full if s['seq']['base'] == seq['base']]
        self.assertEqual(len(parts), 2)
        self.assertTrue(all(not part['whole_sequence'] for part in parts))

    def test_failed_rollouts_are_counted_and_state_is_restored(self):
        model = self.model()
        with patch('vid2player.motion_vae.reconstruction.reconstruct',
                   side_effect=ValueError('Non-finite prediction at source frame 3.')):
            with self.assertLogs(level='WARNING'):
                report = model.validation.evaluate(model, 0, 0, 0)
        self.assertTrue(model.model.training)
        self.assertTrue(all(v['failed_rollouts'] == v['rollouts'] for v in report['summary'].values()))
        self.assertTrue(all(v['predicted_frames'] == 0 for v in report['summary'].values()))
        self.assertTrue(all('error' in rollout for rollout in report['rollouts']))

    def test_physical_metric_units_masks_and_conditioning_exclusion(self):
        from vid2player.motion_vae.reconstruction import feature_slices
        fields = feature_slices(self.opt)
        truth = np.zeros((2, 288), dtype=np.float32)
        truth[1, 75 + 19 * 3] = .3  # Left wrist 3 m/s at 10 fps.
        pred = truth.copy()
        pred[1, 3] = .1  # 1 m/s predicted root motion.
        pred[1, 6 + 19 * 3] = 1.
        roots = np.zeros((2, 2, 3), dtype=np.float32)
        roots[1, 1, 0] = .1
        joints = np.zeros((2, 2, 24, 3), dtype=np.float32)
        joints[1, :, 1:] = pred[:, 6:75].reshape(2, 23, 3)
        joints += roots[:, :, None, :]
        rows = [dict(recon=None, root_error_m=999.), dict(recon=0., root_error_m=.1)]
        result = dict(rows=rows, roots=roots, joints=joints, predicted_features=pred)
        add_motion_metrics(result, truth, fields, 10, self.opt)
        summary = summarize_rows(rows)
        self.assertEqual(summary['predicted_frames'], 1)
        self.assertEqual(summary['root_error_m_mean'], .1)
        self.assertEqual(summary['stationary_frames'], 1)
        self.assertEqual(summary['fast_motion_frames'], 1)
        self.assertAlmostEqual(summary['local_mpjpe_m_mean'], 1 / 23)
        self.assertAlmostEqual(summary['left_wrist_error_m_mean'], 1.)
        self.assertAlmostEqual(summary['right_wrist_error_m_mean'], 0.)
        self.assertAlmostEqual(summary['root_velocity_bias_x_mps_mean'], 1.)
        self.assertAlmostEqual(summary['root_consistency_m_mean'], .1)
        self.assertAlmostEqual(summary['gt_wrist_speed_mps_mean'], 3.)
        self.assertAlmostEqual(summary['pred_wrist_speed_mps_mean'], 4.)
        rows[1]['stationary'] = 0
        self.assertIsNone(summarize_rows(rows)['stationary_root_error_m_mean'])


if __name__ == '__main__':
    unittest.main()
