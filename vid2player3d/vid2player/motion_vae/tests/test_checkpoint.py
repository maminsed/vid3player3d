"""CPU regression tests for MVAE epoch-boundary checkpoint resumption.

Run from vid2player3d:
    python -m unittest discover -s vid2player/motion_vae/tests -v
"""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from vid2player.motion_vae.base import MotionVAEModel
from vid2player.motion_vae.tests.options import motion_options


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        data = self.root / 'dataset'
        data.mkdir()
        rng = np.random.RandomState(10)
        np.save(str(data / 'joint_pos.npy'), rng.randn(14, 72).astype(np.float32))
        np.save(str(data / 'joint_rot.npy'), np.zeros((14, 24, 3), dtype=np.float32))
        np.save(str(data / 'joint_rotmat.npy'),
                np.tile(np.eye(3, dtype=np.float32), (14, 24, 1, 1)))
        np.save(str(data / 'valid.npy'), np.ones(14, dtype=bool))
        (data / 'manifest.json').write_text(json.dumps([{
            'name': 'test', 'background': 'usopen', 'gender': 'mens',
            'is_orig': False, 'sequences': {'bg': [], 'fg': [{
                'base': 0, 'start': 0, 'length': 14, 'beta': [0.] * 10,
                'player': 'Federer', 'handness': 'right',
            }]},
        }]))
        self.opt = motion_options()
        self.opt.update(dataset_dir=str(data), checkpoint_dir=str(self.root),
                        model_ver='model', gpu_ids=[], hidden_size=8, latent_size=2,
                        num_experts=2, batch_size=2, nframes_seq=3, nseqs=4,
                        num_threads=0, n_epochs=1, n_epochs_decay=4,
                        save_freq_epoch=1, log_freq=4, no_log=True)
        self.batch = {'feature': torch.randn(2, 3, 288)}

    def model(self, **changes):
        opt = copy.deepcopy(self.opt)
        opt.update(**changes)
        return MotionVAEModel(opt)

    def assert_weights_equal(self, first, second):
        for name, value in first.model.state_dict().items():
            self.assertTrue(torch.equal(value, second.model.state_dict()[name]), name)

    def test_resume_restores_adam_and_identical_next_update(self):
        original = self.model()
        original.train_batch(self.batch)
        original.update_learning_rate()
        original.start_epoch = 3
        original.total_iters = 12
        original.save_checkpoint('latest')

        resumed = self.model(continue_train=True)
        self.assertEqual(resumed.start_epoch, 3)
        self.assertEqual(resumed.total_iters, 12)
        self.assertEqual(resumed.old_lr, original.old_lr)
        self.assertEqual(resumed.optimizer.param_groups[0]['lr'], original.old_lr)
        self.assertTrue(resumed.optimizer.state)
        self.assert_weights_equal(original, resumed)
        np.testing.assert_array_equal(resumed.dataset.avg, original.dataset.avg)
        np.testing.assert_array_equal(resumed.dataset.std, original.dataset.std)

        # Hold the random latent samples fixed to isolate optimizer restoration.
        torch.manual_seed(123)
        original_loss = original.train_batch(self.batch, regressive=True)
        torch.manual_seed(123)
        resumed_loss = resumed.train_batch(self.batch, regressive=True)
        self.assertEqual(original_loss, resumed_loss)
        self.assert_weights_equal(original, resumed)

    def test_feed_vae_uses_feature_weights_and_preserves_kl(self):
        baseline = self.model()
        weighted = self.model(recon_feature_weights={'root_velo': 5.0})
        weighted.model.load_state_dict(baseline.model.state_dict())
        truth = self.batch['feature'][:, 1:2]
        condition = self.batch['feature'][:, :1]
        torch.manual_seed(123)
        (output, _, _, _), old_losses = baseline.feed_vae(truth, condition)
        torch.manual_seed(123)
        (weighted_output, _, _, _), losses = weighted.feed_vae(truth, condition)
        self.assertTrue(torch.equal(output, weighted_output))
        self.assertTrue(torch.equal(losses['kl'], old_losses['kl']))
        errors = (output - truth).square()
        self.assertTrue(torch.equal(old_losses['recon'], errors.mean(dim=(0, -1)).mul(
            baseline.future_weights).sum() * baseline.opt.weights['recon']))
        expected = old_losses['recon'] + 4 * errors[..., 3:6].sum() / errors.numel() * baseline.opt.weights['recon']
        torch.testing.assert_close(losses['recon'], expected)

    def test_zero_trajectory_weight_preserves_losses_updates_and_optimizer_steps(self):
        baseline = self.model()
        disabled = self.model(accumulate_trajectory_loss=True, weights={'trajectory': 0})
        disabled.model.load_state_dict(baseline.model.state_dict())
        torch.manual_seed(123)
        old_losses = baseline.train_batch(self.batch, regressive=True)
        with patch.object(disabled.optimizer, 'step', wraps=disabled.optimizer.step) as step:
            torch.manual_seed(123)
            losses = disabled.train_batch(self.batch, regressive=True)
        self.assertEqual(step.call_count, 2)
        self.assertEqual(losses, old_losses)
        self.assert_weights_equal(disabled, baseline)

    def test_trajectory_training_updates_per_chunk_and_weights_short_tail_by_frames(self):
        for regressive in (False, True):
            for futures in (1, 2):
                for horizon in (1, 2, 9):
                    with self.subTest(regressive=regressive, futures=futures, horizon=horizon):
                        model = self.model(accumulate_trajectory_loss=True, weights={'trajectory': 2},
                                           trajectory_window_frames=horizon, num_future_predictions=futures)
                        batch = {'feature': torch.randn(2, 6, 288)}
                        predictions, targets, step_losses, updates = [], [], [], []
                        original_feed, original_optimize = model.feed_vae, model._optimize_loss

                        def record_feed(truth, *args):
                            output, losses = original_feed(truth, *args)
                            predictions.append(output[0][:, 0, 3:6].detach().clone())
                            targets.append(truth[:, 0, 3:6].detach().clone())
                            step_losses.append({key: value.item() for key, value in losses.items()})
                            return output, losses

                        def record_update(total):
                            updates.append((len(predictions), total.item()))
                            original_optimize(total)

                        with patch.object(model, 'feed_vae', side_effect=record_feed), patch.object(
                                model, '_optimize_loss', side_effect=record_update):
                            losses = model.train_batch(batch, regressive=regressive)
                        count = len(predictions)
                        self.assertEqual(len(updates), (count + horizon - 1) // horizon)
                        for key in step_losses[0]:
                            self.assertAlmostEqual(losses[key], np.mean([item[key] for item in step_losses]))
                        error_m = (torch.stack(predictions, dim=1) - torch.stack(targets, dim=1)) * model.std[3:6]
                        expected_sum = 0.
                        for update, start in zip(updates, range(0, count, horizon)):
                            stop = min(start + horizon, count)
                            self.assertEqual(update[0], stop)
                            trajectory = error_m[:, start:stop].cumsum(dim=1).square().mean().item() * 2
                            expected_sum += trajectory * (stop - start)
                            reconstruction_kl = np.mean([sum(item.values()) for item in step_losses[start:stop]])
                            self.assertAlmostEqual(update[1], reconstruction_kl + trajectory, places=5)
                        self.assertAlmostEqual(losses['trajectory'], expected_sum / count)
                        self.assertTrue(np.isfinite(losses['trajectory']))
                        self.assertTrue(model.optimizer.state)

    def test_train_continues_epoch_counter_and_schedules(self):
        with patch('vid2player.motion_vae.base.SummaryWriter'):
            original = self.model(no_log=False)
            original.train_batch = Mock(return_value={'recon': 1.})
            original.train()
            # Simulate restarting from the completed third epoch.
            checkpoint_dir = Path(original.checkpoint_dir)
            shutil.copyfile(str(checkpoint_dir / 'epoch_3.tar'),
                            str(checkpoint_dir / 'latest.tar'))
            resumed = self.model(no_log=False, continue_train=True)
            self.assertEqual(resumed.start_epoch, 3)
            self.assertAlmostEqual(resumed.old_lr, self.opt.lr * 0.75)
            resumed.train_batch = Mock(return_value={'recon': 1.})
            resumed.schedual_regressive_training = Mock(
                wraps=resumed.schedual_regressive_training)
            resumed.train()
            self.assertEqual([c.args[0] for c in resumed.schedual_regressive_training.call_args_list],
                             [3, 3, 4, 4])
            self.assertEqual(resumed.start_epoch, 5)
            self.assertEqual(resumed.total_iters, original.total_iters)
            self.assertEqual(resumed.old_lr, original.old_lr)
            checkpoint = torch.load(str(checkpoint_dir / 'latest.tar'), map_location='cpu')
            self.assertEqual(checkpoint['next_epoch'], 5)
            self.assertEqual(checkpoint['optimizer_state_dict']['param_groups'][0]['lr'], original.old_lr)
            finished = self.model(no_log=False, continue_train=True)
            finished.train_batch = Mock()
            finished.train()
            finished.train_batch.assert_not_called()

    def test_inference_and_resume_use_embedded_normalization(self):
        original = self.model()
        original.save_checkpoint('latest')
        checkpoint_dir = Path(original.checkpoint_dir)
        # External statistics may belong to a different training run.
        np.save(str(checkpoint_dir / 'avg.npy'), np.full(288, 999.))
        np.save(str(checkpoint_dir / 'std.npy'), np.full(288, 999.))
        inference = self.model(test_only=True)
        self.assertFalse(inference.model.training)
        self.assert_weights_equal(original, inference)
        np.testing.assert_array_equal(inference.avg.numpy(), original.dataset.avg.astype(np.float32))
        resumed = self.model(continue_train=True)
        np.testing.assert_array_equal(resumed.dataset.avg, original.dataset.avg)
        np.testing.assert_array_equal(resumed.dataset.std, original.dataset.std)

    def test_legacy_inference_and_explicit_warm_start_warning(self):
        original = self.model()
        checkpoint_dir = Path(original.checkpoint_dir)
        torch.save(original.model.state_dict(), str(checkpoint_dir / 'latest.tar'))
        np.save(str(checkpoint_dir / 'avg.npy'), original.dataset.avg)
        np.save(str(checkpoint_dir / 'std.npy'), original.dataset.std)
        inference = self.model(test_only=True)
        self.assert_weights_equal(original, inference)
        with self.assertLogs(level='WARNING') as logs:
            resumed = self.model(continue_train=True)
        self.assertIn('Legacy weights-only', ' '.join(logs.output))
        self.assertEqual(resumed.start_epoch, 0)
        self.assertEqual(resumed.total_iters, 0)
        self.assertFalse(resumed.optimizer.state)
        self.assert_weights_equal(original, resumed)

    def test_inference_falls_back_to_cpu_when_config_requests_hidden_gpu(self):
        original = self.model()
        original.save_checkpoint('latest')
        with patch('torch.cuda.is_available', return_value=False):
            inference = self.model(test_only=True, gpu_ids=[0])
        self.assertEqual(inference.device, torch.device('cpu'))
        self.assertEqual(next(inference.model.parameters()).device.type, 'cpu')
        self.assertEqual(inference.avg.device.type, 'cpu')
        self.assert_weights_equal(original, inference)

    def test_amp_scaler_state_is_restored_after_creation(self):
        # Check scaler serialization on CPU without requiring a CUDA device.
        scaler = Mock()
        scaler.state_dict.return_value = {'scale': 1024., '_growth_tracker': 7}
        with patch('torch.cuda.amp.GradScaler', return_value=scaler):
            original = self.model(use_amp=True)
            original.save_checkpoint('latest')
        resumed_scaler = Mock()
        with patch('torch.cuda.amp.GradScaler', return_value=resumed_scaler):
            self.model(use_amp=True, continue_train=True)
        resumed_scaler.load_state_dict.assert_called_once_with(scaler.state_dict.return_value)

    def test_failed_write_preserves_previous_checkpoint(self):
        original = self.model()
        original.save_checkpoint('latest')
        checkpoint_path = Path(original.checkpoint_dir) / 'latest.tar'
        saved_bytes = checkpoint_path.read_bytes()

        def fail_write(checkpoint, path):
            Path(path).write_bytes(b'incomplete checkpoint')
            raise OSError('simulated write failure')

        with patch('torch.save', side_effect=fail_write):
            with self.assertRaises(OSError):
                original.save_checkpoint('latest')
        self.assertEqual(checkpoint_path.read_bytes(), saved_bytes)
        self.assertFalse(Path(str(checkpoint_path) + '.tmp').exists())


if __name__ == '__main__':
    unittest.main()
