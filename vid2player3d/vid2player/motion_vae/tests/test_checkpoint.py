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
from vid2player.motion_vae.config import MotionVAEOption


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
        self.opt = MotionVAEOption()
        self.opt.load('federer_physics')
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
