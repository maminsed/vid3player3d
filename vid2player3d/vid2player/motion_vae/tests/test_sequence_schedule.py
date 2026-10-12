"""CPU coverage for window curricula, workers, and checkpoint resumption."""
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from vid2player.motion_vae.config import MotionVAEOption
from vid2player.motion_vae.scheduling import sequence_length
from vid2player.motion_vae.tests import test_checkpoint


class SequenceScheduleTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.previous_threads)
        self.fixture = test_checkpoint.CheckpointTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        data = self.fixture.root / 'dataset'
        count = 80
        np.save(str(data / 'joint_pos.npy'), np.random.RandomState(10).randn(count, 72).astype(np.float32))
        np.save(str(data / 'joint_rot.npy'), np.zeros((count, 24, 3), dtype=np.float32))
        np.save(str(data / 'joint_rotmat.npy'), np.tile(np.eye(3, dtype=np.float32), (count, 24, 1, 1)))
        np.save(str(data / 'valid.npy'), np.ones(count, dtype=bool))
        manifest = json.loads((data / 'manifest.json').read_text())
        manifest[0]['sequences']['fg'][0]['length'] = count
        (data / 'manifest.json').write_text(json.dumps(manifest))
        self.fixture.opt.update(nframes_seq=10, n_epochs=3, n_epochs_decay=0,
                                nframes_seq_schedule=[(0, 10), (1/3, 30), (2/3, 60)])

    def test_interpolation_endpoints_and_fixed_default(self):
        opt = MotionVAEOption()
        opt.load('usopen_6_1')
        self.assertEqual([sequence_length(opt, e) for e in (0, 50, 100, 150, 200, 250, 300, 350, 499)],
                         [10, 10, 10, 10, 20, 30, 45, 60, 60])
        opt.nframes_seq_schedule = [(0.3, 10), (0.5, 30)]
        self.assertEqual(sequence_length(opt, 499), 30)
        opt.nframes_seq_schedule = None
        self.assertEqual(sequence_length(opt, 499), 10)

    def test_invalid_schedules_fail_before_loading_data(self):
        for schedule in ([], [(0, 10)], [(0.5, 10), (0.5, 30)],
                         [(-0.1, 10), (1, 30)], [(0, 10), (1.1, 30)],
                         [(0, 30), (1, 10)], [(0, 1), (1, 30)],
                         [(0, 10), (1, 30.5)], [(0, 10), (float('nan'), 30)],
                         [(0, 10), (1, True)], [(0, 10), 'invalid']):
            with self.subTest(schedule=schedule), self.assertRaisesRegex(ValueError, 'nframes_seq_schedule'):
                self.fixture.model(dataset_dir='/missing', nframes_seq_schedule=schedule)

    def test_real_cpu_training_rebuilds_worker_windows_and_logs_lengths(self):
        # Workers are recreated on each epoch iterator and must see the new pool.
        model = self.fixture.model(num_threads=1, no_log=False)
        # Spawn avoids forking the test runner's initialized CPU thread pools.
        model.trainset = torch.utils.data.DataLoader(
            model.dataset, batch_size=2, shuffle=True, drop_last=True,
            num_workers=1, multiprocessing_context='spawn', timeout=30)
        lengths = []
        train_batch = model.train_batch

        def record(batch, *args, **kwargs):
            lengths.append(batch['feature'].shape[1])
            self.assertTrue(all(start + model.dataset.nframes_seq < 80 for start in model.dataset.rollouts))
            return train_batch(batch, *args, **kwargs)

        with patch.object(model, 'train_batch', side_effect=record):
            model.train()
        self.assertEqual(lengths, [10, 10, 30, 30, 60, 60])
        self.assertEqual(model.start_epoch, 3)
        self.assertEqual(model.total_iters, 12)
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        events = EventAccumulator(str(self.fixture.root / 'model' / 'logs'))
        events.Reload()
        self.assertEqual([event.value for event in events.Scalars('train/nframes_seq')], [10, 30, 60])

    def test_resume_initializes_saved_epoch_length_and_preserves_normalization(self):
        model = self.fixture.model()
        model.start_epoch = 2
        model.total_iters = 8
        model.save_checkpoint('latest')
        resumed = self.fixture.model(continue_train=True)
        self.assertEqual(resumed.dataset.nframes_seq, 60)
        self.assertEqual(next(iter(resumed.trainset))['feature'].shape[1], 60)
        np.testing.assert_array_equal(model.dataset.avg, resumed.dataset.avg)
        np.testing.assert_array_equal(model.dataset.std, resumed.dataset.std)
        resumed.train()
        self.assertEqual(resumed.total_iters, 12)

    def test_longest_length_without_valid_windows_fails_before_training(self):
        with self.assertRaisesRegex(ValueError, 'nframes_seq=80'):
            self.fixture.model(nframes_seq_schedule=[(0, 10), (1, 80)])


if __name__ == '__main__':
    unittest.main()
