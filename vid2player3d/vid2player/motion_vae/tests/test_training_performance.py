"""Regression coverage for cached windows, masked metrics, and throughput logging."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator



class TrainingPerformanceTests(unittest.TestCase):
    def setUp(self):
        from test_validation import ValidationTests
        self.fixture = ValidationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)

    def test_cache_matches_uncached_windows_and_refreshes_normalization(self):
        for root_axes, frame_size in (({}, 288), ({'condition_root_x_only': True}, 286),
                                      ({'no_condition_root_y': True}, 287)):
            model = self.fixture.model(**root_axes, frame_size=frame_size, validation_num_videos=0)
            cached = model.dataset
            uncached = copy.deepcopy(cached)
            uncached.opt.precompute_normalized_features = False
            uncached.set_normalization_stats(cached.avg, cached.std)
            self.assertIsNone(uncached.normalized_features)
            for length in (3, 7):
                cached.init_rollouts(length)
                for start in cached.rollouts:
                    np.testing.assert_array_equal(cached.get_window(start, length)['feature'],
                                                  uncached.get_window(start, length)['feature'])
            self.assertEqual(cached.normalized_features.dtype, np.float32)
            changed_avg = cached.avg + .75
            cached.set_normalization_stats(changed_avg, cached.std * 2)
            uncached.set_normalization_stats(changed_avg, uncached.std * 2)
            start = cached.rollouts[-1]
            np.testing.assert_array_equal(cached.get_window(start, 7)['feature'],
                                          uncached.get_window(start, 7)['feature'])

    def test_phase_metrics_match_labelled_reference_with_nan_placeholders(self):
        model = self.fixture.model(predict_phase=True)
        feature = torch.zeros(2, 4, 288)
        phase = torch.randn(2, 4, 2)
        mask = torch.tensor([[False, True, False, True], [False, False, True, False]])
        phase[~mask] = float('nan')
        prediction = torch.tensor([.6, .8]).expand(2, 1, 2)
        def feed(truth, condition, *args):
            return (truth, prediction, torch.zeros(2, 2), torch.zeros(2, 2)), {'recon': torch.tensor(0.)}
        with patch.object(model, 'feed_vae', side_effect=feed), patch.object(model, '_optimize_loss') as update:
            result = model.train_batch({'feature': feature, 'phase': phase, 'phase_mask': mask})
        target = phase[:, 1:][mask[:, 1:]]
        pred = prediction[0, 0].expand_as(target)
        angle = torch.atan2(pred[:, 0], pred[:, 1]) - torch.atan2(target[:, 0], target[:, 1])
        self.assertEqual(result['_phase_labelled_frames'], 3)
        self.assertAlmostEqual(result['phase_mse'], (pred-target).square().mean().item(), places=6)
        self.assertAlmostEqual(result['phase_angular_error_rad'],
                               torch.atan2(angle.sin(), angle.cos()).abs().mean().item(), places=6)
        self.assertEqual(update.call_count, 3)
        with patch.object(model, 'feed_vae', side_effect=feed), patch.object(model, '_optimize_loss'):
            result = model.train_batch({'feature': feature, 'phase': phase,
                                        'phase_mask': torch.zeros_like(mask)})
        self.assertEqual(result['_phase_labelled_frames'], 0)
        self.assertNotIn('phase_mse', result)

    def test_throughput_is_one_positive_tensorboard_metric(self):
        model = self.fixture.model(no_log=False, validation_num_videos=0)
        self.addCleanup(model.writer.close)
        model.train()
        model.writer.flush()
        events = EventAccumulator(model.checkpoint_dir + '/logs').Reload()
        performance = [tag for tag in events.Tags()['scalars'] if 'per_second' in tag]
        self.assertEqual(performance, ['train/predicted_frames_per_second'])
        self.assertGreater(events.Scalars(performance[0])[0].value, 0)
