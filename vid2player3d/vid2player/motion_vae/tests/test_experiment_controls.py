"""Behavior checks for checkpoint choice, runtime root feedback and best selection."""
import copy
import json
import runpy
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from vid2player.motion_vae.metrics import pool_latent_statistics


class ExperimentControlsTests(unittest.TestCase):
    def setUp(self):
        from test_checkpoint import CheckpointTests
        self.fixture = CheckpointTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_selected_checkpoint_uses_its_own_weights_and_normalization(self):
        original = self.fixture.model()
        original.save_checkpoint('epoch_50')
        expected = copy.deepcopy(original.model.state_dict())
        expected_avg = original.avg.clone()
        with torch.no_grad():
            next(original.model.parameters()).add_(10)
        original.dataset.avg = original.dataset.avg + 99
        original.save_checkpoint('latest')
        selected = self.fixture.model(test_only=True, checkpoint='epoch_50.tar')
        for name, value in expected.items():
            torch.testing.assert_close(selected.model.state_dict()[name], value)
        torch.testing.assert_close(selected.avg, expected_avg)
        self.assertTrue(selected.loaded_checkpoint_path.endswith('epoch_50.tar'))
        with self.assertRaises(FileNotFoundError):
            self.fixture.model(test_only=True, checkpoint='missing')
        with self.assertRaisesRegex(ValueError, 'label'):
            self.fixture.model(test_only=True, checkpoint='../latest')

    def test_cli_passes_selected_label_and_rejects_training_override(self):
        script = Path(__file__).resolve().parents[1]/'train.py'
        with patch('sys.argv',['train','--exp','usopen_4','--test-only','--name','clip','--checkpoint','epoch_50']), \
                patch('vid2player.motion_vae.reconstruction.record_reconstruction') as record:
            runpy.run_path(str(script),run_name='__main__')
            self.assertEqual(record.call_args.args[0].checkpoint,'epoch_50')
        with patch('sys.argv',['train','--exp','usopen_4','--checkpoint','epoch_50']), \
                patch('sys.stderr'), self.assertRaises(SystemExit) as error:
            runpy.run_path(str(script),run_name='__main__')
        self.assertEqual(error.exception.code,2)

    def test_integrated_feedback_accumulates_metres_with_history_and_future_overlap(self):
        for axes in ([0, 1, 2], [0], [0, 2]):
            for history, futures in ((1, 1), (2, 2)):
                for mode in ('predicted', 'integrated'):
                    for regressive in (False, True):
                        with self.subTest(axes=axes, history=history, futures=futures, mode=mode, regressive=regressive):
                            size = len(axes)
                            model = self.fixture.model(train_root_feedback=mode,
                                condition_root_x_only=axes == [0], no_condition_root_y=axes == [0, 2],
                                frame_size=285+size, num_condition_frames=history, num_future_predictions=futures)
                            model.avg = torch.zeros(model.opt.frame_size)
                            model.std = torch.ones(model.opt.frame_size)
                            model.avg[:size] = 10
                            model.std[:size] = 20
                            model.avg[size:size+3] = torch.tensor([.2,.3,.4])
                            model.std[size:size+3] = torch.tensor([2.,3.,4.])
                            batch = torch.arange(6.).view(1,6,1).expand(2,6,model.opt.frame_size).clone()
                            conditions = []
                            def feed(truth, condition, *args):
                                conditions.append(condition.clone())
                                output = torch.full_like(truth, 99.)
                                output[:,:,size:size+3] = torch.tensor([1.,2.,3.])
                                return (output, None, torch.zeros(2,2), torch.zeros(2,2)), {'recon':torch.tensor(0.)}
                            with patch.object(model, 'feed_vae', side_effect=feed), patch.object(model, '_optimize_loss'):
                                model.train_batch({'feature':batch},regressive=regressive)
                            for step, condition in enumerate(conditions):
                                if not regressive or step == 0:
                                    expected = batch[:,history-1+step,:size]
                                elif mode == 'predicted':
                                    expected = torch.full((2,size),99.)
                                else:
                                    delta = torch.tensor([2.2,6.3,12.4])[axes]
                                    expected = batch[:,history-1,:size] + step*delta/20
                                torch.testing.assert_close(condition[:,-1,:size],expected)
                            self.assertTrue(all(not c.requires_grad for c in conditions))

    def test_best_uses_complete_finite_world_error_and_restores_selection_on_resume(self):
        with patch('vid2player.motion_vae.base.SummaryWriter'):
            model = self.fixture.model(no_log=False)
            def report(score, step=3, failed=0, frames=100):
                return dict(train_step=step, sampled_windows=step*2,epoch=.5,
                    summary={'window_100/autoregressive':dict(world_mpjpe_m_mean=score,
                        failed_rollouts=failed,predicted_frames=frames,recon_mean=0.)})
            model.consider_best_validation(report(2.))
            path = Path(model.checkpoint_dir)/'best.tar'
            saved = path.read_bytes()
            for item in (report(3.),report(2.),report(float('nan')),report(1.,failed=1),report(1.,frames=0),report(1.,step=0)):
                model.consider_best_validation(item)
                self.assertEqual(path.read_bytes(),saved)
            model.consider_best_validation(report(1.))
            ckpt = torch.load(path,map_location='cpu')
            self.assertEqual(ckpt['best_validation']['value'],1.)
            self.assertEqual(ckpt['total_iters'],6)
            self.assertEqual(ckpt['train_steps'],3)
            self.assertTrue(ckpt['inference_only'])
            selected = self.fixture.model(test_only=True,checkpoint='best')
            self.fixture.assert_weights_equal(model,selected)
            model.save_checkpoint('latest')
            resumed = self.fixture.model(continue_train=True,no_log=False)
            self.assertEqual(resumed.best_validation,model.best_validation)
            with self.assertRaisesRegex(ValueError,'train_root_feedback changed'):
                self.fixture.model(continue_train=True,train_root_feedback='integrated')
            with self.assertRaisesRegex(ValueError,'best_validation_metric changed'):
                self.fixture.model(continue_train=True,best_validation_metric='local_mpjpe_m_mean')

    def test_training_saves_best_at_periodic_and_final_validation(self):
        from test_validation import ValidationTests
        fixture=ValidationTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        with patch('vid2player.motion_vae.base.SummaryWriter'):
            model=fixture.model(no_log=False,validation_freq_steps=1)
            model.train()
            checkpoint=torch.load(Path(model.checkpoint_dir)/'best.tar',map_location='cpu')
            self.assertEqual(checkpoint['best_validation']['metric'],'world_mpjpe_m_mean')
            self.assertGreater(checkpoint['train_steps'],0)
            latest=torch.load(Path(model.checkpoint_dir)/'latest.tar',map_location='cpu')
            self.assertEqual(latest['best_validation'],model.best_validation)
            final=fixture.model(no_log=False,model_ver='final',validation_freq_steps=3)
            final.train()
            best=torch.load(Path(final.checkpoint_dir)/'best.tar',map_location='cpu')
            self.assertEqual(best['train_steps'],2)
            latest=torch.load(Path(final.checkpoint_dir)/'latest.tar',map_location='cpu')
            self.assertEqual(latest['best_validation'],final.best_validation)

    def test_phase_probability_delays_and_ramps_and_handles_zero_exactly(self):
        model=self.fixture.model(mixed_phase_schedule=[(.2,0),(.4,1)],n_epochs=250,n_epochs_decay=250)
        self.assertEqual([model.phase_batch_probability(e) for e in (0,99,100,150,200,499)],[0,0,0,.5,1,1])
        with patch('numpy.random.rand',return_value=0.):
            self.assertFalse(model.schedual_mixed_phase(0))
        with patch('numpy.random.rand',return_value=.999):
            self.assertTrue(model.schedual_mixed_phase(200))
        with self.assertRaisesRegex(ValueError,'mixed_phase_schedule'):
            self.fixture.model(mixed_phase_schedule=[(.4,0),(.2,1)])

    def test_phase_gate_preserves_motion_updates_and_label_diagnostics(self):
        disabled = self.fixture.model(predict_phase=True)
        zero_weight = self.fixture.model(predict_phase=True)
        zero_weight.model.load_state_dict(disabled.model.state_dict())
        zero_weight.opt.weights['recon_phase'] = 0
        batch = dict(self.fixture.batch,
                     phase=torch.tensor([0., 1.]).expand(2, 3, 2).clone(),
                     phase_mask=torch.ones(2, 3, dtype=torch.bool))
        torch.manual_seed(123)
        disabled_losses = disabled.train_batch(batch, train_phase=False)
        torch.manual_seed(123)
        reference_losses = zero_weight.train_batch(batch, train_phase=True)
        self.fixture.assert_weights_equal(disabled, zero_weight)
        self.assertNotIn('recon_phase', disabled_losses)
        self.assertEqual(reference_losses['recon_phase'], 0.)
        self.assertEqual(disabled_losses['_phase_labelled_frames'], 4)
        for key in ('recon', 'kl', 'phase_mse', 'phase_angular_error_rad'):
            self.assertEqual(disabled_losses[key], reference_losses[key])
        torch.manual_seed(123)
        enabled_losses = disabled.train_batch(batch, train_phase=True)
        self.assertGreater(enabled_losses['recon_phase'], 0.)
        self.assertTrue(torch.equal(batch['phase_mask'], torch.ones(2, 3, dtype=torch.bool)))

    def test_phase_schedule_gates_loss_on_original_batches_without_second_loader(self):
        for probability in (0, 1):
            with self.subTest(probability=probability):
                model = self.fixture.model(predict_phase=True,
                    mixed_phase_schedule=[(0, probability), (1, probability)],
                    n_epochs=1, n_epochs_decay=0)
                batch = dict(self.fixture.batch,
                    phase=torch.tensor([0., 1.]).expand(2, 3, 2).clone(),
                    phase_mask=torch.ones(2, 3, dtype=torch.bool))
                model.trainset = [batch]
                self.assertFalse(hasattr(model, 'trainset_no_phase'))
                with patch.object(model, 'train_batch', return_value={'recon': 0.}) as train:
                    model.train()
                self.assertIs(train.call_args.args[0], batch)
                self.assertEqual(train.call_args.kwargs['train_phase'], bool(probability))

    def test_latent_pool_uses_weighted_second_moments_not_mean_of_rms(self):
        stats=pool_latent_statistics([
            dict(frames=1,mu_mean_per_dim=[2.,0.],mu_rms_per_dim=[2.,0.],posterior_std_mean_per_dim=[1.,1.],posterior_variance_mean_per_dim=[1.,1.]),
            dict(frames=3,mu_mean_per_dim=[-2.,4.],mu_rms_per_dim=[2.,4.],posterior_std_mean_per_dim=[2.,2.],posterior_variance_mean_per_dim=[4.,4.])])
        np.testing.assert_allclose(stats['mu_mean_per_dim'],[-1.,3.])
        np.testing.assert_allclose(stats['mu_rms_per_dim'],[2.,np.sqrt(12)])
        np.testing.assert_allclose(stats['mu_std_per_dim'],[np.sqrt(3),np.sqrt(3)])
        self.assertAlmostEqual(stats['mu_rms'],np.sqrt(8))
        self.assertEqual(stats['posterior_std_mean'],1.75)
        np.testing.assert_allclose(stats['z_rms_per_dim'],np.sqrt([7.25,15.25]))
        self.assertAlmostEqual(stats['z_rms'],np.sqrt(11.25))
        self.assertIsNone(pool_latent_statistics([]))


if __name__ == '__main__':
    unittest.main()
