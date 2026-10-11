"""Masked phase supervision, CSV alignment and weak hit detection on CPU."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from vid2player.motion_vae.phase import (
    labels_between_hits, align_phase, canonical_name, project_ball,
    infer_csv_phase, COURT_POINTS, NET_CENTER, METERS_PER_PIXEL,
)
from vid2player.motion_vae.loss import masked_phase_loss
from vid2player.motion_vae.tests.options import motion_options
from vid2player.motion_vae.dataset import Video3DPoseDataset
from vid2player.motion_vae.reconstruction import reconstruct, render_frame, CourtCamera, save_report
from vid2player.motion_vae.metrics import summarize_rows


class PhaseTests(unittest.TestCase):
    def test_masked_loss_ignores_nan_and_has_zero_unlabelled_gradients(self):
        output = torch.ones(2, 2, 2, requires_grad=True)
        target = torch.tensor([[[0., 0.], [float('nan'), float('nan')]],
                               [[float('nan'), float('nan')], [3., 3.]]])
        mask = torch.tensor([[True, False], [False, True]])
        loss = masked_phase_loss(output, target, mask, torch.tensor([.75, .25]))
        self.assertAlmostEqual(loss.item(), 1.75)
        loss.backward()
        self.assertTrue(torch.isfinite(output.grad).all())
        self.assertEqual(output.grad[~mask].abs().sum().item(), 0)
        zero = masked_phase_loss(output, target, torch.zeros_like(mask), torch.tensor([.75, .25]))
        self.assertEqual(zero.item(), 0)

    def test_all_labelled_loss_matches_legacy(self):
        output, target = torch.randn(3, 4, 2), torch.randn(3, 4, 2)
        weights = torch.softmax(torch.arange(4.).float(), 0)
        expected = ((output-target).square().mean(dim=(0, -1)) * weights).sum()
        self.assertTrue(torch.allclose(masked_phase_loss(output, target, None, weights), expected))

    def test_phase_convention_gaps_boundaries_and_background(self):
        frames = np.arange(15)
        observed = np.ones(15, dtype=bool)
        observed[7] = False
        hits = [dict(frame=3, side='bg'), dict(frame=8, side='fg'), dict(frame=12, side='bg')]
        phase, mask = labels_between_hits(frames, observed, hits)
        self.assertFalse(mask[:3].any())
        self.assertFalse(mask[13:].any())
        self.assertFalse(mask[7])
        np.testing.assert_allclose(phase[3], [0, 1], atol=1e-6)
        np.testing.assert_allclose(phase[8], [0, -1], atol=1e-6)
        np.testing.assert_allclose(phase[12], [0, 1], atol=1e-6)
        bg, _ = labels_between_hits(frames, observed, hits, side='bg')
        np.testing.assert_allclose(bg[8], [0, 1], atol=1e-6)
        aligned, valid = align_phase(np.arange(5, 10), frames, phase, mask)
        np.testing.assert_array_equal(valid, mask[5:10])
        np.testing.assert_array_equal(aligned, phase[5:10])
        self.assertEqual(canonical_name('demo_0-A_vs._B.pk'), 'A_vs._B')

    def test_csv_scale_axis_reversal_and_unanswered_tail(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'clip.csv'
            # Ground-plane synthetic trajectory: near -> far -> near -> far.
            y = np.r_[np.linspace(-12, 12, 31), np.linspace(12, -12, 31)[1:],
                      np.linspace(-12, 12, 31)[1:], np.linspace(12, 6, 11)[1:]]
            rows = []
            for i, value in enumerate(y):
                ref = NET_CENTER + np.array([0., value]) / METERS_PER_PIXEL
                rows.append(dict(global_frame=i, x_ball=ref[0]*3, y_ball=ref[1]*3,
                    inv_matrix=str(np.eye(3)), court_kps=str((COURT_POINTS*3).astype(int).tolist()),
                    person_point_bottom=str(tuple(NET_CENTER+np.array([0., -12.])/METERS_PER_PIXEL)),
                    person_point_top=str(tuple(NET_CENTER+np.array([0., 12.])/METERS_PER_PIXEL)), is_bounce='False'))
            np.testing.assert_allclose(project_ball(rows[0]), [0, -12], atol=1e-6)
            rows[45]['x_ball'] = rows[45]['y_ball'] = ''
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            frames, phase, mask, info = infer_csv_phase(path)
            self.assertEqual([h['side'] for h in info['hits']], ['bg', 'fg', 'bg'])
            self.assertFalse(mask[:info['hits'][0]['frame']].any())
            self.assertFalse(mask[45])
            self.assertFalse(mask[info['hits'][-1]['frame']+1:].any())
            self.assertGreater(mask.sum(), 40)

    def test_loader_retains_weak_unlabelled_frames_and_mask(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)
            count = 14
            np.save(data/'joint_pos.npy', np.zeros((count, 72)))
            np.save(data/'joint_rot.npy', np.zeros((count, 24, 3)))
            np.save(data/'joint_rotmat.npy', np.tile(np.eye(3), (count, 24, 1, 1)))
            np.save(data/'valid.npy', np.ones(count, dtype=bool))
            phase = np.tile([0., 1.], (count, 1))
            mask = np.zeros(count, dtype=bool); mask[4:9] = True; mask[6] = False
            phase[~mask] = np.nan
            np.save(data/'phase.npy', phase); np.save(data/'phase_mask.npy', mask)
            seq = dict(base=0, start=0, length=count, beta=[0.]*10, player='Federer', handness='right')
            (data/'manifest.json').write_text(json.dumps([dict(name='clip', background='usopen', gender='mens',
                is_orig=False, sequences=dict(fg=[seq], bg=[]))]))
            opt = motion_options(predict_phase=True); opt.dataset_dir = temp
            dataset = Video3DPoseDataset(opt)
            self.assertEqual(len(dataset.sequences), 1)
            window = dataset.get_window(0, 13)
            np.testing.assert_array_equal(window['phase_mask'], mask[1:])
            self.assertTrue(np.isfinite(window['phase']).all())
            self.assertEqual(len(dataset.rollouts), 4)

    def test_converter_optional_csv_alignment_after_trimming(self):
        import joblib
        from vid2player.motion_vae import convert_export_dataset as converter
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            export = root/'demo_clip.pk'
            joblib.dump({'demo_export_0': dict(pose_aa=np.zeros((16, 72)), trans_orig=np.zeros((16, 3)), fps=30)}, export)
            (root/'motions.yaml').write_text('- file: ' + str(export))
            csv_dir = root/'csv'; csv_dir.mkdir(); (csv_dir/'clip.csv').touch()
            out = root/'dataset'
            frames = np.arange(16)
            phases = np.tile([0., 1.], (16, 1))
            mask = (frames >= 4) & (frames <= 10); mask[6] = False
            def fake_fk(**kwargs):
                n = len(kwargs['pose_aa'])
                return np.zeros((n, 72)), np.zeros((n, 24, 3)), np.tile(np.eye(3), (n, 24, 1, 1))
            argv = ['convert', '--input-yaml', str(root/'motions.yaml'), '--out-dir', str(out),
                    '--tp_dir', str(csv_dir), '--trim-start', '2', '--trim-end', '2']
            with patch('sys.argv', argv), patch.object(converter, 'compute_joint_arrays', side_effect=fake_fk), \
                 patch('vid2player.motion_vae.phase.infer_csv_phase', return_value=(frames, phases, mask, dict(hits=[]))):
                converter.main()
            np.testing.assert_array_equal(np.load(out/'phase_mask.npy'), mask[2:14])
            summary = json.loads((out/'conversion_summary.json').read_text())
            self.assertEqual(summary['num_phase_labelled_frames'], 6)
            self.assertTrue(summary['predict_phase'])
            gvhmr = root/'gvhmr'; clip = gvhmr/'clip'; clip.mkdir(parents=True)
            (clip/'0_input_video.json').write_text(json.dumps(dict(source=dict(
                path='/inputs/clip.mp4', start_frame=4, end_frame=14, fps=30))))
            mapped_out = root/'mapped_dataset'
            mapped_argv = argv.copy(); mapped_argv[mapped_argv.index('--out-dir')+1] = str(mapped_out)
            mapped_argv += ['--gvhmr_dir', str(gvhmr)]
            with patch('sys.argv', mapped_argv), patch.object(converter, 'compute_joint_arrays', side_effect=fake_fk), \
                 patch('vid2player.motion_vae.phase.infer_csv_phase', return_value=(frames, phases, mask, dict(hits=[]))):
                converter.main()
            mapped_mask = np.load(mapped_out/'phase_mask.npy')
            expected = np.zeros(12, dtype=bool); expected[:4] = True  # source frames 7..10
            np.testing.assert_array_equal(mapped_mask, expected)
            mapped_summary = json.loads((mapped_out/'conversion_summary.json').read_text())
            self.assertEqual(mapped_summary['frame_alignments'][0]['first_source_frame'], 7)
            self.assertEqual(mapped_summary['frame_alignments'][0]['out_of_range_frames'], 5)
            with patch('sys.argv', ['convert', '--input-yaml', str(root/'motions.yaml')]):
                self.assertIsNone(converter.parse_args().tp_dir)

    def test_training_and_validation_with_partial_labels(self):
        # Reuse the small motion fixture; exercise the real model/optimizer on CPU.
        from test_validation import ValidationTests
        fixture = ValidationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        data = Path(fixture.opt.dataset_dir)
        count = len(np.load(data/'valid.npy'))
        np.save(data/'phase.npy', np.tile([0., 1.], (count, 1)).astype(np.float32))
        mask = np.arange(count) % 3 == 0
        np.save(data/'phase_mask.npy', mask)
        model = fixture.model(predict_phase=True, mixed_phase_schedule=None)
        window = next(model.dataset.get_window(start, model.opt.nframes_seq)
                      for start in model.dataset.rollouts
                      if model.dataset.get_window(start, model.opt.nframes_seq)['phase_mask'][1:].any())
        batch = {key: torch.as_tensor(np.stack([value, value]))
                 for key, value in window.items() if key != 'start'}
        metrics = model.train_batch(batch)
        self.assertIn('phase_mse', metrics)
        self.assertGreater(metrics['phase_label_fraction'], 0)
        self.assertLess(metrics['phase_label_fraction'], 1)
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
        batch['phase_mask'][:] = False
        unlabelled = model.train_batch(batch)
        self.assertNotIn('recon_phase', unlabelled)
        self.assertEqual(unlabelled['phase_label_fraction'], 0)
        report = model.validation.evaluate(model, 1, 2, 0)
        self.assertTrue(any('phase_mse_mean' in values for values in report['summary'].values()))
        model.train()  # Includes a fully masked minibatch, logging and validation.

    def test_reconstruction_phase_mask_circular_error_and_overlay(self):
        opt = motion_options(predict_phase=True)
        features = np.zeros((6, opt.frame_size), dtype=np.float32)
        phases = np.tile([np.sin(.05), np.cos(.05)], (6, 1))
        mask = np.array([False, True, False, True, True, False])
        class Fake:
            def encode(self, target, condition):
                return torch.zeros(1, opt.latent_size), torch.zeros(1, opt.latent_size)
            def sample(self, latent, condition):
                return torch.cat((torch.zeros(1, opt.frame_size), torch.tensor([[np.sin(-.05), np.cos(-.05)]])), dim=-1).float()
        model = SimpleNamespace(opt=opt, model=Fake(), device=torch.device('cpu'), normalize=lambda x:x, unnormalize=lambda x:x)
        result = reconstruct(model, features, np.zeros((6, 3)), np.arange(6), 30,
                             phases=phases, phase_mask=mask, latent_diagnostics=False)
        summary = summarize_rows(result['rows'])
        self.assertEqual(summary['phase_labelled_frames'], 3)
        self.assertAlmostEqual(summary['phase_angular_error_rad_mean'], .1, places=5)
        self.assertIsNone(result['rows'][2]['phase_mse'])
        self.assertIsNotNone(result['rows'][2]['pred_phase_rad'])
        camera = CourtCamera()
        frame = render_frame(result, 2, camera, 'Phase test', None)
        self.assertEqual(frame.shape[-1], 3)
        with tempfile.TemporaryDirectory() as temp:
            save_report(Path(temp), result, dict(name='phase', resample=None))
            report = json.loads((Path(temp)/'summary.json').read_text())
            self.assertEqual(report['phase_labelled_frames'], 3)
            self.assertAlmostEqual(report['phase_angular_error_rad_mean'], .1, places=5)
            with (Path(temp)/'history.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[2]['phase_mse'], '')
            self.assertNotEqual(rows[2]['pred_phase_rad'], '')
            self.assertTrue((Path(temp)/'metrics.png').exists())


if __name__ == '__main__':
    unittest.main()
