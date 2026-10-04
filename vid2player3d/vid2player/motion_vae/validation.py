"""Fixed individual-video holdout and deterministic posterior reconstruction evaluation."""
import copy
import hashlib
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch

from vid2player.motion_vae.dataset import Video3DPoseDataset
from vid2player.motion_vae.metrics import summarize_rows
from vid2player.motion_vae.monitoring import validation_scalars


def prepare_validation_split(opt, resumed_split=None):
    """Choose individual manifest videos; checkpoints preserve the split on resume."""
    count = opt.validation_num_videos
    if not isinstance(count, int) or count < 0:
        raise ValueError('validation_num_videos must be a nonnegative integer.')
    if count == 0:
        if resumed_split:
            raise ValueError('Cannot disable a checkpoint\'s validation holdout on resume.')
        return None
    for key in ('validation_freq_steps', 'validation_window_frames', 'validation_windows_per_video'):
        if not isinstance(getattr(opt, key), int) or getattr(opt, key) < 1:
            raise ValueError(key + ' must be a positive integer.')
    from vid2player.motion_vae.reconstruction import feature_slices
    feature_slices(opt)  # Fail before training for unsupported feature layouts.
    if opt.metric_stationary_speed_mps <= 0 or opt.metric_fast_wrist_speed_mps <= 0:
        raise ValueError('Motion metric speed thresholds must be positive.')
    manifest_path = Path(opt.dataset_dir) / 'manifest.json'
    signature = dict(manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                     num_videos=count, seed=opt.validation_seed,
                     split_unit='manifest_video',
                     filters={k: getattr(opt, k) for k in ('sport', 'gender', 'background',
                         'player_name', 'player_handness', 'side', 'split_annotation',
                         'database_ratio', 'predict_phase', 'mixed_phase_schedule')})
    signature = json.loads(json.dumps(signature))
    # Phase-free pool includes weakly annotated data used by mixed-phase training.
    probe_opt = copy.deepcopy(opt)
    probe_opt.predict_phase = False
    probe_opt.database_ratio = 1.0
    python_state = random.getstate()
    try:
        probe = Video3DPoseDataset(probe_opt)
    finally:
        random.setstate(python_state)
    names = [v['name'] for v in probe.manifest]
    if len(names) != len(set(names)):
        raise ValueError('Manifest video names must be unique for individual-video validation.')
    videos = sorted({s['_video_name'] for s in probe.sequences})
    if count >= len(videos):
        raise ValueError('validation_num_videos={} leaves no training videos ({} eligible videos).'
                         .format(count, len(videos)))
    held_out = sorted(random.Random(opt.validation_seed).sample(videos, count))
    split = dict(signature=signature, validation_videos=held_out,
                 training_videos=[v for v in videos if v not in held_out])
    if opt.continue_train and resumed_split is None:
        raise ValueError('This checkpoint has no held-out split. Start a new model_ver for honest validation.')
    if resumed_split is not None and resumed_split != split:
        raise ValueError('Validation split/config/manifest changed; use a new model_ver.')
    logging.info('Held-out individual videos (%d): %s', count, ', '.join(held_out))
    return split


def valid_segments(dataset):
    """Never concatenate across clips or across invalid gaps."""
    segments = []
    for seq in dataset.sequences:
        base, length = seq['base'], seq['length']
        valid = dataset.valid_arr[base:base + length].astype(bool)
        boundaries = np.flatnonzero(np.diff(np.r_[False, valid, False]))
        for lo, hi in zip(boundaries[::2], boundaries[1::2]):
            if hi - lo > dataset.opt.num_condition_frames + 1:
                segments.append(dict(seq=seq, start=base + int(lo), length=int(hi - lo - 1),
                                     whole_sequence=bool(lo == 0 and hi == length)))
    return segments


class MotionValidation:
    def __init__(self, opt, videos):
        self.opt = opt
        data_opt = copy.deepcopy(opt)
        data_opt.predict_phase = False
        data_opt.database_ratio = 1.0
        python_state = random.getstate()
        try:
            self.dataset = Video3DPoseDataset(data_opt, split='validation', validation_videos=videos)
        finally:
            random.setstate(python_state)
        self.full = valid_segments(self.dataset)
        self.windows = []
        length = opt.num_condition_frames + opt.validation_window_frames
        for video in videos:
            candidates = [(segment, offset) for segment in self.full
                          if segment['seq']['_video_name'] == video
                          for offset in range(max(0, segment['length'] - length + 1))]
            if not candidates:
                raise ValueError('Held-out video {} has no valid {}-prediction window.'
                                 .format(video, opt.validation_window_frames))
            indices = np.linspace(0, len(candidates) - 1,
                                  min(opt.validation_windows_per_video, len(candidates)), dtype=int)
            for index in indices:
                segment, offset = candidates[index]
                self.windows.append(dict(segment, start=segment['start'] + offset, length=length,
                                         whole_sequence=False))

        self.reference_stats = self._reference_stats()

    def _reference_stats(self):
        """Compute fixed dataset facts without model predictions or failure masks."""
        from vid2player.motion_vae.reconstruction import feature_slices
        fields = feature_slices(self.opt)
        coverage = dict(selected_sequences=len(self.dataset.sequences),
                        selected_frames=sum(s['length'] for s in self.dataset.sequences),
                        invalid_frames=sum(int((~self.dataset.valid_arr[s['base']:s['base'] + s['length']].astype(bool)).sum())
                                           for s in self.dataset.sequences),
                        evaluated_full_segments=len(self.full),
                        complete_sequences=sum(s['whole_sequence'] for s in self.full),
                        fixed_windows=len(self.windows))
        stats = {'coverage': coverage}
        for scope, segments in [('window_{}'.format(self.opt.validation_window_frames), self.windows),
                                ('whole_sequence', self.full)]:
            roots, wrists = [], []
            for segment in segments:
                raw = self.dataset.get_window(segment['start'], segment['length'])['feature'][self.opt.num_condition_frames:]
                fps = float(segment['seq'].get('fps', 30))
                displacement = raw[:, fields['root_velo']]
                roots.append(np.linalg.norm(displacement[:, :2], axis=-1) * fps)
                if 'joint_velo' in fields:
                    local = raw[:, fields['joint_velo']].reshape(-1, 23, 3)[:, [19, 20]]
                    wrists.append(np.linalg.norm(local + displacement[:, None, :], axis=-1).max(axis=-1) * fps)
            root = np.concatenate(roots)
            stationary = root < self.opt.metric_stationary_speed_mps
            values = dict(rollouts=len(segments), predicted_frames=len(root),
                          stationary_frames=int(stationary.sum()), gt_root_speed_mps_mean=float(root.mean()))
            if wrists:
                wrist = np.concatenate(wrists)
                fast = wrist >= self.opt.metric_fast_wrist_speed_mps
                values.update(fast_motion_frames=int(fast.sum()), gt_wrist_speed_mps_mean=float(wrist.mean()),
                              stationary_gt_wrist_speed_mps_mean=float(wrist[stationary].mean()) if stationary.any() else None,
                              fast_motion_gt_wrist_speed_mps_mean=float(wrist[fast].mean()) if fast.any() else None)
            stats[scope] = values
        return stats

    def evaluate(self, model, step, sampled_windows, epoch):
        """No gradients or changes to mode, normalization, optimizer, or RNG streams."""
        from vid2player.motion_vae.reconstruction import reconstruct
        previous_mode = model.model.training
        python_state, numpy_state = random.getstate(), np.random.get_state()
        devices = [model.device.index if model.device.index is not None else torch.cuda.current_device()] \
            if model.device.type == 'cuda' else []
        records, buckets = [], {}
        logging.info('Starting validation at batch %d: %d fixed windows, %d full valid sequences/segments.',
                     step, len(self.windows), len(self.full))
        model.model.eval()
        try:
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                for scope, segments in [('window_{}'.format(self.opt.validation_window_frames), self.windows),
                                        ('whole_sequence', self.full)]:
                    for segment in segments:
                        seq, start, length = segment['seq'], segment['start'], segment['length']
                        raw = self.dataset.get_window(start, length)['feature']
                        roots = np.array(self.dataset.joint_pos_arr[start + 1:start + length + 1, :3])
                        first = seq['start'] + start - seq['base'] + 1
                        frames = np.arange(first, first + length)
                        fps = float(seq.get('fps', 30))
                        for mode, reset in [('teacher_forced', 1), ('autoregressive', None)]:
                            record = dict(scope=scope, mode=mode, video=seq['_video_name'],
                                          sequence=seq['_video_name'], sequence_base=seq['base'],
                                          first_source_frame=int(first), predicted_frames=length - self.opt.num_condition_frames,
                                          whole_sequence=segment['whole_sequence'], fps=fps)
                            bucket = buckets.setdefault((scope, mode), dict(rows=[], endpoints=[], failed=0, rollouts=0))
                            bucket['rollouts'] += 1
                            try:
                                result = reconstruct(model, raw, roots, frames, fps, reset,
                                                     latent_diagnostics=(mode == 'teacher_forced'))
                            except ValueError as error:
                                if 'Non-finite' not in str(error):
                                    raise
                                record['error'] = str(error)
                                bucket['failed'] += 1
                                logging.warning('Validation rollout failed: %s: %s', record['sequence'], error)
                            else:
                                rows = [r for r in result['rows'] if r['recon'] is not None]
                                record['metrics'] = summarize_rows(rows)
                                record['metrics']['root_endpoint_error_m'] = rows[-1]['root_error_m']
                                bucket['rows'].extend(rows)
                                bucket['endpoints'].append(rows[-1]['root_error_m'])
                            records.append(record)
        finally:
            model.model.train(previous_mode)
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        summaries = {}
        for (scope, mode), bucket in buckets.items():
            values = summarize_rows(bucket['rows'])
            values.update(rollouts=bucket['rollouts'], failed_rollouts=bucket['failed'],
                          root_endpoint_error_m_mean=float(np.mean(bucket['endpoints'])) if bucket['endpoints'] else None)
            summaries[scope + '/' + mode] = values
        coverage = self.reference_stats['coverage']
        report = dict(coverage=coverage, train_step=step, sampled_windows=sampled_windows, epoch=epoch,
                      semantics='Posterior mean; encoder sees target every step; integrated-root feedback. '
                                'Means pool predicted frames; endpoints average rollouts. Whole sequences are '
                                'evaluated separately across scene cuts/invalid gaps. Failed rollouts are counted '
                                'and excluded from means. Wrist errors are root-relative; speeds use displacement heads.',
                      thresholds=dict(stationary_speed_mps=self.opt.metric_stationary_speed_mps,
                                      fast_wrist_speed_mps=self.opt.metric_fast_wrist_speed_mps),
                      summary=summaries, rollouts=records)
        # Only curated aggregate metrics go to TensorBoard. Static facts live in
        # run_config.json; detailed rows/records remain in memory for callers.
        for scope_mode, values in summaries.items():
            logging.info('Validation step %d %s: recon=%s root_m=%s local_mpjpe_m=%s failed=%d/%d',
                         step, scope_mode, values.get('recon_mean'), values.get('root_error_m_mean'),
                         values.get('local_mpjpe_m_mean'), values['failed_rollouts'], values['rollouts'])
        if not model.opt.no_log:
            for tag, value in validation_scalars(summaries).items():
                model.writer.add_scalar(tag, value, sampled_windows)
            model.writer.flush()
        return report
