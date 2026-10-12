from vid2player.motion_vae.config import *
from vid2player.motion_vae.dataset import Video3DPoseDataset
from vid2player.motion_vae.model import PoseMixtureVAE
from vid2player.motion_vae.monitoring import train_tag, write_run_config
from vid2player.motion_vae.scheduling import sequence_length
from vid2player.motion_vae.loss import (
    build_recon_feature_weights, feature_squared_error,
    trajectory_loss_settings, accumulated_root_trajectory_loss, masked_phase_loss,
)
from vid2player.utils.konia_transform import quaternion_to_angle_axis
from vid2player.utils.torch_transform import rot6d_to_angle_axis

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import normalize
import time
from collections import defaultdict
import logging
import math
import re


###############################################################################
# Train pipeline
###############################################################################

class MotionVAEModel(object):

    def __init__(self, opt, device=None):
        self.opt = opt
        sequence_length(opt, 0)  # Validate before allocating/loading data.
        if opt.train_root_feedback not in ('predicted', 'integrated'):
            raise ValueError("train_root_feedback must be 'predicted' or 'integrated'.")
        if opt.train_root_feedback == 'integrated' and not {'root_pos', 'root_velo'}.issubset(opt.pose_feature):
            raise ValueError('Integrated training root feedback requires root_pos and root_velo.')
        if opt.mixed_phase_schedule is not None:
            self.phase_batch_probability(0)  # Validate before allocating/loading data.
        if device is not None:
            self.device = device
        else:
            if opt.gpu_ids and (not opt.test_only or torch.cuda.is_available()):
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
                if opt.gpu_ids and opt.test_only:
                    logging.info('CUDA unavailable or hidden; running MVAE test-only on CPU.')

        self.recon_feature_weights = build_recon_feature_weights(opt, self.device)
        self.accumulate_trajectory_loss, self.trajectory_displacement_slice = trajectory_loss_settings(opt)
        frame_size_condition = frame_size_truth = frame_size_pred = opt.frame_size
        if opt.predict_phase:
            frame_size_pred += 2
        self.model = PoseMixtureVAE(
            frame_size_condition,
            frame_size_truth,
            frame_size_pred,
            opt.latent_size,
            opt.hidden_size,
            opt.num_condition_frames,
            opt.num_future_predictions,
            opt.num_experts,
        )
        self.model.to(self.device)
        if opt.test_only: 
            self.model.eval()

        self.checkpoint_dir = os.path.join(opt.checkpoint_dir, opt.model_ver)
        self.start_epoch = 0  # Number of fully completed epochs.
        self.total_iters = 0  # Number of sampled training windows.
        self.train_steps = 0  # Completed batches; trajectory mode updates once per bounded chunk.
        self.validation_split = None
        self.best_validation = None
        if not opt.test_only and not opt.continue_train:
            if os.path.exists(self.checkpoint_dir):
                logging.warning("Checkpoint already exists!")
                # raise Exception("Checkpoint already exists!")
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            logging.info('Checkpoint saved into {}'.format(self.checkpoint_dir))
      
        if opt.test_only: 
            self.load_checkpoint(label=opt.checkpoint)
            return 

        if not opt.no_log:
            self.writer = SummaryWriter(os.path.join(self.checkpoint_dir, 'logs'))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr)
        self.old_lr = opt.lr
        self.scaler = torch.cuda.amp.GradScaler() if opt.use_amp else None

        normalization = None
        if opt.continue_train:
            normalization = self.load_checkpoint(resume=True)
            logging.info("Continue training after epoch %d (%d sampled windows)",
                         self.start_epoch, self.total_iters)
        
        from vid2player.motion_vae.validation import prepare_validation_split, MotionValidation
        self.validation_split = prepare_validation_split(opt, self.validation_split)
        held_out = self.validation_split['validation_videos'] if self.validation_split else []
        # Fail before training if the final curriculum length has no valid windows.
        maximum_length = sequence_length(opt, opt.n_epochs + opt.n_epochs_decay)
        self.dataset = Video3DPoseDataset(opt, split='train', validation_videos=held_out,
                                         nframes_seq=maximum_length)
        if not self.dataset.rollouts:
            raise ValueError('No training windows remain after dataset filtering/holdout '
                             'at nframes_seq={}.'.format(maximum_length))
        self.update_sequence_length(self.start_epoch)
        if len(self.dataset) < opt.batch_size:
            raise ValueError('nseqs must be at least batch_size.')
        if normalization is None:
            self.dataset.get_normalization_stats()
        else:
            self.dataset.set_normalization_stats(*normalization)
  
        self.avg = torch.as_tensor(self.dataset.avg, dtype=torch.float32, device=self.device)
        self.std = torch.as_tensor(self.dataset.std, dtype=torch.float32, device=self.device)
        self.validation = MotionValidation(opt, held_out) if held_out else None

        self.trainset = DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=int(opt.num_threads))
        
        if opt.softmax_future:
            self.future_weights = torch.softmax(
                torch.linspace(1, 0, opt.num_future_predictions), dim=0
            ).to(self.device)
        else:
            self.future_weights = (
                torch.ones(opt.num_future_predictions)
                .to(self.device)
                .div_(opt.num_future_predictions)
            ).to(self.device)

        write_run_config(self)

    #-------------------------------------------------- Training ------------------------------------------------------------

    def train(self):
        opt = self.opt
        total_iters = self.total_iters
        train_start_time = time.time()
        train_loss_dict = defaultdict(float)
        num_batch_phase = 0
        logged_phase_frames = 0
        last_validation_step = None
        if self.validation is not None and self.start_epoch < opt.n_epochs + opt.n_epochs_decay:
            self.validation.evaluate(self, self.train_steps, total_iters, self.start_epoch)
            last_validation_step = self.train_steps

        for epoch in range(self.start_epoch, opt.n_epochs + opt.n_epochs_decay):
            epoch_start_time = time.time()
            self.update_sequence_length(epoch)
            for _, batch_data in enumerate(self.trainset):
                train_phase = (self.schedual_mixed_phase(epoch)
                               if opt.mixed_phase_schedule else True)

                total_iters += opt.batch_size
                regressive = self.schedual_regressive_training(epoch)
                loss_dict = self.train_batch(batch_data, regressive, train_phase=train_phase)
                if opt.mixed_phase_schedule:
                    loss_dict['phase_batch_probability'] = self.phase_batch_probability(epoch)
                self.train_steps += 1
                if self.validation is not None and self.train_steps % opt.validation_freq_steps == 0:
                    validation_epoch = epoch + (total_iters - self.total_iters) / (len(self.trainset) * opt.batch_size)
                    report = self.validation.evaluate(self, self.train_steps, total_iters, validation_epoch)
                    self.consider_best_validation(report)
                    last_validation_step = self.train_steps
                num_batch_phase += 'recon_phase' in loss_dict
                labelled_count = loss_dict.pop('_phase_labelled_frames', 0)
                logged_phase_frames += labelled_count

                for k, v in loss_dict.items():
                    train_loss_dict[k] += v * labelled_count if k in ('phase_mse', 'phase_angular_error_rad') else v

                if total_iters % opt.log_freq == 0:
                    for k, v in train_loss_dict.items():
                        if k in ('phase_mse', 'phase_angular_error_rad'):
                            train_loss_dict[k] = v / max(1, logged_phase_frames)
                        elif k != 'recon_phase':
                            train_loss_dict[k] = v / (opt.log_freq / opt.batch_size)
                        else:
                            train_loss_dict[k] = v / max(1, num_batch_phase)

                    # Report the active length, even if a log interval spans epochs.
                    train_loss_dict['nframes_seq'] = self.dataset.nframes_seq
                    # dump loss to log
                    if not opt.no_log:
                        for k, v in train_loss_dict.items():
                            self.writer.add_scalar(train_tag(k), v, total_iters)
                    # print loss to console
                    self.print_loss(epoch + 1, total_iters, time.time() - train_start_time, train_loss_dict)
                    train_loss_dict = defaultdict(float)
                    num_batch_phase = 0
                    logged_phase_frames = 0

            # Store the learning rate and progress for the next epoch, so a
            # resumed run follows the same LR and curriculum schedules.
            if epoch > opt.n_epochs:
                self.update_learning_rate()
            self.start_epoch = epoch + 1
            self.total_iters = total_iters

            # save latest model
            if not opt.no_log:
                self.save_checkpoint(label='latest')
                # save checkpoint
                if (epoch + 1) % opt.save_freq_epoch == 0:
                    logging.info('Saving the latest model (epoch %d, total_iters %d)\n' % (epoch + 1, total_iters))
                    self.save_checkpoint(label='epoch_{}'.format(epoch + 1))

            logging.info('End of epoch %d / %d \t Time Taken: %d sec\n' % (
                epoch + 1, opt.n_epochs + opt.n_epochs_decay, time.time() - train_start_time))

        if self.validation is not None and last_validation_step is not None and last_validation_step != self.train_steps:
            report = self.validation.evaluate(self, self.train_steps, total_iters, self.start_epoch)
            self.consider_best_validation(report)
            if not opt.no_log:
                self.save_checkpoint(label='latest')  # Persist final best-selection state too.
        if not opt.no_log:
            self.writer.close()
        logging.info('End of training Time Taken: %d sec\n' % (time.time() - train_start_time))


    def train_batch(self, batch_data, regressive=False, train_phase=True):
        """Always learn motion; optionally omit phase loss without changing the batch."""
        opt = self.opt
        B, L, F = batch_data['feature'].shape
        T = opt.num_condition_frames
        S = opt.num_future_predictions
        batch_feature = batch_data['feature'].float().to(self.device)
        batch_phase = None
        if batch_data.get('phase') is not None:
            batch_phase = batch_data['phase'].float().to(self.device)
        batch_phase_mask = None
        if batch_data.get('phase_mask') is not None:
            batch_phase_mask = batch_data['phase_mask'].bool().to(self.device)
        phase_error_sum, phase_angle_sum, phase_count, phase_total = 0., 0., 0, 0
        batch_action = None
        if batch_data.get('action') is not None:
            batch_action = batch_data['action'].float().to(self.device)

        self.model.train()
        self.optimizer.zero_grad()
        loss_dict_seq = defaultdict(float)
        if self.accumulate_trajectory_loss:
            window_losses, predicted_displacements, target_displacements = [], [], []
            trajectory_loss_sum = 0.0
        predicted_steps = L - S - T + 1
        if self.accumulate_trajectory_loss and predicted_steps < 1:
            raise ValueError('Trajectory training requires at least one predicted step per window.')

        condition = batch_feature[:, :T].clone()

        for i in range(T-1, L - S):
            # set input
            if i >= T:
                previous_root = condition[:, -1, :self._root_condition_size()].clone()
                condition = condition.roll(-1, dims=1)
                condition[:, -1].copy_(output[:, 0].detach() if regressive else batch_feature[:, i])
                if regressive and opt.train_root_feedback == 'integrated':
                    size = self._root_condition_size()
                    axes = [0] if opt.condition_root_x_only else [0, 2] if opt.no_condition_root_y else [0, 1, 2]
                    # All values in the training buffer are normalized. Integrate
                    # raw metres, then normalize root only; other outputs remain
                    # unchanged and detached as in the original feedback rule.
                    root = previous_root * self.std[:size] + self.avg[:size]
                    displacement = output[:, 0, size:size+3].detach() * self.std[size:size+3] + self.avg[size:size+3]
                    condition[:, -1, :size] = (root + displacement[:, axes] - self.avg[:size]) / self.std[:size]

            gt_feature = batch_feature[:, i+1:i+1+S]
            if batch_phase is not None:
                gt_phase = batch_phase[:, i+1:i+1+S]
            else:
                gt_phase = None
            if batch_action is not None:
                gt_action = batch_action[:, i+1:i+1+S]
            else:
                gt_action = None

            # forward pass
            with torch.cuda.amp.autocast(enabled=opt.use_amp):
                (output, output_phase, mu, logvar), loss_dict = self.feed_vae(gt_feature, condition, gt_phase, gt_action,
                    None if batch_phase_mask is None else batch_phase_mask[:, i+1:i+1+S])
            if not train_phase:
                loss_dict.pop('recon_phase', None)
            with torch.no_grad():
                loss_dict_seq['latent_mu_rms'] += mu.square().mean().sqrt().item()
                loss_dict_seq['latent_posterior_std_mean'] += (.5 * logvar).exp().mean().item()
                loss_dict_seq['latent_z_rms'] += (mu.square() + logvar.exp()).mean().sqrt().item()

            if opt.predict_phase:
                phase_total += B * S
                if gt_phase is not None:
                    mask = torch.ones((B, S), dtype=torch.bool, device=self.device) if batch_phase_mask is None else batch_phase_mask[:, i+1:i+1+S]
                    with torch.no_grad():
                        pred, target = output_phase[mask].float(), gt_phase[mask].float()
                        phase_count += int(mask.sum())
                        phase_error_sum += float((pred - target).square().mean(dim=-1).sum())
                        angle = torch.atan2(pred[:, 0], pred[:, 1]) - torch.atan2(target[:, 0], target[:, 1])
                        phase_angle_sum += float(torch.atan2(torch.sin(angle), torch.cos(angle)).abs().sum())

            if self.accumulate_trajectory_loss:
                window_losses.append(sum(loss_dict.values()))
                # Only the first prediction advances the autoregressive window;
                # subsequent future predictions overlap with the next step.
                indices = self.trajectory_displacement_slice
                predicted_displacements.append(output[:, 0, indices])
                target_displacements.append(gt_feature[:, 0, indices])
            else:
                self._optimize_loss(sum(loss_dict.values()))

            for k, v in loss_dict.items():
                loss_dict_seq[k] += v.item()

            if self.accumulate_trajectory_loss and (
                    len(window_losses) == opt.trajectory_window_frames or i == L - S - 1):
                trajectory = accumulated_root_trajectory_loss(
                    torch.stack(predicted_displacements, dim=1),
                    torch.stack(target_displacements, dim=1),
                    self.std[self.trajectory_displacement_slice])
                weighted_trajectory = trajectory * opt.weights['trajectory']
                self._optimize_loss(torch.stack(window_losses).mean() + weighted_trajectory)
                # Log a frame-weighted mean, including the shorter final chunk.
                trajectory_loss_sum += weighted_trajectory.item() * len(window_losses)
                window_losses, predicted_displacements, target_displacements = [], [], []

        for k, v in loss_dict_seq.items():
            loss_dict_seq[k] = v / predicted_steps
        if self.accumulate_trajectory_loss:
            loss_dict_seq['trajectory'] = trajectory_loss_sum / predicted_steps
        if opt.predict_phase:
            loss_dict_seq['_phase_labelled_frames'] = phase_count
            loss_dict_seq['phase_label_fraction'] = phase_count / max(1, phase_total)
            if phase_count:
                loss_dict_seq['phase_mse'] = phase_error_sum / phase_count
                loss_dict_seq['phase_angular_error_rad'] = phase_angle_sum / phase_count
            else:
                loss_dict_seq.pop('recon_phase', None)
        return loss_dict_seq

    def _root_condition_size(self):
        if 'root_pos' not in self.opt.pose_feature:
            return 0
        return 1 if self.opt.condition_root_x_only else 2 if self.opt.no_condition_root_y else 3

    def consider_best_validation(self, report):
        """Keep the exact validation weights; failed/empty rollouts cannot win."""
        if self.opt.no_log or report['train_step'] == 0:
            return
        key = 'window_{}/autoregressive'.format(self.opt.validation_window_frames)
        values = report['summary'][key]
        metric = self.opt.best_validation_metric
        score = values.get(metric)
        if (values['failed_rollouts'] or not values['predicted_frames']
                or score is None or not math.isfinite(score)):
            return
        if self.best_validation is not None:
            if self.best_validation['metric'] != metric:
                raise ValueError('best_validation_metric changed on resume; use a new model_ver.')
            if score >= self.best_validation['value']:
                return
        previous_best = self.best_validation
        self.best_validation = dict(metric=metric, value=float(score),
                                    train_step=report['train_step'],
                                    sampled_windows=report['sampled_windows'], epoch=report['epoch'])
        try:
            self.save_checkpoint(label='best', validation_progress=report)
        except Exception:
            self.best_validation = previous_best
            raise
        logging.info('Saved best MVAE: %s=%g at batch %d', metric, score, report['train_step'])

    def _optimize_loss(self, loss_total):
        self.optimizer.zero_grad()
        if self.opt.use_amp:
            self.scaler.scale(loss_total).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss_total.backward()
            self.optimizer.step()
    

    def feed_vae(self, ground_truth, condition, gt_phase=None, gt_action=None, gt_phase_mask=None):
        opt = self.opt
        condition = condition.flatten(start_dim=1, end_dim=2)
        flattened_truth = ground_truth.flatten(start_dim=1, end_dim=2)
        if gt_action is not None:
            condition = torch.cat([condition, gt_action.flatten(start_dim=1, end_dim=2)], dim=1)

        output, mu, logvar = self.model(flattened_truth, condition)
        output_phase = None
        if not opt.predict_phase:
            output = output.view(-1, opt.num_future_predictions, opt.frame_size)
        else:
            output = output.view(-1, opt.num_future_predictions, opt.frame_size + 2)
            output_phase = output[:, :, -2:]
            output = output[:, :, :-2]

        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum().clamp(max=0)
        kl_loss /= logvar.numel()

        recon_feature_loss = feature_squared_error(
            output, ground_truth, self.recon_feature_weights).mean(dim=(0, -1))
        recon_feature_loss = recon_feature_loss.mul(self.future_weights).sum()

        if opt.predict_phase and gt_phase is not None:
            recon_phase_loss = masked_phase_loss(output_phase, gt_phase, gt_phase_mask, self.future_weights)

        loss_dict = {
            'recon': recon_feature_loss * self.opt.weights['recon'],
            'kl': kl_loss * self.opt.weights['kl'],
        }
        if opt.predict_phase and gt_phase is not None:
            loss_dict['recon_phase'] = recon_phase_loss * self.opt.weights['recon_phase']
        return (output, output_phase, mu, logvar), loss_dict


    def save_checkpoint(self, label=None, validation_progress=None):
        checkpoint = {
            'format_version': 2,
            'train_steps': self.train_steps,
            'validation_split': self.validation_split,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'next_epoch': self.start_epoch,
            'total_iters': self.total_iters,
            'old_lr': self.old_lr,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler is not None else None,
            'avg': torch.as_tensor(self.dataset.avg),
            'std': torch.as_tensor(self.dataset.std),
            'best_validation': self.best_validation,
            'train_root_feedback': self.opt.train_root_feedback,
        }
        if validation_progress is not None:
            epoch = validation_progress['epoch']
            checkpoint.update(next_epoch=int(epoch), total_iters=validation_progress['sampled_windows'],
                              train_steps=validation_progress['train_step'],
                              inference_only=True, validation_epoch=epoch)
            # Validation may occur mid-epoch, before LR/counter bookkeeping.
            # Preserve its exact weights for inference, resume from latest.
        checkpoint_path = os.path.join(self.checkpoint_dir, '{}.tar'.format(label))
        # An interrupted write must not destroy the previous complete checkpoint.
        temporary_path = checkpoint_path + '.tmp'
        try:
            torch.save(checkpoint, temporary_path)
            os.replace(temporary_path, checkpoint_path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
        # Retain standalone statistics for existing tooling. New checkpoints
        # also embed them, keeping weights and normalization paired per epoch.
        np.save(os.path.join(self.checkpoint_dir, 'avg.npy'), self.dataset.avg)
        np.save(os.path.join(self.checkpoint_dir, 'std.npy'), self.dataset.std)


    def load_checkpoint(self, label='latest', resume=False):
        label = label[:-4] if label.endswith('.tar') else label
        if re.fullmatch(r'[A-Za-z0-9_-]+', label) is None:
            raise ValueError('Checkpoint must be a label such as latest, best, or epoch_50.')
        model_state_dict_path = os.path.join(self.checkpoint_dir, '{}.tar'.format(label))
        self.loaded_checkpoint_path = model_state_dict_path
        checkpoint = torch.load(model_state_dict_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            avg = checkpoint['avg'].cpu().numpy()
            std = checkpoint['std'].cpu().numpy()
            if resume:
                if checkpoint.get('inference_only'):
                    raise ValueError('Validation best checkpoint is for inference; resume from latest or an epoch checkpoint.')
                if checkpoint.get('train_root_feedback', 'predicted') != self.opt.train_root_feedback:
                    raise ValueError('train_root_feedback changed on resume; use a new model_ver.')
                self.best_validation = checkpoint.get('best_validation')
                if self.best_validation and self.best_validation['metric'] != self.opt.best_validation_metric:
                    raise ValueError('best_validation_metric changed on resume; use a new model_ver.')
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['next_epoch']
                self.total_iters = checkpoint['total_iters']
                self.train_steps = checkpoint.get('train_steps', self.total_iters // self.opt.batch_size)
                self.validation_split = checkpoint.get('validation_split')
                self.old_lr = checkpoint['old_lr']
                if self.scaler is not None and checkpoint['scaler_state_dict']:
                    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        else:
            # Legacy checkpoints contain model weights only.
            self.model.load_state_dict(checkpoint)
            avg = np.load(os.path.join(self.checkpoint_dir, 'avg.npy'))
            std = np.load(os.path.join(self.checkpoint_dir, 'std.npy'))
            if resume:
                logging.warning(
                    "Legacy weights-only MVAE checkpoint: optimizer and epoch state "
                    "are unavailable. Warm-starting with a fresh optimizer and "
                    "epoch/sample counters at zero; schedules restart.")
        logging.info("MotionAE checkpoint loaded from {}!".format(model_state_dict_path))

        self.avg = torch.from_numpy(avg).float().to(self.device)
        self.std = torch.from_numpy(std).float().to(self.device)
        return avg, std


    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.n_epochs_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        logging.info('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr


    def update_sequence_length(self, epoch):
        """Update before iterating: DataLoader workers get the new window pool."""
        length = sequence_length(self.opt, epoch)
        if self.dataset.nframes_seq != length:
            self.dataset.init_rollouts(length)
            logging.info('Training window length at epoch %d: %d frames (%d valid windows)',
                         epoch + 1, length, len(self.dataset.rollouts))

    def schedual_regressive_training(self, epoch):
        opt = self.opt
        if opt.curriculum_schedule is None:
            return True
        l = int((opt.n_epochs + opt.n_epochs_decay) * opt.curriculum_schedule[0])
        h = int((opt.n_epochs + opt.n_epochs_decay) * opt.curriculum_schedule[1])
        thresh = min(h-l, max(0, epoch - l)) / (h - l)
        return np.random.rand() <= thresh
    

    def phase_batch_probability(self, epoch):
        (e1, t1), (e2, t2) = self.opt.mixed_phase_schedule
        opt = self.opt
        if not (0 <= e1 < e2 <= 1 and 0 <= t1 <= 1 and 0 <= t2 <= 1):
            raise ValueError('mixed_phase_schedule needs increasing epoch fractions and probabilities in [0,1].')
        l = int((opt.n_epochs + opt.n_epochs_decay) * e1)
        h = int((opt.n_epochs + opt.n_epochs_decay) * e2)
        if h <= l:
            raise ValueError('mixed_phase_schedule endpoints must span at least one epoch.')
        p = min(h-l, max(0, epoch - l)) / (h - l)
        if p == 0:
            thresh = t1
        elif p == 1:
            thresh = t2
        else:
            thresh = t1 + (t2 - t1) * p
        return thresh

    def schedual_mixed_phase(self, epoch):
        return np.random.rand() < self.phase_batch_probability(epoch)


    def print_loss(self, epoch, iter, t, loss_dict):
        message = '(epoch: %d, iters: %d, time: %d) ' % (epoch, iter, t)
        for k, v in loss_dict.items():
            if v != 0:
                message += '%s: %.5f ' % (k, v)
        logging.info(message)
    

    def normalize(self, tensor, inds=None):
        if inds is None:
            return (tensor - self.avg) / self.std
        else:
            return (tensor - self.avg[inds]) / self.std[inds]


    def unnormalize(self, tensor):
        return tensor * self.std + self.avg


    def infer_single(self, latent, condition, action=None):
        opt = self.opt
        assert latent.shape == (opt.latent_size,)
        assert condition.shape == (opt.num_condition_frames, opt.frame_size)
        frame = {}
        with torch.no_grad():
            z = latent.unsqueeze(0).to(self.device)
            c = self.normalize(condition.to(self.device)). \
                unsqueeze(0).flatten(start_dim=1, end_dim=2)
            output = self.model.sample(z, c)
            if not opt.predict_phase:
                output = output.view(opt.num_future_predictions, opt.frame_size)
            else:
                output = output.view(opt.num_future_predictions, opt.frame_size + 2)
                output_phase = output[:, -2:]
                output = output[:, :-2]
            output = self.unnormalize(output).detach().cpu()
            root_dim = 3 if 'root_pos' not in opt.pose_feature else 6
            frame = {
                'root_velo': output[0, root_dim-3 : root_dim],
                'feature': output[0]
            }
            if opt.update_joint_pos:
                frame['joint_pos'] = output[0, root_dim:root_dim+24*3]
            else:
                joint_rot_dim = root_dim
                if 'joint_pos' in opt.pose_feature:
                    joint_rot_dim += 23*3
                if 'joint_velo' in opt.pose_feature:
                    joint_rot_dim += 23*3
                
                if 'joint_rotmat' in opt.pose_feature:
                    # Need to switch from row major to column major
                    joint_rot6d = output[0, joint_rot_dim:joint_rot_dim+24*6].reshape(24, 6)
                    frame['joint_rot'] = rot6d_to_angle_axis(joint_rot6d)
                if 'joint_quat' in opt.pose_feature:
                    joint_quat = output[0, joint_rot_dim:joint_rot_dim+24*4].view(24, 4)
                    # TO CHECK: Do we need to normalize?
                    joint_quat = normalize(joint_quat, dim=-1)
                    frame['joint_rot'] = quaternion_to_angle_axis(joint_quat)
            if opt.predict_phase:
                phase = output_phase[0].detach().cpu().reshape(-1)
                phase_rad = torch.atan2(phase[0], phase[1]).numpy()
                if phase_rad < 0: phase_rad += np.pi * 2
                frame['phase'] = phase
                frame['phase_rad'] = phase_rad
                
        return frame


    def forward(self, action, condition):
        opt = self.opt
        z = action.float()
        c = condition.float() # normalized already

        # use amp to speed up inference
        with torch.cuda.amp.autocast():
            output = self.model.sample(z, c)
        output = output.float()

        output_phase = None
        if not opt.predict_phase:
            output = output.view(-1, opt.num_future_predictions, opt.frame_size)
        else:
            output = output.view(-1, opt.num_future_predictions, opt.frame_size + 2)
            output_phase = output[:, :, -2:]
            output = output[:, :, :-2]
        phase = None if output_phase is None else output_phase[:, 0]
        return output[:, 0], phase
