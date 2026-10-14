# Motion VAE (MVAE)

The motion embedding learns from exported low-level imitation motions and supplies
the motion model used by the high-level tennis controller. See the
[project README](../../README.md) for environment setup and the full training pipeline.
All commands below run from `vid2player3d`; command and output paths are relative
to that directory.

## Experiment controls

Choose a reconstruction checkpoint by label. Omission means `latest`; `best` and
`epoch_50` select their own embedded normalization as well as weights. A `.tar`
suffix is accepted; paths and silent fallback to another label are not.

```bash
PYTHONPATH=. python vid2player/motion_vae/train.py --exp usopen_4 --test-only \
  --checkpoint epoch_50 \
  --name demo_Adrian_Mannarino_vs._Jiri_Lehecka_reencoded-scene000-001
```

The recorded `summary.json` identifies the checkpoint actually loaded, including
the zero and optional unit-action comparisons. `--checkpoint` applies only to
`--test-only`; training resumption continues to use `latest`.

Set these options in a new experiment entry in [config.py](config.py):

```python
'train_root_feedback': 'integrated',
'best_validation_metric': 'world_mpjpe_m_mean',
```

`train_root_feedback` controls **self-fed training** conditions:

- `predicted` (default): feed the decoder's absolute root-position output back,
  preserving previous training behavior.
- `integrated`: unnormalize the first predicted root displacement, add it to the
  previous conditioning root, and normalize that position into the next condition.
  This matches reconstruction and MVAEPlayer's integration rule. It also supports
  X-only / XZ root conditioning, multiple history frames, and overlapping future
  predictions; only the first prediction advances the condition.

Teacher-forced batches always use true previous frames. Other feedback features
and the existing detachment of self-fed predictions remain unchanged. This option
does not itself lengthen the training window or enable gradients through feedback.
The root feedback mode is stored in checkpoints; changing it on `--continue_train`
raises an error to avoid silently mixing experiments. Older checkpoints are treated
as `predicted`. Inference always uses runtime integration regardless of this option.

Training with validation enabled automatically saves `best.tar` when a complete,
finite, uninterrupted validation-window evaluation improves. With 100 predicted
frames, the default selection variable is `val_100/total_error_m`: mean Euclidean
**world-space joint error**, in metres, pooled across predicted frames and joints.
It includes accumulated root movement and local pose error together. The score
excludes phase loss, KL, and feature-normalization weights; changing those weights
does not change what physical error means. Any failed rollout disqualifies the
evaluation. Initial untrained validation is not eligible. `--no_log` retains its
existing behavior of disabling checkpoint saves.

`best_validation_metric` names a lower-is-better validation summary field; its
default is `world_mpjpe_m_mean`. Selection metadata is embedded in `best.tar` and
subsequent checkpoints, and restored on continuation. `best.tar` stores the exact
weights at validation time and is for inference: periodic evaluation can occur
mid-epoch, so resume training from `latest` or an epoch checkpoint rather than
pretending the best snapshot completed an epoch. Selection uses the configured
validation window length (normally 100), not full-sequence validation.

This is a motion-representation selection criterion, not a high-level tennis
success metric. The reconstruction encoder still sees the target each step, and
the skeleton metrics use decoded joint positions rather than physical SMPL tracking.
Useful latent action influence, plausible motion under prior/policy actions, and
ball-contact/return results need their own evaluations.

`usopen_4_5.mixed_phase_schedule` now uses `[(0.2, 0), (0.4, 1)]` with phase weight
2. With its 500-epoch budget, the probability of enabling phase loss on a
batch is zero through epoch index 100, ramps linearly to one at index 200, and
stays one afterward. A selected batch uses coefficient 2 in full; this is **batch
frequency scheduling**, not continuous loss-weight annealing. All batches learn
motion and KL on the same sampled frames, including phase-labelled frames when
phase supervision is off. The schedule gates only the phase loss; it does not
switch datasets or filter frames. Phase MSE/angular diagnostics remain available
even when phase loss is disabled, and validation still evaluates phase normally.
Log `train/phase_batch_probability` shows the actual scheduled
probability. The previous `[(0, 1), (0.5, 0.1)]` did the opposite: start with phase
on every batch, then reduce frequency to 10% at epoch 250. Fractions scale with
`n_epochs + n_epochs_decay`. This delay is epoch-based, not a detector that latent
control has become useful; check latent diagnostics before interpreting it that way.

Encoder diagnostics now include:

- Training: `train/latent_mu_rms`, `train/latent_posterior_std_mean`, and
  `train/latent_z_rms` (means of batch/step statistics).
- Validation: `diagnostics/val_100/latent/` and `diagnostics/val_full/latent/`,
  using teacher-forced inputs to measure the encoder on real motion states.
  Per-dimension cards report mean, RMS, variation of the mean across frames,
  posterior standard deviation, and expected sampled-latent RMS.
- Reconstruction: `summary.json.latent_statistics` contains the same vectors
  for the selected reset mode; `history.csv` includes per-frame mean RMS and
  posterior standard deviation. Zero/prior-action reports have no encoder statistics.

Validation moments are pooled by predicted-frame count. For each dimension,
`mu_rms = sqrt(E[mu^2])`, `mu_std = sqrt(E[mu^2] - E[mu]^2)`, and
`z_rms = sqrt(E[mu^2 + exp(logvar)])`. The last quantity describes the stochastic
latents actually used in MVAE training. A tiny posterior **mean** RMS does not
mean the decoder was trained only near zero: posterior std near one still gives
sampled latents with RMS near one. Avoid scaling high-level actions down solely
to match posterior-mean RMS. Pair these diagnostics with latent reconstruction
benefit, action-induced motion changes, and sustained controllability.

`--resample 10` replaces motion history and root with ground truth every ten
predictions. It is a useful short-horizon reconstruction diagnostic, but changing
the high-level action every ten frames does not provide those ground-truth resets.
Keep both short-horizon accuracy and 30/60/100-frame stability checks; do not force
an arbitrary prior action to reproduce a particular ground-truth future motion.

## Training

We provide code for training the motion embedding in [motion_vae](.) (assuming the motion data is organized in the format described in [Video3DPoseDataset](dataset.py)).
```bash
python embodied_pose/run.py --play --cfg general_tennis_motion_filtered --rl_device cuda:0 --headless --results_dir ./output/ --checkpoint latest --export_dataset --num_envs 32  &> export_dataset_logs.txt

python vid2player/motion_vae/convert_export_dataset.py --input-yaml exports/motion_export_motion.yaml --out-dir tennis_dataset_physics

python vid2player/motion_vae/train.py  --exp federer_physics &> mvae_logs.txt
```

## Optional phase labels from TennisProject

Add `--tp_dir` explicitly when converting exports; it has no default folder:

```bash
python vid2player/motion_vae/convert_export_dataset.py \
  --input-yaml exports/tennis_motion_2/motions.yaml \
  --out-dir exports/vae_input/tennis_motion_dataset_phase \
  --tp_dir ../TennisProject/res_2 \
  --gvhmr_dir ../GVHMR/outputs/demo_2

PYTHONPATH=..:. python -u vid2player/motion_vae/train.py --exp usopen_4
```

Set `usopen_4.dataset_dir` to the converted folder before training. `usopen_4`
enables the phase head with phase-loss weight 10 and mixed-phase scheduling off:
labelled and unlabelled frames can now coexist in one batch. Existing phase-free
checkpoints cannot resume this model because its output dimension changes.

CSV matching uses the export filename, stripping `demo_` / `0-` prefixes and
known file extensions without stripping dots inside player names. Missing CSVs
produce fully unlabelled sequences.

`--gvhmr_dir` is optional and has no default. It accepts the GVHMR output root or
one matching clip directory. The converter matches the export filename to the
clip directory, reads `0_input_video.json`'s `source.start_frame`, `end_frame`
(exclusive) and `fps`, and validates the source-video identity. Missing or
ambiguous metadata fails explicitly rather than guessing an offset.

TennisProject records input-video `global_frame` without changing temporal
sampling. `GVHMR/extraction_pipeline.sh` trims each source using the CSV range;
the JSON stores the actual retained range. Ground correction preserves those
motion frames. Low-level exports introduce another timing change: they save
**after** stepping the simulation. The normal `--export_dataset` configuration
forces a Start reset, so the first saved frame corresponds to reference frame 1
at 30 FPS. Alternate Hybrid/Random resets can start later in the reference
motion. New exports therefore include one `reference_times`
timestamp (seconds in the reference motion) per saved pose and `export_fps`.
With GVHMR metadata, the CSV frame is:

```
source.start_frame + reference_times[export_frame] * source.fps
```

Fractional source times use circular phase interpolation only when both adjacent
CSV frames have valid labels. Frames outside the JSON range remain phase-masked.
Converter `--trim-start`/`--trim-end` slice the mapping along with the poses;
a short export is never stretched to the full source-video range.

Legacy exports without timestamps assume Start reset and 30 FPS control steps:
`CSV frame = JSON start_frame + export_frame + 1`. Override
`--export-frame-offset` (default 1) if their first saved reference frame is known
to differ. Existing randomly initialized exports need that known offset or
re-exporting with timestamps; their exact reset time cannot be recovered from
GVHMR JSON alone. Mapping provenance and out-of-range counts are recorded in
`conversion_summary.json` and each manifest sequence's `frame_alignment`.

Without `--gvhmr_dir`, the previous mapping remains available:
`CSV frame = export_frame + --source-start-frame` (default 0). This manual absolute
mapping cannot be combined with `--gvhmr_dir`.
`--side fg` labels the near/bottom player; `bg` labels the far/top player.

CSV ball predictions are **image coordinates**, not a recovered 3D flight path.
The converter infers their output-image scale from court keypoints and the
homography, then projects to court coordinates using the ground-correction
convention (near player at negative Y). This ground-plane projection of an
airborne ball is approximate and is used only to detect weak hit timings.
Detection requires sustained Y reversals near a player, excluding annotated
bounces. A stationary ball beginning sustained travel toward the other side can
also supply an initial hit. It does not treat the first visible moving ball as a
confirmed hit. Detection can bridge missing observations of at most 0.2 seconds;
those missing frames still receive **no** phase supervision.

Phase is interpolated in **time** between alternating inferred hits: opponent
contact = 0/2π, controlled-player contact = π. It is not proportional to ball
position or distance travelled. Frames before the first hit, after the last
(unanswered) hit, and missing/unusable ball observations are masked. Intervals
containing longer tracking gaps, missing source frames, same-player consecutive
hits, or more than 3 seconds between hits are left unlabelled internally.
These conservative heuristics can miss or misidentify hits; inspect inferred
`hits`, detection methods and per-sequence label counts in
`conversion_summary.json` before relying on them as supervision.

The dataset adds `phase.npy` (N×2 `[sin φ, cos φ]`) and `phase_mask.npy` (N booleans),
aligned with the existing motion arrays. With `predict_phase=True`, the loader
retains all selected sequences; it also supports legacy manifest hit annotations
without extrapolating at their boundaries. Reconstruction/KL use every valid
motion frame. Phase loss uses only labelled frames and renormalizes observed
future weights; a fully unlabelled batch has no phase loss. Missing phase is
represented by the mask, not a zero angle.

Training logs `train/phase_loss`, `train/phase_mse`,
`train/phase_angular_error_rad` and `train/phase_label_fraction`. MSE/angular error
pool labelled prediction positions; coverage includes all prediction positions.
Validation reports phase loss, circular angular error and label coverage in its
main autoregressive TensorBoard groups when reference labels are available.
Reconstruction `history.csv`, `summary.json` and `metrics.png` include phase
coverage, raw/weighted phase MSE, circular angular error, and predicted/reference
phase. The video shows phase in radians and multiples of π at the top left;
reference phase reads `unavailable` on masked frames. Initial conditioning frames
have no model-predicted phase or evaluation loss.

## Named reconstruction recordings

```bash
# Record a named sequence against its MVAE reconstruction (no display required).
PYTHONPATH=..:. python -u vid2player/motion_vae/train.py --exp usopen --test_only \
  --name demo_Adrian_Mannarino_vs._Jiri_Lehecka_reencoded-scene000-000 --resample 20
```

`--test_only` (also `--test-only`) now loads `latest.tar` and records a named
sequence from the experiment's `dataset_dir`. `--name` accepts a manifest video
name, source key, or unique export filename/stem. For ambiguous names, use
`VIDEO:fg:INDEX` or `VIDEO:bg:INDEX` (zero-based sequence index). The experiment's
normal dataset filters still apply.

The test always writes `reconstruction.mp4`, `metrics.png`, `history.csv`, and
`summary.json` into a timestamped directory under
`out/motionVAE/<model_ver><suffix>/`. The MP4 overlays the orange ground truth at 35% opacity and the opaque blue
reconstruction on one tennis court, with a fixed perspective camera centered behind
the near baseline, root trails, and weighted reconstruction/KL losses in the upper
left. It uses the same PIL/ImageIO approach as EmbodiedPose's headless debug video;
no viewer, OpenGL, browser, or X server is needed. Compact `metrics.png` files show
the diagnostics relevant to each rollout; root-level `experiments.png` compares
the rollouts and explicitly labels their reset schedules.

Without `--resample`, only the initial conditioning state comes from ground truth.
`--resample N` resets the entire conditioning history and accumulated root to ground
truth after each N predictions, before predicting the next frame. `--resample 1`
is teacher forcing. Targets still enter the encoder on **every** step; the decoder
uses the posterior mean latent for repeatable reconstruction, not random prior
samples. This evaluates reconstruction with autoregressive conditioning, not a
blind forecast. Losses use the training normalization and weights; CSV includes
raw MSE and raw/weighted KL. The KL value describes the encoder posterior even
though decoding uses its mean.

The saved root position seeds playback; predicted root displacements are then
integrated as in motion generation. Displayed joints are the decoder's direct
root-relative joint-position outputs, not SMPL forward kinematics from its separate
rotation outputs. All pose features contribute to reconstruction loss, measured
before replacing the conditioning root with the integrated root. No court
recentering, ground snapping, or boundary teleporting hides drift.

By default the full sequence is recorded. Its first frame is reserved for velocity
calculation, and the following `num_condition_frames` frames seed the model and
have no evaluation loss. `--nframes N` caps recorded frames. Sequences with invalid
frames are rejected. This recorder supports the existing one-step prediction
configs with full XYZ root features and joint-position features. Phase labels are
not required for pose/KL evaluation. With a phase-enabled model, predicted phase
is recorded on every prediction, and phase loss/error are evaluated only where
reference labels are valid. `--run_test`
retains the old random-motion preview after training; its viewer/HTML flags do not
apply to `--test_only`.

## Training window length curriculum

`nframes_seq_schedule` optionally gives `(epoch_fraction, frames)` knots, where
fractions refer to `n_epochs + n_epochs_decay`. Lengths interpolate linearly,
round down to whole frames, and hold the first/last value outside the knots.
`None` preserves fixed `nframes_seq`. Knots need increasing fractions in `[0, 1]`
and nondecreasing integer lengths with room for conditioning plus a prediction.

`usopen_6_1` uses `[(0.3, 10), (0.5, 30), (0.7, 60)]`: with its 500 epochs,
it holds 10 frames for the first 150 epochs, reaches 30 at zero-based epoch 250,
and reaches 60 at epoch 350, holding 60 for the remaining epochs. To stop at 30,
use `[(0.3, 10), (0.5, 30)]` instead.

The inherited self-fed sampling curriculum finishes at epoch 100, and phase-loss
activation finishes at epoch 150. Window growth starts after both finish. Learning
rate decay starts around epoch 250; the final 150 epochs train at 60 frames with
a decreasing learning rate. These timings are a training choice to evaluate,
not a guarantee of better long-horizon stability. All schedules use epoch counts;
longer windows perform more optimizer updates per epoch in the default mode.

Run from `vid2player3d`:
```bash
PYTHONPATH=..:. python -u vid2player/motion_vae/train.py --exp usopen_6_1
# CPU-only execution (also disables AMP):
PYTHONPATH=..:. python -u vid2player/motion_vae/train.py --exp usopen_6_1 --cpu
```

Training rebuilds valid windows before each epoch; windows cannot cross sequence
boundaries or invalid frames. The longest length is checked against the filtered
training split before training starts. Normalization and validation windows stay
fixed. Resuming derives the active length from the saved next epoch, so keep the
schedule and total epoch count unchanged. TensorBoard `train/nframes_seq` logs
the active length; `run_config.json` records the schedule and initialization
statistics. Longer windows increase batch memory, work per batch, and (in the
default per-step optimizer mode) optimizer updates per batch. This extends
self-fed training while feedback between prediction steps remains detached.

## Resuming training

Resume MVAE training from the latest completed epoch with:
```bash
PYTHONPATH=. python vid2player/motion_vae/train.py --exp federer_physics --continue_train
```
Setting `'continue_train': True` in the experiment's Python options has the same
behavior. New `.tar` checkpoints bundle model weights, Adam state, the next epoch,
sampled-window count, learning rate, AMP scaler state (when enabled), and feature
normalization statistics. Resumption uses the saved normalization and continues
through the configured total `n_epochs + n_epochs_decay`; it does not add that
many new epochs. Keep the dataset, architecture, and schedule options unchanged
when continuing a run. Checkpoints are saved at epoch boundaries, so work after
the last saved epoch is repeated. Random-number and data-loader states are not
restored, so resumed sampling need not exactly match an uninterrupted run.

Older weights-only checkpoints remain usable for inference. Continuing training
from one logs a warning and starts a fresh optimizer and epoch schedule, because
those checkpoints never saved the missing state. `--no_log` also disables
checkpoint saving. Standalone `avg.npy` and `std.npy` files are still written;
new checkpoints additionally contain their own matching statistics.

## Validation and motion diagnostics

The `usopen` experiment now writes a fresh baseline to `results/motionVAE/usopen_3`.
It retains the existing model architecture, curriculum, KL weight, ten-frame
training windows, and training feedback rule. Start the new baseline without
`--continue_train`; resuming deliberately preserves a checkpoint's old statistics.

```bash
PYTHONPATH=..:. python -u vid2player/motion_vae/train.py --exp usopen > logs/mvae_logs.txt 2>&1
tensorboard --logdir results/motionVAE/usopen_3/logs --port 6006
```

Reconstruction feature weights are configurable in
`vid2player/motion_vae/config.py` through `recon_feature_weights`. The baseline
default `{}` and explicit all-one weights use the exact original MSE calculation.
Available keys are `root_pos`, `root_velo`, `joint_pos`, `joint_velo`,
`joint_rotmat`, and `joint_quat`; only enabled features can be weighted.
Unspecified features have weight 1. Each multiplier applies to squared errors
in **normalized feature space**, before averaging over all feature coordinates
and applying the existing future-frame weights and outer `weights['recon']`.
The denominator stays `frame_size`, rather than the sum of feature weights, so
increasing a root multiplier leaves local-pose and KL coefficients unchanged.

Optional accumulated root trajectory loss is controlled by
`accumulate_trajectory_loss` (default `False`), `weights['trajectory']`
(default `0`; a positive coefficient is required to enable it), and
`trajectory_window_frames` (default `9` predicted steps). `usopen_6` inherits
the current `usopen_5` feature/KL weights, enables the term with coefficient
`1.0`, and uses nine-step trajectory chunks. `weights` overrides merge across
inherited configs, so `usopen_6` only needs `{'trajectory': 1.0}`. The separate
`trajectory_loss_weight` option has been replaced by `weights['trajectory']`.
No training is started by changing these settings.

For each trajectory chunk, let `e_t` be predicted minus target root displacement
in metres. The added loss is
`weights['trajectory'] * mean_{batch,t,xyz}((sum_{j<=t} e_j)^2)`.
It penalizes **every trajectory prefix**, not just the endpoint, and recovers
metres from normalized displacement differences using the training std (the
mean cancels). It is independent of reconstruction feature weights and the
outer reconstruction weight. Both trajectories begin at the same ground-truth
root, so no absolute court-position offset enters the loss. Only the first of
each step's future predictions is accumulated, avoiding overlapping steps when
`num_future_predictions > 1`.

With the term enabled, training splits the predicted sequence into consecutive,
nonoverlapping chunks of at most `trajectory_window_frames` predictions and
performs **one combined optimizer update per chunk**. That update averages
the existing reconstruction/KL/phase losses within the chunk and adds its
trajectory penalty. All trainable parameters receive this combined gradient;
there is no separate root-only optimizer. The shorter final chunk is also
trained. Conditioning carries forward across chunks, while the accumulated
displacement error restarts at zero. Pose conditioning remains detached, as
in baseline training; this does not add backpropagation through that history.

Increasing `nframes_seq` therefore increases the rollout length without retaining
more than one trajectory chunk's network activations for backward. The input
batch itself still grows with `nframes_seq`. This bounds the additional activation
memory; it does not guarantee a particular GPU memory budget. Longer trajectory
horizons can increase drift loss and gradient magnitudes, but cumulative sums
have unit derivative and do not themselves cause recurrent vanishing gradients.
The tradeoff is that drift across chunk boundaries is not directly penalized.
Use full-rollout root error to assess it. For example, 30 predicted steps with
`trajectory_window_frames=9` produce chunks of 9, 9, 9, and 3 predictions and
four optimizer updates. The default nine-prediction training sequence still
produces one update.

With trajectory loss disabled, or `weights['trajectory']=0`, the original
per-step optimizer updates and losses are unchanged. Comparing `usopen_6` with
`usopen_5` changes the optimizer update schedule as well as the loss; improvement
would not isolate the effect of trajectory supervision alone.

Training logs the weighted term at `train/trajectory_loss`. Validation logs it
under `diagnostics/val_*/trajectory_loss`, and reports store
`trajectory_mse_m2` and `trajectory_weighted` separately from reconstruction.
Training, validation, and reconstruction use the same bounded trajectory horizon;
reconstruction resample resets also restart the penalty. The **displayed root
and root-error metrics** continue over the entire evaluation horizon independently
of chunk boundaries. The training trajectory scalar is a frame-weighted mean
over chunks, so the shorter final chunk does not receive equal logging weight
to a full chunk. Resolved options and optimizer updates per batch are saved in
`run_config.json`; reconstruction and zero/action summaries include the settings.

The MVAE paper ([§3–3.4](https://arxiv.org/pdf/2103.14274)) uses next-frame
reconstruction MSE plus KL and scheduled sampling. It does not report an
accumulated root trajectory loss ablation establishing that the loss should be
disabled. Keep this an optional experiment: consistent displacement bias can
produce drift, but tracking a particular recorded trajectory can also trade off
with generative motion diversity. Weight `1.0` is an initial setting, not a tuned
value; evaluate held-out drift, local pose, and latent controllability together.

Configure validation in `vid2player/motion_vae/config.py`:

| Option | `usopen` default | Meaning |
| --- | --- | --- |
| `validation_num_videos` | `2` | Hold out exactly this many individual manifest videos/clips; `0` disables validation. |
| `validation_seed` | `1234` | Local deterministic split seed; does not seed the training run. |
| `validation_freq_steps` | `3000` | Evaluate every N completed training **batches**, plus before training and at the end. Baseline batches update per prediction; accumulated trajectory training updates per bounded chunk. |
| `validation_window_frames` | `100` | Number of predicted frames in each short evaluation, excluding conditioning history. |
| `validation_windows_per_video` | `3` | Fixed, evenly spaced valid windows per held-out video; windows can overlap. |
| `metric_stationary_speed_mps` | `0.1` | A frame is stationary when GT horizontal root speed is below this threshold. |
| `metric_fast_wrist_speed_mps` | `2.0` | A frame is fast when either GT world-space wrist speed meets this threshold. |

The current dataset has 95 individual manifest videos/clips. Setting
`validation_num_videos=2` holds out two and leaves 93 for training. Selection uses
exact manifest video names; scene suffixes and source-match prefixes are not
collapsed. Other clips from the same match remain eligible for training.
`validation_video_regex` has been removed. Held-out names appear in the console and
`run_config.json`. The split, count, seed,
filters, and manifest identity are stored in checkpoints; changing the split on
resume is rejected. Changing dataset arrays in place is not detected, so keep a
run's dataset immutable.

Normalization now uses only selected **training** sequences and valid consecutive
frame pairs within each sequence. It excludes the first frame of each sequence,
invalid predecessors, and jumps between unrelated clips. Validation and recording
use the checkpoint's training statistics. Do not transplant new statistics into
an old model: its learned weights expect the original feature scaling.

TensorBoard uses a compact, aggregate dashboard (14 main cards for `usopen`):

| Group | Cards |
| --- | --- |
| `train/` | `recon_loss`, `kl_loss` (training-weighted losses). |
| `val_100/` | `recon_loss`, `kl_loss`, `root_error_m`, `local_joint_error_m`, `wrist_error_m`, `root_endpoint_error_m`. |
| `val_full/` | The same six metrics over complete held-out sequences. |
| `diagnostics/` | Secondary aggregate checks: one-step reconstruction, tail errors, velocity bias, root consistency, stationary/fast-motion errors, latent use, and failed rollouts. |

The main validation cards use **autoregressive conditioning** with the posterior
mean latent. One-step teacher-forced results are under `diagnostics/`, not mixed
into those means. Both validation groups pool all held-out videos; wrist error
averages the left and right wrists. If `validation_window_frames` changes, the
short-window group changes accordingly (e.g. `val_200/`). Phase-supervised training
also has `train/phase_loss`.

For everyday monitoring, collapse `diagnostics/` or use TensorBoard's tag filter
`^(train|val_100|val_full)/`. There are no per-video/sequence cards. Logging uses an
explicit allowlist rather than emitting every internal metric, so adding a new
internal diagnostic does not automatically add dashboard cards. The diagnostic
group has at most 34 additional cards; empty subgroups are omitted.

An epoch samples `nseqs=30000` windows in batches of 100, hence there are
`30000 / 100 = 300` batch steps per epoch. `validation_freq_steps=3000` therefore
means ten epochs. Set it to `30` for every tenth of an epoch, or `300` for every
epoch. TensorBoard's x-axis remains **sampled training windows**, matching
`train/*`, so 3,000 batches appear as an interval of 300,000 on that plot.
Validation never updates weights or optimizer state and restores model mode and
random-number streams. It runs without rendering videos.

Model metrics are persisted only in TensorBoard. There are no validation JSON
reports or CSV histories. A single `results/motionVAE/<model_ver>/run_config.json`
stores the resolved options, split membership, training dataset sizes, validation
coverage, expected prediction/rollout counts, stationary/fast frame counts, and
fixed ground-truth speed statistics. These facts are computed independently of
model predictions, so rollout failures do not change the stored reference data.
The file is written at startup (including resume); it is metadata, not a time
series of training measurements. `--no_log` disables TensorBoard and checkpoint
writes, but still saves this metadata and prints short validation summaries.

Fixed dataset facts used to produce flat TensorBoard lines; they now live only in
the JSON. A genuinely flat **model** measurement remains a curve: zero failed
rollouts is useful, and a flat or near-zero KL/latent effect can be diagnostically
important. Raw/weighted duplicates and redundant speed/displacement curves are
not both logged. Existing legacy event files retain their old tags; a clean run
uses the new layout without altering old event files.

"Whole sequence" means the full exported clip, with no frame cap. Invalid gaps
divide a clip into valid segments; segments too short to seed and predict are
omitted. JSON coverage counts include selected frames, invalid frames,
complete sequences, and evaluated segments. There must be at least one full-length
short window in every held-out video. Both evaluation modes encode the GT target
at every step and decode the posterior mean: neither is unconditional generation
or a physics rollout. Phase reconstruction is not evaluated.

The following detailed metrics remain available in named `--test_only` CSV/JSON
reports; TensorBoard uses the smaller selection and shorter names above. Summary
metric names below end in `_mean` unless otherwise indicated:

| Metric | Interpretation |
| --- | --- |
| `recon`, `kl`, `recon_weighted`, `kl_weighted` | Normalized feature MSE and per-latent-dimension KL; both raw and training-weighted. Validation decodes the posterior mean while training samples the posterior. |
| `recon_root_pos`, `recon_root_velo`, `recon_joint_pos`, `recon_joint_velo`, `recon_joint_rotmat` | Normalized MSE within each feature group. Overall reconstruction averages all feature coordinates, not these group means equally. |
| `root_error_m`, `root_xy_error_m` | Accumulated root distance to GT in 3D and on the court plane. |
| `root_endpoint_error_m_mean` | Mean final-frame root error across evaluated rollouts. |
| `local_mpjpe_m`, `world_mpjpe_m` | Mean error over the 23 non-root joints, relative to the root or in world coordinates. |
| `left_wrist_error_m`, `right_wrist_error_m` | Root-relative wrist errors; both sides are measured regardless of player metadata. |
| `root_displacement_error_m`, `root_velocity_error_mps` | Error in predicted per-frame displacement, and the same error multiplied by FPS. |
| `root_velocity_bias_{x,y,z}_mps` | Signed velocity errors; a persistent nonzero mean indicates directional bias. |
| `joint_velocity_error_mps` | Error in the predicted root-relative joint-displacement head, converted to m/s. |
| `root_consistency_m` | Distance between the raw predicted absolute root and the root obtained by integrating the predicted displacement from the rollout's previous root. |
| `gt_root_speed_mps`, `pred_root_speed_mps` | Horizontal root speeds. |
| `gt_wrist_speed_mps`, `pred_wrist_speed_mps` | Maximum world-space speed of the two wrists, using root plus joint displacement features. These are velocity-head diagnostics, not derivatives across reset jumps. |
| `pred_pose_wrist_speed_mps`, `pose_wrist_velocity_error_mps` | Wrist speed/error derived from changes in predicted poses, using the actual preceding conditioning pose at resets. Compare these with the velocity-head speed: the two decoder representations need not agree. |
| `recon_zero_latent`, `latent_joint_effect_m` | Reconstruction MSE with z=0 under the same condition, and change in root-relative joints versus posterior-mean z. Used for the validation latent-benefit/effect diagnostics and available directly in named recordings. |

Means pool **predicted frames**; conditioning frames are excluded. Root and joint/
wrist errors also have `_p95` tail statistics. `stationary_*` and `fast_motion_*`
report subgroup means with explicit frame counts. If a subgroup is empty no TensorBoard scalar is emitted for its means. A non-finite rollout
is reported in `diagnostics/*/{mode}_failed_rollouts` and excluded from means; always inspect this count
before comparing curves. Physical pose errors use direct joint-position outputs,
not forward kinematics from the separately predicted rotations.

Named `--test_only` recordings write compact `metrics.png` reports and a central
`experiments.png` comparing root trajectories, reference feature MSE, root/local
joint errors, root/wrist speeds, and differences from the independent zero baseline.
The per-frame CSV retains only relevant report metrics, and the MP4 overlay includes
root and local joint error in centimeters.
Run the same sequence/checkpoint with `--resample 1`, `--resample 20`, and no reset:

- One-step joint/wrist errors already large: local reconstruction is the bottleneck.
- Small one-step errors but large autoregressive errors: feedback accumulation/stability is the bottleneck.
- Small local joint error but large world/root error: trajectory drift dominates.
- On stationary frames, persistent signed velocity error explains root drift even if the video looks nearly correct with resets every frame.
- Large errors during GT wrist-speed peaks suggest poor fast-motion reconstruction; compare both wrists and inspect timing in `metrics.png`.
- Nearly identical posterior and zero-latent losses plus tiny latent joint effect suggest weak target-specific latent use on that sequence; they do not prove that all latent directions are useless.

Keep the old run and new run separate. Their normalized losses are not directly
comparable after correcting statistics, and the old run trained on the new
holdout videos. Use physical metrics and the new run's fixed validation curves as
the baseline for subsequent window-length or KL experiments.

`kl_free_bits` configures a free KL allowance in **nats per latent dimension**
(despite the technique's name). Its default, `0.0`, preserves the original KL
loss and gradients. With a positive allowance, training computes the diagonal
Gaussian KL per dimension, averages over the batch, floors each dimension at
`kl_free_bits`, then averages over dimensions and applies `weights['kl']`.
Dimensions below the floor receive no KL gradient, allowing reconstruction to
encourage latent use without a competing KL penalty in that range. This does
not force the actual posterior KL to reach the allowance.

`usopen_6_3` inherits `usopen_6_1` and enables `kl_free_bits=0.1`, an initial
tunable allowance. With 32 latent dimensions this corresponds to 3.2 nats total;
the existing per-dimension mean reduction and KL coefficient of 0.5 remain in
place. Train with `python vid2player/motion_vae/train.py -e usopen_6_3` from
`vid2player3d`. TensorBoard `train/kl_loss` shows the weighted, floored objective;
`train/kl_raw` shows the unfloored, unweighted KL mean to monitor collapse.
Validation/reconstruction KL metrics remain unfloored posterior diagnostics.

Named MVAE `--test_only` recordings save an independent **always-zero-latent
rollout** exactly once under `zero/`: `z=0` at every step, with its own predicted
history and integrated root. This baseline always uses GT initialization only,
regardless of `--resample`. The central `experiments.png` legend explicitly labels
the posterior reset schedule, so teacher-forced reconstruction is not mistaken
for uninterrupted generation. The posterior's one-step zero substitution is
computed only once per posterior frame and appears only in its own report.
No `zero_latent_history.csv` or nested `zero_latent_rollout` summaries are written.

Passing `--unit-actions` alongside `--test_only` additionally writes three
independent rollouts under `unit_actions/`: `fixed/` (one Gaussian action held for
the whole sequence), `held/` (a new action every 10 predictions), and `per_frame/`
(a new action each prediction). Unit scale means **N(0,1) per latent coordinate**,
without unit-norm normalization or clipping. These probes use their own predictions
and integrated roots, **ignore `--resample`**, and bypass the encoder. They reuse
the single zero baseline for comparison; no baseline data is copied into them.

Each rollout directory contains `rollout.mp4`, compact `metrics.png`, `history.csv`,
and `summary.json`; the three nonzero schedules also save exact decoder inputs in
`actions.npy` (one row per predicted frame, starting at `actions_first_source_frame`).
The posterior report has six panels; zero/action reports have four physical-motion
panels and omit KL and one-step substitution. CSVs omit undefined metrics,
per-feature losses, redundant weighted MSE, repeated GT root coordinates, and
unused physical diagnostics. `unit_actions/summary.json` summarizes action effects.
The central `experiments.png` overlays all enabled rollouts with consistent colors.
Errors against source motion are reference differences, not evidence that random
actions should reproduce that motion. These recording changes do not affect
training or validation metrics.

Use `--latent_seed 0` (default) to reproduce the draws and
`--latent_hold_frames 10` (default) to change the held-action interval.
The three schedules share their first action for comparison. Repeat with other
seeds and initial sequences to assess whether control effects are consistent.

Unit-action probes are off by default. Named `--test-only` recordings automatically
use CPU when CUDA is unavailable or hidden. To run while someone else uses the GPU,
prefix the command with `CUDA_VISIBLE_DEVICES=""`; add `--unit-actions` only when
you want the extra action-rollout outputs.

Training caches normalized float32 frame features in CPU RAM by default.
Set `precompute_normalized_features: False` in an experiment config to build
features per sampled window instead. The cache uses approximately
`total_dataset_frames * frame_size * 4` bytes, is rebuilt when normalization
changes, and is reused across sequence-length changes. CUDA training uses pinned
DataLoader memory and nonblocking device transfers with either setting.

`train/predicted_frames_per_second` is the only performance scalar. It counts
predicted frames across all examples over each logging interval and includes
DataLoader wait, transfers, and training; validation and checkpoint saving are
excluded. It uses wall-clock timing without adding CUDA synchronization, so
compare sustained intervals at the same batch size and sequence length.
