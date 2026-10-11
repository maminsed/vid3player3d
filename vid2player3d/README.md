# Learning Physically Simulated Tennis Skills from Broadcast Videos

<strong>Haotian zhang</strong>, Ye Yuan, Viktor Makoviychuk, Yunrong Guo, Sanja Fidler, Xue Bin Peng, Kayvon Fatahalian

SIGGRAPH 2023 (best paper honorable mention) 

[Paper](https://research.nvidia.com/labs/toronto-ai/vid2player3d/data/tennis_skills_main.pdf) |
[Project](https://research.nvidia.com/labs/toronto-ai/vid2player3d/) |
[Video](https://youtu.be/ZZVKrNs7_mk) 

<img src="doc/teaser.png"/>

### Note: the current release provides the implementation of the hierarchical controller, including the low-level imitation policy, motion embedding and the high-level planning policy, as well as the environment setup in IsaacGym. Unfortunately, the demo can NOT run because the trained models are currently not available due to license issues. 

# News
[2023/11/28] Training code for the low-level policy is released.

[2023/11/01] Demo code for the hierarchical controller is released.

# Environment setup

### 1. Download IsaacGym and create python virtual env
You can download IsaacGym Preview Release 4 from the official [site](https://developer.nvidia.com/isaac-gym).
Then download Miniconda3 from [here](https://repo.anaconda.com/miniconda/Miniconda3-py37_23.1.0-1-Linux-x86_64.sh).
Create a conda virtual env named `rlgpu` by running `create_conda_env_rlgpu.sh` from IsaacGym, either python3.7 or python3.8 works.
Note you might need to run the following command or add it to your `.bashrc` if you encounter the error `ImportError: libpython3.7m.so.1.0: cannot open shared object file: No such file or directory` when running IsaacGym.
```
export LD_LIBRARY_PATH=<YOUR CONDA PATH>envs/rlgpu/lib/
``` 

### 2. Install dependencies 
Enter the created virtual env and run the install script.
```
conda activate rlgpu
bash install.sh
```
To install additional dependencies for the low-level policy, follow the instructions in [install_embodied_pose.sh](install_embodied_pose.sh). 

### 3. Install [smpl_visualizer](https://github.com/Haotianz94/smpl_visualizer) for visualizing results
Git clone and then run 
```
bash install.sh
```

### 4. Download data/checkpoints

Download [data](https://drive.google.com/drive/folders/1zuDOJWtjjGOL5ZWPlErv39ORfg0NEi_s?usp=sharing) into `vid2player3d/data`.

Download checkpoints of motion embedding and trained polices into `vid2player3d/results` (currently unavailable).

Download SMPL by first registering [here](https://smpl.is.tue.mpg.de/login.php) and then download the [models](https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.0.0.zip) (male and female models) into `smpl_visualizer/data/smpl` and rename the files as `SMPL_MALE.pkl` and `SMPL_FEMALE.pkl`.

For training the low-level policy, also copy the smpl model files into `vid2player3d/data/smpl`.

# Demo
These demos require trained models which are currently unavailable.
### Single player
In the single player setting, the player will react to consecutive incoming tennis balls from the other side.
The script below runs the simulation and renders the result online. The simulation will be reset after 300 frames. You can change the player by chaning`--cfg` to `djokovic` or `nadal`. 
```
python vid2player/run.py --cfg federer --rl_device cuda:0 --test --num_envs 1 --episode_length 300 --seed 0 --checkpoint latest --enable_shadow
```

The script below will run the simulations in batch and render the result videos offline and saved into `out/video`. You can also change `--record` to `--record_scenepic`, which will save the result into an interactive html file under `out/html`. Note that the saved html file is large and may take seconds to load.
```
python vid2player/run.py --cfg federer --rl_device cuda:0 --test --num_envs 8192 --episode_length 300 --seed 0 --checkpoint latest --select_best --enable_shadow --num_rec_frames 300 --num_eg 5 --record --headless
```

### Dual player
In the dual player setting, the two players will play tennis rally against each other.
The script below runs the simulation and renders the result online. The simulation will be reset if the ball is missed or out. You can change the players by changing `--cfg` to `nadal_djokovic`. More player settings will be added soon. 
```
python vid2player/run.py --cfg federer_djokovic --rl_device cuda:0 --test --num_envs 2 --episode_length 10000 --seed 0 --checkpoint latest --enable_shadow
```

The script below will run the simulations in batch and render the result videos offline and saved into `out/video`.
```
python vid2player/run.py --cfg federer_djokovic --rl_device cuda:0 --test --num_envs 8192 --episode_length 10000 --seed 0 --checkpoint latest --enable_shadow --headless --num_rec_frames 600 --num_eg 5 --record
```

# Training

### Low-level policy
We provide the code for training the low-level policy in [embodied_pose](embodied_pose). As described in the paper, the low-level policy is trained in two stages using AMASS motions and tennis motions. You can run the following script to execute the two-stage training (assuming the motion data are available).
```bash
python embodied_pose/run.py --cfg amass_im --rl_device cuda:0 --headless

python -u embodied_pose/run.py --cfg general_tennis_motion_filtered --rl_device cuda:0 --headless --results_dir ./output &> output_logs.txt

python -u embodied_pose/run.py --play --cfg general_tennis_motion_filtered --rl_device cuda:0 --headless --results_dir ./output/ --checkpoint latest --export_dataset &> export_dataset_logs.txt

python -u embodied_pose/run.py --cfg specific_on_general --play --checkpoint latest --record --num_rec_frames 300 --rec_fname saved_video.mp4 --results_dir ./output &> sim_logs.txt


python -u embodied_pose/run.py --cfg 0-Adrian_Mannarino_vs._Jiri_Lehecka_reencoded-scene000-000 --play --checkpoint latest --record --num_rec_frames 300 --rec_fname saved_video.mp4 --results_dir ./output &>> sim_logs.txt

# Try this when in waterloo
python -u embodied_pose/run.py --cfg specific_on_general --play --checkpoint latest --results_dir ./output
python -u embodied_pose/run.py --cfg specific_on_general --test --checkpoint epoch01000 --record --max_test_steps 20 --rec_fname video_saved.mp4 --results_dir  ./output &> output_log.txt

--test

```
[convert_amass_isaac.py](uhc/utils/convert_amass_isaac.py) shows how to convert the AMASS motion dataset into the format that can be used for our training code.

### Motion embedding
We provide code for training the motion embedding in [vid2player/motion_vae](vid2player/motion_vae/) (assuming the motion data is organized in the format described in [Video3DPoseDataset](vid2player/motion_vae/dataset.py)).
```bash
python embodied_pose/run.py --play --cfg general_tennis_motion_filtered --rl_device cuda:0 --headless --results_dir ./output/ --checkpoint latest --export_dataset --num_envs 32  &> export_dataset_logs.txt

python vid2player/motion_vae/convert_export_dataset.py --input-yaml exports/motion_export_motion.yaml --out-dir tennis_dataset_physics

python vid2player/motion_vae/train.py  --exp federer_physics &> mvae_logs.txt

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
not required for pose/KL evaluation, and phase loss is not evaluated. `--run_test`
retains the old random-motion preview after training; its viewer/HTML flags do not
apply to `--test_only`.

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

Motion exports are saved as `exports/<params.config.name>/<source_sequence_basename>.pk`.
The basename comes from the original sequence name stored in the motion library,
not the library shard filename. For `tennis_im_training_2`, the directory is
`exports/tennis_motion_2/`. Files retain the existing joblib payload format.

Each export run also writes `exports/<params.config.name>/metrics.csv`, with one
row per saved motion. It includes the exported filename, motion/environment IDs,
frame count and simulated duration, total/mean reward and reward components,
completion fraction, survival/completion flags, stop reason, tracking errors in
meters, and head-height diagnostics. `survived=1` means no early termination was
observed during the exported rollout; `completed_motion=1` additionally requires
reaching the reference motion's end. Episode/export step limits can leave a motion
incomplete even when it survived. Rows are flushed after each saved motion, and
the CSV is replaced on each export invocation (including single-motion exports).

Player logging defaults to `INFO`, showing progress and one result per motion.
Add `--log_level DEBUG` to include step-by-step diagnostics. `--verbose` retains
its separate meaning of recording a debug video when headless.


### High-level policy
We also provide code for training the high-level policy in [vid2player](vid2player). As described in the paper, we design a curriculum trained in three stages. You can run the following script to execute the curriculum training (assuming the checkpoints for the low-level policy and motion embedding are available).
```bash
python vid2player/run.py --cfg federer_train_stage_1 --rl_device cuda:0 --headless
python vid2player/run.py --cfg federer_train_stage_2 --rl_device cuda:0 --headless
python vid2player/run.py --cfg federer_train_stage_3 --rl_device cuda:0 --headless

#test:
PYTHONPATH=. python -u vid2player/run.py --cfg tennis_test --test --checkpoint latest --num_envs 1 --episode_length 300 --rl_device cuda:0
```


The phase-free US Open curriculum uses `--cfg usopen_train_stage_1`,
`usopen_train_stage_2`, then `usopen_train_stage_3`. See the
[US Open baseline notes](doc/usopen_high_level_baseline.md) for frozen checkpoint
compatibility, initialization, output protection, metrics and the ball-physics
audit. Run from `vid2player3d` with `PYTHONPATH=.`. These configurations select the
`usopen` MVAE option, which loads the `usopen_3` checkpoint directory.

# Citation

```
@article{
  zhang2023vid2player3d,
  author = {Zhang, Haotian and Yuan, Ye and Makoviychuk, Viktor and Guo, Yunrong and Fidler, Sanja and Peng, Xue Bin and Fatahalian, Kayvon},
  title = {Learning Physically Simulated Tennis Skills from Broadcast Videos},
  journal = {ACM Trans. Graph.},
  issue_date = {August 2023},
  numpages = {14},
  doi = {10.1145/3592408},
  publisher = {ACM},
  address = {New York, NY, USA},
  keywords = {physics-based character animation, imitation learning, reinforcement learning},
}
```

# References
This repository is built on top of the following repositories:
* Low-level imitation policy is adapted from [EmbodiedPose](https://github.com/ZhengyiLuo/EmbodiedPose)
* Motion embedding is adapted from [character-motion-vaes](https://github.com/electronicarts/character-motion-vaes)
* RL environment in IsaacGym is adapted from [ASE](https://github.com/nv-tlabs/ASE/)
  
Here are additional references for reproducing the video annotation pipeline:
* Player detection and tracking: [Yolo4](https://github.com/Tianxiaomo/pytorch-YOLOv4)
* 2D Pose keypoint detection: [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)
* 3D Pose estimation and mesh recovery: [HybrIK](https://github.com/Jeff-sjtu/HybrIK)
* 2D foot contact [detection](https://github.com/yul85/movingcam)
* Global root trajectory optimization: [GLAMR](https://github.com/NVlabs/GLAMR)
* Tennis court line [detection](https://github.com/gchlebus/tennis-court-detection)


# Contact
For any question regarding this project, please contact Haotian Zhang via haotianz@nvidia.com.

### Headless imitation debug videos (SSH)

From `vid2player3d`, record a lightweight skeleton video without a display:

```bash
PYTHONPATH=.:.. python embodied_pose/run.py --cfg tennis_im_testing_2 \
  --rl_device cuda:0 --test --headless --verbose \
  --checkpoint results/amass_im/models/Humanoid_latest.pth \
  --rec_fname out/recording/amass_debug.mp4
```

`--headless --verbose` (or `--headless --record`) records environment 0 during
training, `--play`, or `--test`. The 512x256 MP4 shows a perspective 3D skeleton view with a floor grid:
blue is the simulated body and orange is its reference target. It uses Pillow
and imageio/FFmpeg on CPU, with no GUI or OpenGL renderer. The camera follows the
simulated pelvis horizontally and keeps the floor at world Z=0.

For testing/playback, the default video is stored with the evaluation report at
`<results_dir>/<cfg>/evaluations/<timestamp>/motion.mp4`. For training, it remains
`out/recording/debug_<timestamp>.mp4`. `--rec_fname` overrides either location;
relative paths are relative to the working directory. Parent directories are created.
`--num_rec_frames` defaults to 300; reaching the limit closes the video but lets
the run continue. Earlier completion or Ctrl-C finalizes the frames recorded so
far. `--rec_once` instead uses the longest loaded motion's duration as the limit.
The default playback FPS matches the control timestep (30 FPS with these configs);
`--rec_fps` overrides playback speed without changing simulation or sampling.

Playback checkpoints accept a direct `.pth` path, a suffix such as `latest`, or
`base` to use the configuration's `pretrained_model_cp`. Before tennis training,
use the AMASS path above; `tennis_im_testing_2`'s `base` points to the tennis
training output, which may not exist yet.

### Imitation evaluation reports

Every ordinary `--test` or `--play` run writes a new folder under
`<results_dir>/<cfg>/evaluations/<timestamp>/`, whether or not video is enabled:

- `history.csv`: per-step rewards, reward components, body-position error, pelvis
  position error, root-aligned body-position error, reference time and done flags.
- `summary.json`: per-episode reward totals, mean/max errors and stopping reason.
- `metrics.png`: reward, tracking-error and reward-component curves.
- `console.log`: Python console output during runner execution.
- `config.json`: arguments and configuration used for the run.
- `motion.mp4`: when headless recording is enabled and no custom filename is set.

Errors are measured in meters against the reference body origins, averaged over
bodies without reward weights. Root-aligned error removes pelvis translation,
not heading differences. These are tracking errors, not a count of discrete
mistakes. Metrics cover the actual rollout, even after the video frame limit.
The summary distinguishes height-based early termination, motion completion and
episode limits; interrupted runs preserve collected history and partial reports.
Dataset export has its own output workflow and does not create tracking reports.

300 recording frames is a maximum (10 seconds at 30 FPS), not a minimum episode
length. A 14-second source motion can produce a short video if the simulated body
hits its termination height threshold. `--num_rec_frames 450` raises the recording
cap, but does not bypass termination. For deliberate debugging beyond a fall,
set `env.enableEarlyTermination: False` in a separate debug configuration; motion
completion and episode limits still apply, and this changes the evaluation.

Imitation training writes epoch-aggregated TensorBoard metrics under
`<results_dir>/<cfg>/train/<experiment>/summaries`; for example, launch
`tensorboard --logdir output/tennis_im_training_2/train` to inspect training history.

Evaluation history also includes `head_height_m`, `reference_head_height_m`,
`head_termination_height_m` (the effective threshold), and `head_below_threshold`.
The height plot compares simulated and reference head body-origin world Z with
the threshold; this is not the top or bottom of the head mesh. Being below the
threshold only causes termination when early termination is enabled, after the
initial two steps, and when the head is not an excluded contact body.

Training additionally writes `metrics.jsonl` (one JSON object per epoch) next to
its `summaries/` directory. It includes actor/critic and bounds losses, entropy,
KL, clipping fraction, learning rate, rewards/components, alive ratio, episode
lengths when available, and timings/FPS. Losses are averages of optimizer
minibatches collected during that training epoch, not individual batch records.
TensorBoard uses training epoch as the horizontal step; the `frame` metric
retains the rollout-frame counter. Files are flushed each epoch. This persistence
was added to the custom imitation loop; older runs may have empty event files.

MVAE validation and motion diagnostics
------------------------------------

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
