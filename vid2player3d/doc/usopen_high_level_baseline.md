# US Open high-level baseline without phase

Run these commands from `vid2player3d` after the GPU is available:

```bash
PYTHONPATH=. python vid2player/run.py --cfg usopen_train_stage_1 --rl_device cuda:0 --headless
PYTHONPATH=. python vid2player/run.py --cfg usopen_train_stage_2 --rl_device cuda:0 --headless
PYTHONPATH=. python vid2player/run.py --cfg usopen_train_stage_3 --rl_device cuda:0 --headless
```

Run them sequentially. Stage 2 initializes from stage 1 and stage 3 from stage 2.
The high-level agent learns from synthetic tennis balls, using the US Open motion
decoder as its motion representation. It does not load video clips as supervised
PPO targets.

## GPU memory settings

The original stage-1 run in `logs/high_level.txt` failed during MVAE inference
after two epochs: 10,240 environments required a 1.58 GiB mixed-weight temporary
with only 1.07 GiB free on a 15.70 GiB GPU. The mixed decoder creates per-sample
weight matrices; inference under `no_grad` does not eliminate those temporaries.
PyTorch reservation/fragmentation and simulator memory also compete for VRAM.

All US Open stages now use 2,048 environments, PPO minibatches of 4,096, and
`env.vid2player.mvae_inference_batch_size: 512`. The latter splits frozen decoder
inference into independent chunks and concatenates outputs in environment order;
it leaves the checkpoint, conditioning and reward unchanged. The failed mixed
weight allocation scales from about 1.58 GiB at 10,240 samples to about 81 MiB
at 512 samples. Set this knob to a smaller positive integer to reduce decoder
temporaries further; zero disables chunking. Actual total VRAM still depends on
physics, PPO buffers, minibatch activations and other processes.

Horizons remain 64/32/32 steps. Rollouts now contain 131,072/65,536/65,536
transitions, all divisible by the 4,096 minibatch size. Smaller environment counts
change PPO batch statistics and samples per epoch: stage 1 collects one fifth,
and stages 2/3 about one seventh-and-a-half, of their previous samples per epoch.
The epoch limits were retained; compare learning by total frames, not just epochs.
Equal original sample budgets would require 5x/7.5x/7.5x as many epochs.

Before a full run, use a separate output root for a short stage-1 memory check:

```bash
PYTHONPATH=. python vid2player/run.py --cfg usopen_train_stage_1 --rl_device cuda:0 --headless --max_iterations 10 --results_dir results/usopen_memory_check
```

Then use the normal stage commands above. Existing-checkpoint protection remains
enabled. A CPU equivalence test covers chunked versus unchunked mixed decoding,
phase/no-phase outputs, strided inputs and a final partial chunk. Full GPU memory
validation has not yet been performed.

## Headless checkpoint evaluation

Run from `vid2player3d`, with the normal IsaacGym environment active:

```bash
PYTHONPATH=. python vid2player/run.py --cfg usopen_train_stage_1 \
  --eval_controller \
  --eval_checkpoint results/usopen_train_stage_1/models/usopen_no_phase_latest.pth \
  --eval_ball center_low --eval_target center \
  --eval_output out/controller_eval/stage1_center \
  --eval_steps 120 --seed 123 --rl_device cuda:0 --headless
```

One environment launches one ball, with no timed relaunch. Evaluation does not
train or update weights. It strictly loads the explicit checkpoint and observation
normalization. Rendering uses the same centered rear camera as MVAE reconstruction:
eye `[0,-28,9]`, target `[0,-1,0]`, vertical field of view 45 degrees, image 768x720.
There is one view, no inset, and no saved screenshots. Rendering is CPU/PIL/ffmpeg
with no display server; IsaacGym physics still needs the configured GPU.

Incoming ball preset mapping (`--eval_ball`):

| Preset | Requested X at Y=-13 m | Requested height |
| --- | ---: | ---: |
| `left_low` | -2 m | 0.9 m |
| `left_high` | -2 m | 1.4 m |
| `center_low` | 0 m | 0.9 m |
| `center_high` | 0 m | 1.4 m |
| `right_low` | +2 m | 0.9 m |
| `right_high` | +2 m | 1.4 m |

Left/right denote court X, not forehand/backhand. Optional `--eval_ball_x` and
`--eval_ball_height` override these preset coordinates, for example
`--eval_ball_x 1.0 --eval_ball_height 1.2`. The nearest available paired launch
and observation preview is selected from a deterministic subset of the configured
incoming dataset. This is approximate placement, not an exact physical intercept
constraint. The requested and selected preview positions are recorded in the
config JSON. The ball is still simulated after launch.

`--eval_target left|center|right` sets the desired return location to `[-3,10,0]`,
`[0,10,0]`, or `[3,10,0]`. This is separate from where the incoming ball is sent.
Stage 1 rewards proximity and does not reward that return target.

The output directory must be new; omitting `--eval_output` generates a timestamped
folder under `out/controller_eval/`. Each completed run saves exactly four files:

- `video.mp4`: fixed court view, simulated skeleton/racket/ball, and top-left metrics.
- `metrics.png`: static curves of step reward, cumulative reward, weighted reward
  components, and weighted auxiliary/bounds losses against time. A dashed line
  marks the first detected contact. Evaluation performs no PPO update.
- `evaluation_config.json`: checkpoint, effective configs, seed, camera, requested
  ball coordinates, selected launch row, desired return target, and final results.
- `rollout.npz`: actual body/ball/racket positions and policy means per frame, plus
  the selected launch/preview row and numeric per-frame metrics (`metrics` matrix
  with `metric_names` columns). No extra launch file is retained; the simulator's
  temporary input is cleaned up after the run.

Auxiliary and bounds penalties are calculated from each deterministic policy
output before scaling, with the configured coefficients also reported. PPO actor
and critic training losses are null: no rollout advantage fitting or optimization
is performed in evaluation. Rewards are the real configured task rewards, so their
components show where the policy receives reward during the flight.

The root reset is centered near `[0,-13]`. Recovery retains its configured random
walk, with the fixed seed reset after inner-player initialization. Recording ends
on episode termination or the step limit (120 steps is about four seconds).
To compare checkpoints, keep the same ball/target/seed/config and use a new output
folder for each checkpoint. A single case is a diagnostic, not a success benchmark.
Contact detection and estimated return validity retain the existing implementation
and its limitations, including the absence of a physical net collider.

CPU tests cover the four-file output contract, case mapping and custom overrides,
exact MVAE-camera projection equivalence, and reward/loss recording with simulated
inputs. Live GPU physics remains unverified in this session.

## Configuration and frozen components

- `mvae_ver: usopen` selects the Python option entry whose `model_ver` is
  `usopen_3`; it loads `results/motionVAE/usopen_3/latest.tar`.
- All three stages load the low-level checkpoint
  `output/tennis_im_training_2/models/tennis_motion_2_latest.pth`.
- `env.physics.config: usopen_im` selects a dedicated inference configuration
  beneath `vid2player/cfg/im`. Do not copy the EmbodiedPose training YAML wholesale:
  its `pose_im_rnn` builder expects context preprocessing that this inference
  runner does not use. The checkpoint's actual learned control network is an MLP.
- The runtime MLP takes the same 734 features and emits 75 values. Its raw output
  is residual joint angles plus six residual force/torque values. Runtime
  `no_scale_action: True` adds joint residuals to the generated reference angles,
  reproducing the training builder's `residual_action=True`. Both clamp PD targets
  to current joint angle +/- pi/2.
- The runtime uses `smpl_mesh_humanoid_djokovic.xml`, matching the training YAML,
  and `player: djokovic` for matching proxy body shape. This is a fixed proxy
  character, not an assertion that the dataset contains Djokovic alone.
- `grip: semi_western` and `righthand: True` are explicit. The Djokovic asset's
  racket-head cylinder axis is proportional to `[0,1,1]`; this matches the
  semi-western normal used by observations and decoded-motion racket geometry.
  The grip config does not rotate the asset's physical collider. YAML `null` for
  grip is not accepted by the physical racket-normal dictionary.
- PPO exploration remains nonzero and fixed: sigma approximately 0.502 in stages
  1/2, 0.055 in stage 3, before the latent action scale of 1.5.

## Removing phase

All stages set `env.vid2player.predict_phase: False` and
`reward_scales.phase: 0`. The runtime now keeps placeholder phase/swing buffers
for reset and rendering compatibility while disabling phase prediction and
phase-dependent grip presets. Residual elbow twist and wrist adjustments are
bounded, additive corrections to the decoded rotations, in both training and
evaluation. Zero residual action preserves the decoded pose.

Stage 1 reward becomes reaction-mask times `exp(-5 * racket_ball_distance^2)`.
Stages 2/3 retain contact and estimated-return rewards, with the phase multiplier
equal to one. Contact stays latched until the next incoming ball. Estimated
landing reward stays latched too. Neither of these temporal reward conventions
was changed for this baseline.

This removes both phase reward shaping and procedural phase-dependent motion
adjustments. A later phase comparison must state which of these mechanisms is
enabled; simply comparing two different motion decoders does not isolate phase.

## Initialization

`data/vae/vae_init_condition_usopen_ready.pt` was created on CPU from a valid,
upright, low-motion, near-baseline frame in the MVAE training set. The checkpoint's
held-out videos were excluded. Its adjacent JSON records the source and selection.
The selection is a heuristic, not a semantic ready-pose annotation. The runtime
relocates the root near `[0, -13]` while retaining pose and world orientation.
This is a raw `[1,1,288]` motion-feature tensor, not a checkpoint or a ball
trajectory. The 288 features contain root position/velocity, 24 joint rotations
in 6D form, and 23 root-relative joint positions/velocities. It seeds the
autoregressive decoder's previous-frame condition and initial reference pose
on episode reset. The same seed is repeated across environments; reset root XY,
incoming balls and policy actions still vary. Without `vae_init_conditions`,
the code can sample initial frames from the dataset instead. This explicit seed
avoids reusing a Federer-specific condition and makes resets reproducible, but
does not establish that the selected frame is an optimal tennis ready stance.

To reproduce on a checkout lacking this generated data artifact:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python -m vid2player.motion_vae.prepare_controller_init
```

The generator refuses to overwrite an existing initializer or metadata file.

## Output and logging

Default output root is `results`, relative to the working directory:

```text
results/usopen_train_stage_1/models/usopen_no_phase_latest.pth
results/usopen_train_stage_2/models/usopen_no_phase_latest.pth
results/usopen_train_stage_3/models/usopen_no_phase_latest.pth
```

Each stage also writes best and numbered epoch checkpoints beneath its `models/`
directory. Existing US Open stage checkpoints cause a fresh training invocation
to stop before creating the simulator. Explicit continuation uses
`--resume --checkpoint latest`; continuation intentionally updates that run's
checkpoints. A separate experiment should use a new `--results_dir`. Update the
stage 2/3 previous-stage paths as well: CLI output overrides do not rewrite YAML
input checkpoint paths. Frozen low-level/MVAE checkpoints are read only.

Previously, the custom high-level loop populated `log_dict` without persisting
it. It now writes metrics only as TensorBoard scalars through the existing writer,
and saves the separate config snapshot at `results/<stage>/logs/run_config.json`. `--no_log`
suppresses these metrics; its existing checkpoint-saving behavior is unchanged.

```bash
tensorboard --logdir results/usopen_train_stage_1/train
```

Metrics include PPO losses, entropy, KL, clipping fraction, learning rate, reward
components, episode length and FPS. Additional `task/` metrics report:

- Hit rate and estimated in-court rate per ended ball cycle. Episode truncations
  count as ended cycles, so unfinished final cycles can lower these rates.
- Estimated landing error among estimated in-court cycles, and estimated flight
  time for those cycles.
- Detected outgoing bounce count/in-court rate, and XY discrepancy between
  estimated landing and detected bounce for estimated-in shots observed to bounce.
- Mean stored-preview versus simulated-ball position discrepancy before contact.

Ratios use event totals across the PPO rollout. Metrics with zero denominator are
omitted. The detected-bounce metrics use the existing height-threshold detector;
they are not precise floor-contact measurements. Training is not a fixed-seed,
held-out evaluation. Later evaluation should use fixed ball launches and targets,
multiple policy seeds, actual contact/landing measurements and short videos.

## Coordinates and physics audit

The ground/XY converter maps the net center to XY zero, uses meters and Z-up,
and places the near player at negative Y. Its court length is 23.78 m; the baseline
is at +/-11.89 m. High-level targets `[-3,10]`, `[0,10]`, `[3,10]` lie inside the
opposite singles half, whose bounds are X +/-4.11 and Y 0..11.89. Stage 3's target
rectangle X -3..3, Y 9..11 also lies inside it.

The local exported dataset has median root Y -12.69 m, and the MVAE normalization
mean Y is -12.05 m. Sampled incoming data launches at Y 12..13 m and reaches
Y approximately -13 with heights around 0.81..1.45 m (six-substep file) or
0.88..1.43 m (two-substep file). Some exported motion roots lie outside the
controller movement rectangle; geometric convention agreement does not guarantee
every decoded motion remains within bounds.

`env.vid2player.court_min: [-5,-16]` and `court_max: [5,-10]` bound the
player's root XY, not the ball. This is a 10 m wide, 6 m deep baseline strip:
4.11 m behind the baseline and 1.89 m inside it. Crossing a bound terminates
the episode even when `enableEarlyTermination: False`. For a separate broader
movement experiment, e.g. `[-6,-18]` to `[6,-6]`, change these two existing
knobs consistently across stages. They do not move ball launches, targets or
reset positions. The baseline retains its original bounds.

`reward_type: return` uses simulated ball/bounce positions, but its landing
component is `has_contact * clamp((400 - squared_distance_to_target)/400,0,1)`:
it rewards current ball position before the detected bounce and holds the
bounce-position score afterward. It lacks the estimated version's in-court/net
and flight-time gates and ignores `bounce_pos`/`bounce_time` scales. It still
loads the estimator for diagnostics. It is not an equivalent simulator-backed
replacement for `return_w_estimate`.

Assigning a future bounce result retroactively to a contact transition would
require retaining that transition until the result arrives and recomputing GAE
and value targets before PPO optimization. A hit and its landing can straddle
rollout boundaries, and a reset can erase the trajectory before landing. For a
simulator-reward experiment, delayed landing reward plus causal GAE is simpler;
retroactive assignment deliberately changes the reward's timing/objective.

The live ball and offline generator load the same tennis-ball URDF: mass 0.057 kg
and radius 0.032 m. Gravity is -9.81 m/s^2. Drag is explicitly applied every physics
step as `-0.5 * rho * pi * R^2 * Cd * |v| * v`, with rho 1.21 and Cd 0.55. Lift is
also explicitly applied. The simulator divides applied forces by the URDF mass;
the force expression correctly does not divide by mass again. Incoming generation
and live configs use restitution 0.9, ball friction 0.2, plane restitution 0.5,
and spin scale 5. Stage 1 uses two substeps; stages 2/3 use six, matching the named
incoming datasets' intended generation settings. Stored tables have no provenance
metadata, so their full agreement with current code has not been demonstrated.

Known limitations retained for this baseline:

- Live lift and outgoing lookup use angular-velocity magnitude, losing spin sign
  and axis. The generator can model signed backspin, creating an inconsistency
  for negative-spin launches. Current incoming files contain positive launch spin.
  General sidespin/backspin after racket contact is still approximated poorly.
- With six substeps, contact is inferred from outgoing Y velocity and a large Y
  velocity change; body/handle contact can be counted as racket contact.
- Bounce detection triggers at ball-center height <= 4R or 6R (0.128 or 0.192 m),
  above the actual floor-contact height R. This biases detected landing position.
- The outgoing estimator rounds velocity/spin to table bins and clamps certain
  out-of-range values. It rejects other states, including outgoing Y speed <=10
  m/s, vertical speed outside -5..10 m/s, and launch height >=3 m. Such states can
  occur physically but receive no estimated-return reward or a clipped estimate.
- The environment creates a player, ball and floor; there is no physical net
  collider in this task. Net failures are classified through trajectory estimates.

Changing physics should be followed by regenerating trajectories/lookups and a
small comparison against live ball flight. It was not done while the GPU is busy.

## Verification performed

### CUDA ball-material fix (2026-10-06)

The two `stage*_latest_initial_failure` rollouts expose a collision-response
failure, not merely a rendering problem. Before the first impact, recorded ball
positions agree with the paired stored previews within 0.00005 m. After impact,
stage 1 rebounds only to 0.169 m and stage 2 settles at approximately the ball
radius (0.032 m), although both previews predict a rebound near 1 m.

A controlled drop from 1.5 m reproduced a GPU-specific material initialization
bug. With ball restitution 0.9 and floor restitution 0.5, initializing only the
actor's material produces rebound apices 0.121/0.124 m for two/six substeps.
Initializing the asset material before creating the actor produces 0.748/0.751 m.
CPU PhysX gives the latter result for both initialization methods, so CPU-only
tests do not expose this bug. NVIDIA has also documented an underlying GPU
friction/restitution update issue in its
[restitution discussion](https://forums.developer.nvidia.com/t/contact-restitution-with-physx-in-isaacgym/219637/11).

The live environment and offline generator now initialize ball materials on the
asset before creating actors. Racket-head materials are initialized on the asset
too; the body-to-shape mapping selects the head without changing the handle or
humanoid materials. Optional `ball_restitution` and `racket_restitution` settings
fall back to the existing shared `restitution` value. This corrects how existing
parameters reach GPU PhysX; no stage YAML or checkpoint was changed by this fix.

The same checkpoints, incoming launches, seed 42 and original targets were
evaluated on CUDA, with separate outputs:

| Case | Original first rebound | Patched first rebound | Patched detected racket contact |
| --- | ---: | ---: | --- |
| `out/controller_eval/stage1_material_fix` | 0.169 m | 1.100 m | None |
| `out/controller_eval/stage2_material_fix` | effectively none | 1.008 m | None |

Both complete 120 steps. Stage 2's maximum preview discrepancy through frame 43
is 0.0065 m; stage 1's is 0.1484 m. Stage 1 later changes direction near the
player without detected racket contact, so its later trajectory discrepancy must
not be attributed entirely to incoming-ball physics. Restoring a bounce does not
repair policies learned under the faulty physics: neither checkpoint hits the
ball in these cases, and stage-1 reach reward decreases. No training was run.

The standalone diagnostic uses the real tennis-ball asset and material helper,
without the humanoid or policy. Run in the IsaacGym environment:

```bash
PYTHONPATH=. python -m vid2player.utils.check_ball_bounce --gpu --compare-actor-only \
  --evaluation out/controller_eval/stage1_latest_initial_failure \
  --evaluation out/controller_eval/stage2_latest_initial_failure \
  --output /tmp/usopen_ball_physics_gpu_audit.json
```

Use a new output filename on each run. The diagnostic exits unsuccessfully if the
patched material path fails to rebound above 0.3 m; the old actor-only path is
reported as a comparison. Omit `--gpu` for CPU PhysX. This is a restitution
isolation check, not validation of full incoming/outgoing trajectories.

Remaining work before a corrected training experiment:

- Resolve the floor mismatch deliberately: low-level training uses floor
  restitution 0.0, while the controller and ball generator use 0.5. A single
  global plane affects feet as well as the ball. Changing it blindly also changes
  the incoming bounce. `plane_restitution: 0.0` now works as an explicit override;
  the previous truthiness check silently ignored zero. Zero-valued contact/rest
  offsets, bounce threshold and velocity-iteration overrides also work.
- Compare the low-level controller at its trained two-substep setting with
  stages 2/3's six-substep setting, using matching incoming trajectories. The
  material fix itself retains the configured substep counts.
- Resolve stage 1's remaining preview mismatch and regenerate/validate incoming
  data on the chosen physics backend. Verify outgoing tables against live flight
  too; their provenance is still unknown. Do not overwrite the original tables.
- Address signed/axis-aware lift and the early height-threshold bounce/contact
  detectors described above as separate, measured changes. They were not changed
  by the material fix.
- Start a separate training run after those checks, retaining the original
  checkpoints for comparison. Continuing an old policy is an experiment, not
  evidence that the old training used correct physics.

The new material/override tests and existing controller-baseline checks pass
(10 tests). The full CPU suite also exposed a camera-equivalence failure in the
pre-existing camera/reconstruction edits; that is outside this physics patch.

Before the material-fix investigation above, no GPU simulation or training had
been launched for baseline verification. Those CPU checks verified strict low-level
checkpoint loading, identical training/runtime observation formulas, finite MLP
outputs, phase-free rewards/residuals, config chaining, event aggregation, output
protection, metric persistence and batched decoder equivalence. Run the CPU test suite with:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python -m unittest discover -s vid2player/tests -v
```

The first simulator smoke test remains necessary when the GPU becomes available.
For stages 2/3, a small check can use `--num_envs 64 --horizon_length 32
--minibatch_size 2048 --max_iterations 1` with a separate `--results_dir`, after
the configured previous-stage checkpoint exists. This avoids creating a smoke
checkpoint in the full training directory.
