# Regenerating incoming previews and outgoing reward tables

Normal runs now default to **all three artifacts**: `ball_traj_in.npy`,
`ball_traj_out_x.npy`, and `ball_traj_out_y.npy`. Select `--artifacts incoming`
or `--artifacts outgoing` to regenerate only that family. `--test-only` always
writes video/config metadata only, regardless of the selected artifact family.

Run in the `vid2player3d` conda environment, from `vid2player3d/`:

```bash
# Inspect one random existing shot under the selected physics. No pool is written.
python vid2player/tools/regenerate_ball_previews.py \
  --cfg usopen_train_stage_1 --test-only --seed 123

# Sample NEW incoming launches AND regenerate outgoing reward tables.
python vid2player/tools/regenerate_ball_previews.py \
  --cfg usopen_train_stage_1 --resample --num-shots 200000 \
  --batch-size 2048 --output-dir data/ball_traj/generated/practical_v1

# Preview a random accepted NEW shot before generating that pool.
python vid2player/tools/regenerate_ball_previews.py \
  --cfg usopen_train_stage_1 --resample --test-only --seed 123

# Or preserve every existing launch and only recalculate its preview.
python vid2player/tools/regenerate_ball_previews.py \
  --cfg usopen_train_stage_2 --artifacts incoming \
  --output-dir data/ball_traj/generated/stage2_replay_v1

# Incoming pool already generated? Create only the new outgoing x/y tables.
python vid2player/tools/regenerate_ball_previews.py \
  --cfg usopen_train_stage_2 --artifacts outgoing \
  --output-dir data/ball_traj/generated/outgoing_v1

# An illustrative outgoing shot as MP4, without generating lookup tables.
python vid2player/tools/regenerate_ball_previews.py \
  --cfg usopen_train_stage_2 --artifacts outgoing --test-only
```

`--cfg` accepts a controller name or a YAML path. Paths in the controller follow
the usual `vid2player3d/` project convention. `--shots path/to/pool.npy` overrides
the source for replay; `--num-shots N` replays a seeded subset. These options
control incoming data only; outgoing tables always use the full runtime grid.
Default incoming replay
includes every source row, preserves its launch XYZ/velocity/signed spin, and
sorts by launch x, as required by the offline sampler. This is **not** a guarantee
that those launches still land in the intended zone. The manifest reports net
clearance, first bounces, target-zone counts and landing bounds. Under the current
physics, sampled old launches landed too deep; prefer `--resample` for a new
training pool.

`--resample` reads `origin_min/max`, `bounce_min/max`, `vel_range`, `theta_range`
and `vspin_range` from `env.vid2player`. The stage YAMLs contain explicit proposal
ranges for the current baseline. Speed is m/s, elevation is degrees, and signed
spin is rev/s. Candidates must clear the reference net before bouncing, have an
observed first bounce inside the configured XY zone, and reach more than 1 m
center height after that bounce within the 100-frame window. These are the
existing generator's acceptance rules; the target bounce used to aim the launch
is only a proposal, and acceptance uses its actual simulated bounce. No landing
or net constraints are silently imposed in replay mode. There is no physical net
collider in this ball-only simulation.

The command defaults to CUDA device 0 and uses the full controller's solver,
outer timestep, substeps, ball asset, aerodynamics and contact settings through
the shared incoming generator. `--device cuda:N` selects another GPU;
`--device cpu` is for debugging and should not be used to build GPU training
artifacts. The generator enforces 30 Hz recording: frame zero is the launch state,
then positions at `1/30, 2/30, ... 99/30` seconds. It does not change the controller
physics to speed up generation. Batched simulation and memory-mapped writes keep
memory bounded and avoid repeatedly copying a growing pool. Resampling performs
one additional streamed sort on disk. `--max-batches` bounds failed sampling;
adjust the YAML proposals if acceptance becomes poor after future physics changes.

Every invocation creates a **new directory**. Without `--output-dir`, it uses
`data/ball_traj/generated/<cfg>_<UTC timestamp>_<unique id>/`. An existing directory
is always rejected, even if empty. Original YAMLs and pools are never overwritten
or appended. Interrupted/failed runs retain a failed manifest and may leave an
`.incomplete.npy`. Each artifact is published after its generation succeeds; a
later outgoing failure can leave a completed incoming pool in a run marked failed.
Check the whole-run manifest before adopting data.

Normal output contains:

- `ball_traj_in.npy`: compatible float32 `(N, 307)` pool with 7 launch values and
  100 XYZ preview samples per row, sorted by launch x.
- `ball_traj_out_x.npy` and `ball_traj_out_y.npy`: outgoing lookup tables, unless
  `--artifacts incoming` was selected. `--artifacts outgoing` omits the incoming pool.
- `controller.source.yaml`: exact parsed input configuration snapshot.
- `controller.generated.yaml`: reviewable configuration snapshot pointing to the
  newly generated pool/table paths. It is **not automatically installed or selected
  for training**; paths for artifact families not selected remain as in the input.
- `manifest.json`: completion status, command arguments/seed, source pool hash,
  controller/source/ball-asset hashes, effective solver/model settings, units,
  output hash and sampling statistics. Results can vary across hardware/PhysX
  versions; the seed does not promise cross-platform bitwise reproducibility.

`--test-only` instead saves `random_shot.mp4`, its selected launch in the manifest,
and the source config snapshot. It writes **no `.npy` pool**, and never creates
`controller.generated.yaml`. The movie shows a top view and height profile at
real-time 30 fps for 3.33 seconds. The ball marker is enlarged and the height plot
uses separate horizontal/vertical scales; both axes are labeled in meters.
Matplotlib and an `ffmpeg` executable with H.264 support are required only for video.
Choose a different `--seed` to inspect another shot. The default/all selection
shows an incoming shot; `--artifacts outgoing --test-only` shows an illustrative
random outgoing launch, not a policy/racket rollout.

After reviewing generated data, copy the selected absolute `ball_traj_file`,
`ball_traj_out_x_file`, and `ball_traj_out_y_file` values from
`controller.generated.yaml` into the desired stage YAML(s). Stages 1–3
currently share the same physics and generation proposals, so one new resampled
pool can be shared if the same incoming curriculum is intended. Recheck a sample
against the actual GPU controller before a long training run. Recreate artifacts
after changing physics; manifests preserve the previous version for comparison.

## Outgoing table generation and limits

The CLI uses the existing estimator's full grid and row order: horizontal speed
10–64.9 m/s at 0.1 steps, vertical speed -5–9.9 m/s at 0.1 steps, signed transverse
spin -10–9.8 rev/s at 0.2 steps (spin varies fastest). There are **8,250,000 rows**.
X stores relative center height at 60 distances from 0 to 29.5 m; Y stores distance
and seconds at 30 drops from 0 to 2.9 m. Shapes are `(8250000, 60)` and
`(8250000, 30, 2)`, float32, **about 4 GB combined**. Generation is substantial work;
`--num-shots` does not reduce the outgoing grid. No complete 8.25-million-row run
is part of the small smoke validation.

The new generator decodes just the current batch of grid rows, including a padded
partial final batch, and streams to `.incomplete.npy` memory maps. It uses the cfg
ball asset, aerodynamic force and native solver at the actual outer physics dt.
It simulates unobstructed flight at an elevated launch altitude and interpolates
the descending height crossings, including the return to launch height for an
upward shot. A batch ends once all supported launch heights have landed. The
default 8 s horizon is an upper bound, configurable with `--outgoing-max-seconds`;
an uncovered height query fails instead of being extrapolated. Distance bins
after the simulated flight ends, once every supported launch height has landed,
use height `-1e6`, which makes the existing net-clearance test reject those bins.
This avoids extending slow-ball trajectories to unreachable distances. All valid
landing/clearance queries remain interpolated from simulated flight. Native
physics substeps are never reduced for generation.

The lookup retains the existing consumer's center-height-zero floor convention,
height/speed binning, and scalar transverse-spin approximation. **The existing
runtime estimator still converts the angular-velocity vector to unsigned spin
magnitude**, so backspin and sidespin predictions are not made fully consistent
merely by replacing its tables. Fixing that representation or introducing a
vector-aware predictor remains separate prediction work. No humanoid dynamics
or runtime reward code is changed by the outgoing-generation extension.

`ball_traj_in_dual.npy` is a separate incoming estimator table used by the dual-player
task, not the incoming launch pool selected by stages 1–3. It is not regenerated
by this command. There is no single `ball_traj_out.npy` in the stage configs: the
outgoing data is the x/y pair above. Original files and stage YAML paths are never
overwritten by the regeneration command.
