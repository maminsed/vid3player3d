# High-level policy validation

From `vid2player3d`, run a fixed suite against an explicit high-level checkpoint:

```bash
python vid2player/run.py --validation --cfg usopen_train_stage_1/high_level_1 \
  --eval_checkpoint results/usopen_train_stage_1__high_level_1/models/high_level_1_best.pth \
  --rl_device cpu --sim_device cpu --pipeline cpu
```

For CUDA simulation and inference:

```bash
python vid2player/run.py --validation --cfg usopen_train_stage_1 \
  --eval_checkpoint /path/to/high_level.pth \
  --rl_device cuda:0 --sim_device cuda:0 --pipeline gpu
```

Use the config matching the checkpoint's architecture and motion VAE. Validation strictly loads the given checkpoint and its observation normalization; it never falls back to a different high-level policy. The frozen low-level controller and motion VAE still load from the config. No training or video rendering takes place.

Training, evaluation and validation use the same phase-dependent arm corrections by default, including learned residuals outside swing gates. Add `--env-flag legacy_eval_residual_gating` to restore the historical evaluation/validation-only suppression of those residuals. The flag leaves training behavior unchanged; the equivalent YAML setting is `env.vid2player.legacy_eval_residual_gating: true`. Existing checkpoints can therefore produce different evaluation results with the new default.

The default suite contains **90 shots**: five distinct incoming rows per preset (left/center/right × low/high), each paired with three return targets. `--validation_shots_per_preset N` changes the total to `18*N`. `--eval_steps` limits each shot (default 120 control steps); `--seed` defaults to 123. Every shot fully resets the actor and recurrent policy state, with the same launch and initial motion pose across the three targets for a given incoming row. All slots use the same initial motion pose. Each batch seeds its RNG with its last real shot's seed. Recovery follows the config; timed ball relaunch is disabled. Early termination based on a missed ball or estimated return is disabled to observe outgoing flights. Physical and movement-boundary termination remains active.

Physics and both policies run in batches across independent IsaacGym environments. By default validation allocates up to **90 concurrent environments on CUDA**, or **8 on CPU**, capped at the suite size. Thus the default CUDA suite runs in one batch of at most 120 physics/control calls, rather than 90 separate rollouts. This is vectorized simulation and inference in one process, with no Python worker threads or subprocesses. Actual speedup depends on hardware and simulator initialization costs.

Use `--validation_num_envs 32` to limit memory usage, or `--validation_num_envs 1` for sequential execution. For example:

```bash
python vid2player/run.py --validation --cfg usopen_train_stage_1 \
  --eval_checkpoint /path/to/high_level.pth --rl_device cuda:0 \
  --sim_device cuda:0 --pipeline gpu --validation_num_envs 32
```

Larger suites run in consecutive batches. Each batch advances until all its shots finish or the step cap is reached. Finished slots immediately stop contributing metrics; unused slots in the final batch are excluded. Slots with terminated/invalid states are reset before subsequent batched inference, while surviving shots keep their own flight and target. Results retain the original suite order. The chosen environment count and number of batches are saved in the config. Validation controls `numEnvs` with `--validation_num_envs`, independently of the training `--num_envs` setting.

Use the **same concurrency and seed** for checkpoint or CPU/CUDA comparisons. Configured stochastic recovery uses batched random draws; changing batch width can change those draws and floating-point physics/inference results. Bitwise equivalence between sequential and parallel runs is not guaranteed.

Protocol version 2 (`execution.protocol_version` in the saved config) resets all slots in one combined simulator call. Older validation used one reset call per slot between physics steps, which violates IsaacGym's GPU tensor setter contract: only the final slot's pending reset took effect. Those CUDA results can show balls near the actor creation origin and immediate movement-boundary failures in the other slots; rerun them. CPU results were unaffected by the deferred GPU setter bug. Reset-time random draws now use a batch, so stochastic recovery sequences can differ from older CPU runs too.

Protocol version 3 also reconstructs the racket's fixed attachment to the reset wrist before computing policy observations, and clears instantaneous bounce/contact events on ball reset. Previously only the 24 SMPL body poses were reset: the separate racket body retained its creation pose in the first batch and its previous shot's pose in later batches. That changed the first policy observation depending on batch width and shot order. This correction applies to controller actor resets during both training and evaluation. Existing checkpoints are preserved, but validation results from older protocols should be rerun.

Presets use nearest trajectories from a deterministic sample of at most 8192 stored incoming rows. All chosen source rows are distinct, even across presets. Labels refer to **requested** intercepts: inspect actual preview intercepts and selection distances in the config to assess dataset coverage. An insufficient candidate pool raises an error. This benchmark samples the configured training pool; it does not claim held-out validation or coverage of all speeds/spins.

Each run creates exactly three files under `out/controller_validation/<config>_<checkpoint>_<timestamp>/` (or an unused `--eval_output` directory):

- `validation_config.json`: arguments, checkpoint, source dataset, exact shot suite, seeds, effective high-level/training config, and metric definitions.
- `metrics.png`: contact/in-rate/error heatmaps by incoming preset and target; per-shot reward, closest distance and landing error; weighted reward components, weighted regularization losses and termination counts.
- `values.json`: overall and grouped summaries, per-shot outcomes, per-step rewards/components/losses/ball positions and the exact launch/preview rows. Values missing because contact/landing never occurred are JSON `null`. Body meshes and full skeleton trajectories are omitted to keep this modest in size.

`values.json` stores both raw and weighted reward components per step. Summary reward-component and regularization values are mean **per-shot totals**. Episode reward totals can reflect different durations because shots stop at the first measured landing, a physical termination or the step cap.

Contact rate and measured return-in rate divide by **all shots**. A measured return requires the first ground bounce in a later control step than contact; incoming and same-step ambiguous bounces are excluded. A landing is in when `-4.11 < x < 4.11` and `0 < y < 11.89`. This verifies landing location, not net clearance. Contact without an observed landing is separately counted as unresolved and is not counted as in. Landing-to-target error uses XY Euclidean distance over observed first landings, including out landings; missing landings are excluded. Estimated return-in is reported separately in the numeric summaries and per-step values.

Regularization uses the policy's mean actions: squared elbow/wrist residuals and squared excess beyond action bounds ±1, with configured weights. PPO actor/critic training losses are not computed during inference.

CPU checks without IsaacGym:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python -m unittest vid2player.tests.test_controller_validation
```

For faster small CPU batches, consider `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` when running validation. CPU and CUDA run the same suite and metric code; numerical physics differences between devices can change outcomes.
