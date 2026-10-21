# Racket phase experiments

`usopen_train_stage_1/high_level_5` copies high_level_4 and enables smoothly gated
Djokovic racket presets. `high_level_6` copies high_level_5 and disables those
presets while retaining decoded elbow/wrist rotations plus additive learned
policy corrections. Both keep phase prediction, phase reward, ball-history
observations, physics, recovery settings, decoder and training seed unchanged.

From `vid2player3d`, start separate training runs with:

```bash
PYTHONPATH=. python vid2player/run.py --cfg usopen_train_stage_1/high_level_5 --headless
PYTHONPATH=. python vid2player/run.py --cfg usopen_train_stage_1/high_level_6 --headless
```

They write separate named training outputs. Neither config resumes high_level_4
automatically. Existing checkpoint protection still applies.

The optional `env.vid2player.racket_phase_correction` mapping supports:

- `mode: legacy` (default if omitted): existing replacement behavior, including
  the `legacy_eval_residual_gating` compatibility option.
- `mode: interception_window`: preserve decoded rotations plus additive policy
  corrections outside the allowed window, smoothly introducing the original
  elbow/wrist presets when predicted phase and swing gates also permit them.
- `mode: none`: preserve decoded rotations plus additive policy corrections
  everywhere. Does not disable phase prediction or its reward.
- `blend_s`: envelope ramp duration and relative-preset rotation filter time
  scale. Filtering acts on relative rotations, leaving decoded body motion and
  policy residuals responsive. Preset removal also fades; it is not instantaneous.

Optional preset modes currently require single-player Djokovic and Euler arm
residuals. Training, evaluation and validation execute the same control code.

`prediction.enabled: true` reuses the full paired incoming launch preview already
in the trajectory pool. This is privileged information used by the hand-written
controller; it is **not** passed to the actor's ball-history observations. No new
physics simulation, model inference, trajectory generation or lookup table is
needed. Use pools generated for the current physics configuration.

The heuristic considers preview height, lateral reach plus possible movement,
and a forward hitting region (default 0.60 m ahead of current body position,
tolerance 0.60 m). Current root velocity is extrapolated for at most 0.25 s.
It selects a plausible contact point; it does not solve full-body reachability.
The predicted forehand/backhand side is a diagnostic and does not replace the
decoder's swing classification.

Candidate selection is batched on the simulation device every three control
steps by default. At 30 Hz this is 10 Hz; the window and blend update every
control step. The selected time locks once preparation starts (0.45 s before
predicted contact). A 0.10 s late tolerance allows timing variation without
chasing later points after a miss. Contact, recovery, the ball passing behind
the player, an exhausted preview, or live/preview discrepancy above 1 m closes
the gate. The relative preset fades out over subsequent steps. All geometric
and timing values are provisional YAML parameters, not calibrated biomechanics.

The preview buffer costs 2.34 MiB for 2,048 environments. A single-thread CPU
microbenchmark measured about 2.9 ms/control step for the predictor at that batch
size, averaged over the default update interval; this is not a measurement of
end-to-end training slowdown or CUDA performance. The arm correction helper
separately took about 5.6 ms for the gated mode and 0.8 ms for additive-only mode
at that CPU batch size; these are whole-helper times, not overhead relative to
the legacy arm code. Configs omitting the feature
allocate no interception preview and run no predictor.

Evaluation keeps four plot panels. Extra `interception_*` and `racket_*` columns
are stored in `rollout.npz`, including validity, contact XYZ, time from launch,
time remaining at action start, side, preview error, window/blend weights and
actual relative elbow/wrist preset rotation magnitudes. `racket_preset_gate`
is the requested preset weight including decoded phase gates; the filtered
rotation may retain a decaying tail after this weight reaches zero.
`interception_contact_error_s` is recorded at first detected contact, with
control-step timing uncertainty. Reward/phase values describe action end;
`interception_eta_at_action_s` describes action start.

For a useful comparison, evaluate both trained policies on the same launch suite,
seeds, checkpoint selection and validation concurrency. Existing-policy smoke
tests only verify wiring and diagnostics; they do not establish which experiment
will learn better. The fixed stage-1 phase/proximity reward is unchanged and is
still not an explicit reward for a successful return.
