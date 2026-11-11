# Viewing remote 3D applications through MobaXterm

These notes record tests on this workstation over SSH (October–November 2026).
They describe the tested setup; another machine, Python environment, or SSH
session may use a different graphics implementation.

## X11 forwarding

Run graphics commands on the remote Linux machine in the MobaXterm SSH session.
Activate the Python environment needed by the application first. A separate VS
Code SSH terminal does not automatically share the same forwarded display.

```bash
echo "DISPLAY=$DISPLAY"
command -v glxinfo glxgears xdpyinfo
xeyes
```

The tested display was `localhost:10.0`. Keep the value provided by SSH rather
than hardcoding this example. `xeyes` working confirms X11 forwarding, but does
not establish OpenGL or Isaac Gym viewer compatibility.

MobaXterm's X11 settings include OpenGL acceleration options. Changes to these
settings may require restarting the X server and reconnecting. See the
[MobaXterm documentation](https://mobaxterm.mobatek.net/documentation.html).

## OpenGL capability check

```bash
env -u LD_LIBRARY_PATH -u LIBGL_DRIVERS_PATH -u LIBGL_ALWAYS_INDIRECT \
  /usr/bin/glxinfo -B
glxgears
```

The tested system reported Mesa 23.2.1, OpenGL 4.5, and the `llvmpipe` renderer
(LLVM 15.0.7), with `Accelerated: no`. Gears displayed successfully. This is
software rendering even though `direct rendering: Yes` also appears. It does
not demonstrate use of the remote NVIDIA GPU for OpenGL rendering.

## Python / PyVista / VTK

Check imports inside the intended Python environment:

```bash
python - <<'PY'
import numpy
import vtk
import pyvista
print('NumPy:', numpy.__version__)
print('VTK:', vtk.vtkVersion.GetVTKVersion())
print('PyVista:', pyvista.__version__)
PY
```

The initial PyVista window failed with `failed to load driver: swrast`, an
OpenGL 3.2 error from VTK, and a segmentation fault. The following invocation
successfully displayed an interactive sphere:

```bash
env -u LD_LIBRARY_PATH -u LIBGL_ALWAYS_INDIRECT \
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri \
  python - <<'PY'
import pyvista as pv
plotter = pv.Plotter(off_screen=False, window_size=(900, 700))
plotter.add_mesh(pv.Sphere(), color='orange', show_edges=True)
plotter.add_axes()
plotter.show()
PY
```

A Conda/system C++ runtime mismatch is a plausible cause, but this test changed
both the runtime and driver search path, so it did not isolate the cause. A
generic VTK OpenGL error does not by itself mean the machine lacks modern
OpenGL. Keep these overrides local to the command; do not globally replace
Conda libraries or export these settings for unrelated applications. A separate
viewer environment is another option, but was not tested here.

## Isaac Gym native viewer

Use the evaluation environment, which on this machine is `vid2player3d`:

```bash
conda activate vid2player3d
echo "DISPLAY=$DISPLAY"
which python
python -c "from isaacgym import gymapi; print('Isaac Gym import OK')"
nvidia-smi
cd /pub2/amin/IsaacGym_Preview_4_Package/isaacgym/python/examples
python joint_monkey.py --asset_id 0 --graphics_device_id 0
```

The tested Python was `/pub2/amin/miniconda3/envs/vid2player3d/bin/python`.
Isaac Gym Preview 4 loaded `gym_38.so`. The remote machine reported an RTX 4080
and NVIDIA driver 535.309.01. The humanoids rendered and zoom worked. Camera
translation/rotation was not confirmed; this remains a controls/input issue to
investigate rather than a confirmed capability.

The example reported GPU PhysX with the GPU pipeline disabled, and explicitly
forced its CPU pipeline. This is the example's configuration, not evidence that
evaluation's GPU tensor pipeline has been tested with a viewer. Its log ended
with `Done`, followed by an X11 `BadDrawable` / `X_PutImage` error. If that occurs
only at window close, investigate shutdown separately from rendering failures.

If the normal example fails, a diagnostic retry is:

```bash
env -u LIBGL_ALWAYS_INDIRECT \
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  python joint_monkey.py --asset_id 0 --graphics_device_id 0
```

Keep the normal `LD_LIBRARY_PATH` for Isaac Gym unless diagnosing a specific
library problem. The successful PyVista workaround is not automatically an
appropriate Isaac Gym configuration. For Vulkan/device errors, collect:

```bash
command -v vulkaninfo
vulkaninfo --summary
```

An available CUDA GPU or successful `nvidia-smi` alone does not prove that a
forwarded GUI works. Test each application's actual window, since OpenGL and
Isaac Gym's graphics path can behave differently through SSH.

## High-level controller evaluation

Run from `vid2player3d` in the evaluation environment and forwarded SSH session:

```bash
PYTHONPATH=..:. python vid2player/run.py \
  --cfg YOUR_CONFIG --eval_controller \
  --eval_checkpoint /path/to/high_level_checkpoint.pth \
  --eval_ball center_low --eval_steps 120 --seed 123 --rl_device cuda:0
```

Use your usual config and checkpoint. Omit `--headless` to open the native Isaac
Gym physics viewer, or add `--headless` to disable graphics. Both modes still
save the court video, rollout, configuration/report, and metrics plot. Output
defaults to a fresh timestamped directory; an explicit `--eval_output` must name
a directory that does not already exist.

The viewer starts running near the player with automatic camera-follow disabled
so manual camera input is not overwritten each frame. Click the viewer to focus
it before using these keys:

Native camera movement uses **hold right mouse button + drag** to look around,
and **hold right mouse button + W/A/S/D** to move forward/left/back/right.
Plain mouse movement or left dragging is not the free-camera navigation gesture.
These controls should also work while evaluation is paused with Space. See
[NVIDIA's Isaac Gym camera navigation guidance](https://forums.developer.nvidia.com/t/i-want-to-move-the-viewpoint-close-to-the-actor-between-all-envs-with-the-arrow-keys/194624).

Evaluation additionally provides a WASD translation fallback (right mouse is
optional for translation). It uses the current native camera orientation and
only translates when the native renderer did not already move the camera that
frame. Release movement keys to stop. Preset keys trigger once per press, so
holding one does not repeatedly overwrite manual movement. Native mouse-look
and scroll remain available after snapping to a preset.

| Key | Action |
| --- | --- |
| Space | Pause/resume policy and physics together; camera stays interactive |
| Right arrow | Advance one complete control step, then stay paused |
| `-` / `=` | Halve/double target speed, from 0.125x to 4x (starts at 1x) |
| `1` | Close back view: 2 m behind the player's current position |
| `2` | Close right view: 2 m to the player's court-right (+X) |
| `3` | Close left view: 2 m to the player's court-left (-X) |
| Escape / window close | Stop early and save the completed steps |

All three camera presets look toward the player at pelvis height + 25 cm.
Back means the near-baseline side (-Y); side directions are relative to the
court, independent of the player's torso rotation. Presets snap once rather
than following continuously. Pause and switch between the three views to inspect
the same pose, or press a preset again to recenter after the player moves.

### Control-step metrics and contact history

The terminal prints one metrics line after each completed control step and
repeats the latest line when Space pauses. Camera redraws while paused do not
continually print. `t` is simulation time in seconds (step / FPS), matching the
horizontal axis in `metrics.png`; it excludes wall-clock pauses and slow motion.
The line includes reward, cumulative reward, current ball–racket position
distance, registered contacts in this step and in total, the persistent contact
latch, and the ID/time of the last registered evidence interval.

Every evidence-bearing physics interval in the step is printed beneath it with
an ID, a time bracket, sensor/impulse evidence, and status: `registered`,
`ignored_already_latched`, or `rejected_ground_impact` (or `not_registered` if
the available evidence does not explain rejection). IDs persist across control
steps. The detector currently accepts only the first racket contact in a flight;
later evidence remains visible in this history without changing physics or the
registration rule. Consecutive evidence-bearing intervals may belong to one
impact, so their count is not a count of distinct physical contacts.

These records are retained under
`evaluation_config.json` → `evaluation_results.contact_evidence_intervals`.
`rollout.npz` adds per-step registration counts, `last_registered_time_s`, and
`physics_time_start_s` / `physics_time_end_s`. Timestamp brackets identify outer
physics intervals, not exact internal PhysX impact times. Legacy detectors
without interval-resolved evidence report a new registration at control-step
resolution and cannot expose later candidate intervals the same way.

This is live forward stepping, not saved replay or backward scrubbing. Pausing
does not add samples to the rollout/video. Speed changes affect wall-clock
pacing, not the simulation timestep or recorded video frame rate. Actual speed
is limited by policy inference, physics, diagnostics, drawing, and SSH latency.

Evaluation uses a bright pink ball for visibility. The viewer draws once per
control step rather than at every inner physics interval. Court-video rendering
and encoding run after interactive evaluation, followed by the usual artifact
save. Expect a short saving delay at the end. Headless mode retains its streaming
video generation. If you close before the first step, the report and rollout
contain zero samples; no video frames have been generated.

Validation (`--validation`) remains headless. Other training/playback modes keep
their previous display behavior.
