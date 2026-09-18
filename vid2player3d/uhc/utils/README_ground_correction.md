# Ground and court correction

Both scripts use `convert_amass_isaac_correct_ground.py` for input validation,
camera reconstruction, body alignment, SMPL reconstruction, and Z correction.
Importing the module does not load simulation dependencies or run conversion.

## Inputs and commands

Run from `vid2player3d` in the project's Python environment. Supply **one GVHMR
clip output folder** and its matching TennisProject CSV:

```bash
python uhc/utils/convert_amass_isaac_correct_ground.py \
  --gvhmr_dir ../GVHMR/outputs/demo_2/Arthur_Rinderknech_vs._Carlos_Alcaraz_reencoded-scene007-000 \
  --csv ../TennisProject/res_2/Arthur_Rinderknech_vs._Carlos_Alcaraz_reencoded-scene007-000.csv \
  --out_dir data/motion_lib/court_example --no_show

python uhc/utils/plot_ground_contacts_simple.py \
  --gvhmr_dir ../GVHMR/outputs/demo_2/Arthur_Rinderknech_vs._Carlos_Alcaraz_reencoded-scene007-000 \
  --csv ../TennisProject/res_2/Arthur_Rinderknech_vs._Carlos_Alcaraz_reencoded-scene007-000.csv \
  --out_dir /tmp/court_plots --no_show
```

`--tennisproject_data` is an alias for `--csv`. The folder must contain exactly
one `*_amass.pkl`, `0_input_video.json`, `0_input_video.mp4`,
`preprocess/vitpose.pt`, and `hmr4d_results.pt`. Use the matching video artifacts;
lengths, source identity, FPS, poses, translations, and body shape are checked.
The loader reads these local pickle/PyTorch artifacts as trusted project data.

The converter's `--num_motion_libs` defaults to 1 and controls output groups,
not frame sampling. Each folder currently represents one sequence; empty groups
are skipped. `--num_seq` limits sequences and the plotter also supports
`--sequence_name` substring matching. `--disable_xy_correction` runs the existing
Z correction alone, retaining the original GVHMR horizontal coordinates.

The previous `--amass_data`, `--tp_*`, and `--trim_frames` options are removed.
All GVHMR-selected frames are retained. `--no_show` saves without opening windows.

### Corrected court video over SSH

Add `--verbose` to the converter to save `<sequence>_corrected_court.mp4` in
`--out_dir`. This also suppresses plot windows, so no display or GUI is needed:

```bash
python uhc/utils/convert_amass_isaac_correct_ground.py \
  --gvhmr_dir ../GVHMR/outputs/demo_2/Adrian_Mannarino_vs._Jiri_Lehecka_reencoded-scene000-002 \
  --csv ../TennisProject/res_2/Adrian_Mannarino_vs._Jiri_Lehecka_reencoded-scene000-002.csv \
  --out_dir data/motion_lib/court_video_example --verbose
```

The 960x540 H.264 video retains every frame and the input FPS. It shows only the
corrected body, with a fixed elevated rear camera on the negative-Y side, court
lines and a simple net matching the existing tennis viewer. Camera framing fits
all motion and the net; it never follows or rotates with the player. The court
may extend outside the image so the body stays large enough to inspect.
The mesh is exactly the one saved in `*_render.pkl`: no additional grounding,
recentering, or heading changes are applied. Camera settings and video details
are saved under the sequence's `debug_video` entry in `args.yml`.

Rasterization and shading require CUDA and the existing PyTorch3D installation
in `vid2player3d`. There is **no CPU rendering fallback**. H.264 encoding runs
through FFmpeg on the CPU after GPU rendering. An early CUDA kernel check fails
explicitly if the process cannot access the GPU (including a sandbox that hides
it). No environment switch or dependency changes are required. Court rendering
requires XY correction, so `--verbose` and `--disable_xy_correction` cannot be
combined. Without `--verbose`, conversion does not import or run the renderer.

## Coordinates and frame matching

The downstream `embodied_pose/run.py` creates `HumanoidSMPLIM` or its visualizer.
`HumanoidSMPL` sets Z-up and an upward ground normal; `MotionLib` passes stored
root XY and rotations through. Tennis visualization uses a net at `(0, 0)`,
width 10.97 m, and baselines at Y=+/-11.89 m. The controller places the near player
at negative Y. Conversion therefore uses a right-handed, meter coordinate system:

- X: reference-court right.
- Y: toward the reference court's far baseline (near side is negative).
- Z: upward; ground Z=0.
- Quaternion order: XYZW, as required by PoseLib/Isaac Gym.

CSV `global_frame = JSON source.start_frame + motion_frame`. The JSON end frame
is exclusive. No guessed offsets, neighboring-frame averaging, or additional
trimming are used. CSV `res_2/0-<name>.mp4` names are normalized before matching.
Missing/duplicate rows and unusable calibration produce an explicit error.

ViTPose contains full input-video pixel coordinates and COCO17 joints. Indices
11 and 12 are left/right hips; their 2D midpoint approximates the pelvis image
location. There is no pelvis joint. The image coordinates and saved intrinsics
are scaled to TennisProject's **640x360** homography input coordinates. CSV
`court_kps` use the original drawing resolution and are scaled as well, then
checked against the homography to catch mismatched artifacts/resolutions.

TennisProject's reference net endpoints are `(286,1748)` and `(1379,1748)`, so the
center is `(832.5,1748)`. Reference pixel coordinates map to court meters with
`[10.97/1093, -23.78/2374]`. These and the quality thresholds are file-level
constants. The court length follows the downstream +/-11.89 m baseline drawing.

## Camera-aware hip reconstruction and heading

1. Fit each frame's court-to-camera rotation/translation with OpenCV `solvePnP`,
   using the CSV homography's subpixel court correspondences and GVHMR's saved
   `K_fullimg`. CSV court keypoints independently check the resolution convention.
2. Infer GVHMR-world to court rotation from the camera extrinsics and the saved
   camera-space/global SMPL root orientations. Apply its closest yaw rotation
   to the body root orientation; preserve its gravity alignment and joint poses.
3. Run the existing Z correction on the **original mesh and trajectory**, before
   changing XY or yaw. Reconstruct the hip-midpoint height and its pose-dependent
   offset from the pelvis root using the SMPL joints.
4. Intersect the hip image ray with the plane at that reconstructed hip height.
   Subtract the rotated hip-to-root offset to obtain the root's court XY.
5. Rebuild the mesh and skeleton using the corrected root position and orientation.

A ground homography applied directly to hips would incorrectly treat their
height as zero. Here, reconstructed hip height is used even during crouches or
jumps; the midpoint is still an anatomical/keypoint approximation.

Camera intrinsics are usually **estimated** by GVHMR, not measured calibration.
Zero lens distortion is assumed. Fits require a camera above the court, positive
point depths, and at most 8 pixels RMS reprojection error at 640x360. This checks
consistency, not ground-truth accuracy. No temporal XY smoothing is added. Saved
ViTPose may already contain interpolated low-confidence joints (validity=1 is
not model confidence), and upstream tracking/court inference has its own filters.

The SMPL model translation and skeleton pelvis translation are different:

```python
root_trans = trans + skeleton_tree.local_translation[0]
corrected_trans = corrected_root_trans - skeleton_tree.local_translation[0]
```

The skeleton uses `corrected_root_trans`; the mesh uses `corrected_trans`.
Heading alignment rotates about SMPL's true pelvis, accounting for the tiny
rounding difference in the XML skeleton's pelvis offset.

## Existing Z correction

The vertical algorithm is unchanged: identify candidate foot contacts, fit
RANSAC ground models in overlapping 15/30-frame windows, then combine local
height corrections. Its three-frame height averaging for contact detection is
retained. This is not a hard foot constraint; corrected feet can still penetrate
or float above Z=0, and the method does not establish ground-truth jump heights.
Downstream `MotionLib.get_motion_state(adjust_height=True)` may additionally
subtract `min_verts_h - ground_tolerance` from Z; no XY transform is applied there.

## Outputs and diagnostics

The converter writes `mlib_part_*.pth`, matching `*_render.pkl`, shape/split
metadata, `args.yml`, and `ground_correction_<sequence>.png`. That image now
includes XY trajectories and X/Y versus time alongside the Z diagnostics.
The plotting script saves `trajectory_and_ground_contacts_<sequence>.png` with
XY, X/Y over time, corrected 3D motion, and original/corrected foot heights.

For meaningful trajectory comparisons, the **original** GVHMR path is registered
using a single first-frame yaw and XY offset to the corrected path. Its relative
motion is unchanged. Labels distinguish this reference from the corrected path;
the per-frame correction is never used to warp the reference trajectory.

Both scripts keep `args.yml` short: input paths, source start/end frames,
coordinate conventions, constants, calibration error summaries, comparison
registration, and optional video settings. `source.end_frame` is exclusive;
every frame is retained, so no explicit frame-index array is saved.

Per-frame arrays go in `correction_diagnostics.pkl` next to `args.yml`, which
references it through `diagnostics_file`. Its `camera` dictionary contains
intrinsics/extrinsics, reprojection errors, yaw corrections, and discarded tilt.
Its `sequences[sequence_name]` dictionary contains `reconstructed_hip_midpoint_m`,
`original_root_m`, and `corrected_root_m`. Positions are meters; original roots
use the GVHMR frame and corrected roots use the court frame. Row `i` corresponds
to `source.start_frame + i`. Load with `joblib.load(path)` to get NumPy arrays.
With XY correction disabled there are no such arrays and no diagnostics file
is written. Use distinct output directories to keep separate runs' metadata.

## Shared API and verification

- `load_correction_inputs(folder, csv)` validates artifacts and prepares cameras.
- `smpl_robot_context()` owns and cleans up temporary geometry.
- `correct_smpl_sequence(entry, robot, xy_context=...)` runs the shared pipeline.
- `build_motion_output(...)` serializes every frame with consistent poses/positions.
- `draw_xy_comparison(...)` supplies the shared XY plotting logic.

```bash
MPLBACKEND=Agg python -m unittest discover -s uhc/utils/tests -v
```

Tests cover synthetic elevated-hip camera geometry, body heading, mesh/skeleton
alignment, unchanged Z behavior, frame matching/resolution, stale inputs, full
frame retention, import safety, and plot output. Existing motion libraries must
be regenerated to include court correction.
