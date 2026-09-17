# Advanced Ground Correction for AMASS Data

This document explains the advanced ground correction algorithm implemented in `convert_amass_isaac_correct_ground.py`.

## 1. Problem Statement

When processing long motion sequences, simple height normalization based on the first frame can lead to a "drifting" effect where the character's feet gradually move up or down from the ground plane. This is especially problematic for animations that need to interact realistically with a flat ground surface.

## 2. Solution: Dynamic Ground Correction

This script implements an advanced ground correction algorithm to dynamically adjust the character's height throughout the animation, ensuring the feet stay properly grounded. The algorithm analyzes the motion sequence to identify ground contact points and adjusts the root translation to maintain a consistent ground level.

## 3. Key Features

- **Multi-Frequency Windowing**: Instead of analyzing the entire sequence at once, the algorithm uses multiple overlapping windows of different sizes (e.g., 15, 30 frames). This captures both local, short-term ground interactions and longer-term trends, ensuring smooth transitions without discrete jumps.

- **Foot-Aware Ground Contact Detection**:
    - It uses specific SMPL foot vertex indices to accurately track the position of the left and right feet.
    - It detects "valleys" in the foot height data, which correspond to moments of ground contact.
    - For added robustness, it checks if both feet are on the ground simultaneously and uses both for a more stable ground plane estimation when they are.

- **Robust RANSAC Plane Fitting**:
    - For each window, it fits a 3D plane (`Z = a*X + b*Y + c*T + d`) to the detected ground contact points.
    - It uses the RANSAC (RANdom SAmple Consensus) algorithm, which is highly effective at ignoring outliers (e.g., noisy detections or frames where the character is airborne).

- **Two-Step Correction for Smoothness and Consistency**:
    1.  **Local Correction**: First, it corrects the character's height relative to the *local* plane fitted for each window. This preserves the nuances of the original motion (e.g., the character walking up a slight incline within the window).
    2.  **Global Alignment**: Second, it calculates the offset of each local plane from the global `z=0` origin and applies a global shift. This ensures that while local motion is preserved, the character remains at a consistent global height, preventing long-term drift.

## 4. Algorithm Flow

1.  **Generate Windows**: Create a set of overlapping, multi-sized windows covering the entire animation.
2.  **Process Each Window**: For each window:
    a. **Detect Valleys**: Identify potential ground contact frames by finding valleys in the foot height data.
    b. **Validate Foot Contact**: For each valley, check if one or both feet are on the ground.
    c. **Fit Local Plane**: Use RANSAC to fit a robust ground plane to the validated contact points within the window.
    d. **Store Predictions**: For each frame in the window, predict the local ground height and calculate the local plane's offset from `z=0`.
3.  **Combine and Correct**: For each frame in the entire sequence:
    a. **Weighted Average**: Collect predictions from all windows that overlap the current frame and compute a weighted average for both the local ground height and the plane offset. Weights are higher for predictions from windows where the frame is closer to the center.
    b. **Apply Two-Step Correction**: Adjust the character's root translation `Z` value by first aligning to the local plane and then shifting the local plane to `z=0`.

## 5. Visualization and Debugging

The script automatically generates and saves plots (`ground_correction_{sequence_name}.png`) for each processed sequence. These plots help visualize:
-   The original vs. corrected foot heights.
-   The detected ground contact points (green dots).
-   The sliding windows used for estimation (vertical dashed lines).
-   The magnitude of the height correction applied at each frame.

## 6. Usage

To use the script, run it from the main project directory:
```bash
python uhc/utils/convert_amass_isaac_correct_ground.py --amass_data <path_to_your_amass.pkl> --out_dir <your_output_directory>
```
For example:
```bash
python uhc/utils/convert_amass_isaac_correct_ground.py --amass_data outputs/demo/my_video/my_video_amass.pkl --out_dir data/motion_lib/my_video_corrected
```

Run these commands from `vid2player3d` in the project's environment. Add
`--no_show` to save plots without opening windows. For the original and corrected
left/right foot-height visualization, use:

```bash
python uhc/utils/plot_ground_contacts_simple.py \
  --amass_data /path/to/video_amass.pkl \
  --num_seq 1 --out_dir /path/to/plots --no_show
```

### Shared functions and translation conventions

Both scripts parse arguments inside `main(argv=None)` and can be imported without
starting conversion or initializing SMPL/MuJoCo. The plotting script imports its
foot-contact detection and sequence-correction functions from
`convert_amass_isaac_correct_ground.py`; maintain the correction algorithm there.

`correct_smpl_sequence()` is the shared pipeline for reconstructing the original
SMPL mesh, applying `correct_ground_height()`, and reconstructing the corrected
mesh. It returns the original and corrected vertices and translations.
`smpl_robot_context()` provides the SMPL robot and owns temporary XML/geometry
files, cleaning them up when processing finishes or fails.

SMPL model translation and skeleton root position use different origins:

```python
root_trans = trans + skeleton_tree.local_translation[0].numpy()
corrected_trans = corrected_root_trans - skeleton_tree.local_translation[0].numpy()
```

The corrected mesh uses `corrected_trans`, while `SkeletonState` uses
`corrected_root_trans`. `build_motion_output()` preserves this distinction in the
motion dictionary's `trans` and `root_trans` fields, trims all frame arrays
consistently, and computes `min_verts_h` from the corrected render mesh. Existing
motion libraries must be regenerated to incorporate this fix.

For array-only callers, `correct_ground_height()` accepts `out_dir` and
`show_plots` explicitly; it does not read global CLI arguments. The plotting
script uses vertical correction with the default settings; the converter also
supports optional court-based XY correction through its existing CLI flags.

Regression checks (from `vid2player3d`):

```bash
MPLBACKEND=Agg python -m unittest discover -s uhc/utils/tests -v
```

## 7. Tunable Parameters

The following parameters in the script can be adjusted to fine-tune the algorithm's behavior:

-   In `detect_ground_contacts_with_feet`:
    -   `window_size=3`: The smoothing window for valley detection. Smaller is more sensitive.
    -   `prominence=0.02`: The minimum prominence for a valley to be detected. Smaller is more sensitive.
    -   `foot_threshold=0.05`: The maximum height difference between feet to be considered a dual-foot contact.

-   In `get_multi_frequency_windows`:
    -   `base_window=15`: The smallest window size for plane fitting.
    -   `frequencies=[1, 2]`: The multipliers for the base window size (creates windows of 15 and 30 frames).

-   In `fit_ground_plane_ransac_with_feet`:
    -   The `RANSACRegressor` parameters can be tuned for more or less aggressive outlier rejection. 
