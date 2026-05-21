# Trajectory Analysis for HMR4D Demo

This document explains the trajectory analysis functionality added to the `demo_amass_gravity_and_xy_correct.py` script.

## Overview

The trajectory analysis feature provides comprehensive visualization and analysis of:
1. **Horizontal trajectory** (X-Z plane movement)
2. **Ground contact detection** based on static confidence logits
3. **Gravity correction analysis** showing how the Y-axis is corrected to prevent "flying" poses
4. **Detailed motion analysis** including velocity, contact patterns, and trajectory curvature

## Usage

### Basic Usage with Trajectory Analysis

```bash
cd GVHMR
python tools/demo/demo_amass_gravity_and_xy_correct.py --video inputs/demo/your_video.mp4 --plot_trajectory
```

### Interactive Trajectory Analysis

For interactive exploration of the trajectory data:

```bash
cd GVHMR
python tools/demo/demo_amass_gravity_and_xy_correct.py --video inputs/demo/your_video.mp4 --plot_trajectory --interactive
```

This will:
- Display plots in interactive matplotlib windows
- Allow zooming, panning, and exploration of the data
- Show hover information and detailed views
- Continue processing after you close the plot windows

This will:
1. Process the video using HMR4D (with gravity correction)
2. Generate the standard output videos (incam, global, merged)
3. Save AMASS format data (if `--save_amass` is used)
4. **Generate trajectory analysis plots** in the output directory

### All Available Options

```bash
python tools/demo/demo_amass_gravity_and_xy_correct.py \
    --video inputs/demo/your_video.mp4 \
    --output_root /custom/output/dir \
    --save_amass \
    --amass_output /custom/amass/output.pkl \
    --plot_trajectory \
    --interactive \
    --static_cam \
    --use_dpvo \
    --f_mm 24 \
    --verbose
```

## Output Files

When `--plot_trajectory` is used, the following files are generated in the output directory:

**Note**: If `--interactive` is also used, plots will be displayed in interactive windows instead of being saved to files.

### 1. `trajectory_analysis.png`
Main trajectory analysis with 4 subplots:
- **Top-left**: Horizontal trajectory (X-Z plane)
- **Top-right**: Trajectory with ground contact points highlighted
- **Bottom-left**: Ground contact confidence over time for foot joints
- **Bottom-right**: Height over time (Y-axis, gravity direction)

### 2. `detailed_trajectory_analysis.png`
Comprehensive analysis with 6 subplots:
- **3D trajectory view** with ground contact points
- **Ground contact confidence heatmap** for all joints
- **Velocity analysis** showing movement speed over time
- **Height distribution** histogram
- **Ground contact duration distribution**
- **Trajectory curvature analysis**

## Gravity Correction Explanation

The HMR4D system applies gravity correction to prevent "flying" poses by:

1. **Detecting ground contact** using static confidence logits for foot joints (L_Ankle, L_foot, R_Ankle, R_foot)
2. **Correcting the Y-axis** (gravity direction) by subtracting the minimum Y value from all frames
3. **Ensuring the person stays on the ground** level throughout the sequence

The correction is applied in the post-processing step:
```python
# Put the sequence on the ground by -min(y)
ground_y = post_w_j3d[..., 1].flatten(-2).min(dim=-1)[0]
post_w_transl[..., 1] -= ground_y
```

## Ground Contact Detection

Ground contact is detected using the static confidence logits from the HMR4D model:

- **Joint IDs**: [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]
- **Foot joints**: [0, 1, 2, 3] (L_Ankle, L_foot, R_Ankle, R_foot)
- **Threshold**: 0.8 (80% confidence) for ground contact detection
- **Method**: Maximum confidence across foot joints per frame

## Analysis Metrics

The trajectory analysis provides several key metrics:

### Trajectory Metrics
- **Total frames**: Number of frames in the sequence
- **Trajectory length**: Euclidean distance from start to end point
- **Height range**: Minimum and maximum Y values (gravity axis)
- **Ground contact percentage**: Percentage of frames with detected ground contact

### Ground Contact Analysis
- **Contact intervals**: Continuous sequences of ground contact frames
- **Contact duration distribution**: Histogram of contact interval lengths
- **Contact confidence patterns**: Time series of confidence values for each joint

### Motion Analysis
- **Velocity magnitude**: Speed of movement over time
- **Trajectory curvature**: Angular changes in movement direction
- **Height distribution**: Statistical distribution of Y-axis values

## Coordinate System

The analysis uses the HMR4D coordinate system:
- **X-axis**: Left-right movement
- **Y-axis**: Up-down movement (gravity direction)
- **Z-axis**: Forward-backward movement

The gravity correction ensures that Y=0 represents the ground level.

## Testing

You can test the trajectory plotting functionality using the provided test script:

```bash
cd tools/demo
python test_trajectory_plotting.py
```

This will:
1. Create mock prediction data simulating a walking motion
2. Test the plotting functions
3. Generate sample plots in the `test_output` directory

## Dependencies

The trajectory analysis requires:
- **matplotlib**: For creating plots and visualizations
- **numpy**: For numerical computations
- **torch**: For tensor operations (already required by HMR4D)

If matplotlib is not available, the script will display a warning and skip the trajectory plotting.

## Example Output

The analysis provides detailed insights into the motion:

```
[Trajectory Analysis] Saved to outputs/demo/video_name/trajectory_analysis.png
[Trajectory Analysis] Total frames: 150
[Trajectory Analysis] Ground contact frames: 89 (59.3%)
[Trajectory Analysis] Trajectory length: 2.456 meters
[Trajectory Analysis] Height range: [0.000, 0.234] meters
[Gravity Analysis] Gravity correction applied to prevent 'flying' poses
[Gravity Analysis] Y-axis (gravity) is corrected by subtracting minimum Y value
[Gravity Analysis] This ensures the person stays on the ground level
[Ground Contact] Found 5 contact intervals:
[Ground Contact] Interval 1: frames 0-15 (duration: 16 frames)
[Ground Contact] Interval 2: frames 30-45 (duration: 16 frames)
...
```

## Integration with AMASS Format

The trajectory analysis works seamlessly with the AMASS format output:

1. **Gravity correction** is applied to the global SMPL parameters
2. **Ground contact detection** uses the same static confidence logits
3. **Coordinate transformation** for AMASS format preserves the corrected trajectory
4. **Analysis plots** show the final corrected motion that will be used in AMASS format

This ensures that the trajectory analysis accurately represents the motion that will be exported to the AMASS format for use in embodied pose systems. 