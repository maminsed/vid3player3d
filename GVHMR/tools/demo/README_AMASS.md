# AMASS Format Output for demo_amass.py

This document explains how to use the modified `demo_amass.py` script to output AMASS format files that can be used with `convert_amass_isaac.py` in the vid2player3d project.

## Overview

The `demo_amass.py` script has been modified to include an option to save HMR4D predictions in AMASS format. This allows you to convert video-based motion capture results into a format that can be processed by the `convert_amass_isaac.py` script for use in the vid2player3d embodied pose system.

## Usage

### Basic Usage with AMASS Output

```bash
cd GVHMR
python tools/demo/demo_amass.py --video inputs/demo/your_video.mp4 --save_amass
```

This will:
1. Process the video using HMR4D
2. Generate the standard output videos (incam, global, merged)
3. Save an additional AMASS format file at `outputs/demo/your_video_amass.pkl`

### Custom AMASS Output Path

```bash
python tools/demo/demo_amass.py --video inputs/demo/your_video.mp4 --save_amass --amass_output /path/to/custom/output.pkl
```

### All Available Options

```bash
python tools/demo/demo_amass.py \
    --video inputs/demo/your_video.mp4 \
    --output_root /custom/output/dir \
    --save_amass \
    --amass_output /custom/amass/output.pkl \
    --static_cam \
    --use_dpvo \
    --f_mm 24 \
    --verbose
```

## AMASS Format Structure

The generated AMASS format file contains:

```python
{
    "demo_video_name": {
        "pose_aa": np.array,      # (L, 72) - SMPL pose parameters in axis-angle format
        "trans_orig": np.array,   # (L, 3) - translation parameters
        "beta": np.array,         # (10,) - body shape parameters
        "gender": str,            # gender information (default: "neutral")
        "fps": float             # frame rate (default: 30.0)
    }
}
```

Where:
- `L` is the number of frames in the video
- `pose_aa` contains 72 parameters: 3 for root rotation + 69 for body joints
- `trans_orig` contains 3 translation parameters (x, y, z)
- `beta` contains 10 body shape parameters
- The sequence name is automatically generated as `demo_<video_name>`

## Integration with convert_amass_isaac.py

After generating the AMASS format file, you can use it with the `convert_amass_isaac.py` script:

```bash
cd vid2player3d
python uhc/utils/convert_amass_isaac.py \
    --amass_data GVHMR/outputs/demo/your_video_amass.pkl \
    --out_dir data/motion_lib/amass \
    --num_seq 1 \
    --num_motion_libs 1
```

## Testing

Use the provided test script to verify that your AMASS format file is compatible:

```bash
cd GVHMR
python tools/demo/test_amass_conversion.py outputs/demo/your_video_amass.pkl
```

This will:
1. Check the AMASS format structure
2. Simulate the `convert_amass_isaac.py` process
3. Verify compatibility

## Technical Details

### SMPL Parameter Mapping

The HMR4D output contains:
- `smpl_params_global["global_orient"]`: (L, 3) - root rotation in axis-angle
- `smpl_params_global["body_pose"]`: (L, 69) - body joint rotations in axis-angle
- `smpl_params_global["transl"]`: (L, 3) - translation
- `smpl_params_global["betas"]`: (L, 10) - body shape parameters

These are combined into the AMASS format:
- `pose_aa = np.concatenate([global_orient, body_pose], axis=1)` - (L, 72)
- `trans_orig = transl` - (L, 3)
- `beta = betas[0]` - (10,) - using first frame's shape parameters

### Coordinate System

The AMASS format uses the global coordinate system from HMR4D, which provides motion in a world coordinate frame suitable for embodied pose applications.

## Troubleshooting

### Common Issues

1. **File not found**: Ensure the video file exists and the path is correct
2. **Memory issues**: For long videos, consider processing shorter segments
3. **Shape mismatch**: If you see shape errors, check that the HMR4D model output structure hasn't changed

### Debugging

Use the `--verbose` flag to see intermediate processing steps:

```bash
python tools/demo/demo_amass.py --video your_video.mp4 --save_amass --verbose
```

### Validation

The test script will help identify any issues with the generated AMASS format file before using it with `convert_amass_isaac.py`.

## Example Workflow

1. **Process video with HMR4D**:
   ```bash
   cd GVHMR
   python tools/demo/demo_amass.py --video inputs/demo/dance.mp4 --save_amass
   ```

2. **Test AMASS format**:
   ```bash
   python tools/demo/test_amass_conversion.py outputs/demo/dance_amass.pkl
   ```

3. **Convert for vid2player3d**:
   ```bash
   cd ../vid2player3d
   python uhc/utils/convert_amass_isaac.py --amass_data ../GVHMR/outputs/demo/dance_amass.pkl --out_dir data/motion_lib/amass
   ```

4. **Use in embodied pose system**:
   The motion library files will be available for use in the vid2player3d embodied pose training and evaluation. 
