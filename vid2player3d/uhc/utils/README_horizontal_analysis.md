# Horizontal Trajectory Drift Analysis

This document explains how to use the `analyze_horizontal_drift.py` script to analyze horizontal trajectory drifting in AMASS motion data.

## Overview

The script analyzes horizontal (X-Y plane) trajectories of characters and identifies potential drifting issues that are not corrected by the vertical ground correction algorithm. It provides interactive visualizations to help you understand:

1. **Horizontal Trajectory**: The path the character takes in the X-Y plane
2. **Ground Contact Points**: Where the character's feet make contact with the ground
3. **Velocity Profile**: How the character's speed changes over time
4. **Trajectory Statistics**: Efficiency, total distance, and other metrics

## Key Features

### 1. Ground Contact Detection
- Uses the same foot vertex detection as the ground correction algorithm
- Identifies when both feet are on the ground vs. single foot contact
- Marks contact points on the trajectory plot

### 2. Interactive Visualization
- **Main Trajectory Plot**: Shows the character's path with color-coded time progression
- **Velocity Plot**: Shows speed over time with ground contact markers
- **Statistics Panel**: Displays trajectory metrics and contact information
- **Hover Interaction**: Hover over the trajectory to see frame and time information

### 3. Drift Analysis
- **Trajectory Efficiency**: Ratio of straight-line distance to total path distance
- **Stationary Period Detection**: Identifies when the character is not moving
- **Contact Point Distribution**: Shows where ground contacts occur relative to movement

## Usage

### Basic Usage
```bash
python uhc/utils/analyze_horizontal_drift.py --amass_data data/amass/your_data.pkl
```

### Analyze Specific Sequence
```bash
python uhc/utils/analyze_horizontal_drift.py --amass_data data/amass/your_data.pkl --sequence_name "sequence_name"
```

### Analyze Multiple Sequences
```bash
python uhc/utils/analyze_horizontal_drift.py --amass_data data/amass/your_data.pkl --num_seq 10
```

## Understanding the Plots

### Main Trajectory Plot
- **Green dot**: Starting position
- **Red square**: Ending position
- **Blue triangles (^)**: Left foot ground contacts
- **Orange triangles (v)**: Right foot ground contacts
- **Purple X**: Single foot ground contacts
- **Color gradient**: Time progression (blue → yellow → red)

### Velocity Plot
- **Blue line**: Instantaneous velocity
- **Red line**: Smoothed velocity
- **Green dots**: Stationary periods
- **Red circles**: Ground contact points

### Statistics Panel
- **Total Distance**: Actual path length
- **Straight Line Distance**: Direct distance from start to end
- **Efficiency**: How direct the path is (higher = more direct)
- **Ground Contacts**: Number and type of detected contacts

## Interpreting Results

### Good Trajectory (Low Drift)
- High efficiency (> 0.8)
- Smooth velocity profile
- Ground contacts distributed along the path
- Minimal zigzagging or circular motion

### Problematic Trajectory (High Drift)
- Low efficiency (< 0.5)
- Erratic velocity changes
- Ground contacts clustered in certain areas
- Excessive back-and-forth movement

### Common Drift Patterns
1. **Circular Drift**: Character moves in circles
2. **Linear Drift**: Character gradually moves in one direction
3. **Oscillating Drift**: Character moves back and forth
4. **Stationary Drift**: Character stays in place but rotates

## Potential Solutions

If you identify horizontal drifting issues, consider:

1. **Root Translation Correction**: Apply similar multi-frequency windowing to X-Y coordinates
2. **Contact-Based Anchoring**: Use ground contact points as anchors for horizontal position
3. **Velocity Smoothing**: Apply smoothing to reduce erratic movement
4. **Directional Constraints**: Add constraints to maintain forward movement direction

## Example Output

The script will generate:
- Interactive matplotlib plots for each sequence
- Saved PNG files of the plots
- Console output with statistics

You can interact with the plots by:
- Hovering over the trajectory to see frame information
- Zooming and panning to examine details
- Using the navigation toolbar for additional controls

## Troubleshooting

### No Ground Contacts Detected
- Check if the foot vertex indices are correct for your SMPL model
- Adjust the `prominence` and `foot_threshold` parameters
- Verify the motion data contains realistic foot movements

### Poor Visualization Quality
- Increase the DPI when saving plots
- Adjust the figure size for better detail
- Use the interactive mode for better exploration

### Memory Issues
- Process fewer sequences at once
- Reduce the sequence length for analysis
- Close plots between sequences 
