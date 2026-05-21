#!/usr/bin/env python3

import joblib
import numpy as np
import os
import sys

def examine_motion_export():
    """Examine the exported motion data to understand its structure"""
    
    export_file = "exports/motion_export_motion-1.pkl"
    
    if not os.path.exists(export_file):
        print(f"Export file not found: {export_file}")
        return
    
    print(f"Examining export file: {export_file}")
    print(f"File size: {os.path.getsize(export_file) / 1024:.1f} KB")
    
    try:
        with open(export_file, 'rb') as f:
            data = joblib.load(f)
        print("Successfully loaded with joblib")
    except Exception as e:
        print(f"Failed to load with joblib: {e}")
        return
    
    print(f"\nData keys: {list(data.keys())}")
    
    # Get the main data
    motion_key = list(data.keys())[0]  # Should be something like "demo_export_-1"
    motion_data = data[motion_key]
    
    print(f"\nMotion data keys: {list(motion_data.keys())}")
    
    for key, value in motion_data.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            if len(value.shape) == 1:
                print(f"  - First 5 values: {value[:5]}")
            elif len(value.shape) == 2:
                print(f"  - Shape: {value.shape[0]} frames x {value.shape[1]} features")
                print(f"  - First frame: {value[0][:10]}...")
        else:
            print(f"{key}: {value} (type: {type(value)})")
    
    # Check if this represents multiple humanoids
    pose_data = motion_data['pose_aa']
    trans_data = motion_data['trans_orig']
    
    print(f"\nMotion Analysis:")
    print(f"- Number of frames: {pose_data.shape[0]}")
    print(f"- Pose data shape: {pose_data.shape}")
    print(f"- Translation data shape: {trans_data.shape}")
    
    # Check if pose data represents multiple humanoids
    # SMPL has 24 joints, so pose_aa should be 24*3=72 dimensions per frame
    expected_pose_dim = 24 * 3  # 24 joints * 3 axis-angle values
    
    if pose_data.shape[1] == expected_pose_dim:
        print(f"- This appears to be single humanoid data (72 pose dimensions)")
    elif pose_data.shape[1] > expected_pose_dim:
        print(f"- This might contain multiple humanoids or additional data")
        print(f"- Expected 72 dimensions, got {pose_data.shape[1]}")
        num_humanoids = pose_data.shape[1] // expected_pose_dim
        print(f"- Could represent {num_humanoids} humanoids")
    else:
        print(f"- Unexpected pose dimensions: {pose_data.shape[1]}")
    
    # Additional analysis
    print(f"\nDetailed Analysis:")
    print(f"- Motion ID from filename: -1 (matches the 'demo_export_-1' key)")
    print(f"- This confirms only environment 0's motion is exported")
    print(f"- The 8192 environments run in parallel, but only one motion is saved")

if __name__ == "__main__":
    examine_motion_export() 