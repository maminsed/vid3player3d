#!/usr/bin/env python3
"""
Test script to verify AMASS format conversion from demo_amass.py to convert_amass_isaac.py
"""

import joblib
import numpy as np
import os
import sys
from pathlib import Path

def test_amass_format(amass_file_path):
    """Test if the AMASS format file has the correct structure"""
    print(f"Testing AMASS format file: {amass_file_path}")
    
    if not os.path.exists(amass_file_path):
        print(f"Error: File {amass_file_path} does not exist")
        return False
    
    try:
        # Load the AMASS data
        amass_data = joblib.load(amass_file_path)
        print(f"Successfully loaded AMASS data with {len(amass_data)} sequences")
        
        # Check structure
        for seq_name, seq_data in amass_data.items():
            print(f"\nSequence: {seq_name}")
            
            # Check required fields
            required_fields = ['pose_aa', 'trans_orig', 'beta', 'gender', 'fps']
            for field in required_fields:
                if field not in seq_data:
                    print(f"Error: Missing required field '{field}'")
                    return False
                print(f"  {field}: {type(seq_data[field])} - {seq_data[field].shape if hasattr(seq_data[field], 'shape') else seq_data[field]}")
            
            # Check specific requirements
            pose_aa = seq_data['pose_aa']
            trans_orig = seq_data['trans_orig']
            beta = seq_data['beta']
            
            # Check shapes
            if pose_aa.shape[1] != 72:
                print(f"Error: pose_aa should have 72 parameters, got {pose_aa.shape[1]}")
                return False
            
            if trans_orig.shape[1] != 3:
                print(f"Error: trans_orig should have 3 parameters, got {trans_orig.shape[1]}")
                return False
            
            if len(beta) != 10:
                print(f"Error: beta should have 10 parameters, got {len(beta)}")
                return False
            
            print(f"  Sequence length: {pose_aa.shape[0]} frames")
            print(f"  All checks passed!")
        
        return True
        
    except Exception as e:
        print(f"Error loading AMASS file: {e}")
        return False

def simulate_convert_amass_isaac(amass_file_path):
    """Simulate the convert_amass_isaac.py process to verify compatibility"""
    print(f"\nSimulating convert_amass_isaac.py process...")
    
    try:
        # Load AMASS data (same as convert_amass_isaac.py)
        amass_data = joblib.load(amass_file_path)
        
        # Process each sequence (simplified version of convert_amass_isaac.py)
        for key_name, smpl_data_entry in amass_data.items():
            print(f"\nProcessing sequence: {key_name}")
            
            # Extract data (same as convert_amass_isaac.py)
            pose_aa = smpl_data_entry['pose_aa'].copy()
            trans = smpl_data_entry['trans_orig'].copy()
            beta = smpl_data_entry['beta'][:10].copy()
            gender = smpl_data_entry['gender']
            fps = smpl_data_entry.get('fps', 30.0)
            
            print(f"  pose_aa shape: {pose_aa.shape}")
            print(f"  trans shape: {trans.shape}")
            print(f"  beta shape: {beta.shape}")
            print(f"  gender: {gender}")
            print(f"  fps: {fps}")
            
            # Verify data types and ranges
            if not np.isfinite(pose_aa).all():
                print(f"  Warning: pose_aa contains non-finite values")
            
            if not np.isfinite(trans).all():
                print(f"  Warning: trans contains non-finite values")
            
            if not np.isfinite(beta).all():
                print(f"  Warning: beta contains non-finite values")
            
            print(f"  Sequence processed successfully!")
        
        print(f"\nAll sequences processed successfully!")
        return True
        
    except Exception as e:
        print(f"Error in convert_amass_isaac simulation: {e}")
        return False

def main():
    """Main test function"""
    print("AMASS Format Conversion Test")
    print("=" * 50)
    
    # Check if AMASS file path is provided
    if len(sys.argv) < 2:
        print("Usage: python test_amass_conversion.py <amass_file_path>")
        print("Example: python test_amass_conversion.py outputs/demo/video_amass.pkl")
        return
    
    amass_file_path = sys.argv[1]
    
    # Test 1: Check AMASS format structure
    print("\nTest 1: AMASS Format Structure")
    print("-" * 30)
    if not test_amass_format(amass_file_path):
        print("Test 1 FAILED")
        return
    print("Test 1 PASSED")
    
    # Test 2: Simulate convert_amass_isaac.py process
    print("\nTest 2: Convert AMASS Isaac Compatibility")
    print("-" * 30)
    if not simulate_convert_amass_isaac(amass_file_path):
        print("Test 2 FAILED")
        return
    print("Test 2 PASSED")
    
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED!")
    print("The AMASS format file is compatible with convert_amass_isaac.py")
    print("\nYou can now use the file with:")
    print(f"python uhc/utils/convert_amass_isaac.py --amass_data {amass_file_path} --out_dir data/motion_lib/amass")

if __name__ == "__main__":
    main() 