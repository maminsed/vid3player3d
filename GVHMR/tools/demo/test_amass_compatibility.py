#!/usr/bin/env python3
"""
Test script to verify AMASS format compatibility with convert_amass_isaac.py
"""

import joblib
import numpy as np
import os
import sys
from pathlib import Path

def test_amass_compatibility(amass_file_path):
    """Test if the AMASS format file is compatible with convert_amass_isaac.py"""
    print(f"Testing AMASS format file: {amass_file_path}")
    
    if not os.path.exists(amass_file_path):
        print(f"Error: File {amass_file_path} does not exist")
        return False
    
    try:
        # Load the AMASS data
        amass_data = joblib.load(amass_file_path)
        print(f"Successfully loaded AMASS data with {len(amass_data)} sequences")
        
        # Check structure for each sequence
        for seq_name, seq_data in amass_data.items():
            print(f"\nSequence: {seq_name}")
            
            # Check required fields
            required_fields = ['pose_aa', 'trans_orig', 'beta', 'gender']
            for field in required_fields:
                if field not in seq_data:
                    print(f"  ERROR: Missing required field '{field}'")
                    return False
                else:
                    print(f"  ✓ {field}: {type(seq_data[field])} - {seq_data[field].shape if hasattr(seq_data[field], 'shape') else 'scalar'}")
            
            # Check specific requirements
            pose_aa = seq_data['pose_aa']
            trans_orig = seq_data['trans_orig']
            beta = seq_data['beta']
            gender = seq_data['gender']
            
            # Check pose_aa shape (should be L x 72)
            if pose_aa.shape[1] != 72:
                print(f"  ERROR: pose_aa should have 72 columns, got {pose_aa.shape[1]}")
                return False
            
            # Check trans_orig shape (should be L x 3)
            if trans_orig.shape[1] != 3:
                print(f"  ERROR: trans_orig should have 3 columns, got {trans_orig.shape[1]}")
                return False
            
            # Check beta shape (should be 10)
            if len(beta) != 10:
                print(f"  ERROR: beta should have 10 elements, got {len(beta)}")
                return False
            
            # Check gender (should be string)
            if not isinstance(gender, str):
                print(f"  ERROR: gender should be string, got {type(gender)}")
                return False
            
            # Check sequence length consistency
            if pose_aa.shape[0] != trans_orig.shape[0]:
                print(f"  ERROR: pose_aa and trans_orig should have same sequence length")
                return False
            
            print(f"  ✓ Sequence length: {pose_aa.shape[0]}")
            print(f"  ✓ All checks passed for sequence {seq_name}")
        
        print(f"\n✓ All sequences passed compatibility checks!")
        print(f"✓ This AMASS file should work with convert_amass_isaac.py")
        return True
        
    except Exception as e:
        print(f"Error loading or processing AMASS file: {e}")
        return False

def main():
    """Main function to test AMASS compatibility"""
    # Look for AMASS files in the outputs directory
    outputs_dir = Path("outputs/demo")
    
    if not outputs_dir.exists():
        print(f"Error: Outputs directory {outputs_dir} does not exist")
        return
    
    # Find AMASS files
    amass_files = list(outputs_dir.rglob("*_amass.pkl"))
    
    if not amass_files:
        print(f"No AMASS files found in {outputs_dir}")
        return
    
    print(f"Found {len(amass_files)} AMASS files:")
    for amass_file in amass_files:
        print(f"  - {amass_file}")
    
    # Test each file
    for amass_file in amass_files:
        print(f"\n{'='*60}")
        success = test_amass_compatibility(amass_file)
        if success:
            print(f"✓ {amass_file} is compatible with convert_amass_isaac.py")
        else:
            print(f"✗ {amass_file} has compatibility issues")

if __name__ == "__main__":
    main() 