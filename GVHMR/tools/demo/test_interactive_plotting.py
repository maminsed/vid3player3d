#!/usr/bin/env python3
"""
Test script for interactive trajectory plotting functionality.
This script demonstrates the interactive plotting features.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

def create_mock_prediction_data():
    """Create mock prediction data for testing interactive plotting."""
    # Create mock translation data (L, 3) - simulating a walking motion
    L = 100  # 100 frames
    t = np.linspace(0, 4*np.pi, L)
    
    # Create a figure-8 pattern for X-Z, with some height variation
    transl = np.zeros((L, 3))
    transl[:, 0] = 2 * np.sin(t)  # X: figure-8 pattern
    transl[:, 2] = np.sin(2*t)    # Z: figure-8 pattern
    transl[:, 1] = 0.1 + 0.05 * np.sin(3*t)  # Y: slight height variation
    
    # Create mock static confidence logits (L, 6)
    # Simulate ground contact patterns
    static_conf_logits = np.zeros((L, 6))
    
    # Simulate walking pattern: alternating foot contact
    for i in range(L):
        # Left foot contact (frames 0-25, 50-75)
        if (i < 25) or (50 <= i < 75):
            static_conf_logits[i, 0] = 5.0  # L_Ankle high confidence
            static_conf_logits[i, 1] = 4.0  # L_foot high confidence
            static_conf_logits[i, 2] = -2.0  # R_Ankle low confidence
            static_conf_logits[i, 3] = -2.0  # R_foot low confidence
        else:
            static_conf_logits[i, 0] = -2.0  # L_Ankle low confidence
            static_conf_logits[i, 1] = -2.0  # L_foot low confidence
            static_conf_logits[i, 2] = 5.0   # R_Ankle high confidence
            static_conf_logits[i, 3] = 4.0   # R_foot high confidence
        
        # Wrist joints always low confidence (not in contact)
        static_conf_logits[i, 4] = -3.0  # L_wrist
        static_conf_logits[i, 5] = -3.0  # R_wrist
    
    # Create mock prediction structure
    pred = {
        "smpl_params_global": {
            "transl": torch.from_numpy(transl)
        },
        "net_outputs": {
            "static_conf_logits": torch.from_numpy(static_conf_logits).unsqueeze(0)  # Add batch dimension
        }
    }
    
    return pred

def test_interactive_plotting():
    """Test the interactive trajectory plotting functionality."""
    print("Creating mock prediction data...")
    pred = create_mock_prediction_data()
    
    # Create output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Create mock config
    class MockConfig:
        def __init__(self):
            self.output_dir = output_dir
    
    cfg = MockConfig()
    
    # Create mock amass settings with interactive mode
    amass_settings = {'interactive': True}
    
    print("Testing interactive trajectory plotting...")
    print("Interactive plots will open in separate windows.")
    print("You can zoom, pan, and explore the data interactively.")
    print("Close the plot windows to continue...")
    
    # Import and test the plotting function
    try:
        from vid3player.GVHMR.tools.demo.demo_amass_gravity_and_xy_correct import plot_trajectory_and_ground_contact
        plot_trajectory_and_ground_contact(cfg, pred)
        print("✓ Interactive trajectory plotting test completed successfully!")
        
    except Exception as e:
        print(f"✗ Interactive trajectory plotting test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_interactive_plotting() 
