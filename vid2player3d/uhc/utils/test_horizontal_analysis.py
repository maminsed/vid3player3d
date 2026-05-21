#!/usr/bin/env python3
"""
Test script for horizontal trajectory analysis.
This script demonstrates how to use the analyze_horizontal_drift.py functionality
on a single sequence for quick testing.
"""

import joblib
import numpy as np
import matplotlib.pyplot as plt
from vid3player.vid2player3d.uhc.utils.analyze_horizontal_drift import analyze_horizontal_trajectory, create_interactive_trajectory_plot

def test_single_sequence(amass_data_path, sequence_name=None):
    """
    Test the horizontal trajectory analysis on a single sequence.
    
    Args:
        amass_data_path: Path to the AMASS data file
        sequence_name: Specific sequence to analyze (optional)
    """
    
    print(f"Loading AMASS data from: {amass_data_path}")
    amass_data = joblib.load(amass_data_path)
    
    # Select a sequence to analyze
    if sequence_name:
        sequences = [seq for seq in amass_data.keys() if sequence_name in str(seq)]
        if not sequences:
            print(f"No sequence found containing '{sequence_name}'")
            return
        key_name = sequences[0]
    else:
        # Use the first sequence
        key_name = list(amass_data.keys())[0]
    
    print(f"Analyzing sequence: {key_name}")
    
    # Get sequence data
    smpl_data_entry = amass_data[key_name]
    pose_aa = smpl_data_entry['pose_aa'].copy()
    trans = smpl_data_entry['trans_orig'].copy()
    
    # For this test, we'll use the root translations directly
    # In the full script, you would process through SMPL to get vertices
    root_trans = trans  # Simplified for testing
    
    # Create dummy vertices for testing (in real usage, these come from SMPL)
    T = root_trans.shape[0]
    # Create a simple mesh-like structure for testing
    dummy_verts = np.zeros((T, 100, 3))
    for t in range(T):
        # Create a simple character-like mesh
        dummy_verts[t, :, 0] = root_trans[t, 0] + np.random.normal(0, 0.1, 100)  # X
        dummy_verts[t, :, 1] = root_trans[t, 1] + np.random.normal(0, 0.1, 100)  # Y
        dummy_verts[t, :, 2] = root_trans[t, 2] + np.random.normal(0, 0.05, 100)  # Z
    
    # Add some foot-like vertices at the bottom
    dummy_verts[:, 90:95, 2] -= 0.1  # Left foot
    dummy_verts[:, 95:100, 2] -= 0.1  # Right foot
    
    # Analyze the trajectory
    trajectory_data = analyze_horizontal_trajectory(
        root_trans, 
        dummy_verts, 
        fps=30.0,
        sequence_name=str(key_name)
    )
    
    # Create and display the plot
    fig = create_interactive_trajectory_plot(trajectory_data)
    
    # Save the plot
    plot_filename = f"test_horizontal_trajectory_{str(key_name).replace('/', '_')}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved test plot: {plot_filename}")
    
    # Show the plot
    plt.show()
    
    # Print summary statistics
    print("\nTrajectory Analysis Summary:")
    print(f"Sequence: {key_name}")
    print(f"Duration: {trajectory_data['time_coords'][-1]:.2f} seconds")
    print(f"Total Distance: {trajectory_data['total_distance']:.2f}")
    print(f"Straight Line Distance: {trajectory_data['straight_line_distance']:.2f}")
    print(f"Efficiency: {trajectory_data['efficiency']:.3f}")
    print(f"Ground Contacts: {len(trajectory_data['ground_contacts'])}")
    
    return trajectory_data

def create_synthetic_drift_example():
    """
    Create a synthetic example showing different types of horizontal drift.
    """
    print("Creating synthetic drift examples...")
    
    # Create time coordinates
    T = 300  # 10 seconds at 30 fps
    time_coords = np.arange(T) / 30.0
    
    # Example 1: Linear drift
    linear_drift = np.zeros((T, 3))
    linear_drift[:, 0] = np.linspace(0, 5, T)  # X: linear increase
    linear_drift[:, 1] = np.linspace(0, 2, T)  # Y: slight drift
    linear_drift[:, 2] = np.zeros(T)  # Z: constant
    
    # Example 2: Circular drift
    circular_drift = np.zeros((T, 3))
    radius = 2.0
    angular_velocity = 0.5  # radians per second
    circular_drift[:, 0] = radius * np.cos(angular_velocity * time_coords)
    circular_drift[:, 1] = radius * np.sin(angular_velocity * time_coords)
    circular_drift[:, 2] = np.zeros(T)
    
    # Example 3: Oscillating drift
    oscillating_drift = np.zeros((T, 3))
    oscillating_drift[:, 0] = np.linspace(0, 3, T)  # Forward movement
    oscillating_drift[:, 1] = 0.5 * np.sin(2 * np.pi * time_coords / 2)  # Side-to-side
    oscillating_drift[:, 2] = np.zeros(T)
    
    # Create dummy vertices for each example
    def create_dummy_verts(root_trans):
        T = root_trans.shape[0]
        dummy_verts = np.zeros((T, 100, 3))
        for t in range(T):
            dummy_verts[t, :, 0] = root_trans[t, 0] + np.random.normal(0, 0.1, 100)
            dummy_verts[t, :, 1] = root_trans[t, 1] + np.random.normal(0, 0.1, 100)
            dummy_verts[t, :, 2] = root_trans[t, 2] + np.random.normal(0, 0.05, 100)
        # Add foot vertices
        dummy_verts[:, 90:95, 2] -= 0.1
        dummy_verts[:, 95:100, 2] -= 0.1
        return dummy_verts
    
    # Analyze each example
    examples = [
        ("Linear_Drift", linear_drift),
        ("Circular_Drift", circular_drift),
        ("Oscillating_Drift", oscillating_drift)
    ]
    
    for name, root_trans in examples:
        print(f"\nAnalyzing {name}...")
        dummy_verts = create_dummy_verts(root_trans)
        
        trajectory_data = analyze_horizontal_trajectory(
            root_trans, 
            dummy_verts, 
            fps=30.0,
            sequence_name=name
        )
        
        # Create plot
        fig = create_interactive_trajectory_plot(trajectory_data)
        
        # Save plot
        plot_filename = f"synthetic_{name.lower()}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {plot_filename}")
        
        # Show plot
        plt.show()
        
        # Print statistics
        print(f"Efficiency: {trajectory_data['efficiency']:.3f}")
        print(f"Ground Contacts: {len(trajectory_data['ground_contacts'])}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test horizontal trajectory analysis")
    parser.add_argument('--amass_data', type=str, default="data/amass/amass_copycat_take5_5.pkl",
                       help="Path to AMASS data file")
    parser.add_argument('--sequence_name', type=str, default=None,
                       help="Specific sequence to analyze")
    parser.add_argument('--synthetic', action='store_true',
                       help="Create synthetic drift examples instead of using real data")
    
    args = parser.parse_args()
    
    if args.synthetic:
        create_synthetic_drift_example()
    else:
        test_single_sequence(args.amass_data, args.sequence_name) 
