#!/usr/bin/env python3
"""
Simple test for interactive matplotlib plotting.
"""

import matplotlib.pyplot as plt
import numpy as np

def test_interactive():
    """Test basic interactive plotting."""
    print("Testing interactive matplotlib plotting...")
    
    # Create a simple plot
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interactive Test Plot')
    plt.legend()
    plt.grid(True)
    
    print("Plot should open in a new window.")
    print("You can zoom, pan, and interact with it.")
    print("Close the window to continue...")
    
    try:
        plt.show(block=True)
        print("✓ Interactive plotting works!")
    except Exception as e:
        print(f"✗ Interactive plotting failed: {e}")
        print("Falling back to saving plot...")
        plt.savefig('test_interactive.png')
        print("✓ Plot saved as test_interactive.png")
    finally:
        plt.close()

if __name__ == "__main__":
    test_interactive() 