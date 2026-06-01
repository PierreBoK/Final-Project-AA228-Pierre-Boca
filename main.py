import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm  # Added to easily generate 10 colors

# Import the environment and configuration from your files
from ife.environment import IFEEnvironment
from ife.config import TARGET_SPEED_NOM, DT, N_MEASUREMENTS, TAU_LATENCY

def main():
    print("Initializing IFE Environment...")
    # Create the environment. We'll use a fixed seed for reproducibility.
    env = IFEEnvironment(seed=42)
    
    # ── Visual Scaling Factors ───────────────────────────────────────────────
    SCALE_X = 100.0       # Magnify X variation 1,000x
    SCALE_Y = 100.0       # Magnify Y variation 1,000x
    SCALE_Z = 1          # Compress Z to fit the fast longitudinal flight into a readable box
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_sublegend() if hasattr(fig, 'add_sublegend') else fig.add_subplot(111, projection='3d')
    
    # Grab 10 distinct colors from matplotlib's tab10 colormap
    colors = cm.tab10(np.linspace(0, 1, 10))
    
    print("Simulating 10 particles...")
    for i in range(10):
        # Simulate a unique trajectory using the environment's internal kinematics
        states = env._simulate_trajectory()
        
        # Extract the X, Y, Z positions (first 3 columns of the 9D state vector)
        x_raw = states[:, 0]
        y_raw = states[:, 1]
        z_raw = states[:, 2]
        
        # Apply the visual magnification/scaling
        x_scaled = x_raw * SCALE_X
        y_scaled = y_raw * SCALE_Y
        z_scaled = z_raw * SCALE_Z
        
        # Plot the continuous trajectory line
        ax.plot(x_scaled, y_scaled, z_scaled, 
                label=f'Particle {i+1}', 
                color=colors[i], 
                linewidth=2)
        
        # Mark the start (injection point) and end (firing point t_f)
        ax.scatter(x_scaled[0], y_scaled[0], z_scaled[0], color=colors[i], marker='o', s=50)
        ax.scatter(x_scaled[-1], y_scaled[-1], z_scaled[-1], color=colors[i], marker='*', s=100, 
                   label='Firing Point' if i==0 else "")

    # ── Plot Formatting ──────────────────────────────────────────────────────
    ax.set_title("3D Target Trajectories", 
                 fontsize=14, fontweight='bold', pad=20)
    
    ax.set_xlabel(f"X Position (Scaled ×{SCALE_X}) [m]", fontsize=11, labelpad=10)
    ax.set_ylabel(f"Y Position (Scaled ×{SCALE_Y}) [m]", fontsize=11, labelpad=10)
    ax.set_zlabel(f"Z Position (Scaled ×{SCALE_Z}) [m]", fontsize=11, labelpad=10)
    
    # Move the legend outside the 3D plot box so it doesn't cover the data
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Adjust view angle for ideal 3D perspective
    ax.view_init(elev=20, azim=-45)
    
    print("Displaying plot. Close the window to exit.")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()