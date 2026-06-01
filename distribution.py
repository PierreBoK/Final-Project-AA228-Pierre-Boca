import numpy as np
import matplotlib.pyplot as plt

from ife.environment import IFEEnvironment


def main():
    print("Initializing IFE Environment...")
    # fixed seed so the scatter plot looks the same every run
    env = IFEEnvironment(seed=42)

    n_shots = 500
    x_points = []
    y_points = []

    print(f"Simulating {n_shots} shots...")
    for i in range(n_shots):
        states = env._simulate_trajectory()

        # grab x, y at the very last timestep (firing time t_f)
        x_end = states[-1, 0]
        y_end = states[-1, 1]

        # convert to µm so the numbers are readable
        x_points.append(x_end * 1e6)
        y_points.append(y_end * 1e6)

    fig, ax = plt.subplots(figsize=(8, 8))

    # tried drawing target rings but they clutter the plot at this scale
    #circle_10 = plt.Circle((0, 0), 10, color='gray', fill=False, linestyle='--', alpha=0.6)
    #circle_20 = plt.Circle((0, 0), 20, color='green', fill=False, linewidth=2, label='20 µm Requirement Limit')
    #circle_30 = plt.Circle((0, 0), 30, color='gray', fill=False, linestyle='--', alpha=0.6)

    #ax.add_patch(circle_10)
    #ax.add_patch(circle_20)
    #ax.add_patch(circle_30)

    ax.scatter(x_points, y_points, color='red', marker='x', s=35, label=f'Particle Endpoints ({n_shots} shots)')

    ax.axhline(0, color='black', linewidth=0.8, alpha=0.7)
    ax.axvline(0, color='black', linewidth=0.8, alpha=0.7)

    ax.set_aspect('equal')  # keep it square so the spread looks right
    ax.set_xlim(-3000, 3000)
    ax.set_ylim(-3000, 3000)

    ax.set_title("Distribution of the shots with a standard deviation of 1mm\n500 Simulated Injections",
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("X Position [µm]", fontsize=12)
    ax.set_ylabel("Y Position [µm]", fontsize=12)

    ax.legend(loc='upper right')
    ax.grid(False, linestyle=':', alpha=0.5)

    print("Displaying 2D target plot. Close the window to exit.")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()