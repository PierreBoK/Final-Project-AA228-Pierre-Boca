import matplotlib.pyplot as plt

from ife.sensor_placement import compare_schedules_crlb

def main():
    print("Calculating precision gains for Random vs Recommended spread...")

    m_values = [2, 4, 6, 8, 10]

    results = compare_schedules_crlb(m_values=m_values)

    fig, ax = plt.subplots(figsize=(10, 6))

    # dashed red for random (stochastic), solid green for the optimized spread
    ax.plot(m_values, results["random"], marker='o', linestyle='--', linewidth=2,
            label="Random Distribution", color="#FF5733")

    ax.plot(m_values, results["recommended"], marker='*', markersize=12, linewidth=3,
            label="Recommended Spread (Optimized)", color="#33FF57")

    ax.axhline(20, color='red', linestyle=':', alpha=0.8, linewidth=2, label="20 µm Requirement")

    ax.set_title("Gain in Precision vs. Number of Measurements ($m$)\nRandom vs. Recommended Spread (CRLB)",
                 fontsize=14, fontweight='bold', pad=15)

    ax.set_xlabel("Number of Measurements", fontsize=12)
    ax.set_ylabel("Predicted Position Error at $t_f$ [µm]", fontsize=12)

    ax.set_xticks(m_values)

    ax.legend(fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.6)

    print("Displaying plot. Close the window to exit.")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()