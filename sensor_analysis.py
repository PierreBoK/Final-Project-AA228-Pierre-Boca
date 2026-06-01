"""
sensor_analysis.py — compare six sensor placement strategies

Fig 1: CRLB precision vs number of measurements m (analytical)
Fig 2: Monte Carlo scatter of KF prediction error in the XY plane (each panel at its own scale)
Fig 3: Timeline view of each schedule
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

from ife.environment import IFEEnvironment
from ife.sensor_placement import (
    consecutive_schedule,
    random_schedule,
    recommended_spread_schedule,
    end_clustered_schedule,
    two_anchor_schedule,
    chebyshev_schedule,
    geometric_schedule,
    _t_f_idx,
    crlb_position_error,
)
from ife.config import DT, TAU_LATENCY, SIGMA_TRANSVERSE, ACCURACY_3SIGMA

matplotlib.rcParams.update({
    "text.usetex":        False,
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Times New Roman", "serif"],
    "font.size":          10,
    "axes.labelsize":     11,
    "axes.titlesize":     10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "legend.fontsize":    8.5,
    "legend.framealpha":  0.92,
    "legend.edgecolor":   "0.75",
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "figure.dpi":         130,
    "savefig.dpi":        300,
    "axes.linewidth":     0.8,
    "grid.linewidth":     0.4,
    "grid.alpha":         0.35,
    "lines.linewidth":    1.8,
    "lines.markersize":   6,
})

N_REF    = 10           # reference measurement count (sets t_f_idx)
M_FIXED  = 6            # fixed m for Fig 2 scatter
M_VALUES = list(range(2, 11))
N_MC     = 500

REQ_UM = ACCURACY_3SIGMA * 1e6   # 20 µm

# each entry: (label, schedule_fn, color, marker, linestyle)
# SCATTER_STRATEGIES drops Random since it's non-deterministic

CRLB_STRATEGIES = [
    ("Consecutive (default)", consecutive_schedule,         "#C0392B", "o",  ":" ),
    ("End-Clustered",          end_clustered_schedule,       "#E67E22", "^",  "-."),
    ("Uniform Spread",         recommended_spread_schedule,  "#27AE60", "s",  "-" ),
    ("Geometric",              geometric_schedule,           "#2980B9", "P",  "--"),
    ("Chebyshev",              chebyshev_schedule,           "#8E44AD", "D",  "--"),
    ("Two-Anchor (D-Optimal)", two_anchor_schedule,          "#1A5276", "v",  "-" ),
    ("Random",                 random_schedule,              "#7F8C8D", "x",  ":" ),
]

SCATTER_STRATEGIES = [s for s in CRLB_STRATEGIES if s[0] != "Random"]


# Figure 1 — CRLB

def compute_crlb() -> dict:
    t_f = _t_f_idx(N_REF, TAU_LATENCY, DT)
    results = {}
    for label, fn, *_ in CRLB_STRATEGIES:
        errs = []
        for m in M_VALUES:
            steps = fn(m, n_meas_ref=N_REF)
            err   = crlb_position_error(steps, SIGMA_TRANSVERSE, t_f, DT)
            errs.append(err * 1e6)
        results[label] = errs
    return results


def plot_crlb(results: dict,
              save_path: str = "figures/fig1_sensor_crlb.pdf") -> plt.Figure:

    fig, ax = plt.subplots(figsize=(6.8, 4.4))

    # shade the "good" region first so it sits behind the curves
    ax.fill_between(
        [M_VALUES[0] - 0.5, M_VALUES[-1] + 0.5],
        [1e-4, 1e-4], [REQ_UM, REQ_UM],
        alpha=0.06, color="#27AE60", zorder=0,
    )
    ax.axhline(REQ_UM, color="black", linestyle="--",
               linewidth=1.0, alpha=0.75, zorder=4,
               label=r"20 $\mu$m engagement requirement")

    for label, _, color, marker, ls in CRLB_STRATEGIES:
        ax.semilogy(
            M_VALUES, results[label],
            marker=marker, linestyle=ls, color=color,
            linewidth=1.8, markersize=6,
            label=label, clip_on=False, zorder=3,
        )

    ax.text(M_VALUES[-1] + 0.12, REQ_UM * 1.12,
            r"$\leq 20\,\mu$m", fontsize=8.5, va="bottom",
            ha="right", color="black", alpha=0.75)

    ax.set_xlabel(r"Number of measurements, $m$")
    ax.set_ylabel(r"CRLB per-axis prediction error at $t_f$ [$\mu$m]")
    ax.set_title(
        "Effect of Sensor Placement on Tracking Precision\n"
        r"Cramér–Rao Lower Bound  ($\sigma_{\rm meas}=" +
        f"{SIGMA_TRANSVERSE*1e6:.0f}" + r"\,\mu$m, "
        r"$\tau_{\rm latency}=" + f"{TAU_LATENCY*1e3:.0f}" + r"\,$ms)",
        pad=8,
    )
    ax.set_xticks(M_VALUES)
    ax.set_xlim(M_VALUES[0] - 0.4, M_VALUES[-1] + 0.4)
    ax.set_ylim(bottom=5e-3)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: (f"{x:.3f}" if x < 0.1 else
                      f"{x:.2f}" if x < 1   else
                      f"{x:.1f}" if x < 10  else f"{x:.0f}")
    ))
    ax.yaxis.set_minor_locator(mticker.NullLocator())

    ax.legend(loc="upper right", ncol=1, fontsize=8.5,
              handlelength=2.4, columnspacing=1.0)
    ax.grid(True, which="major", linestyle=":", alpha=0.35)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"  [saved] {save_path}")
    return fig


# Figure 2 — MC scatter

def run_mc(m_fixed: int = M_FIXED, n_mc: int = N_MC) -> dict:
    """Run n_mc episodes per strategy, returns label → (Δx_µm, Δy_µm)."""
    t_f = _t_f_idx(N_REF, TAU_LATENCY, DT)
    print(f"  t_f = step {t_f},  max_step = {t_f-1},  m = {m_fixed}")
    print(f"  Running {len(SCATTER_STRATEGIES)} strategies × {n_mc} episodes …\n")

    data = {}
    for i, (label, fn, color, *_) in enumerate(SCATTER_STRATEGIES):
        steps = fn(m_fixed, n_meas_ref=N_REF)
        t_meas = np.array(steps) * DT * 1e3
        S_tt   = np.sum((t_meas - t_meas.mean()) ** 2)
        tau_r  = (t_f - steps[-1]) * DT * 1e3

        env = IFEEnvironment(
            n_measurements    = N_REF,
            measurement_steps = steps,
            seed              = 200 + i * 997,
        )
        dx_arr = np.empty(n_mc)
        dy_arr = np.empty(n_mc)
        for ep in range(n_mc):
            res      = env.run_episode()
            err      = (res["r_hat"] - res["r_true_tf"]) * 1e6
            dx_arr[ep] = err[0]
            dy_arr[ep] = err[1]

        r2d  = np.sqrt(dx_arr**2 + dy_arr**2)
        rms  = np.sqrt(np.mean(r2d ** 2))
        frac = np.mean(r2d <= REQ_UM) * 100
        print(f"  {label:<28s}  steps={steps}")
        print(f"    S_tt={S_tt:.1f} ms2  tau_rem={tau_r:.0f} ms  "
              f"2D-RMS={rms:.1f} um  within 20 um: {frac:.0f}%\n")
        data[label] = (dx_arr, dy_arr)

    return data


def plot_scatter(data: dict,
                 m_fixed: int = M_FIXED,
                 save_path: str = "figures/fig2_prediction_scatter.pdf"
                 ) -> plt.Figure:
    """2×3 grid of scatter plots, each at its own scale so all clouds are visible."""
    ncols = 3
    nrows = 2

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6.8, 4.8),
        gridspec_kw={"hspace": 0.54, "wspace": 0.35},
    )
    axes_flat = axes.flatten()

    for idx, (label, _, color, marker, _) in enumerate(SCATTER_STRATEGIES):
        ax    = axes_flat[idx]
        dx, dy = data[label]
        r2d   = np.sqrt(dx**2 + dy**2)
        rms2d = np.sqrt(np.mean(r2d ** 2))
        frac  = np.mean(r2d <= REQ_UM) * 100

        # axis limit: at least 1.7× the requirement, but stretch if needed
        p98  = np.percentile(r2d, 98)
        lim  = max(REQ_UM * 1.7, p98 * 1.15)
        lim  = (np.ceil(lim / 5) * 5)   # round to nearest 5 µm

        circ_req = Circle((0, 0), REQ_UM,
                          fill=False, edgecolor="black",
                          linewidth=0.9, linestyle="--", zorder=4)
        ax.add_patch(circ_req)

        sigma_1 = np.sqrt(np.mean(dx**2 + dy**2) / 2)   # isotropic σ
        ax.add_patch(Circle((0, 0), sigma_1,
                            facecolor=color, alpha=0.20,
                            linewidth=0, zorder=1))
        ax.add_patch(Circle((0, 0), 2 * sigma_1,
                            facecolor=color, alpha=0.08,
                            linewidth=0, zorder=1))

        # rasterized keeps PDF file size manageable with 500 points
        ax.scatter(dx, dy, s=2, alpha=0.28, color=color,
                   linewidths=0, rasterized=True, zorder=2)

        ax.axhline(0, color="0.60", linewidth=0.45, alpha=0.7, zorder=3)
        ax.axvline(0, color="0.60", linewidth=0.45, alpha=0.7, zorder=3)

        for v in [-REQ_UM, REQ_UM]:
            ax.axvline(v, color="0.88", linewidth=0.3, zorder=0)
            ax.axhline(v, color="0.88", linewidth=0.3, zorder=0)

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7.5, length=3)

        half = round(lim / 2 / 5) * 5
        ax.set_xticks([-half, 0, half])
        ax.set_yticks([-half, 0, half])

        short = (label.replace("(default)", "")
                      .replace("(D-Optimal)", "D-opt.")
                      .strip())
        ax.set_title(
            f"{short}\n"
            rf"$\sigma_{{2D}}$={rms2d:.1f} µm  |  {frac:.0f}% $\leq$ 20 µm",
            fontsize=8, pad=3,
        )
        ax.set_xlabel(r"$\hat{r}_x - x_{\rm true}$ [$\mu$m]", fontsize=8)
        ax.set_ylabel(r"$\hat{r}_y - y_{\rm true}$ [$\mu$m]", fontsize=8)

        ax.text(0.97, 0.03, rf"$\pm${lim:.0f} µm",
                transform=ax.transAxes, fontsize=6.5,
                ha="right", va="bottom", color="0.45")

    fig.suptitle(
        (r"KF Prediction Error  ($\hat{r} - r_{\rm true}$, transverse XY),  "
         rf"$m={m_fixed}$ sensors,  {N_MC} Monte Carlo episodes — "
         r"Each panel at its own scale"),
        fontsize=9.5, y=1.01,
    )

    legend_handles = [
        Line2D([0], [0], linestyle="--", color="black",
               linewidth=0.9, label=r"20 µm engagement requirement"),
        matplotlib.patches.Patch(facecolor="grey", alpha=0.25,
                                 label=r"1$\sigma$, 2$\sigma$ density contours"),
    ]
    fig.legend(handles=legend_handles,
               loc="lower center", ncol=2,
               fontsize=8.5, framealpha=0.92,
               bbox_to_anchor=(0.5, -0.05))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"  [saved] {save_path}")
    return fig


# Figure 3 — schedule timelines

def plot_schedules(m: int = M_FIXED,
                   save_path: str = "figures/fig3_schedule_timelines.pdf"
                   ) -> plt.Figure:
    """Horizontal timeline showing when each strategy places its measurements."""
    t_f      = _t_f_idx(N_REF, TAU_LATENCY, DT)
    max_step = t_f - 1

    fig, axes = plt.subplots(
        len(SCATTER_STRATEGIES), 1,
        figsize=(6.8, 3.8),
        sharex=True,
        gridspec_kw={"hspace": 0.20},
    )

    for i, (label, fn, color, marker, _) in enumerate(SCATTER_STRATEGIES):
        ax    = axes[i]
        steps = fn(m, n_meas_ref=N_REF)
        tau_r = (t_f - steps[-1]) * DT * 1e3

        ax.hlines(0, 0.5, t_f + 0.3, color="0.80", linewidth=1.2, zorder=1)

        # red shading = latency window (no measurements allowed here)
        ax.axvspan(steps[-1] + 0.5, t_f + 0.3, alpha=0.12, color="#C0392B",
                   zorder=0)

        ax.vlines(t_f, -0.45, 0.45, color="black",
                  linewidth=1.5, linestyle="--", zorder=3)

        ax.scatter(steps, [0] * len(steps),
                   color=color, marker=marker, s=55, zorder=4, clip_on=False)

        ax.set_xlim(0.3, t_f + 0.7)
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])
        ax.spines[["left", "bottom", "top", "right"]].set_visible(False)

        short = (label.replace("(default)", "")
                      .replace("(D-Optimal)", "D-opt.")
                      .strip())
        ax.text(0.35, 0.02, short,
                va="center", ha="left",
                fontsize=8, color=color, fontweight="bold",
                transform=ax.get_yaxis_transform())

        ax.text(t_f + 0.4, 0,
                rf"$\tau_r={tau_r:.0f}$ ms",
                va="center", ha="left", fontsize=7, color="0.40")

    axes[-1].set_xlabel(r"Measurement step  ($\times\Delta t = 1$ ms)")
    axes[-1].tick_params(bottom=True, labelbottom=True, direction="in")
    axes[-1].set_xticks(range(1, t_f + 1))
    axes[-1].xaxis.set_tick_params(labelsize=8)

    axes[0].text(t_f, 0.55, r"$t_f$",
                 ha="center", fontsize=9, fontweight="bold", color="black")

    axes[0].annotate(
        "", xy=(t_f, 0.38), xytext=(6.5, 0.38),
        arrowprops=dict(arrowstyle="<->", color="#C0392B",
                        lw=0.9, mutation_scale=8),
    )
    axes[0].text((t_f + 6.5) / 2, 0.50, r"$\tau_{\rm latency}$",
                 ha="center", va="bottom", fontsize=7.5, color="#C0392B")

    fig.suptitle(
        rf"Measurement Schedules: $m={m}$ sensors  "
        rf"(window steps 1–{max_step},  firing at step {t_f})",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"  [saved] {save_path}")
    return fig


def main():
    os.makedirs("figures", exist_ok=True)

    # Fig 3 first — it's instant (no MC needed)
    print("=" * 65)
    print("Figure 3: Measurement schedule timelines")
    print("=" * 65)
    fig3 = plot_schedules()

    print("\n" + "=" * 65)
    print("Figure 1: Cramér–Rao Lower Bound (analytical)")
    print("=" * 65)
    crlb = compute_crlb()
    header = f"  {'Strategy':<28}  " + "  ".join(f"m={v:2d}" for v in M_VALUES)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, *_ in CRLB_STRATEGIES:
        row = (f"  {label:<28}  " +
               "  ".join(f"{v:5.2f}" for v in crlb[label]) + "  µm")
        print(row)
    fig1 = plot_crlb(crlb)

    print("\n" + "=" * 65)
    print("Figure 2: Monte Carlo prediction-error scatter")
    print("=" * 65)
    mc_data = run_mc()
    fig2    = plot_scatter(mc_data)

    plt.show()


if __name__ == "__main__":
    main()
