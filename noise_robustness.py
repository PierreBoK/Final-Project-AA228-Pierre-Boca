"""
noise_robustness.py — how badly does KF prediction degrade as sensor noise grows?

m = 8, uniformly spread. Sweeps σ_transverse from 0.05× to 20× nominal.
σ_axial scales with it to keep the 2:1 ratio fixed.

Fig 1: log-log error curves + compliance fraction
Fig 2: violin plots at 6 noise levels
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

from ife.environment import IFEEnvironment
from ife.dynamics import build_R
from ife.sensor_placement import (
    recommended_spread_schedule,
    _t_f_idx,
    crlb_position_error,
)
from ife.config import (
    DT, TAU_LATENCY, SIGMA_TRANSVERSE, SIGMA_AXIAL,
    ACCURACY_3SIGMA,
)

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
    "lines.markersize":   5,
})

N_REF    = 10
M_FIXED  = 8
N_MC     = 400
REQ_UM   = ACCURACY_3SIGMA * 1e6   # 20 µm

N_LEVELS = 20
MULTS    = np.geomspace(0.05, 20, N_LEVELS)

SIG_T_NOM = SIGMA_TRANSVERSE   # 5 µm
SIG_A_NOM = SIGMA_AXIAL        # 10 µm

VIOLIN_TARGETS = [0.05, 0.2, 1.0, 3.0, 10.0, 20.0]

def run_robustness():
    t_f   = _t_f_idx(N_REF, TAU_LATENCY, DT)
    steps = recommended_spread_schedule(M_FIXED, n_meas_ref=N_REF)

    t_meas = np.array(steps) * DT * 1e3
    S_tt   = np.sum((t_meas - t_meas.mean()) ** 2)
    tau_r  = (t_f - steps[-1]) * DT * 1e3

    print(f"  m={M_FIXED} uniform spread  |  steps={steps}")
    print(f"  t_f=step {t_f}  S_tt={S_tt:.1f} ms²  tau_r={tau_r:.0f} ms")
    print(f"  {N_LEVELS} noise levels × {N_MC} episodes = "
          f"{N_LEVELS*N_MC} total\n")

    records = []
    for k, mult in enumerate(MULTS):
        sig_t = SIG_T_NOM * mult
        sig_a = SIG_A_NOM * mult

        crlb_1ax = crlb_position_error(steps, sig_t, t_f, DT) * 1e6
        crlb_2d  = crlb_1ax * np.sqrt(2)

        env = IFEEnvironment(
            n_measurements    = N_REF,
            measurement_steps = steps,
            seed              = 77,      # fixed seed → same trajectories across all σ levels
        )
        env.R      = build_R(sig_t, sig_a)
        env.kf.R   = env.R

        errs_2d    = np.empty(N_MC)
        pr_trace   = np.empty(N_MC)   # KF's own uncertainty estimate (for consistency check)
        for ep in range(N_MC):
            res         = env.run_episode()
            e           = (res["r_hat"] - res["r_true_tf"]) * 1e6
            errs_2d[ep] = np.sqrt(e[0]**2 + e[1]**2)
            pr_trace[ep]= np.sqrt(np.trace(res["P_r"][:2, :2])) * 1e6

        rms  = np.sqrt(np.mean(errs_2d ** 2))
        frac = np.mean(errs_2d <= REQ_UM) * 100
        kf_predicted = np.mean(pr_trace)

        records.append({
            "mult":        mult,
            "sig_t_um":    sig_t * 1e6,
            "crlb_1ax":    crlb_1ax,
            "crlb_2d":     crlb_2d,
            "rms":         rms,
            "median":      np.median(errs_2d),
            "p16":         np.percentile(errs_2d, 16),
            "p84":         np.percentile(errs_2d, 84),
            "p95":         np.percentile(errs_2d, 95),
            "p99":         np.percentile(errs_2d, 99),
            "frac_20":     frac,
            "kf_sigma":    kf_predicted,
            "errs":        errs_2d.copy(),
        })
        print(f"  x{mult:6.2f}  sig_t={sig_t*1e6:7.2f} um  "
              f"CRLB_2D={crlb_2d:7.3f} um  "
              f"MC-RMS={rms:7.2f} um  "
              f"KF-sig={kf_predicted:7.2f} um  "
              f"<=20um: {frac:.0f}%")

    return records, steps


# Figure 1 — main robustness curves

def plot_robustness(records, steps,
                    save_path="figures/fig_noise_robustness.pdf"):

    sig_t = np.array([r["sig_t_um"]  for r in records])
    crlb  = np.array([r["crlb_2d"]   for r in records])
    rms   = np.array([r["rms"]        for r in records])
    p16   = np.array([r["p16"]        for r in records])
    p84   = np.array([r["p84"]        for r in records])
    p95   = np.array([r["p95"]        for r in records])
    frac  = np.array([r["frac_20"]    for r in records])
    kfsig = np.array([r["kf_sigma"]   for r in records])

    nom_sig = SIG_T_NOM * 1e6   # 5 µm

    C_MC    = "#2980B9"
    C_CRLB  = "#27AE60"
    C_KF    = "#8E44AD"
    C_P95   = "#E67E22"

    fig = plt.figure(figsize=(9.0, 4.2))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.38,
                   left=0.08, right=0.97, top=0.88, bottom=0.14)

    ax_main = fig.add_subplot(gs[0, :2])
    ax_frac = fig.add_subplot(gs[0, 2])

    ax = ax_main

    ax.axhline(REQ_UM, color="black", linestyle="--",
               linewidth=1.0, alpha=0.75, zorder=4,
               label=r"20 µm requirement")

    # floor is wherever MC-RMS saturates at very low noise (process noise dominated)
    floor = rms[0]
    ax.axhline(floor, color="0.55", linestyle=":",
               linewidth=0.9, alpha=0.7, zorder=3,
               label=rf"Process-noise floor ≈{floor:.1f} µm")

    ax.axvline(nom_sig, color="0.50", linestyle="--",
               linewidth=0.7, alpha=0.60, zorder=3)
    ax.text(nom_sig * 1.08, ax_main.get_ylim()[0] if ax_main.get_ylim()[0] > 0
            else 0.05,
            r"$\sigma_{\rm nom}$", fontsize=8, color="0.45",
            va="bottom", ha="left")

    ax.loglog(sig_t, kfsig, color=C_KF, linestyle="--",
              linewidth=1.4, marker="^", markersize=4,
              label=r"KF covariance $\sqrt{{\rm tr}(P_r)}$ (filter self-estimate)")

    ax.loglog(sig_t, p95, color=C_P95, linestyle=":",
              linewidth=1.3, marker="v", markersize=4, alpha=0.85,
              label="MC 95th-percentile error")

    ax.fill_between(sig_t, p16, p84, alpha=0.18, color=C_MC, zorder=1)
    ax.loglog(sig_t, rms, color=C_MC, linestyle="-",
              linewidth=2.0, marker="o", markersize=5, zorder=5,
              label=r"MC 2D-RMS (±1$\sigma$ band)")

    nom_idx = np.argmin(np.abs(sig_t - nom_sig))
    ax.scatter([sig_t[nom_idx]], [rms[nom_idx]],
               s=80, color=C_MC, zorder=6,
               edgecolors="black", linewidths=0.8)
    ax.annotate(
        rf"Nominal  ({rms[nom_idx]:.1f} µm)",
        xy=(sig_t[nom_idx], rms[nom_idx]),
        xytext=(sig_t[nom_idx] * 3.5, rms[nom_idx] * 1.5),
        arrowprops=dict(arrowstyle="-|>", color="0.4",
                        lw=0.8, mutation_scale=9),
        fontsize=8, color="0.30",
    )

    ax.set_xlabel(r"Transverse sensor noise $\sigma_t$ [µm]")
    ax.set_ylabel(r"2D prediction error $\|\hat{r} - r_{\rm true}\|_{xy}$ [µm]")
    ax.set_title(
        rf"Observation Noise Robustness  ($m={M_FIXED}$, Uniform Spread)",
        pad=6,
    )
    ax.set_xlim(sig_t[0] * 0.85, sig_t[-1] * 1.15)
    ax.set_ylim(bottom=rms[0] * 0.4)

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:.1f}" if x < 1 else (f"{x:.0f}" if x < 100 else f"{x:.0f}")
    ))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:.1f}" if x < 1 else f"{x:.0f}"
    ))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.grid(True, which="major", linestyle=":", alpha=0.35)

    ax.legend(loc="upper left", fontsize=8, handlelength=2.2,
              borderaxespad=0.5, ncol=1)

    ax2 = ax_frac

    ax2.axhspan(50, 100, alpha=0.06, color="#27AE60")
    ax2.axhspan(0,  50,  alpha=0.06, color="#E74C3C")
    ax2.axhline(50, color="0.55", linestyle=":", linewidth=0.8, alpha=0.7)
    ax2.axhline(frac[nom_idx], color=C_MC, linestyle="--",
                linewidth=0.8, alpha=0.6)

    ax2.semilogx(sig_t, frac, color=C_MC, linestyle="-",
                 linewidth=2.0, marker="o", markersize=5, zorder=4)
    ax2.fill_between(sig_t, 0, frac, alpha=0.15, color=C_MC, zorder=1)

    ax2.scatter([sig_t[nom_idx]], [frac[nom_idx]],
                s=80, color=C_MC, zorder=5,
                edgecolors="black", linewidths=0.8)

    ax2.axvline(nom_sig, color="0.50", linestyle="--",
                linewidth=0.7, alpha=0.60)
    ax2.text(nom_sig * 1.1, 2,
             r"$\sigma_{\rm nom}$", fontsize=8, color="0.45", va="bottom")

    ax2.set_xlabel(r"$\sigma_t$ [µm]")
    ax2.set_ylabel(r"Episodes with error $\leq 20$ µm  [%]")
    ax2.set_title(r"Requirement Compliance", pad=6)
    ax2.set_xlim(sig_t[0] * 0.85, sig_t[-1] * 1.15)
    ax2.set_ylim(0, 102)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:.1f}" if x < 10 else f"{x:.0f}"
    ))
    ax2.grid(True, which="major", linestyle=":", alpha=0.35)

    ax2.annotate(
        rf"{frac[nom_idx]:.0f}% @ $\sigma_{{nom}}$",
        xy=(sig_t[nom_idx], frac[nom_idx]),
        xytext=(sig_t[nom_idx] / 4.5, frac[nom_idx] + 8),
        arrowprops=dict(arrowstyle="-|>", color="0.4",
                        lw=0.8, mutation_scale=9),
        fontsize=8, color="0.30",
    )

    fig.suptitle(
        rf"Robustness to Observation Noise — KF Tracking Prediction, "
        rf"$m={M_FIXED}$ Uniform-Spread Sensors",
        fontsize=11, y=0.995,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"  [saved] {save_path}")
    return fig


# Figure 2 — violin distributions

def plot_violins(records,
                 save_path="figures/fig_noise_violins.pdf"):
    """Violin plots at 6 selected noise levels."""
    chosen = []
    for target in VIOLIN_TARGETS:
        idx = int(np.argmin(np.abs(MULTS - target)))
        chosen.append(records[idx])

    n     = len(chosen)
    fig, axes = plt.subplots(1, n, figsize=(9.0, 3.8),
                              sharey=False,
                              gridspec_kw={"wspace": 0.08})

    colors = plt.cm.plasma_r(np.linspace(0.15, 0.85, n))

    # set y-limit from the noisiest case so panels are comparable
    all_errs = np.concatenate([r["errs"] for r in chosen])
    ymax     = np.percentile(all_errs, 99.5) * 1.12
    ymax     = np.ceil(ymax / 10) * 10

    for col, (rec, ax, color) in enumerate(zip(chosen, axes, colors)):
        errs   = rec["errs"]
        sig_t  = rec["sig_t_um"]
        mult   = rec["mult"]
        rms    = rec["rms"]
        frac   = rec["frac_20"]

        vp = ax.violinplot(
            errs, positions=[0], widths=[0.7],
            showmedians=True, showextrema=False,
        )
        for pc in vp["bodies"]:
            pc.set_facecolor(color)
            pc.set_edgecolor("0.3")
            pc.set_alpha(0.70)
            pc.set_linewidth(0.6)
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(1.2)

        ax.axhline(REQ_UM, color="black", linestyle="--",
                   linewidth=0.9, alpha=0.75, zorder=4)

        q25, q75 = np.percentile(errs, [25, 75])
        ax.vlines(0, q25, q75, color="0.2", linewidth=2.5, zorder=5)

        ax.scatter([0], [rms], marker="D", s=30,
                   color="white", edgecolors="0.2",
                   linewidths=0.8, zorder=6)

        ax.set_ylim(0, ymax)
        ax.set_xlim(-0.55, 0.55)
        ax.set_xticks([])
        ax.tick_params(labelsize=8)

        mult_str = f"×{mult:.2f}" if mult < 1 else f"×{mult:.1f}" if mult < 10 else f"×{mult:.0f}"
        ax.set_title(
            rf"$\sigma_t={sig_t:.1f}$ µm" + "\n" + rf"({mult_str} nom.)",
            fontsize=8.5, pad=3,
        )

        ax.text(0, -ymax * 0.09,
                rf"RMS={rms:.0f}" + "\n" + rf"{frac:.0f}% $\leq$20",
                ha="center", va="top", fontsize=7.5,
                transform=ax.transData, clip_on=False)

        if col > 0:
            ax.spines["left"].set_visible(False)
            ax.set_yticks([])
        else:
            ax.set_ylabel(r"2D prediction error $\|\hat{r}-r_{\rm true}\|_{xy}$ [µm]")

        if abs(mult - 1.0) < 0.05:  # highlight the nominal noise level panel
            ax.set_facecolor("#FAFAD2")
            ax.text(0.5, 0.97, "nominal", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7.5, color="0.50",
                    style="italic")

    fig.suptitle(
        rf"Prediction Error Distribution at Selected Noise Levels "
        rf"($m={M_FIXED}$, Uniform Spread)",
        fontsize=10.5, y=1.01,
    )

    # Legend
    legend_handles = [
        Line2D([0], [0], linestyle="--", color="black",
               linewidth=0.9, label="20 µm requirement"),
        Line2D([0], [0], linestyle="-", color="0.3",
               linewidth=2.5, label="IQR (25–75th pct.)"),
        Line2D([0], [0], linestyle="none", marker="D",
               markerfacecolor="white", markeredgecolor="0.2",
               markersize=6, label="RMS"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.10),
               framealpha=0.92)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"  [saved] {save_path}")
    return fig


def main():
    os.makedirs("figures", exist_ok=True)
    print("=" * 65)
    print(f"Noise robustness analysis  (m={M_FIXED}, uniform spread)")
    print("=" * 65)

    records, steps = run_robustness()

    print("\nGenerating figures …")
    fig1 = plot_robustness(records, steps)
    fig2 = plot_violins(records)

    plt.show()


if __name__ == "__main__":
    main()
