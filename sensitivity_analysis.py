"""
sensitivity_analysis.py — what does it take for m=8 uniform spread to hit P95 ≤ 20 µm?

Primary metric: 2D transverse KF prediction error  ε_2D = ||(r_hat - r_true)_xy|| [µm]

Note: slew-rate and n_beamlets are both flat because (1) slew-rate has huge margin at
nominal theta_ddot (~4 µrad needed vs 4.5 mrad available), and (2) n_beamlets doesn't
enter the KF prediction. Flat is informative — these aren't the bottleneck.

Outputs:
  figures/fig_sensitivity_grid.pdf — 3×3 sweep grid + tornado chart
  figures/fig_sensitivity_2d.pdf   — 2D heat map sigma_a × tau_l + slice
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

from ife.environment import IFEEnvironment
from ife.dynamics import build_R
from ife.sensor_placement import recommended_spread_schedule, _t_f_idx, _resolve
from ife.config import (
    DT, TAU_LATENCY, SIGMA_TRANSVERSE, SIGMA_AXIAL,
    SIGMA_A_PROCESS, ACCURACY_3SIGMA, THETA_DDOT_MAX,
)

matplotlib.rcParams.update({
    "text.usetex":        False,
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Times New Roman", "serif"],
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     8.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "legend.fontsize":    8,
    "xtick.labelsize":    7.5,
    "ytick.labelsize":    7.5,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "figure.dpi":         130,
    "savefig.dpi":        300,
    "axes.linewidth":     0.7,
    "grid.linewidth":     0.35,
    "grid.alpha":         0.30,
    "lines.linewidth":    1.6,
    "lines.markersize":   4.5,
})

N_REF   = 10
M       = 8
N_MC    = 120          # per 1D point
N_MC2D  = 80           # per 2D cell
SEED    = 42
REQ_UM  = ACCURACY_3SIGMA * 1e6   # 20 µm

STEPS = recommended_spread_schedule(M, n_meas_ref=N_REF)


def phys_spread_schedule(m: int, n_meas_ref: int = N_REF,
                         tau_latency: float = TAU_LATENCY,
                         dt: float = DT) -> list:
    """Measurements end at step n_meas_ref (before the latency window), so tau_rem = tau_latency."""
    max_step = n_meas_ref
    if m == 1:
        return [max_step]
    raw = np.round(np.linspace(1, max_step, m)).astype(int)
    raw = np.clip(raw, 1, max_step)
    return _resolve(raw, m, max_step)


PARAMS = {
    "sigma_a": dict(
        values  = np.array([0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 20, 25]),
        nominal = 20.0,
        xlabel  = r"Process noise $\sigma_a$ [m/s²]",
        xscale  = "log",
        color   = "#C0392B",
        short   = r"$\sigma_a$",
    ),
    "tau_latency_ms": dict(
        values  = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]),
        nominal = 3.0,
        xlabel  = r"System latency $\tau_l$ [ms]  (physical sched.)",
        xscale  = "linear",
        color   = "#E67E22",
        short   = r"$\tau_l$",
    ),
    "sigma_t_um": dict(
        values  = np.array([0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]),
        nominal = 5.0,
        xlabel  = r"Transverse noise $\sigma_t$ [µm]",
        xscale  = "log",
        color   = "#27AE60",
        short   = r"$\sigma_t$",
    ),
    "theta_ddot": dict(
        values  = np.array([5, 20, 50, 100, 200, 500, 1000, 5000]),
        nominal = 1000.0,
        xlabel  = r"Slew rate $\ddot\theta_{\rm max}$ [rad/s²]",
        xscale  = "log",
        color   = "#16A085",
        short   = r"$\ddot\theta_{\rm max}$",
    ),
}

SA_2D  = np.array([0.5, 1, 2, 4, 8, 12, 20])   # m/s²
TAU_2D = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 5.0])  # ms


def make_env(pname: str, val):
    """Build an environment for a single parameter value; also returns a pre-episode drift fn if needed."""
    pre_fn = None

    if pname == "sigma_a":
        env = IFEEnvironment(
            n_measurements=N_REF, measurement_steps=STEPS,
            sigma_a=float(val), seed=SEED,
        )

    elif pname == "tau_latency_ms":
        # use physical schedule so tau_rem = tau_latency (genuine lever)
        tau   = float(val) * 1e-3
        steps = phys_spread_schedule(M, n_meas_ref=N_REF)
        env   = IFEEnvironment(
            n_measurements=N_REF, measurement_steps=steps,
            tau_latency=tau, seed=SEED,
        )

    elif pname == "sigma_t_um":
        sig_t = float(val) * 1e-6
        sig_a = SIGMA_AXIAL * (float(val) / (SIGMA_TRANSVERSE * 1e6))
        env   = IFEEnvironment(
            n_measurements=N_REF, measurement_steps=STEPS, seed=SEED,
        )
        env.R = build_R(sig_t, sig_a)
        env.kf.R = env.R

    elif pname == "theta_ddot":
        env = IFEEnvironment(
            n_measurements=N_REF, measurement_steps=STEPS,
            theta_ddot_max=float(val), seed=SEED,
        )

    else:
        raise ValueError(pname)

    return env, pre_fn


def _run_one(pname: str, val, n_mc: int = N_MC) -> dict:
    env, pre_fn = make_env(pname, val)
    errs = np.empty(n_mc)
    for ep in range(n_mc):
        res = env.run_episode()
        e   = (res["r_hat"] - res["r_true_tf"]) * 1e6   # µm, 3-D vector
        dx, dy = e[0], e[1]
        if pre_fn is not None:
            cal = pre_fn(ep)
            dx += cal[0]
            dy += cal[1]
        errs[ep] = np.sqrt(dx**2 + dy**2)
    return dict(val=val, p25=np.percentile(errs, 25), p50=np.percentile(errs, 50),
                p75=np.percentile(errs, 75), p95=np.percentile(errs, 95),
                rms=float(np.sqrt(np.mean(errs**2))),
                frac=float(np.mean(errs <= REQ_UM) * 100))


def run_all_sweeps(n_mc: int = N_MC) -> dict:
    all_res = {}
    for pname, cfg in PARAMS.items():
        vals   = cfg["values"]
        print(f"  {pname:<18} ({len(vals)} pts) … ", end="", flush=True)
        all_res[pname] = [_run_one(pname, v, n_mc) for v in vals]
        nom_i = int(np.argmin(np.abs(vals - float(cfg["nominal"]))))
        p_nom = all_res[pname][nom_i]["p95"]
        p_min = min(r["p95"] for r in all_res[pname])
        print(f"P95@nom={p_nom:.1f}  best={p_min:.1f} µm")
    return all_res


def run_2d(n_mc: int = N_MC2D) -> np.ndarray:
    n_sa, n_tau = len(SA_2D), len(TAU_2D)
    grid  = np.full((n_tau, n_sa), np.nan)
    total = n_sa * n_tau
    done  = 0
    print(f"  2-D grid {n_sa}×{n_tau} = {total} cells × {n_mc} ep …")
    for j, tau_ms in enumerate(TAU_2D):
        tau   = tau_ms * 1e-3
        steps = phys_spread_schedule(M, n_meas_ref=N_REF)  # physical schedule so tau_rem = tau
        for i, sa in enumerate(SA_2D):
            env = IFEEnvironment(
                n_measurements=N_REF, measurement_steps=steps,
                tau_latency=tau, sigma_a=float(sa), seed=SEED,
            )
            errs = []
            for _ in range(n_mc):
                res = env.run_episode()
                e   = (res["r_hat"] - res["r_true_tf"]) * 1e6
                errs.append(np.sqrt(e[0]**2 + e[1]**2))
            grid[j, i] = np.percentile(errs, 95)
            done += 1
            print(f"    [{done:2d}/{total}] sa={sa:5.1f} tau={tau_ms:.1f}ms "
                  f"P95={grid[j,i]:.1f} um")
    return grid


def _crossover(x, y, threshold=REQ_UM):
    """Linear interpolation to find x where y crosses threshold (for decreasing y)."""
    for k in range(len(y) - 1):
        if (y[k] - threshold) * (y[k+1] - threshold) < 0:
            frac = (threshold - y[k]) / (y[k+1] - y[k])
            xv = float(x[k] + frac * (x[k+1] - x[k]))
            return xv
    return None


# Figure 1 — 3×3 grid

def _panel(ax, pname: str, results: list):
    cfg   = PARAMS[pname]
    col   = cfg["color"]
    vals  = np.array([r["val"]  for r in results], dtype=float)
    p25   = np.array([r["p25"]  for r in results])
    p50   = np.array([r["p50"]  for r in results])
    p75   = np.array([r["p75"]  for r in results])
    p95   = np.array([r["p95"]  for r in results])
    nom   = float(cfg["nominal"])

    ax.set_xscale(cfg["xscale"])

    ax.fill_between(vals, p25, p75, alpha=0.18, color=col, zorder=1)
    ax.plot(vals, p50, color=col, lw=0.9, alpha=0.65, ls="--", zorder=3)
    ax.plot(vals, p95, color=col, lw=1.8, marker="o", ms=4, zorder=4)

    ax.axhline(REQ_UM, color="black", ls="--", lw=0.85, alpha=0.80, zorder=5)

    ax.axvline(nom, color="0.50", ls=":", lw=0.75, alpha=0.65, zorder=2)
    y_top = max(p95.max() * 1.12, REQ_UM * 1.55)
    y_top = np.ceil(y_top / 10) * 10
    if cfg["xscale"] == "linear":
        ax.text(nom, -y_top * 0.09, "nom",
                ha="center", va="top", fontsize=6.0, color="0.40",
                clip_on=False)

    xover = _crossover(vals, p95)
    if xover is not None and vals[0] < xover < vals[-1]:
        ax.axvline(xover, color=col, ls="-.", lw=0.8, alpha=0.50, zorder=2)
        ax.scatter([xover], [REQ_UM], marker="*", s=65, color=col, zorder=6)
        fmt = (f"{xover:.2f}" if xover < 1 else
               f"{xover:.1f}" if xover < 10 else f"{xover:.0f}")
        ax.annotate(fmt, xy=(xover, REQ_UM),
                    xytext=(0, 7), textcoords="offset points",
                    fontsize=6.5, ha="center", color=col)

    ax.set_ylim(0, y_top)
    pad = 0.85 if cfg["xscale"] == "log" else 0
    ax.set_xlim(vals[0] * (pad if cfg["xscale"] == "log" else 1.0) or vals[0] - (vals[-1]-vals[0])*0.05,
                vals[-1] * (1.15 if cfg["xscale"] == "log" else 1.0) or vals[-1] + (vals[-1]-vals[0])*0.05)
    ax.set_xlabel(cfg["xlabel"], fontsize=7.5, labelpad=2)
    ax.set_ylabel(r"P95 $\varepsilon_{2D}$ [µm]", fontsize=7.5)
    extra = " + cal." if pname == "sigma_cal_um" else ""
    ax.set_title(cfg["xlabel"].split("[")[0].strip() + extra, fontsize=8.0, pad=3)
    ax.tick_params(labelsize=7)
    ax.grid(True, ls=":", alpha=0.28)


def _tornado(ax, all_res: dict):
    labels, deltas, colors = [], [], []
    for pname, results in all_res.items():
        cfg  = PARAMS[pname]
        vals = np.array([r["val"] for r in results], dtype=float)
        p95s = np.array([r["p95"] for r in results])
        ni   = int(np.argmin(np.abs(vals - float(cfg["nominal"]))))
        labels.append(cfg["short"])
        deltas.append(max(0.0, p95s[ni] - p95s.min()))
        colors.append(cfg["color"])

    order   = np.argsort(deltas)[::-1]
    labels  = [labels[i]  for i in order]
    deltas  = [deltas[i]  for i in order]
    colors  = [colors[i]  for i in order]
    y_pos   = np.arange(len(labels))

    bars = ax.barh(y_pos, deltas, color=colors, alpha=0.82, height=0.62, zorder=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel(r"$\Delta$P95 = P95(nom) $-$ P95(best)  [µm]", fontsize=7.5)
    ax.set_title("Improvement potential\n(sensitivity rank)", fontsize=8, pad=3)
    ax.axvline(0, color="black", lw=0.5)
    ax.grid(True, axis="x", ls=":", alpha=0.28)
    ax.tick_params(labelsize=7)

    for bar, val in zip(bars, deltas):
        if val > 0.5:
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}", va="center", ha="left", fontsize=6.5, color="0.3")


def plot_grid(all_res: dict,
              save_path: str = "figures/fig_sensitivity_grid.pdf") -> plt.Figure:
    pkeys = list(PARAMS.keys())

    fig = plt.figure(figsize=(11.0, 7.0))
    gs  = GridSpec(2, 3, figure=fig,
                   hspace=0.58, wspace=0.40,
                   left=0.07, right=0.97, top=0.91, bottom=0.10)

    for idx, pkey in enumerate(pkeys):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        _panel(ax, pkey, all_res[pkey])

    _tornado(fig.add_subplot(gs[1, 1:3]), all_res)

    legend_handles = [
        mpatches.Patch(facecolor="0.6", alpha=0.30, label="IQR (P25–P75)"),
        Line2D([0], [0], ls="--", color="0.5", lw=0.9, label="Median (P50)"),
        Line2D([0], [0], ls="-",  color="0.4", lw=1.6,
               marker="o", ms=4, label="P95  (primary metric)"),
        Line2D([0], [0], ls="--", color="black", lw=0.85,
               label="20 µm requirement"),
        Line2D([0], [0], ls=":",  color="0.5", lw=0.75,
               label="Nominal value"),
        Line2D([0], [0], ls="none", marker="*", color="0.4", ms=8,
               label="P95 = 20 µm crossover"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=7.5, framealpha=0.92,
               bbox_to_anchor=(0.5, -0.005))

    fig.suptitle(
        rf"Parameter Sensitivity — P95 Transverse KF Prediction Error  ($m={M}$, {N_MC} MC ep/pt)"
        "\n"
        r"$\tau_l$: physically-correct schedule ($\tau_{\rm rem}=\tau_l$);  "
        r"$\sigma_a$, $\sigma_t$, $\ddot\theta_{\rm max}$: uniform spread over full window",
        fontsize=10.5, y=0.98,
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"  [saved] {save_path}")
    return fig


# Figure 2 — 2D heat map

def plot_2d(grid: np.ndarray,
            save_path: str = "figures/fig_sensitivity_2d.pdf") -> plt.Figure:
    from matplotlib.colors import LogNorm

    fig = plt.figure(figsize=(9.5, 4.2))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.38,
                   left=0.09, right=0.97, top=0.88, bottom=0.15)
    ax_hm = fig.add_subplot(gs[0, :2])
    ax_mg = fig.add_subplot(gs[0, 2])

    vmin, vmax = 5, 250
    levels = np.geomspace(vmin, vmax, 24)
    im = ax_hm.contourf(SA_2D, TAU_2D, grid, levels=levels,
                        norm=LogNorm(vmin=vmin, vmax=vmax), cmap="RdYlGn_r")
    cb = fig.colorbar(im, ax=ax_hm,
                      label=r"P95 transverse prediction error [µm]",
                      fraction=0.046, pad=0.04,
                      ticks=[5, 10, 20, 50, 100, 250])
    cb.ax.set_yticklabels(["5", "10", "20", "50", "100", "250"])
    cb.ax.tick_params(labelsize=8)

    cs = ax_hm.contour(SA_2D, TAU_2D, grid, levels=[REQ_UM],
                       colors=["white"], linewidths=[2.0], linestyles=["--"])
    ax_hm.clabel(cs, fmt=r"P95 = 20 µm", fontsize=8.5, colors="white",
                 inline=True, inline_spacing=4)

    ax_hm.scatter([SIGMA_A_PROCESS], [TAU_LATENCY * 1e3],
                  s=110, color="white", edgecolors="black",
                  linewidths=1.0, zorder=5)
    ax_hm.annotate(
        "Nominal",
        xy=(SIGMA_A_PROCESS, TAU_LATENCY * 1e3),
        xytext=(SIGMA_A_PROCESS / 2.5, TAU_LATENCY * 1e3 + 0.9),
        arrowprops=dict(arrowstyle="-|>", color="white",
                        lw=0.9, mutation_scale=9),
        fontsize=8.5, color="white",
    )

    ax_hm.set_xscale("log")
    ax_hm.set_xlabel(r"Process noise $\sigma_a$ [m/s²]", fontsize=10)
    ax_hm.set_ylabel(r"System latency $\tau_l$ [ms]", fontsize=10)
    ax_hm.set_title(
        r"Operating Envelope: P95 prediction error" "\n"
        r"($m=8$, physical schedule — $\tau_{\rm rem}=\tau_l$ — white dashed = 20 µm boundary)",
        pad=6,
    )
    ax_hm.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:.1f}" if x < 1 else f"{x:.0f}"
    ))

    # slice at nominal latency
    tau_nom_idx = int(np.argmin(np.abs(TAU_2D - TAU_LATENCY * 1e3)))
    row = grid[tau_nom_idx, :]

    ax_mg.semilogx(SA_2D, row, color="#C0392B", lw=1.8, marker="o", ms=5, zorder=4)
    ax_mg.axhline(REQ_UM, color="black", ls="--", lw=0.9, alpha=0.80,
                  label=r"20 µm req.")
    ax_mg.axvline(SIGMA_A_PROCESS, color="0.50", ls=":", lw=0.8, alpha=0.70)

    xover = _crossover(SA_2D, row)
    if xover is not None:
        ax_mg.axvline(xover, color="#C0392B", ls="-.", lw=0.8, alpha=0.55)
        ax_mg.scatter([xover], [REQ_UM], marker="*", s=70, color="#C0392B", zorder=5)
        ax_mg.annotate(
            rf"$\sigma_a^* \approx {xover:.1f}$ m/s²",
            xy=(xover, REQ_UM),
            xytext=(xover * 2.0, REQ_UM + 10),
            arrowprops=dict(arrowstyle="-|>", color="#C0392B",
                            lw=0.8, mutation_scale=8),
            fontsize=8.5, color="#C0392B",
        )

    ax_mg.set_xlabel(r"$\sigma_a$ [m/s²]", fontsize=9.5)
    ax_mg.set_ylabel(r"P95 [µm]", fontsize=9.5)
    ax_mg.set_title(rf"Slice at $\tau_l = {TAU_2D[tau_nom_idx]:.0f}$ ms",
                    fontsize=9, pad=4)
    ax_mg.set_ylim(0, row.max() * 1.12)
    ax_mg.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:.1f}" if x < 1 else f"{x:.0f}"
    ))
    ax_mg.grid(True, ls=":", alpha=0.28)
    ax_mg.legend(fontsize=7.5, loc="upper left")

    fig.suptitle(
        r"2-D Sensitivity: $\sigma_a$ × $\tau_l$  —  "
        r"P95 Transverse KF Prediction Error [µm]",
        fontsize=11, y=0.99,
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"  [saved] {save_path}")
    return fig


def main():
    os.makedirs("figures", exist_ok=True)
    matplotlib.use("Agg")

    print("=" * 65)
    print(f"Sensitivity analysis  m={M} Uniform Spread  STEPS={STEPS}")
    print("=" * 65)

    print("\n--- 1D sweeps ---")
    sweeps = run_all_sweeps()
    fig1 = plot_grid(sweeps)
    plt.close(fig1)

    print("\n--- 2D heat map ---")
    grid = run_2d()
    fig2 = plot_2d(grid)
    plt.close(fig2)

    print("\nDone.")


if __name__ == "__main__":
    main()
