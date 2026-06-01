"""
deep_dive_analysis.py — follow-up investigations from the sensitivity results

1. GP warmup: why the P95 spike and why doesn't it recover?
2. Physically-correct schedule: measurements must finish before the latency window.
   Once we enforce this, τ_l and the velocity prior actually matter.
3. Shot rate: is SHOT_RATE a real lever?
4. Slew rate: why is it flat in sensitivity? Shows the right metric: targeting gap ||p_i - r_hat||.
"""

import os
import numpy as np
import warnings
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

from ife.environment import IFEEnvironment
from ife.dynamics import build_Q_kinematic
from ife.sensor_placement import (
    recommended_spread_schedule, _t_f_idx, _resolve, crlb_position_error,
)
from ife.config import (
    DT, TAU_LATENCY, N_MEASUREMENTS,
    SIGMA_TRANSVERSE, SIGMA_A_PROCESS,
    ACCURACY_3SIGMA, THETA_DDOT_MAX, SHOT_RATE,
)

matplotlib.rcParams.update({
    "text.usetex":        False,
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Times New Roman", "serif"],
    "font.size":          10,
    "axes.labelsize":     10,
    "axes.titlesize":     9.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "legend.fontsize":    8.5,
    "legend.framealpha":  0.92,
    "xtick.labelsize":    8.5,
    "ytick.labelsize":    8.5,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "figure.dpi":         130,
    "savefig.dpi":        300,
    "axes.linewidth":     0.8,
    "grid.linewidth":     0.4,
    "grid.alpha":         0.3,
    "lines.linewidth":    1.8,
    "lines.markersize":   5,
})

N_REF  = 10
M      = 8
SEED   = 42
REQ_UM = ACCURACY_3SIGMA * 1e6

NOMINAL_STEPS = recommended_spread_schedule(M, n_meas_ref=N_REF)


def phys_spread_schedule(m: int,
                         n_meas_ref: int = N_REF,
                         tau_latency: float = TAU_LATENCY,
                         dt: float = DT) -> list:
    """
    Spread within [1, n_meas_ref] only — measurements must finish before the latency window.
    This makes tau_remaining = tau_l always, so latency is a real precision lever.
    """
    max_step = n_meas_ref           # = t_f − latency_steps
    if m == 1:
        return [max_step]
    raw = np.round(np.linspace(1, max_step, m)).astype(int)
    raw = np.clip(raw, 1, max_step)
    return _resolve(raw, m, max_step)


# Figure 1 — GP warmup

def _run_gp_sweep(n_wu_values, n_mc=150, eval_seed=9999):
    """
    Train the GP on n_wu warmup episodes, then reset the RNG to a fixed eval seed
    so every warmup size sees the same trajectories — fair comparison.
    """
    records = []
    for n_wu in n_wu_values:
        env = IFEEnvironment(
            n_measurements=N_REF, measurement_steps=NOMINAL_STEPS,
            use_gp=(n_wu > 0), seed=SEED,
        )
        for _ in range(n_wu):
            env.run_episode()
        if n_wu > 0:
            env.update_gp()
            if n_wu >= 10:
                env.gp.optimize_hyperparams()

        # probe Q_pred at a representative mid-flight position
        states = env._simulate_trajectory()
        r_probe = states[1, :3]
        if n_wu > 0 and env.gp.X_train is not None:
            Q = env.gp.predict_Q(r_probe)
            q_pos = Q[0, 0] if Q is not None else float(env.Q_nom[0, 0])
            ell   = env.gp.ell
            sigma_f = env.gp.sigma_f
        else:
            q_pos   = float(env.Q_nom[0, 0])
            ell     = float("nan")
            sigma_f = float("nan")

        env.rng = np.random.default_rng(eval_seed)

        pred_errs = np.empty(n_mc)
        for ep in range(n_mc):
            res = env.run_episode()
            e   = (res["r_hat"] - res["r_true_tf"]) * 1e6
            pred_errs[ep] = np.sqrt(e[0]**2 + e[1]**2)

        records.append(dict(
            n_wu    = n_wu,
            p95     = np.percentile(pred_errs, 95),
            p50     = np.median(pred_errs),
            q_pos   = q_pos,
            ell     = ell,
            sigma_f = sigma_f,
        ))
        print(f"  n_wu={n_wu:4d}: P95={records[-1]['p95']:7.1f} um  "
              f"Q_pos={q_pos:.2e}  ell={ell:.4f}  sf={sigma_f:.3f}")

    return records


def plot_gp_figure(records, save_path="figures/fig_gp_deep_dive.pdf"):
    n_wu  = [r["n_wu"]   for r in records]
    p95   = [r["p95"]    for r in records]
    p50   = [r["p50"]    for r in records]
    q_pos = [r["q_pos"]  for r in records]
    ell   = [r["ell"]    for r in records]

    q_nom = float(build_Q_kinematic(DT, SIGMA_A_PROCESS)[0, 0])

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0),
                             gridspec_kw={"wspace": 0.40})

    ax = axes[0]
    ax.plot(n_wu, p95, color="#D35400", lw=2.0, marker="o", ms=5.5, zorder=4,
            label="P95 prediction error")
    ax.plot(n_wu, p50, color="#D35400", lw=0.9, ls="--", alpha=0.6, zorder=3,
            label="Median (P50)")

    baseline = p95[0]   # no-GP baseline
    ax.axhline(baseline, color="0.45", ls=":", lw=1.0, alpha=0.75,
               label=f"No-GP baseline ({baseline:.0f} µm)")
    ax.axhline(REQ_UM, color="black", ls="--", lw=0.9, alpha=0.80,
               label="20 µm requirement")

    ax.axvspan(0.5, 8, alpha=0.06, color="#C0392B")
    ax.axvspan(8, 15, alpha=0.06, color="#E67E22")
    ax.axvspan(15, max(n_wu)+5, alpha=0.05, color="#27AE60")

    ax.text(4, baseline * 1.55,
            "Insufficient\ntraining:\nQ_pred ≫ Q_nom",
            ha="center", va="bottom", fontsize=7.5, color="#C0392B",
            style="italic")
    ax.text(11, baseline * 1.20,
            "hyperparams\ncollapse (ℓ→0)",
            ha="center", va="bottom", fontsize=7.5, color="#E67E22",
            style="italic")
    ax.text(60, baseline * 0.80,
            "Stable but\nnever improves\n(uniform noise)",
            ha="center", va="bottom", fontsize=7.5, color="#27AE60",
            style="italic")

    ax.set_xlabel("Number of GP warmup episodes")
    ax.set_ylabel(r"P95 prediction error $\varepsilon_{2D}$ [µm]")
    ax.set_title("GP Warmup: P95 vs Training Episodes\n(fair evaluation — fixed eval seed)")
    ax.set_xlim(-2, max(n_wu) + 5)
    ax.set_ylim(0, max(p95) * 1.15)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, ls=":", alpha=0.28)

    ax2 = axes[1]
    ratio = [q / q_nom for q in q_pos]

    ax2.semilogy(n_wu, ratio, color="#D35400", lw=2.0, marker="s", ms=5.5,
                 zorder=4, label=r"$Q_{\rm pred,pos} / Q_{\rm nom,pos}$")
    ax2.axhline(1.0, color="black", ls="--", lw=0.9, alpha=0.80,
                label="Ideal ratio = 1  (no inflation)")
    ax2.axhspan(0.1, 10, alpha=0.08, color="#27AE60")
    ax2.text(50, 5, "≈ ideal", ha="right", fontsize=8, color="#27AE60")

    ax2.annotate(
        r"$Q_{\rm pred} \approx \sigma_f^2 \cdot \mathbf{I}_9$"
        "\n" r"$= 1.0\,\mathrm{m}^2$ (GP uncertainty)"
        "\n" r"vs $Q_{\rm nom,pos} \approx 10^{-12}\,\mathrm{m}^2$",
        xy=(10, ratio[3] if len(ratio) > 3 else 1e10),
        xytext=(35, 1e8),
        arrowprops=dict(arrowstyle="-|>", color="#C0392B",
                        lw=0.9, mutation_scale=9),
        fontsize=7.5, color="#C0392B",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#C0392B", lw=0.7),
    )

    for i, (nw, r_val, el) in enumerate(zip(n_wu, ratio, ell)):
        if not np.isnan(el) and el < 0.005 and nw >= 10:
            ax2.annotate(f"ℓ→{el:.4f}\n(collapsed)",
                         xy=(nw, r_val),
                         xytext=(nw + 8, r_val * 3),
                         arrowprops=dict(arrowstyle="-|>", color="#E67E22",
                                         lw=0.8, mutation_scale=8),
                         fontsize=7, color="#E67E22")
            break

    ax2.set_xlabel("Number of GP warmup episodes")
    ax2.set_ylabel(r"$Q_{\rm pred,pos}\ /\ Q_{\rm nom,pos}$  (log scale)")
    ax2.set_title("Process-Noise Inflation by GP\n"
                  r"($\sigma_f^2 \gg$ actual residual variance → KF velocity diverges)")
    ax2.set_xlim(-2, max(n_wu) + 5)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, ls=":", alpha=0.28)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"10$^{{{int(np.log10(x)):+d}}}$" if x > 0 else "0"
    ))

    fig.suptitle(
        r"Why the GP Warmup Spike? — Noise-Scale Mismatch ($\sigma_f = 1$ m, "
        r"$Q_{\rm nom,pos} \approx 10^{-12}$ m²)",
        fontsize=10.5, y=1.01,
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"  [saved] {save_path}")
    return fig


# Figure 2 — physically-correct schedule

def _run_param_both_schedules(param: str, values, n_mc=120, eval_seed=SEED):
    """1D parameter sweep comparing the old (cheating) and new (physical) schedules."""
    old_res, new_res = [], []
    for val in values:
        for label, schedule_fn, results in [
            ("old", recommended_spread_schedule, old_res),
            ("new", phys_spread_schedule, new_res),
        ]:
            tau = float(val) * 1e-3 if param == "tau_latency_ms" else TAU_LATENCY
            steps = schedule_fn(M, n_meas_ref=N_REF, tau_latency=tau)
            kw = dict(n_measurements=N_REF, measurement_steps=steps,
                      tau_latency=tau, seed=eval_seed)
            if param == "p0_vel_std":
                kw["p0_vel_std"] = float(val)
            env = IFEEnvironment(**kw)
            errs = []
            for _ in range(n_mc):
                res = env.run_episode()
                e   = (res["r_hat"] - res["r_true_tf"]) * 1e6
                errs.append(np.sqrt(e[0]**2 + e[1]**2))
            errs = np.array(errs)
            results.append(dict(
                val = val,
                p50 = np.median(errs),
                p75 = np.percentile(errs, 75),
                p95 = np.percentile(errs, 95),
            ))
    return old_res, new_res


def plot_schedule_comparison(tau_old, tau_new, p0v_old, p0v_new,
                              tau_values, p0v_values,
                              save_path="figures/fig_phys_schedule.pdf"):
    C_OLD = "#7F8C8D"
    C_NEW = "#C0392B"

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0),
                              gridspec_kw={"wspace": 0.38})

    def _draw(ax, old, new, vals, nom, xlabel, xscale):
        v   = np.array(vals, dtype=float)
        ax.set_xscale(xscale)

        for data, color, label, ls in [
            (old, C_OLD, "Cheating spread (steps 1–t_f−1)",   "--"),
            (new, C_NEW, "Physical spread (steps 1–N_ref)",    "-"),
        ]:
            p50 = np.array([r["p50"] for r in data])
            p75 = np.array([r["p75"] for r in data])
            p95 = np.array([r["p95"] for r in data])
            ax.fill_between(v, p50, p75, alpha=0.15, color=color)
            ax.plot(v, p50, color=color, lw=0.9, ls=ls, alpha=0.6)
            ax.plot(v, p95, color=color, lw=1.8, marker="o", ms=4.5,
                    ls=ls, label=label)

        ax.axhline(REQ_UM, color="black", ls="--", lw=0.9, alpha=0.80,
                   label="20 µm requirement")
        ax.axvline(nom, color="0.50", ls=":", lw=0.75, alpha=0.65)
        y_top = max(max(r["p95"] for r in new) * 1.12, REQ_UM * 1.5)
        ax.set_ylim(0, np.ceil(y_top / 10) * 10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"P95 $\varepsilon_{2D}$ [µm]")
        ax.grid(True, ls=":", alpha=0.28)
        ax.legend(fontsize=8)

    _draw(axes[0], tau_old, tau_new,
          tau_values, 3.0,
          r"System latency $\tau_l$ [ms]", "linear")
    axes[0].set_title(
        r"Latency $\tau_l$ sensitivity" + "\n"
        r"Cheating: τ_rem=1ms always → flat line" + "\n"
        r"Physical: τ_rem = τ_l → latency matters",
        fontsize=9,
    )

    _draw(axes[1], p0v_old, p0v_new,
          p0v_values, 2.0,
          r"Initial velocity prior $\sigma_{v_0}$ [m/s]", "log")
    axes[1].set_title(
        r"Velocity prior $\sigma_{v_0}$ sensitivity" + "\n"
        r"Cheating (τ_rem=1ms): prior irrelevant → flat" + "\n"
        r"Physical (τ_rem=3ms): prior matters at large σ",
        fontsize=9,
    )

    fig.suptitle(
        r"Impact of the Physically-Correct Measurement Schedule" + "\n"
        r"'Cheating' spread includes steps inside the latency window"
        r" → $\tau_{\rm rem}=1$ ms always",
        fontsize=10.5, y=1.03,
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"  [saved] {save_path}")
    return fig


# Figure 3 — slew rate + shot rate

def _run_slew_sweep(td_values, n_mc=150):
    """
    For each theta_ddot_max, measure:
      - targeting gap ||p_i - r_hat|| (mirror vs. where KF says to point)
      - prediction error (should be flat since slew rate doesn't affect the KF)
    """
    records = []
    for td in td_values:
        env = IFEEnvironment(
            n_measurements=N_REF, measurement_steps=NOMINAL_STEPS,
            theta_ddot_max=float(td), seed=SEED,
        )
        env.u_prev[:] = 0   # fresh start each value
        gaps, pred = [], []
        for _ in range(n_mc):
            res  = env.run_episode()
            r_hat = res["r_hat"]
            u_opt = res["u_opt"]
            # Per-beamlet: unconstrained focal point vs actual focal point
            g = 0.0
            for i, (L, o) in enumerate(zip(env.L_list, env.o_list_nominal)):
                u_free = np.linalg.solve(L.T @ L, L.T @ (r_hat - o))
                p_free = L @ u_free + o
                p_act  = L @ u_opt[2*i:2*i+2] + o
                g += np.linalg.norm(p_act - p_free)
            gaps.append(g / env.n_beamlets * 1e6)
            e = (res["r_hat"] - res["r_true_tf"]) * 1e6
            pred.append(np.sqrt(e[0]**2 + e[1]**2))
        gaps, pred = np.array(gaps), np.array(pred)
        delta_urad = 0.5 * td * (TAU_LATENCY**2) * 1e6
        records.append(dict(
            td       = td,
            delta    = delta_urad,
            gap_p95  = np.percentile(gaps, 95),
            gap_p50  = np.median(gaps),
            pred_p95 = np.percentile(pred, 95),
        ))
        print(f"  tdot={td:7.1f}  delta={delta_urad:8.1f} urad  "
              f"gap_P95={records[-1]['gap_p95']:8.1f} um  "
              f"pred_P95={records[-1]['pred_p95']:6.1f} um")
    return records


def _shot_rate_critical(theta_ddot_values, required_p95_urad=117.0,
                        tau_l=TAU_LATENCY, theta_max=0.5):
    """
    For each theta_ddot_max: compute the available steering during tau_l,
    and the critical shot rate above which inter-shot gap starts to limit repositioning.
    """
    delta_lat = 0.5 * np.array(theta_ddot_values) * tau_l**2 * 1e6  # µrad
    # f_crit: shot rate where inter-shot delta = required steering
    # delta_intershot = 0.5 * theta_ddot * (1/f_s)^2 >= required_p95_urad
    # → f_crit = sqrt(0.5 * theta_ddot * 1e6 / required_p95_urad)
    f_crit = np.sqrt(0.5 * np.array(theta_ddot_values) * 1e6 / required_p95_urad)
    return delta_lat, f_crit


def plot_slew_figure(slew_records, save_path="figures/fig_slew_deep_dive.pdf"):
    td_vals   = np.array([r["td"]       for r in slew_records])
    gap_p95   = np.array([r["gap_p95"]  for r in slew_records])
    gap_p50   = np.array([r["gap_p50"]  for r in slew_records])
    pred_p95  = np.array([r["pred_p95"] for r in slew_records])
    delta_urad = np.array([r["delta"]   for r in slew_records])

    crit_td = None   # theta_ddot where the targeting gap first collapses to zero
    for k in range(len(td_vals) - 1):
        if gap_p95[k] > 0 and gap_p95[k+1] == 0:
            crit_td = (td_vals[k] + td_vals[k+1]) / 2

    td_range = np.geomspace(1, 2000, 50)
    delta_lat_arr, f_crit_arr = _shot_rate_critical(td_range)
    required_delta = 117.0  # P95 required angle in µrad

    fig = plt.figure(figsize=(9.5, 7.5))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.38,
                   left=0.09, right=0.97, top=0.93, bottom=0.09)

    ax = fig.add_subplot(gs[0, 0])
    ax.fill_between(td_vals, gap_p50, gap_p95, alpha=0.18, color="#C0392B")
    ax.loglog(td_vals, np.maximum(gap_p95, 0.1), color="#C0392B", lw=2.0,
              marker="o", ms=5, label=r"Targeting gap P95  $\|p_i - \hat{r}\|$")
    ax.semilogx(td_vals, pred_p95, color="#2980B9", lw=1.6, ls="--",
                marker="s", ms=4, label=r"Prediction error P95  $\|\hat{r}-r_{\rm true}\|$")
    ax.axhline(REQ_UM, color="black", ls="--", lw=0.9, alpha=0.80)
    if crit_td:
        ax.axvline(crit_td, color="#E67E22", ls="-.", lw=1.2, alpha=0.7)
        ax.text(crit_td * 1.15, gap_p95.max() * 0.6,
                rf"$\ddot\theta^*\approx${crit_td:.0f}",
                color="#E67E22", fontsize=8)
    ax.text(3, REQ_UM * 1.35, "20 µm req.", fontsize=7.5, color="0.35")
    ax.text(800, 1.5, "Flat: KF prediction\nindependent of mirror",
            fontsize=7.5, color="#2980B9", ha="center")
    ax.set_xlabel(r"Mirror slew rate $\ddot\theta_{\rm max}$ [rad/s²]")
    ax.set_ylabel(r"Error [µm]")
    ax.set_title("Slew Rate Impact on Two Metrics\n"
                 "Prediction error: zero sensitivity (KF ≠ mirror)\n"
                 "Targeting gap: collapses at critical threshold")
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.28)
    ax.set_ylim(bottom=0.1)

    ax2 = fig.add_subplot(gs[0, 1])
    td_fine = np.geomspace(1, 2000, 80)
    delta_fine = 0.5 * td_fine * (TAU_LATENCY**2) * 1e6
    ax2.loglog(td_fine, delta_fine, color="#2980B9", lw=2.0,
               label=r"Available: $\delta = \frac{1}{2}\ddot\theta\,\tau_l^2$")
    ax2.axhline(required_delta, color="#C0392B", ls="--", lw=1.2,
                label=f"P95 required ≈ {required_delta:.0f} µrad")
    ax2.fill_between(
        td_fine, delta_fine, required_delta,
        where=delta_fine < required_delta,
        alpha=0.15, color="#C0392B", label="Slew-limited zone",
    )
    ax2.fill_between(
        td_fine, delta_fine, required_delta,
        where=delta_fine >= required_delta,
        alpha=0.10, color="#27AE60",
    )
    ax2.scatter([THETA_DDOT_MAX], [0.5 * THETA_DDOT_MAX * TAU_LATENCY**2 * 1e6],
                s=90, color="#27AE60", zorder=5, edgecolors="black", lw=0.8,
                label=f"Nominal θ̈={THETA_DDOT_MAX:.0f} rad/s² (38× margin)")
    # Critical point
    td_crit = required_delta / (0.5 * (TAU_LATENCY**2) * 1e6)
    ax2.scatter([td_crit], [required_delta], marker="*", s=90,
                color="#E67E22", zorder=5,
                label=rf"Critical $\ddot\theta^*={td_crit:.0f}$ rad/s²")
    ax2.set_xlabel(r"$\ddot\theta_{\rm max}$ [rad/s²]")
    ax2.set_ylabel(r"Available steering $\delta$ [µrad]  (log)")
    ax2.set_title(r"Available Steering Budget During $\tau_l$" + "\n"
                  r"Below critical: mirror cannot reach $\hat{r}$")
    ax2.legend(fontsize=7.5)
    ax2.grid(True, ls=":", alpha=0.28)
    ax2.set_ylim(bottom=0.5)

    ax3 = fig.add_subplot(gs[1, 0])
    t_intersot = 1.0 / SHOT_RATE
    delta_intershot = 0.5 * td_range * t_intersot**2 * 1e6   # µrad
    ax3.loglog(td_range, delta_lat_arr, color="#2980B9", lw=2.0,
               label=rf"During latency ($\tau_l={TAU_LATENCY*1e3:.0f}$ ms)")
    ax3.loglog(td_range, delta_intershot, color="#8E44AD", lw=2.0, ls="-.",
               label=rf"Inter-shot gap ($f_s={SHOT_RATE:.0f}$ Hz → {t_intersot*1000:.0f} ms)")
    ax3.axhline(required_delta, color="#C0392B", ls="--", lw=1.2,
                label=f"P95 required ≈ {required_delta:.0f} µrad")
    ax3.scatter([THETA_DDOT_MAX], [delta_lat_arr[np.argmin(np.abs(td_range-THETA_DDOT_MAX))]],
                s=90, color="#27AE60", zorder=5, edgecolors="black", lw=0.8)
    ax3.text(1.2, required_delta * 3.0, "Inter-shot gap >300×\nlarger than latency\n→ shot rate irrelevant",
             fontsize=7.5, color="#8E44AD")
    ax3.set_xlabel(r"$\ddot\theta_{\rm max}$ [rad/s²]")
    ax3.set_ylabel(r"Available $\delta$ [µrad]")
    ax3.set_title(rf"Shot Rate ($f_s={SHOT_RATE:.0f}$ Hz) vs Latency Budget" + "\n"
                  "Inter-shot gap is 200 ms vs 3 ms latency → 66× more time")
    ax3.legend(fontsize=7.5)
    ax3.grid(True, ls=":", alpha=0.28)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.loglog(td_range, f_crit_arr, color="#D35400", lw=2.0,
               label=r"Critical $f_s^*(\ddot\theta)$: inter-shot starts to bind")
    ax4.axhline(SHOT_RATE, color="#27AE60", ls="--", lw=1.2,
                label=rf"Nominal $f_s={SHOT_RATE:.0f}$ Hz")
    ax4.axhline(1000, color="#C0392B", ls=":", lw=0.9, alpha=0.6,
                label="1 kHz (physically extreme)")
    ax4.fill_between(td_range, SHOT_RATE, f_crit_arr,
                     where=f_crit_arr > SHOT_RATE,
                     alpha=0.08, color="#27AE60", label="Safe zone (current)")
    ax4.scatter([THETA_DDOT_MAX], [SHOT_RATE], s=90, color="#27AE60",
                zorder=5, edgecolors="black", lw=0.8)
    ax4.text(2, SHOT_RATE * 1.5, "At 5 Hz, shot rate\nonly matters for\nθ̈ < 0.01 rad/s²",
             fontsize=7.5, color="0.35")
    ax4.set_xlabel(r"$\ddot\theta_{\rm max}$ [rad/s²]")
    ax4.set_ylabel(r"Critical shot rate $f_s^*$ [Hz]")
    ax4.set_title("Critical Shot Rate Where Inter-Shot\n"
                  "Repositioning Starts Limiting Accuracy")
    ax4.legend(fontsize=7.5)
    ax4.grid(True, ls=":", alpha=0.28)

    fig.suptitle(
        r"Slew Rate and Shot Rate Deep Dive — "
        r"Why $\dot{\theta}_{\rm max}$ Shows No Effect on Prediction Error",
        fontsize=10.5, y=1.00,
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"  [saved] {save_path}")
    return fig


def main():
    os.makedirs("figures", exist_ok=True)
    matplotlib.use("Agg")

    print("=== GP Warmup Deep Dive ===")
    gp_warmup_vals = [0, 5, 10, 15, 20, 30, 50, 75, 100]
    gp_records = _run_gp_sweep(gp_warmup_vals, n_mc=150)
    fig1 = plot_gp_figure(gp_records)
    plt.close(fig1)

    print("\n=== Physically-Correct Schedule: tau_l and p0_vel_std ===")
    tau_vals = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    p0v_vals = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    print("  Sweeping tau_latency_ms …")
    tau_old, tau_new = _run_param_both_schedules("tau_latency_ms", tau_vals)
    print("  Sweeping p0_vel_std …")
    p0v_old, p0v_new = _run_param_both_schedules("p0_vel_std", p0v_vals)

    fig2 = plot_schedule_comparison(tau_old, tau_new, p0v_old, p0v_new,
                                     tau_vals, p0v_vals)
    plt.close(fig2)

    print("\n=== Slew Rate Deep Dive ===")
    td_vals = [5, 10, 20, 30, 50, 100, 200, 500, 1000]
    slew_recs = _run_slew_sweep(td_vals, n_mc=150)
    fig3 = plot_slew_figure(slew_recs)
    plt.close(fig3)

    print("\nDone.")


if __name__ == "__main__":
    main()
