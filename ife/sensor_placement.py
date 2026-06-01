"""
Sensor placement schedules and CRLB calculation for IFE target tracking.
Each schedule function returns a list of measurement step indices in [1, t_f-1].
"""

import numpy as np
from .config import DT, TAU_LATENCY, N_MEASUREMENTS, SIGMA_TRANSVERSE

def _t_f_idx(n_meas_ref: int = N_MEASUREMENTS,
             tau_latency: float = TAU_LATENCY,
             dt: float = DT) -> int:
    """Step index of firing time t_f."""
    return n_meas_ref + max(1, int(round(tau_latency / dt)))


def _resolve(raw_steps, m: int, max_step: int) -> list:
    """Return exactly m unique steps in [1, max_step], padding from the front if we're short."""
    steps = sorted(set(max(1, min(max_step, int(s))) for s in raw_steps))
    all_v = list(range(1, max_step + 1))
    while len(steps) < m:
        for s in all_v:
            if s not in steps:
                steps = sorted(steps + [s])
                break
    return steps[:m]


def consecutive_schedule(m: int,
                         n_meas_ref: int = N_MEASUREMENTS,
                         tau_latency: float = TAU_LATENCY,
                         dt: float = DT) -> list:
    """Default schedule: steps 1..m all at the start, worst CRLB because of the long gap to t_f."""
    t_f = _t_f_idx(n_meas_ref, tau_latency, dt)
    max_step = t_f - 1
    return list(range(1, min(m, max_step) + 1))


def random_schedule(m: int,
                    n_meas_ref: int = N_MEASUREMENTS,
                    tau_latency: float = TAU_LATENCY,
                    dt: float = DT,
                    seed: int = 42) -> list:
    """Random schedule — stochastic baseline."""
    t_f = _t_f_idx(n_meas_ref, tau_latency, dt)
    max_step = t_f - 1

    if m >= max_step:
        return list(range(1, max_step + 1))

    rng = np.random.default_rng(seed)
    steps = rng.choice(np.arange(1, max_step + 1), size=m, replace=False)
    return sorted(steps.tolist())


def recommended_spread_schedule(m: int,
                                n_meas_ref: int = N_MEASUREMENTS,
                                tau_latency: float = TAU_LATENCY,
                                dt: float = DT) -> list:
    """
    Evenly spread across the whole window — the "doubly-compounded benefit" schedule.
    Both S_tt and tau_remaining improve simultaneously.
    """
    t_f = _t_f_idx(n_meas_ref, tau_latency, dt)
    max_step = t_f - 1
    if m == 1:
        return [max_step]

    raw = np.round(np.linspace(1, max_step, m)).astype(int)
    raw = np.clip(raw, 1, max_step)
    steps = sorted(set(int(s) for s in raw))

    # rounding can produce collisions — fill in extras from the back
    all_valid = list(range(1, max_step + 1))
    while len(steps) < m:
        for s in reversed(all_valid):
            if s not in steps:
                steps.append(s)
                steps = sorted(steps)
                break
    return steps[:m]


def end_clustered_schedule(m: int,
                           n_meas_ref: int = N_MEASUREMENTS,
                           tau_latency: float = TAU_LATENCY,
                           dt: float = DT) -> list:
    """
    Cluster all measurements at the end of the window.
    Minimizes tau_remaining but hurts velocity estimation (small S_tt).
    """
    t_f = _t_f_idx(n_meas_ref, tau_latency, dt)
    max_step = t_f - 1
    span = max(m - 1, max_step // 3)
    start = max(1, max_step - span)
    raw = np.round(np.linspace(start, max_step, m)).astype(int)
    return _resolve(raw, m, max_step)


def two_anchor_schedule(m: int,
                        n_meas_ref: int = N_MEASUREMENTS,
                        tau_latency: float = TAU_LATENCY,
                        dt: float = DT) -> list:
    """
    D-optimal: half at start, half at end.
    Maximizes Fisher info (∝ S_tt) for the position+velocity model.
    """
    t_f = _t_f_idx(n_meas_ref, tau_latency, dt)
    max_step = t_f - 1
    m_start = m // 2
    m_end = m - m_start
    starts = list(range(1, m_start + 1))
    ends = list(range(max(m_start + 1, max_step - m_end + 1), max_step + 1))
    raw = sorted(set(starts + ends))
    return _resolve(raw, m, max_step)


def chebyshev_schedule(m: int,
                       n_meas_ref: int = N_MEASUREMENTS,
                       tau_latency: float = TAU_LATENCY,
                       dt: float = DT) -> list:
    """
    Chebyshev nodes: dense at both ends, sparse in the middle.
    Good S_tt from the end anchors; sits between uniform spread and two-anchor.
    """
    t_f = _t_f_idx(n_meas_ref, tau_latency, dt)
    max_step = t_f - 1
    if m == 1:
        return [max_step]
    k = np.arange(m)
    tc = 0.5 * (1 + max_step) + 0.5 * (max_step - 1) * np.cos(np.pi * k / (m - 1))
    raw = np.clip(np.round(tc).astype(int), 1, max_step)
    return _resolve(raw, m, max_step)


def geometric_schedule(m: int,
                       n_meas_ref: int = N_MEASUREMENTS,
                       tau_latency: float = TAU_LATENCY,
                       dt: float = DT) -> list:
    """
    Log-spaced: sparse early, dense near t_f.
    One early anchor for velocity init; finer resolution close to firing.
    """
    t_f = _t_f_idx(n_meas_ref, tau_latency, dt)
    max_step = t_f - 1
    if m == 1:
        return [max_step]
    # build from the end backwards, then flip to ascending order
    raw = (max_step + 1) - np.round(np.geomspace(1, max_step, m)).astype(int)[::-1]
    raw = np.clip(raw, 1, max_step)
    return _resolve(raw, m, max_step)


def schedule_dt_multiples(measurement_steps: list) -> list:
    """Convert step indices to inter-step gap counts for KalmanFilter.run_filter."""
    s = list(measurement_steps)
    return [s[0]] + [s[k] - s[k - 1] for k in range(1, len(s))]


def crlb_position_error(measurement_steps: list,
                         sigma_meas: float,
                         t_f_idx: int,
                         dt: float = DT) -> float:
    """
    CRLB per-axis position error at t_f: sigma_pred = (sigma_meas / sqrt(S_tt)) * tau_remaining.
    S_tt = sum of squared deviations of measurement times from their mean.
    """
    t_meas = np.array(measurement_steps, dtype=float) * dt
    t_bar = t_meas.mean()
    sum_sq_dev = float(np.sum((t_meas - t_bar) ** 2))

    if sum_sq_dev < 1e-30:
        return np.inf

    sigma_v = sigma_meas / np.sqrt(sum_sq_dev)
    tau_remaining = (t_f_idx - int(measurement_steps[-1])) * dt
    return sigma_v * abs(tau_remaining)


def compare_schedules_crlb(m_values=(2, 4, 6, 8, 10),
                            n_meas_ref: int = N_MEASUREMENTS,
                            tau_latency: float = TAU_LATENCY,
                            sigma_meas: float = SIGMA_TRANSVERSE,
                            dt: float = DT) -> dict:
    """CRLB sweep across all seven schedules for each m in m_values."""
    schedule_fns = {
        "consecutive":   consecutive_schedule,
        "random":        random_schedule,
        "recommended":   recommended_spread_schedule,
        "end_clustered": end_clustered_schedule,
        "two_anchor":    two_anchor_schedule,
        "chebyshev":     chebyshev_schedule,
        "geometric":     geometric_schedule,
    }

    results = {name: [] for name in schedule_fns}

    for m in m_values:
        t_f = _t_f_idx(n_meas_ref, tau_latency, dt)
        for name, fn in schedule_fns.items():
            steps = fn(m, n_meas_ref=n_meas_ref, tau_latency=tau_latency, dt=dt)
            err = crlb_position_error(steps, sigma_meas, t_f, dt)
            results[name].append(err * 1e6)   # store in µm

    return results
