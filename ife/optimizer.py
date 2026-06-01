"""
Beam-pointing optimization for IFE.

Each beamlet is separable, so we solve N independent 2D QPs:
    min_u  ||L u - r_tilde||^2   s.t.  slew-rate box + global angle bounds

Three solvers:
  1. closed_form    — unconstrained LS + box projection (fast)
  2. interior_point — log-barrier Newton (exact)
  3. SAA            — sample-average approximation over p(r(t_f)|z)
  4. robust_ouu     — minimax over worst-case calibration drift
"""

import numpy as np
from typing import List, Optional
from .config import (THETA_MIN, THETA_MAX, THETA_DDOT_MAX, TAU_LATENCY,
                     N_SAMPLES_SAA, CHI2_95_3DOF)


def _ls_solution(L: np.ndarray, r_tilde: np.ndarray) -> np.ndarray:
    """Unconstrained LS: u* = (L^T L)^{-1} L^T r_tilde."""
    return np.linalg.solve(L.T @ L, L.T @ r_tilde)


def _box_project(u: np.ndarray,
                 lb: np.ndarray,
                 ub: np.ndarray) -> np.ndarray:
    return np.clip(u, lb, ub)


def _effective_bounds(u_prev: np.ndarray,
                      delta: float,
                      theta_min: float,
                      theta_max: float):
    """Slew-rate box intersected with the global angle limits."""
    lb = np.maximum(theta_min * np.ones(2), u_prev - delta)
    ub = np.minimum(theta_max * np.ones(2), u_prev + delta)
    return lb, ub


def _log_barrier_newton(A: np.ndarray,
                        b: np.ndarray,
                        lb: np.ndarray,
                        ub: np.ndarray,
                        u0: np.ndarray,
                        mu_init: float = 1.0,
                        mu_factor: float = 0.1,
                        n_outer: int = 20,
                        newton_tol: float = 1e-10,
                        max_newton_iter: int = 60) -> np.ndarray:
    """
    Log-barrier Newton solver for the 2D box-constrained QP:  min ½u^T A u - b^T u  s.t. lb≤u≤ub.
    Shrinks the barrier parameter mu by mu_factor each outer iteration.
    """
    n   = len(u0)
    u   = u0.copy()
    eps = 1e-8 * (ub - lb)
    u   = np.clip(u, lb + eps, ub - eps)

    mu = mu_init
    for _ in range(n_outer):
        for _ in range(max_newton_iter):
            sl = u - lb
            su = ub - u
            if np.any(sl <= 0) or np.any(su <= 0):
                u  = np.clip(u, lb + eps, ub - eps)
                sl = u - lb
                su = ub - u

            grad     = A @ u - b - mu * (1.0 / sl - 1.0 / su)
            H_diag   = 1.0 / sl**2 + 1.0 / su**2
            H_total  = A + mu * np.diag(H_diag)   # positive definite by construction
            step     = np.linalg.solve(H_total, -grad)

            alpha = 1.0
            for _ in range(40):
                u_try = u + alpha * step
                if np.all(u_try > lb) and np.all(u_try < ub):
                    break
                alpha *= 0.5
            else:
                break

            u = u + alpha * step
            if np.linalg.norm(grad) < newton_tol:
                break

        mu *= mu_factor
        if mu < 1e-13:
            break

    return u


def _solve_beamlet(L: np.ndarray,
                   r_tilde: np.ndarray,
                   u_prev: np.ndarray,
                   delta: float,
                   theta_min: float,
                   theta_max: float,
                   method: str = "interior_point") -> np.ndarray:
    lb, ub = _effective_bounds(u_prev, delta, theta_min, theta_max)
    u_star = _ls_solution(L, r_tilde)

    if method == "closed_form":
        return _box_project(u_star, lb, ub)

    # interior_point: if unconstrained solution is already feasible we're done
    if np.all(u_star >= lb) and np.all(u_star <= ub):
        return u_star

    A_i = 2.0 * L.T @ L
    b_i = 2.0 * L.T @ r_tilde
    u0  = _box_project(u_star, lb, ub)
    return _log_barrier_newton(A_i, b_i, lb, ub, u0)


def optimize_beamlets(L_list: List[np.ndarray],
                      o_list: List[np.ndarray],
                      r_hat: np.ndarray,
                      u_prev: np.ndarray,
                      tau_latency: float = TAU_LATENCY,
                      theta_ddot_max: float = THETA_DDOT_MAX,
                      theta_min: float = THETA_MIN,
                      theta_max: float = THETA_MAX,
                      method: str = "interior_point") -> np.ndarray:
    """Solve beam-pointing for all N beamlets; returns stacked u ∈ R^{2N}."""
    N     = len(L_list)
    delta = 0.5 * theta_ddot_max * (tau_latency ** 2)         
    u_opt = np.zeros(2 * N)

    for i, (L, o) in enumerate(zip(L_list, o_list)):
        u_i = _solve_beamlet(L, r_hat - o, u_prev[2*i:2*i+2],
                             delta, theta_min, theta_max, method)
        u_opt[2*i:2*i+2] = u_i

    return u_opt


def optimize_saa(L_list: List[np.ndarray],
                 o_list: List[np.ndarray],
                 r_hat: np.ndarray,
                 P_r: np.ndarray,
                 u_prev: np.ndarray,
                 n_samples: int = N_SAMPLES_SAA,
                 tau_latency: float = TAU_LATENCY,
                 theta_ddot_max: float = THETA_DDOT_MAX,
                 theta_min: float = THETA_MIN,
                 theta_max: float = THETA_MAX,
                 rng: np.random.Generator = None) -> np.ndarray:
    """
    SAA: draw samples from p(r(t_f)|z) = N(r_hat, P_r), optimize against sample mean.
    For linear-Gaussian the mean collapses to r_hat, but this is cleaner under non-Gaussian noise.
    """
    if rng is None:
        rng = np.random.default_rng()

    L_P = np.linalg.cholesky(P_r + 1e-16 * np.eye(3))
    r_samples = r_hat + (L_P @ rng.standard_normal((3, n_samples))).T
    r_saa = r_samples.mean(axis=0)

    return optimize_beamlets(L_list, o_list, r_saa, u_prev,
                             tau_latency=tau_latency,
                             theta_ddot_max=theta_ddot_max,
                             theta_min=theta_min,
                             theta_max=theta_max)


def optimize_robust_ouu(L_list: List[np.ndarray],
                        o_nominal_list: List[np.ndarray],
                        Sigma_cal: np.ndarray,
                        r_hat: np.ndarray,
                        u_prev: np.ndarray,
                        chi2_95: float = CHI2_95_3DOF,
                        tau_latency: float = TAU_LATENCY,
                        theta_ddot_max: float = THETA_DDOT_MAX,
                        theta_min: float = THETA_MIN,
                        theta_max: float = THETA_MAX) -> np.ndarray:
    """
    Minimax robust optimization over worst-case calibration drift in the ellipsoidal set
    U = {Δo : Δo^T Σ^{-1} Δo ≤ χ²_{3,0.95}}.
    The inner max has a closed form (Cauchy-Schwarz), so we just shift the effective target.
    """
    N     = len(L_list)
    delta = 0.5 * theta_ddot_max * (tau_latency ** 2)

    radius     = np.sqrt(chi2_95)
    Sigma_half = np.linalg.cholesky(Sigma_cal)

    u_opt = np.zeros(2 * N)
    for i, (L, o) in enumerate(zip(L_list, o_nominal_list)):
        r_tilde = r_hat - o
        u_prev_i = u_prev[2*i:2*i+2]
        lb, ub   = _effective_bounds(u_prev_i, delta, theta_min, theta_max)

        u_nom = _ls_solution(L, r_tilde)
        resid = L @ np.clip(u_nom, lb, ub) - r_tilde

        if np.linalg.norm(resid) > 1e-12:
            d_hat = resid / np.linalg.norm(resid)
        else:
            d_hat = np.zeros(3)

        # shift the effective target to hedge against the worst-case drift
        drift_mag = radius * np.linalg.norm(Sigma_half.T @ d_hat)
        r_tilde_robust = r_tilde - drift_mag * d_hat

        u_star = _ls_solution(L, r_tilde_robust)
        if np.all(u_star >= lb) and np.all(u_star <= ub):
            u_i = u_star
        else:
            A_i = 2.0 * L.T @ L
            b_i = 2.0 * L.T @ r_tilde_robust
            u_i = _log_barrier_newton(A_i, b_i, lb, ub,
                                      _box_project(u_star, lb, ub))

        u_opt[2*i:2*i+2] = u_i

    return u_opt
