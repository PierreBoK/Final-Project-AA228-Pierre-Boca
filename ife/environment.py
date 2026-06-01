"""
IFEEnvironment — end-to-end simulation for one IFE laser engagement.

Wraps trajectory simulation, noisy measurements, Kalman filtering,
beam-pointing optimization, and Monte Carlo evaluation.

Two offset lists are kept separate:
  o_list_true    — physical focal offsets (used to evaluate real engagement error)
  o_list_nominal — what the optimizer sees (can lag behind after calibration drift)

With use_online_cal=True, the nominal offsets are updated after each shot
via EMA (alpha=0.3 → converges in ~3-4 shots).

Passing measurement_steps explicitly lets you test non-uniform schedules
spread across the full [1, t_f-1] window instead of the default consecutive start.
"""

import numpy as np
from .config import (
    DT, TAU_LATENCY, N_MEASUREMENTS, N_BEAMLETS,
    THETA_DDOT_MAX, THETA_MIN, THETA_MAX,
    SIGMA_TRANSVERSE, SIGMA_AXIAL, SIGMA_A_PROCESS,
    TARGET_SPEED_NOM, N_SAMPLES_SAA,
    CAL_DRIFT_STD, CHI2_95_3DOF, ACCURACY_3SIGMA,
)
from .dynamics import build_F, build_Q_kinematic, build_H, build_R
from .kalman_filter import KalmanFilter
from .beam_model import build_beamlet_geometry, focal_point
from .optimizer import optimize_beamlets, optimize_saa, optimize_robust_ouu
from .gp_surrogate import GPSurrogate
from .sensor_placement import schedule_dt_multiples


class IFEEnvironment:
    """
    Full IFE engagement simulator.
    See module docstring for details on calibration drift and measurement schedules.
    """

    def __init__(self,
                 n_beamlets: int           = N_BEAMLETS,
                 dt: float                 = DT,
                 tau_latency: float        = TAU_LATENCY,
                 n_measurements: int       = N_MEASUREMENTS,
                 measurement_steps: list   = None,
                 target_speed: float       = TARGET_SPEED_NOM,
                 optimizer_mode: str       = "interior_point",
                 use_saa: bool             = False,
                 use_gp: bool              = False,
                 use_robust: bool          = False,
                 use_online_cal: bool      = False,
                 online_cal_alpha: float   = 0.3,
                 seed: int                 = 42,
                 # --- sensitivity-analysis overrides (default = config values) ---
                 sigma_a: float            = SIGMA_A_PROCESS,
                 theta_ddot_max: float     = THETA_DDOT_MAX,
                 p0_vel_std: float         = 2.0):

        self.dt               = dt
        self.tau_latency      = tau_latency
        self.target_speed     = target_speed
        self.optimizer_mode   = optimizer_mode
        self.use_saa          = use_saa
        self.use_gp           = use_gp
        self.theta_ddot_max   = theta_ddot_max
        self.p0_vel_std       = p0_vel_std
        self.use_robust     = use_robust
        self.use_online_cal = use_online_cal
        self.online_cal_alpha = online_cal_alpha
        self.rng            = np.random.default_rng(seed)

        self.latency_steps = max(1, int(round(tau_latency / dt)))
        if measurement_steps is not None:
            self.measurement_steps = sorted(int(s) for s in measurement_steps)
            self.n_measurements    = len(self.measurement_steps)
        else:
            self.n_measurements    = n_measurements
            self.measurement_steps = list(range(1, n_measurements + 1))

        # t_f is fixed by physics; use nominal n_measurements so comparisons
        # across different schedule types share the same firing time
        self.t_f_idx = n_measurements + self.latency_steps

        assert all(1 <= s < self.t_f_idx for s in self.measurement_steps), (
            f"All measurement steps must be in [1, {self.t_f_idx - 1}]; "
            f"got {self.measurement_steps}"
        )

        # how many KF predict steps between each consecutive measurement pair
        all_steps   = self.measurement_steps
        kf_steps    = all_steps[1:]
        self._kf_dt_multiples = (
            [kf_steps[0] - all_steps[0]]
            + [kf_steps[k] - kf_steps[k - 1] for k in range(1, len(kf_steps))]
        ) if len(kf_steps) > 0 else []

        self._remaining_steps = self.t_f_idx - self.measurement_steps[-1]

        self.F        = build_F(dt)
        self.Q_nom    = build_Q_kinematic(dt, sigma_a)
        self.H        = build_H()
        self.R        = build_R(SIGMA_TRANSVERSE, SIGMA_AXIAL)

        # Kalman filter
        self.kf = KalmanFilter(dt=dt, Q=self.Q_nom, R=self.R)

        self.L_list, _o_base = build_beamlet_geometry(
            n_beamlets=n_beamlets, rng=self.rng)
        self.n_beamlets = n_beamlets

        # o_list_true = physical reality; o_list_nominal = what the optimizer sees.
        # Drift goes into true only so the optimizer is genuinely uncertain.
        self.o_list_true    = [o.copy() for o in _o_base]
        self.o_list_nominal = [o.copy() for o in _o_base]

        self.Sigma_cal = (CAL_DRIFT_STD**2) * np.eye(3)

        self.gp = GPSurrogate() if use_gp else None
        self.history_trajectories: list = []

        self.u_prev = np.zeros(2 * n_beamlets)

    def _simulate_trajectory(self) -> np.ndarray:
        tf_idx = self.t_f_idx

        v_mag = self.target_speed + self.rng.normal(0.0, 0.05)

        # mostly +z with µrad-scale angular jitter
        phi_inj   = self.rng.normal(0.0, 4e-6)
        theta_inj = self.rng.uniform(0.0, 2.0 * np.pi)
        v_dir = np.array([
            np.sin(phi_inj) * np.cos(theta_inj),
            np.sin(phi_inj) * np.sin(theta_inj),
            1.0,
        ])
        v_dir /= np.linalg.norm(v_dir)
        v0 = v_mag * v_dir

        # force injection at (x=0, y=0) and back-calculate z so it arrives at t_f
        z_start = -tf_idx * self.dt * v_mag
        r0 = np.array([0.0, 0.0, z_start])

        a0 = self.rng.normal(0.0, 0.1, size=3)
        x0 = np.concatenate([r0, v0, a0])

        total  = tf_idx + 1
        states = np.empty((total, 9))
        states[0] = x0
        for k in range(1, total):
            w         = self.rng.multivariate_normal(np.zeros(9), self.Q_nom)
            states[k] = self.F @ states[k - 1] + w

        return states

    def _generate_measurements(self, states: np.ndarray) -> np.ndarray:
        """Generate noisy observations at each step in self.measurement_steps."""
        zs = np.empty((self.n_measurements, 3))
        for j, step in enumerate(self.measurement_steps):
            nu     = self.rng.multivariate_normal(np.zeros(3), self.R)
            zs[j]  = self.H @ states[step] + nu
        return zs

    def _true_r_at_tf(self, states: np.ndarray) -> np.ndarray:
        """True target position at firing time t_f."""
        return states[self.t_f_idx, :3]

    def _update_calibration(self, p_list: list, r_hat: np.ndarray):
        """EMA update of nominal focal offsets: o_nominal_i += α(p_i − r_hat)."""
        alpha = self.online_cal_alpha
        for i, p_i in enumerate(p_list):
            delta_est = np.array(p_i) - r_hat
            self.o_list_nominal[i] = self.o_list_nominal[i] + alpha * delta_est

    def run_episode(self) -> dict:
        """Run one full engagement episode and return all relevant quantities."""
        states    = self._simulate_trajectory()
        meas      = self._generate_measurements(states)
        r_true_tf = self._true_r_at_tf(states)

        # init KF from first measurement; velocity prior centred on nominal speed
        x0_kf = np.zeros(9)
        x0_kf[:3]  = meas[0]
        x0_kf[3:6] = np.array([0.0, 0.0, self.target_speed])
        P0 = np.diag(
            [1e-3**2] * 3
            + [self.p0_vel_std**2] * 3
            + [1.0**2] * 3
        )
        self.kf.reset(x0_kf, P0)

        # build GP-informed Q sequence if available
        Q_seq = None
        if self.use_gp and self.gp is not None and self.gp.X_train is not None:
            Q_seq = []
            x_tmp = x0_kf.copy()
            for _ in range(self.n_measurements - 1):
                Q_k = self.gp.predict_Q(x_tmp[:3])
                Q_seq.append(Q_k if Q_k is not None else self.Q_nom)
                x_tmp = self.F @ x_tmp

        self.kf.run_filter(meas[1:],
                           dt_multiples=self._kf_dt_multiples if self._kf_dt_multiples else None,
                           Q_seq=Q_seq)

        tau_remaining = self._remaining_steps * self.dt
        r_hat, P_r = self.kf.propagate_to_tf(tau_remaining)

        # optimizer always gets nominal offsets — never the true drifted ones
        if self.use_robust:
            u_opt = optimize_robust_ouu(
                self.L_list, self.o_list_nominal, self.Sigma_cal,
                r_hat, self.u_prev,
                chi2_95=CHI2_95_3DOF,
                tau_latency=self.tau_latency,
                theta_ddot_max=self.theta_ddot_max,
                theta_min=THETA_MIN, theta_max=THETA_MAX,
            )
        elif self.use_saa:
            u_opt = optimize_saa(
                self.L_list, self.o_list_nominal, r_hat, P_r, self.u_prev,
                n_samples=N_SAMPLES_SAA,
                tau_latency=self.tau_latency,
                theta_ddot_max=self.theta_ddot_max,
                theta_min=THETA_MIN, theta_max=THETA_MAX,
                rng=self.rng,
            )
        else:
            u_opt = optimize_beamlets(
                self.L_list, self.o_list_nominal, r_hat, self.u_prev,
                tau_latency=self.tau_latency,
                theta_ddot_max=self.theta_ddot_max,
                theta_min=THETA_MIN, theta_max=THETA_MAX,
                method=self.optimizer_mode,
            )

        # evaluate against TRUE offsets — this is what actually matters
        p_list = [
            focal_point(u_opt[2*i:2*i+2], L, o)
            for i, (L, o) in enumerate(zip(self.L_list, self.o_list_true))
        ]
        errors    = [np.linalg.norm(p - r_true_tf) for p in p_list]
        rms_error = np.sqrt(np.mean([e**2 for e in errors]))

        self.u_prev = u_opt.copy()

        if self.use_gp:
            self.history_trajectories.append(states)

        if self.use_online_cal:
            self._update_calibration(p_list, r_hat)

        return {
            "r_true_tf":        r_true_tf,
            "r_hat":            r_hat,
            "P_r":              P_r,
            "u_opt":            u_opt,
            "p_list":           p_list,
            "errors":           errors,
            "rms_error":        rms_error,
            "prediction_error": np.linalg.norm(r_hat - r_true_tf),
        }

    def update_gp(self):
        """Refit GP on accumulated trajectory history."""
        if not self.use_gp or len(self.history_trajectories) < 3:
            return
        self.gp.fit_from_trajectories(self.history_trajectories, self.F)
        if len(self.history_trajectories) >= 10:
            self.gp.optimize_hyperparams()

    def run_monte_carlo(self,
                        n_episodes: int = 1000,
                        gp_update_interval: int = 20,
                        verbose: bool = False) -> dict:
        """Run n_episodes and return aggregated stats (mean, std, p95, p99, success rate)."""
        rms_arr  = np.empty(n_episodes)
        pred_arr = np.empty(n_episodes)

        for ep in range(n_episodes):
            res          = self.run_episode()
            rms_arr[ep]  = res["rms_error"]
            pred_arr[ep] = res["prediction_error"]

            if self.use_gp and (ep + 1) % gp_update_interval == 0:
                self.update_gp()

            if verbose and (ep + 1) % 100 == 0:
                print(f"  episode {ep+1:4d} | "
                      f"rms={rms_arr[:ep+1].mean()*1e6:.2f} µm")

        return {
            "rms_errors":    rms_arr,
            "pred_errors":   pred_arr,
            "eps_rms_mean":  rms_arr.mean(),
            "eps_rms_std":   rms_arr.std(),
            "eps_rms_p95":   np.percentile(rms_arr, 95),
            "eps_rms_p99":   np.percentile(rms_arr, 99),
            "req_met_frac":  np.mean(rms_arr <= ACCURACY_3SIGMA),
        }
