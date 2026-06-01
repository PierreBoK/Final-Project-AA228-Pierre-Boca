"""
Linear Kalman filter for IFE target state estimation.
Standard predict-update cycle plus forward propagation to t_f.
"""

import numpy as np
from .dynamics import build_F, build_Q_kinematic, build_H, build_R
from .config import DT, SIGMA_TRANSVERSE, SIGMA_AXIAL, SIGMA_A_PROCESS


class KalmanFilter:
    """
    9D Kalman filter (state = [r, v, a]).
    After the last measurement, call propagate_to_tf to extrapolate to firing time.
    """

    def __init__(self, dt: float = DT, Q: np.ndarray = None,
                 R: np.ndarray = None):
        self.dt  = dt
        self.F   = build_F(dt)
        self.H   = build_H()
        self.Q   = Q if Q is not None else build_Q_kinematic(dt)
        self.R   = R if R is not None else build_R()
        self._I9 = np.eye(9)

        self.x_hat = np.zeros(9)
        self.P     = np.eye(9) * 1.0

    def reset(self, x0: np.ndarray, P0: np.ndarray):
        self.x_hat = x0.copy()
        self.P     = P0.copy()

    def predict(self, Q_override: np.ndarray = None):
        Q = Q_override if Q_override is not None else self.Q
        self.x_hat = self.F @ self.x_hat
        self.P     = self.F @ self.P @ self.F.T + Q

    def update(self, z: np.ndarray):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        innov = z - self.H @ self.x_hat
        self.x_hat = self.x_hat + K @ innov
        self.P = (self._I9 - K @ self.H) @ self.P
        return innov

    def run_filter(self, measurements: np.ndarray,
                   dt_multiples: list = None,
                   Q_seq: list = None):
        """
        Run predict-update over all measurements.
        dt_multiples lets you handle non-uniform schedules (steps between measurements).
        """
        if dt_multiples is None:
            dt_multiples = [1] * len(measurements)
        for k, (z, n_steps) in enumerate(zip(measurements, dt_multiples)):
            Q_k = Q_seq[k] if Q_seq is not None else None
            for _ in range(n_steps):
                self.predict(Q_override=Q_k)
            self.update(z)

    def propagate_to_tf(self, tau: float):
        """Propagate forward by tau seconds and return (r_hat, P_r) at t_f."""
        steps   = max(1, int(round(tau / self.dt)))
        x_pred  = self.x_hat.copy()
        P_pred  = self.P.copy()
        for _ in range(steps):
            x_pred = self.F @ x_pred
            P_pred = self.F @ P_pred @ self.F.T + self.Q
        r_hat = x_pred[:3]
        P_r   = P_pred[:3, :3]
        return r_hat, P_r
