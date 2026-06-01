"""
GP surrogate for spatially-varying process noise Q(r).

Learns Q from past trajectory residuals δ_k = x_k - F x_{k-1}.
Per-component normalization is critical: raw residuals span ~12 orders of magnitude
across position/velocity/acceleration, so without it sigma_f is meaningless.
"""

import numpy as np
from scipy.optimize import minimize


class GPSurrogate:
    """GP that predicts Q(r) from past process-noise residuals, with per-component normalization."""

    def __init__(self,
                 sigma_f: float = 1.0,
                 length_scale: float = 0.05,
                 sigma_n: float = 1e-4):
        self.sigma_f = sigma_f
        self.ell     = length_scale
        self.sigma_n = sigma_n

        self.X_train: np.ndarray = None   # (n, 3) positions
        self.Y_train: np.ndarray = None   # (n, 9) raw residuals
        self._K_inv:  np.ndarray = None   # (n, n) cached inverse Gram matrix
        self._alpha:  np.ndarray = None   # (n, 9) K_inv @ Y_norm (cached)
        self._scales: np.ndarray = None   # (9,)   per-component RMS scale factors

    def _k(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """SE kernel: σ_f² exp(-||r-r'||² / 2ℓ²)."""
        diff    = X1[:, None, :] - X2[None, :, :]   # (n1, n2, 3)
        sq_dist = np.sum(diff**2, axis=-1)            # (n1, n2)
        ell2 = max(self.ell, 1e-300) ** 2            # guard against ℓ→0 during opt
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return self.sigma_f**2 * np.exp(-sq_dist / (2.0 * ell2))

    def _compute_scales(self, Y: np.ndarray) -> np.ndarray:
        """Per-component RMS, clamped away from zero to avoid divide-by-zero."""
        scales = np.sqrt(np.mean(Y**2, axis=0))        # (9,)
        scales = np.where(scales > 1e-15, scales, 1.0)
        return scales

    def fit(self, positions: np.ndarray, residuals: np.ndarray):
        """Fit GP to (n,3) positions and (n,9) process-noise residuals."""
        self.X_train = positions.copy()
        self.Y_train = residuals.copy()

        self._scales = self._compute_scales(self.Y_train)
        Y_norm = self.Y_train / self._scales[None, :]

        K = self._k(positions, positions)
        K += self.sigma_n**2 * np.eye(len(positions))
        self._K_inv = np.linalg.inv(K)
        self._alpha  = self._K_inv @ Y_norm            # (n, 9) — cached

    def fit_from_trajectories(self, trajectories: list, F: np.ndarray):
        """Extract residuals from trajectory list and fit."""
        X_all, Y_all = [], []
        for traj in trajectories:
            for k in range(1, len(traj)):
                delta = traj[k] - F @ traj[k - 1]
                X_all.append(traj[k - 1, :3])
                Y_all.append(delta)
        if X_all:
            self.fit(np.array(X_all), np.array(Y_all))

    def predict_Q(self, r: np.ndarray) -> np.ndarray | None:
        """
        Return Q(r) in physical units.
        Works in normalized space then de-normalizes: Q_hat = S @ (Q_emp_norm + σ*² I) @ S.
        Returns None if not yet trained (caller falls back to fixed Q_nom).
        """
        if self.X_train is None or len(self.X_train) == 0:
            return None

        r2d  = np.atleast_2d(r)
        k_s  = self._k(r2d, self.X_train)

        mu_norm = (k_s @ self._alpha)[0]

        k_ss   = float(self.sigma_f**2)
        sigma2 = max(0.0, k_ss - float((k_s @ self._K_inv @ k_s.T).item()))

        # kernel-weighted empirical covariance (stays in normalized space)
        w   = k_s[0]
        w  /= w.sum() + 1e-12
        Y_norm = self.Y_train / self._scales[None, :]
        dc_norm = Y_norm - mu_norm[None, :]
        Q_emp_norm = np.einsum("i,ij,ik->jk", w, dc_norm, dc_norm)

        Q_norm = Q_emp_norm + sigma2 * np.eye(9)
        S = self._scales
        Q_pred = S[:, None] * Q_norm * S[None, :]
        return Q_pred

    def optimize_hyperparams(self):
        """Maximize log marginal likelihood over (ℓ, σ_f) via Nelder-Mead."""
        if self.X_train is None or len(self.X_train) < 5:
            return

        Y_norm = self.Y_train / self._scales[None, :]

        def nll(log_params):
            ell_try, sf_try = np.exp(log_params[0]), np.exp(log_params[1])
            self.ell, self.sigma_f = ell_try, sf_try
            K = self._k(self.X_train, self.X_train)
            K += self.sigma_n**2 * np.eye(len(self.X_train))
            try:
                L_ch   = np.linalg.cholesky(K)
                logdet = 2.0 * np.sum(np.log(np.diag(L_ch)))
                K_inv  = np.linalg.inv(K)
                val = 0.0
                for d in range(Y_norm.shape[1]):
                    y   = Y_norm[:, d]
                    val += 0.5 * (y @ K_inv @ y + logdet)
                return val
            except np.linalg.LinAlgError:
                return 1e10

        res = minimize(nll,
                       x0=np.log([self.ell, self.sigma_f]),
                       method="Nelder-Mead",
                       options={"maxiter": 200, "xatol": 1e-4})
        self.ell     = np.exp(res.x[0])
        self.sigma_f = np.exp(res.x[1])

        # refresh cached K_inv and alpha with the new hyperparams
        K = self._k(self.X_train, self.X_train)
        K += self.sigma_n**2 * np.eye(len(self.X_train))
        self._K_inv = np.linalg.inv(K)
        self._alpha  = self._K_inv @ Y_norm
