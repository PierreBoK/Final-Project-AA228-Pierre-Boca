"""
System matrices for IFE target kinematics (state = [r, v, a], 9D).
"""

import numpy as np
from .config import DT, SIGMA_A_PROCESS, SIGMA_TRANSVERSE, SIGMA_AXIAL


def build_F(dt: float = DT) -> np.ndarray:
    """9×9 constant-acceleration state transition matrix."""
    I3 = np.eye(3)
    Z3 = np.zeros((3, 3))
    return np.block([
        [I3,  dt * I3,  0.5 * dt**2 * I3],
        [Z3,  I3,       dt * I3          ],
        [Z3,  Z3,       I3               ],
    ])


def build_Q_kinematic(dt: float = DT,
                      sigma_a: float = SIGMA_A_PROCESS) -> np.ndarray:
    """
    Process-noise covariance from the Singer white-noise acceleration model.
    Q = σ_a² Γ Γ^T where Γ = [Δt²/2, Δt, 1]^T ⊗ I₃.
    """
    q = sigma_a ** 2
    I3 = np.eye(3)
    g = np.array([0.5 * dt**2, dt, 1.0])   # Γ column (scalar part)
    Q = np.zeros((9, 9))
    for r in range(3):
        for c in range(3):
            Q[3*r:3*(r+1), 3*c:3*(c+1)] = q * g[r] * g[c] * I3
    return Q


def build_H() -> np.ndarray:
    """3×9 measurement matrix that picks out [rx, ry, rz] from the full state."""
    H = np.zeros((3, 9))
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    H[2, 2] = 1.0
    return H


def build_R(sigma_t: float = SIGMA_TRANSVERSE,
            sigma_a: float = SIGMA_AXIAL) -> np.ndarray:
    """3×3 measurement noise covariance."""
    return np.diag([sigma_t**2, sigma_t**2, sigma_a**2])
