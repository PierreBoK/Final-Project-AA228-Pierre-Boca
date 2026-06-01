"""
Beamlet geometry: each beamlet i has p_i(u_i) = L_i u_i + o_i.
u_i are (az, el) pointing angles, L_i is the 3×2 path geometry matrix,
o_i is the nominal focal point offset.
"""

import numpy as np
from .config import N_BEAMLETS, MIRROR_RADIUS


def build_beamlet_geometry(n_beamlets: int = N_BEAMLETS,
                            mirror_radius: float = MIRROR_RADIUS,
                            nominal_offset_std: float = 5e-6,
                            rng: np.random.Generator = None):
    """
    Place beamlets on a sphere via Fibonacci lattice and compute (L_i, o_i).
    Returns L_list (3×2 each) and o_list (3D each).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    L_list, o_list = [], []
    phi_gold = (1.0 + np.sqrt(5.0)) / 2.0   # golden ratio gives good sphere coverage

    for i in range(n_beamlets):
        theta = 2.0 * np.pi * i / phi_gold
        phi   = np.arccos(1.0 - 2.0 * (i + 0.5) / n_beamlets)

        m = mirror_radius * np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
        ])

        d_nom = -m / np.linalg.norm(m)   # beam points toward chamber centre

        # orthonormal steering frame perpendicular to the beam direction
        ref = np.array([0.0, 0.0, 1.0]) if abs(d_nom[2]) < 0.9 \
              else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(d_nom, ref);  e1 /= np.linalg.norm(e1)
        e2 = np.cross(d_nom, e1);   e2 /= np.linalg.norm(e2)

        # paraxial approximation: 1 rad of pointing → mirror_radius metres of focal shift
        L_i = mirror_radius * np.column_stack([e1, e2])

        o_i = rng.normal(0.0, nominal_offset_std, size=3)

        L_list.append(L_i)
        o_list.append(o_i)

    return L_list, o_list


def focal_point(u_i: np.ndarray,
                L_i: np.ndarray,
                o_i: np.ndarray) -> np.ndarray:
    """p_i(u_i) = L_i u_i + o_i  (paraxial affine map)."""
    return L_i @ u_i + o_i
