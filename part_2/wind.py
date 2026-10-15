"""
Wind template

Students should compute generalized BODY-frame wind loads:
    tau_w6 = [Fx, Fy, Fz, Mx, My, Mz]

The simulator uses the 3-DOF subset [Fx, Fy, Mz] = tau_w6 indices [0, 1, 5]
and calls, once per step:

    wind.step(t, dt, eta, nu) -> (tau_w6, info)

Inputs (full 6-DOF state — use what your model needs):
    t    : current simulation time [s]        (gust spectra, time variation)
    dt   : time step [s]                      (slowly-varying components)
    eta  : (6,) vessel state [N, E, z, phi, theta, psi] in NED
           (heading is eta[5])
    nu   : (6,) vessel BODY velocities [u, v, w, p, q, r]
           (RELATIVE wind: compute the loads from V_rw = V_wind - V_vessel,
            using the horizontal components nu[0], nu[1])

Outputs:
    tau_w6 : (6,) BODY loads
    info   : optional dict for logging, e.g.
             {"U": ambient speed, "beta_ned": direction (towards, rad),
              "alpha_body": relative wind angle in BODY (rad)}
             Return {} (or None) if you do not need it.
             NOTE: "beta_ned" is always the direction the wind blows
             TOWARDS, even when the constructor semantics is "from" —
             convert before logging, do not log the raw constructor value.

Wind coefficient data
---------------------
The vessel wind coefficients C(alpha) = [Cx, Cy, Cz, Cphi, Ctheta, Cpsi] are
provided in `data/wind_coeff.csv` (repository root), tabulated against the relative
wind angle alpha in degrees (0..360). Load them with:

    alpha_deg, C6 = load_wind_coefficients()

The wind loads are then computed as F_wind = U_rw^2 * C(alpha_rw), where U_rw
and alpha_rw are the relative wind speed and angle in the BODY frame.
"""
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

_WIND_COEFF_FILE = Path(__file__).resolve().parent.parent / "data" / "wind_coeff.csv"


def load_wind_coefficients() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the vessel wind coefficient table.

    Returns
    -------
    alpha_deg : (M,) ndarray
        Relative wind angle grid [deg], from 0 to 360.
    C6 : (M, 6) ndarray
        Coefficients [Cx, Cy, Cz, Cphi, Ctheta, Cpsi] at each angle.
    """
    table = np.loadtxt(_WIND_COEFF_FILE, delimiter=",", skiprows=1)
    return table[:, 0], table[:, 1:]


class Wind:
    """Template for student wind model.

    Constructor contract — the automated checks (``python check.py``,
    ``pytest``, ``notebooks/part_1_demo.ipynb``) construct your model with
    this signature, so keep it working:

        Wind(mean_speed, beta, semantics=..., sigma_slow=..., seed=...)

    Parameters
    ----------
    mean_speed : mean wind speed [m/s].
    beta : direction [rad] in NED (0 = North, pi/2 = East).
    semantics : ``"from"`` (default, the usual meteorological convention —
        "wind from south" blows northward) or ``"towards"``.
    sigma_slow : standard deviation of the slowly-varying wind speed
        component [m/s] (required in Part 1; 0 disables it).
    tau_slow : time constant of the slow variation [s].
    seed : random seed for the slow component, so runs are reproducible.
    """

    def __init__(self, mean_speed: float = 0.0, beta: float = 0.0, *,
                 semantics: str = "from", sigma_slow: float = 0.0,
                 tau_slow: float = 120.0, seed: int | None = None):
        # TODO: Store and use the parameters above in step().
        self.mean_speed = float(mean_speed)
        self.sigma_slow = float(sigma_slow)
        self.tau_slow = float(tau_slow)

        if semantics == "from":
            offset = np.pi
        elif semantics == "towards":
            offset = 0.0
        else:
            raise ValueError(
                f"semantics must be 'towards' or 'from', got {semantics!r}"
            )
        self.beta_towards = float(beta) + offset

        # Loading the coefficients
        self._alpha_deg, self._C6 = load_wind_coefficients()

        # Seeded RNG
        self._rng = np.random.default_rng(seed)

        # State of slowly warying wind speed component that stays consistent with every run
        self._u_slow = 0.0

    def _interp_coeff(self, alpha_rw_deg: float) -> np.ndarray:
        """Interpolate C(alpha) at given relative wind angle [deg], wrapping correctly across the 0 - 369 deg boundary."""
        alpha = alpha_rw_deg % 360.0

        alpha_ext = np.concatenate(
            ([self._alpha_deg[-1] - 360.0], self._alpha_deg, [self._alpha_deg[0] + 360.0])
        )
        C_ext = np.vstack([self._C6[-1], self._C6, self._C6[0]])
        return np.array([
            np.interp(alpha, alpha_ext, C_ext[:,j]) for j in range(6)
        ])


    def step(
        self,
        t: float,
        dt: float,
        eta: np.ndarray,
        nu: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        psi = eta[5]
        u, v = nu[0], nu[1]

        # Total speed U(t).
        if self.sigma_slow > 0.0 and self.tau_slow > 0.0 and dt > 0.0:
            a = np.exp(-dt / self.tau_slow)
            b = self.sigma_slow * np.sqrt(1.0 - a ** 2)
            self._u_slow = a * self._u_slow + b * self._rng.standard_normal()
        U = self.mean_speed + self._u_slow

        # Ambient wind, NED components
        V_wN = U * np.cos(self.beta_towards)
        V_wE = U * np.sin(self.beta_towards)

        # Rotating ambient wind NED -> BODY.
        c, s = np.cos(psi), np.sin(psi)
        V_wb_x = c * V_wN + s * V_wE
        V_wb_y = -s * V_wN + c * V_wE

        # Relative wind = ambient (body) - vessel velocity.
        V_rw_x = V_wb_x - u
        V_rw_y = V_wb_y - v
        U_rw = np.hypot(V_rw_x, V_rw_y)
        alpha_rw = np.arctan2(V_rw_y, V_rw_x)  # rad, body frame

        # Interpolating coefficients (table is in degrees) and applying force law
        C = self._interp_coeff(np.degrees(alpha_rw))
        tau_w6 = U_rw ** 2 * C

        info = {
            "U": U,
            "beta_ned": self.beta_towards % (2 * np.pi),
            "alpha_body": alpha_rw,
        }
        return tau_w6, info
