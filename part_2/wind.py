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

from simulation.utils import wrap_angle_pi

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
    """Wind loads with slow speed variation, gusts, and bounded direction noise.

    Constructor contract — the automated checks (``python check.py``,
    ``pytest``) construct this model with the parameters below.

        Wind(mean_speed, beta, semantics=..., sigma_slow=..., gust=...,
             gust_params=..., sigma_dir=..., tau_dir=..., dir_limit=..., seed=...)

    Parameters
    ----------
    mean_speed : mean wind speed [m/s].
    beta : direction [rad] in NED (0 = North, pi/2 = East).
    semantics : ``"from"`` (default, the usual meteorological convention —
        "wind from south" blows northward) or ``"towards"``.
    sigma_slow : standard deviation of the slowly-varying wind speed
        component [m/s] (required in Part 1; 0 disables it).
    tau_slow : time constant of the slow variation [s].
    gust : enable zero-mean gust-speed variation.
    gust_params : optional ``{"sigma": ..., "tau": ...}`` gust standard
        deviation [m/s] and time constant [s].
    sigma_dir : standard deviation of the slowly varying direction [rad].
    tau_dir : direction time constant [s].
    dir_limit : maximum absolute direction excursion from the mean [rad].
    seed : random seed for all varying components, so runs are reproducible.
    """

    def __init__(self, mean_speed: float = 0.0, beta: float = 0.0, *,
                 semantics: str = "from", sigma_slow: float = 0.0,
                 tau_slow: float = 120.0, gust: bool = False,
                 gust_params: Dict[str, float] | None = None,
                 sigma_dir: float = 0.0, tau_dir: float = 300.0,
                 dir_limit: float = np.deg2rad(5.0),
                 seed: int | None = None):
        self.mean_speed = float(mean_speed)
        self.sigma_slow = float(sigma_slow)
        self.tau_slow = float(tau_slow)
        self.gust = bool(gust)
        self.gust_params = {"sigma": 1.0, "tau": 5.0,
                    "spectrum": "norsok",   # "markov" = current behaviour, "norsok" = new
                    "U10": None,            # None -> use mean_speed
                    "z": 10.0, "n_freq": 100,
                    "f_min": 0.01,          # [Hz] below this the slow component (sigma_slow) takes over
                    "f_max": 0.5}
        if gust_params is not None:
            unknown = set(gust_params) - self.gust_params.keys()
            if unknown:
                raise ValueError(f"unknown gust parameter(s): {sorted(unknown)}")
            self.gust_params.update(gust_params)
        self.sigma_dir = float(sigma_dir)
        self.tau_dir = float(tau_dir)
        self.dir_limit = float(dir_limit)
        if self.sigma_slow < 0.0 or self.tau_slow <= 0.0:
            raise ValueError("sigma_slow must be non-negative and tau_slow > 0")
        if self.sigma_dir < 0.0 or self.tau_dir <= 0.0:
            raise ValueError("sigma_dir must be non-negative and tau_dir > 0")
        if self.dir_limit < 0.0:
            raise ValueError("dir_limit must be non-negative")
        if self.gust_params["sigma"] < 0.0 or self.gust_params["tau"] <= 0.0:
            raise ValueError("gust sigma must be non-negative and tau > 0")

        if semantics == "from":
            offset = np.pi
        elif semantics == "towards":
            offset = 0.0
        else:
            raise ValueError(
                f"semantics must be 'towards' or 'from', got {semantics!r}"
            )
        self.beta_towards = float(beta) + offset

        self._alpha_deg, self._C6 = load_wind_coefficients()
        self._rng = np.random.default_rng(seed)
        self._u_slow = 0.0
        self._u_gust = 0.0
        self._dir_slow = 0.0
        self._seed = seed
        if self.gust and self.gust_params["spectrum"] == "norsok":
            self._build_norsok_gust(seed)

    def _interp_coeff(self, alpha_rw_deg: float) -> np.ndarray:
        """Interpolate C(alpha) at a relative wind angle, wrapping at 360°."""
        alpha = alpha_rw_deg % 360.0

        alpha_ext = np.concatenate(
            ([self._alpha_deg[-1] - 360.0], self._alpha_deg, [self._alpha_deg[0] + 360.0])
        )
        C_ext = np.vstack([self._C6[-1], self._C6, self._C6[0]])
        return np.array([np.interp(alpha, alpha_ext, C_ext[:, j]) for j in range(6)])

    def _build_norsok_gust(self, seed):
        p = self.gust_params
        U10 = max(p["U10"] if p["U10"] is not None else self.mean_speed, 1e-6)
        z, n = p["z"], 0.468
        edges = np.logspace(np.log10(p["f_min"]), np.log10(p["f_max"]), int(p["n_freq"]) + 1)
        rng = np.random.default_rng(None if seed is None else [seed, 1])  # separate stream
        f = rng.uniform(edges[:-1], edges[1:])      # random frequency in each bin: no periodic repeat
        df = np.diff(edges)
        x = 172.0 * f * (z / 10) ** (2 / 3) * (U10 / 10) ** (-3 / 4)
        S = 320.0 * (U10 / 10) ** 2 * (z / 10) ** 0.45 / (1 + x ** n) ** (5 / (3 * n))
        self._g_amp = np.sqrt(2.0 * S * df)         # a_i = sqrt(2 S(f_i) df_i)
        self._g_w = 2.0 * np.pi * f
        self._g_phi = rng.uniform(0.0, 2.0 * np.pi, f.size)   # seeded random phases

    def step(
        self,
        t: float,
        dt: float,
        eta: np.ndarray,
        nu: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        eta = np.asarray(eta, dtype=float).reshape(6)
        nu = np.asarray(nu, dtype=float).reshape(6)
        if dt <= 0.0:
            raise ValueError("dt must be greater than zero")
        psi = eta[5]
        u, v = nu[0], nu[1]

        if self.sigma_slow > 0.0:
            a = np.exp(-dt / self.tau_slow)
            b = self.sigma_slow * np.sqrt(1.0 - a**2)
            self._u_slow = a * self._u_slow + b * self._rng.standard_normal()
        U_mean = self.mean_speed + self._u_slow

        if self.gust and self.gust_params["spectrum"] == "norsok":
            self._u_gust = float(np.sum(self._g_amp * np.cos(self._g_w * t + self._g_phi)))
        elif self.gust and self.gust_params["sigma"] > 0.0:
            a_gust = np.exp(-dt / self.gust_params["tau"])
            b_gust = self.gust_params["sigma"] * np.sqrt(1.0 - a_gust**2)
            self._u_gust = a_gust * self._u_gust + b_gust * self._rng.standard_normal()
        else:
            self._u_gust = 0.0
        U_gust = self._u_gust
        U = U_mean + U_gust

        if self.sigma_dir > 0.0 and self.dir_limit > 0.0:
            a_dir = np.exp(-dt / self.tau_dir)
            b_dir = self.sigma_dir * np.sqrt(1.0 - a_dir**2)
            self._dir_slow = a_dir * self._dir_slow + b_dir * self._rng.standard_normal()
            self._dir_slow = float(np.clip(self._dir_slow, -self.dir_limit, self.dir_limit))
        else:
            self._dir_slow = 0.0
        beta_towards = wrap_angle_pi(self.beta_towards + self._dir_slow)

        V_wN = U * np.cos(beta_towards)
        V_wE = U * np.sin(beta_towards)

        c, s = np.cos(psi), np.sin(psi)
        V_wb_x = c * V_wN + s * V_wE
        V_wb_y = -s * V_wN + c * V_wE

        V_rw_x = V_wb_x - u
        V_rw_y = V_wb_y - v
        U_rw = np.hypot(V_rw_x, V_rw_y)
        alpha_rw = np.arctan2(V_rw_y, V_rw_x)

        C = self._interp_coeff(np.degrees(alpha_rw))
        tau_w6 = U_rw**2 * C

        info = {
            "U": U,
            "beta_ned": beta_towards % (2 * np.pi),
            "alpha_body": alpha_rw,
            "U_mean": U_mean,
            "U_gust": U_gust,
        }
        return tau_w6, info
