"""
Current template

Students should provide the ambient current as a generalized NED velocity
vector. The simulator calls, once per step:

    current.step(t, dt, eta, nu) -> nu_c_ned

Inputs (full 6-DOF state — use what your model needs):
    t    : current simulation time [s]        (time-varying currents)
    dt   : time step [s]                      (slowly-varying components)
    eta  : (6,) vessel state [N, E, z, phi, theta, psi] in NED
           (heading is eta[5]; position for spatially varying fields)
    nu   : (6,) vessel BODY velocities [u, v, w, p, q, r]
           (only indices [0, 1, 5] are nonzero in the 3-DOF model)

Output:
    nu_c_ned : (6,) generalized NED current velocity [m/s]
               [V_N, V_E, V_D, 0, 0, 0]
               Only the horizontal components are used by the 3-DOF model.
               Direction convention is 'towards' (the direction the current
               flows to): a current with V_N > 0, V_E = 0 pushes the vessel
               North.
"""
import numpy as np


class Current:
    """Template for student current model.

    Constructor contract — the automated checks (``python check.py``,
    ``pytest``, ``notebooks/part_1_demo.ipynb``) construct your model with
    this signature, so keep it working:

        Current(speed, beta, semantics=..., beta_end=..., duration=...)

    Parameters
    ----------
    speed : current speed [m/s].
    beta : direction [rad] in NED (0 = North, pi/2 = East).
    semantics : ``"towards"`` (default) — ``beta`` is the direction the
        current flows to — or ``"from"`` — the direction it comes from.
    beta_end, duration : if given, the direction varies linearly from
        ``beta`` to ``beta_end`` over ``duration`` seconds (Simulation 2),
        then stays at ``beta_end``.  Constant direction if ``beta_end`` is
        ``None``.
    """

    def __init__(self, speed: float = 0.0, beta: float = 0.0, *,
                 semantics: str = "towards",
                 beta_end: float | None = None, duration: float = 0.0):
        # TODO: Store and use the parameters above in step().
        self.speed = float(speed)
        self.duration = float(duration)

        # Normalising everything to the "towards" convention
        if semantics == "from":
            offset = np.pi
        elif semantics == "towards":
            offset = 0.0
        else:
            raise ValueError(
                f"semantics must be 'towards' or 'from', got {semantics!r}"
            )

        self.beta = float(beta) + offset
        self.beta_end = (
            None if beta_end is None else float(beta_end) + offset
        )

    def step(
        self,
        t: float,
        dt: float,
        eta: np.ndarray,
        nu: np.ndarray,
    ) -> np.ndarray:
        
        # Direction at time t constant, or linearly ramped over
        if self.beta_end is None:
            angle = self.beta
        else:
            frac = 0.0 if self.duration <= 0.0 else np.clip(t / self.duration, 0.0, 1.0)
            angle = self.beta + frac * (self.beta_end - self.beta)

        # Speed and direction
        V_N = self.speed * np.cos(angle)
        V_E = self.speed * np.sin(angle)

        # Generalized vector to return
        nu_c_ned = np.zeros(6)
        nu_c_ned[0] = V_N
        nu_c_ned[1] = V_E

        return nu_c_ned
