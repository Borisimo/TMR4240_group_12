"""
Thrust Allocation template

Students should implement an algorithm that maps the desired body-frame
wrench to individual thruster commands. The simulator calls, once per step:

    allocator.allocate(t, dt, tau_d, u_now, alpha_now) -> (u_cmd, alpha_cmd)

Inputs (full actuator state — use what your algorithm needs):
    t         : current simulation time [s]
    dt        : time step [s]              (rate-aware/dynamic allocation)
    tau_d     : (6,) desired BODY wrench [Fx, Fy, Fz, Mx, My, Mz]
                (the 3-DOF wrench to allocate is tau_d[[0, 1, 5]]
                 = [Fx, Fy, Mz]; the other components are zero)
    u_now     : current actual thrusts [N]     (rate-aware allocation)
    alpha_now : current thruster angles [rad]  (minimize azimuth slewing)

Outputs:
    u_cmd     : signed thrust command for each thruster [N]
    alpha_cmd : thruster angle command for each thruster [rad]

Students may implement, for example:
    - pseudo-inverse allocation,
    - weighted least-squares allocation,
    - optimization-based allocation,
    - power-minimizing allocation.
"""


from typing import List, Optional, Tuple

import numpy as np

from models.thruster_dynamics import ThrusterConfig


class ThrustAllocator:
    """Pseudo-inverse thrust allocation with azimuth-angle optimization."""

    def __init__(self, thrusters: List[ThrusterConfig]):
        self.thrusters = thrusters

    def allocate(
        self,
        t: float,
        dt: float,
        tau_d: np.ndarray,
        u_now: Optional[np.ndarray] = None,
        alpha_now: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(self.thrusters)
        if n == 0:
            return np.zeros(0), np.zeros(0)

        # Only the horizontal-plane wrench [Fx, Fy, Mz] is relevant here.
        tau_req = np.asarray(tau_d, dtype=float).reshape(6)[[0, 1, 5]].copy()

        nominal_alpha = np.array([th.alpha0 for th in self.thrusters], dtype=float)
        if np.allclose(tau_req, 0.0):
            return np.zeros(n), nominal_alpha.copy()

        # ------------------------------------------------------------------
        # Build the extended, linear allocation matrix.
        #
        # Column layout: each tunnel thruster gets ONE column (its fixed
        # direction, scalar signed thrust). Each azimuth thruster gets TWO
        # columns (its Fx and Fy components, solved independently). We keep
        # a map from each thruster's index to which column(s) in the
        # extended system belong to it, so we can reassemble u_cmd/alpha_cmd
        # in the original thruster order afterward.
        # ------------------------------------------------------------------
        columns = []          # list of (3,) arrays, one per extended column
        col_owner = []        # thruster index each column belongs to
        col_kind = []         # "tunnel" or "fx"/"fy" for azimuth columns

        for i, th in enumerate(self.thrusters):
            if th.kind == "tunnel":
                a = th.alpha0
                c, s = np.cos(a), np.sin(a)
                columns.append(np.array([c, s, th.x * s - th.y * c]))
                col_owner.append(i)
                col_kind.append("tunnel")
            else:  # azimuth: two Cartesian columns
                columns.append(np.array([1.0, 0.0, -th.y]))   # unit Fx
                col_owner.append(i)
                col_kind.append("fx")
                columns.append(np.array([0.0, 1.0, th.x]))    # unit Fy
                col_owner.append(i)
                col_kind.append("fy")

        B_ext = np.column_stack(columns)  # shape (3, n_columns)

        # Minimum-norm solution: achieves tau_req using the least total
        # force, which is exactly what we want when the system is
        # underdetermined (more force components than wrench equations).
        f_ext = np.linalg.pinv(B_ext) @ tau_req

        # ------------------------------------------------------------------
        # Recover per-thruster (u_i, alpha_i) from the extended solution.
        # ------------------------------------------------------------------
        u_cmd = np.zeros(n)
        alpha_cmd = nominal_alpha.copy()
        fx_by_thruster = {}
        fy_by_thruster = {}

        for col_val, owner, kind in zip(f_ext, col_owner, col_kind):
            if kind == "tunnel":
                u_cmd[owner] = col_val
                # alpha_cmd[owner] stays at nominal_alpha (fixed direction)
            elif kind == "fx":
                fx_by_thruster[owner] = col_val
            elif kind == "fy":
                fy_by_thruster[owner] = col_val

        for i, th in enumerate(self.thrusters):
            if th.kind != "tunnel":
                fx = fx_by_thruster.get(i, 0.0)
                fy = fy_by_thruster.get(i, 0.0)
                u_cmd[i] = float(np.hypot(fx, fy))
                if u_cmd[i] > 1e-9:
                    alpha_cmd[i] = float(np.arctan2(fy, fx))
                # if this azimuth's solved force is ~0, leave it at its
                # nominal angle rather than commanding an arbitrary one

        # ------------------------------------------------------------------
        # Saturation: if any thruster's required MAGNITUDE exceeds its
        # limit, scale every thrust down by a single factor. Uses abs()
        # throughout since tunnel thrust is signed (direction comes from a
        # fixed alpha0, reversal is via sign) while azimuth thrust is kept
        # as a non-negative magnitude (direction comes from alpha instead).
        # Scaling by a single positive factor preserves sign correctly for
        # both conventions at once.
        # ------------------------------------------------------------------
        max_u = np.array([th.u_max for th in self.thrusters], dtype=float)
        abs_u = np.abs(u_cmd)
        if np.any(abs_u > max_u):
            denom = np.maximum(abs_u, 1e-12)
            scale = np.min(max_u / denom)
            u_cmd = u_cmd * float(scale)

        # Tunnel thrusters cannot rotate at all (rot_speed == 0), so their
        # alpha_cmd must stay at alpha0 unconditionally -- their u_cmd is
        # already signed (positive/negative reverses direction along that
        # fixed axis), so no further sign handling is needed here.

        return u_cmd, alpha_cmd