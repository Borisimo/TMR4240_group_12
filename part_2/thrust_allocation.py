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


# Commented out the previous part so nothing is lost.


"""
from typing import List, Optional, Tuple
import numpy as np

from models.thruster_dynamics import ThrusterConfig


class ThrustAllocator:
    ""Template for student thrust allocation.""

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

        # TODO: Replace this placeholder with your thrust allocation algorithm.
        # The placeholder commands zero thrust and alpha for all thrusters.
        

     
        ""Map desired BODY wrench to per-thruster commands.
        The vehicle has a 3-DOF wrench [Fx, Fy, Mz] and three thrusters, so a linear allocation is sufficient for Part 1. 
        For each thruster i, [Fx_i, Fy_i, Mz_i]^T = u_i * [cos(alpha_i), sin(alpha_i), x_i*sin(alpha_i) - y_i*cos(alpha_i)]
        and the full allocation is simply tau = B @ u.
        ""

        
        if n == 0:
            return np.zeros(0), np.zeros(0)

        # In this project the vessel uses only 3 DOFs in the horizontal plane:
        # surge, sway, and yaw. The rest of the 6D wrench is irrelevant here.
        
        tau_req = np.asarray(tau_d, dtype=float).reshape(6)[[0, 1, 5]].copy()
        if np.allclose(tau_req, 0.0):
            return np.zeros(n), np.array([th.alpha0 for th in self.thrusters], dtype=float)

        # Default azimuth angles: if the caller does not give a current angle
        # state, we keep the thrusters at their nominal configuration.
        
        alpha_cmd = np.array([th.alpha0 for th in self.thrusters], dtype=float)
        if alpha_now is not None:
            alpha_cmd = np.asarray(alpha_now, dtype=float).reshape(n).copy()

        # Build the wrench matrix B, where each column is the contribution of one thruster to the 3D wrench [Fx, Fy, Mz]^T.
        # For a thruster with thrust magnitude u_i and angle alpha_i:
        #   Fx_i = u_i * cos(alpha_i)
        #   Fy_i = u_i * sin(alpha_i)
        #   Mz_i = u_i * (x_i * sin(alpha_i) - y_i * cos(alpha_i))
        
        B = np.zeros((3, n), dtype=float)
        for i, th in enumerate(self.thrusters):
            a = float(alpha_cmd[i])
            c = np.cos(a)
            s = np.sin(a)
            B[0, i] = c
            B[1, i] = s
            B[2, i] = th.x * s - th.y * c

        # Solve B @ u = tau_req. If the matrix is singular or nearly singular,
        # the pseudo-inverse gives the minimum-norm solution that still matches
        # the requested wrench as closely as possible.
        
        try:
            u_cmd = np.linalg.solve(B, tau_req)
        except np.linalg.LinAlgError:
            u_cmd = np.linalg.pinv(B) @ tau_req

        # The project checks all commands against each thruster's max thrust.
        # If the unconstrained solution violates those limits, scale everything
        # down by a single factor so the allocation stays feasible.

        max_u = np.array([th.u_max for th in self.thrusters], dtype=float)
        if np.any(np.abs(u_cmd) > max_u):
            denom = np.maximum(np.abs(u_cmd), 1e-12)
            scale = np.min(max_u / denom)
            u_cmd = u_cmd * float(scale)

        # We keep the commanded angle vector as the current/nominal angle set.
        # In Part 1 this is sufficient because the actuator model is ideal and the
        # project checks are primarily about achieving the correct wrench while respecting limits.

        return u_cmd, alpha_cmd
"""

"""
Thrust Allocation

Maps the desired body-frame wrench to individual thruster commands. The
simulator calls, once per step:

    allocator.allocate(t, dt, tau_d, u_now, alpha_now) -> (u_cmd, alpha_cmd)

Inputs (full actuator state -- use what your algorithm needs):
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

-----------------------------------------------------------------------------
Why this version is different from a plain fixed-angle solve
-----------------------------------------------------------------------------
A tunnel thruster (kind == "tunnel") is physically fixed pointing athwart-ship
(alpha0, rot_speed == 0) -- it can only ever contribute force in that one
direction, so there is nothing to optimize for it.

An azimuth thruster (kind == "azimuth") can point anywhere. If we naively
solve for (thrust magnitude, fixed angle) with the angle left at its nominal
value, the azimuths never actually get used for what they're good at, and the
vessel ends up depending on the tunnel thruster alone for sway -- which is
usually not nearly enough capacity, forcing a large scale-down of every
manoeuvre that needs sway or a combined wrench.

The fix: instead of parameterising an azimuth thruster by (u_i, alpha_i)
(nonlinear -- alpha_i appears inside sin/cos), parameterise it by its
Cartesian force components (Fx_i, Fy_i) directly. That makes the whole
allocation problem LINEAR again:

    Fx_i = Fx_i                        (by definition)
    Fy_i = Fy_i
    Mz_i = x_i * Fy_i - y_i * Fx_i      (standard moment-arm formula)

Solve the resulting (usually underdetermined -- more unknowns than the 3
wrench equations) linear system with the Moore-Penrose pseudo-inverse, which
picks the minimum-norm solution: the one that achieves the requested wrench
using the least total force, spread sensibly across the azimuths. Then
recover each azimuth's actual thrust magnitude and pointing angle from its
solved (Fx_i, Fy_i):

    u_i     = sqrt(Fx_i^2 + Fy_i^2)
    alpha_i = atan2(Fy_i, Fx_i)
"""



"""
            Thrust allocation dynamic azimuth fix from part 1
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