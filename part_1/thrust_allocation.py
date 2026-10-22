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
    """Template for student thrust allocation."""

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
        
        """Map desired BODY wrench to per-thruster commands.
        The vehicle has a 3-DOF wrench [Fx, Fy, Mz] and three thrusters, so a linear allocation is sufficient for Part 1. 
        For each thruster i, [Fx_i, Fy_i, Mz_i]^T = u_i * [cos(alpha_i), sin(alpha_i), x_i*sin(alpha_i) - y_i*cos(alpha_i)]
        and the full allocation is simply tau = B @ u.
        """
        
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
