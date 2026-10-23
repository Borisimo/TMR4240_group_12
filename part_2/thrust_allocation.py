"""Rate-aware wrench allocation for the Part 2 Gunnerus thrusters."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from models.thruster_dynamics import ThrusterConfig
from simulation.utils import wrap_angle_pi


class ThrustAllocator:
    """Allocate BODY-frame surge, sway, and yaw requests.

    Azimuth angles are optimization variables. Their per-step change and each
    thruster's signed force are bounded by the actuator's rotation, thrust-rate,
    and maximum-thrust limits. Tunnel thrusters retain their fixed geometry.
    """

    def __init__(self, thrusters: List[ThrusterConfig]):
        self.thrusters = list(thrusters)

    @staticmethod
    def _column(thruster: ThrusterConfig, angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([c, s, thruster.x * s - thruster.y * c])

    @staticmethod
    def _column_derivative(thruster: ThrusterConfig, angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([-s, c, thruster.x * c + thruster.y * s])

    def allocate(
        self,
        t: float,
        dt: float,
        tau_d: np.ndarray,
        u_now: Optional[np.ndarray] = None,
        alpha_now: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not np.isfinite(t) or not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("t must be finite and dt must be finite and greater than zero")

        n = len(self.thrusters)
        if n == 0:
            return np.zeros(0), np.zeros(0)

        target = np.asarray(tau_d, dtype=float).reshape(6)[[0, 1, 5]]
        if not np.all(np.isfinite(target)):
            raise ValueError("tau_d must contain only finite values")

        thrust_now = (
            np.zeros(n) if u_now is None
            else np.asarray(u_now, dtype=float).reshape(n)
        )
        angle_now = (
            np.array([thruster.alpha0 for thruster in self.thrusters], dtype=float)
            if alpha_now is None
            else np.asarray(alpha_now, dtype=float).reshape(n)
        )
        if not np.all(np.isfinite(thrust_now)) or not np.all(np.isfinite(angle_now)):
            raise ValueError("current thrusts and angles must contain only finite values")

        maximum = np.array([thruster.u_max for thruster in self.thrusters], dtype=float)
        rate = np.array([thruster.u_rate for thruster in self.thrusters], dtype=float)
        if np.any(maximum <= 0.0) or np.any(rate <= 0.0):
            raise ValueError("thrusters must have positive maximum thrust and thrust rate")

        low_thrust = np.maximum(-maximum, thrust_now - rate * dt)
        high_thrust = np.minimum(maximum, thrust_now + rate * dt)

        moving: list[int] = []
        angle_bounds: list[tuple[float, float]] = []
        fixed_angles = np.array(
            [thruster.alpha0 for thruster in self.thrusters], dtype=float
        )
        for i, thruster in enumerate(self.thrusters):
            if thruster.kind.lower() == "tunnel" or thruster.rot_speed == 0.0:
                continue
            moving.append(i)
            max_angle_step = float(thruster.rot_speed) * dt
            lower, upper = -max_angle_step, max_angle_step
            if hasattr(thruster, "alpha_min") and hasattr(thruster, "alpha_max"):
                lower = max(lower, float(thruster.alpha_min) - angle_now[i])
                upper = min(upper, float(thruster.alpha_max) - angle_now[i])
            if lower > upper:
                raise ValueError(
                    f"current angle for {thruster.name} is outside its allowed limits"
                )
            angle_bounds.append((lower, upper))

        # Scale thrust decision variables to O(1) and balance translational
        # force and yaw moment residuals by representative wrench magnitudes.
        wrench_scale = np.array([
            max(float(np.sum(maximum)), 1.0),
            max(float(np.sum(maximum)), 1.0),
            max(float(np.sum(maximum * np.maximum(
                1.0, [np.hypot(thruster.x, thruster.y) for thruster in self.thrusters]
            ))), 1.0),
        ])
        thrust_scale = maximum
        angle_count = len(moving)
        x0 = np.zeros(n + angle_count)
        x0[:n] = np.clip(thrust_now, low_thrust, high_thrust) / thrust_scale
        bounds = [
            (low_thrust[i] / thrust_scale[i], high_thrust[i] / thrust_scale[i])
            for i in range(n)
        ] + angle_bounds

        # A small regularizer resolves the redundant azimuth choices in favor
        # of the least-turn solution without materially relaxing wrench tracking.
        angle_regularization = 1e-4

        def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            thrust = x[:n] * thrust_scale
            angles = fixed_angles.copy()
            for j, i in enumerate(moving):
                angles[i] = angle_now[i] + x[n + j]
            achieved = np.zeros(3)
            jacobian = np.zeros((3, n + angle_count))
            for i, thruster in enumerate(self.thrusters):
                column = self._column(thruster, angles[i])
                achieved += thrust[i] * column
                jacobian[:, i] = thrust_scale[i] * column
                if i in moving:
                    j = moving.index(i)
                    jacobian[:, n + j] = (
                        thrust[i] * self._column_derivative(thruster, angles[i])
                    )
            return (achieved - target) / wrench_scale, jacobian / wrench_scale[:, None]

        def objective(x: np.ndarray) -> float:
            residual, _ = evaluate(x)
            return 0.5 * float(residual @ residual) + 0.5 * angle_regularization * float(
                x[n:] @ x[n:]
            )

        def gradient(x: np.ndarray) -> np.ndarray:
            residual, jacobian = evaluate(x)
            result = jacobian.T @ residual
            result[n:] += angle_regularization * x[n:]
            return result

        solution = minimize(
            objective,
            x0,
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 40, "ftol": 1e-10},
        )
        if not solution.success and not np.all(np.isfinite(solution.x)):
            raise RuntimeError(f"thrust allocation failed: {solution.message}")

        thrust_cmd = np.clip(solution.x[:n] * thrust_scale, low_thrust, high_thrust)
        angle_cmd = fixed_angles.copy()
        for j, i in enumerate(moving):
            angle_cmd[i] = wrap_angle_pi(angle_now[i] + solution.x[n + j])
        return thrust_cmd, angle_cmd
