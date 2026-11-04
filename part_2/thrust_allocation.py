"""Rate-aware wrench allocation for the Part 2 Gunnerus thrusters."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import qr
from scipy.optimize import minimize

from models.thruster_dynamics import ThrusterConfig
from simulation.utils import wrap_angle_pi


class ThrustAllocator:
    """Allocate BODY-frame surge, sway, and yaw requests.

    Azimuth angles are optimization variables. Their per-step change and each
    thruster's signed force are bounded by the actuator's rotation, thrust-rate,
    and maximum-thrust limits. Tunnel thrusters retain their fixed geometry.
    Wrench tracking is optimized first; a second stage minimizes weighted
    absolute thrust while preserving the best wrench found in that first stage.
    """

    def __init__(
        self,
        thrusters: list[ThrusterConfig],
        thrust_weights: np.ndarray | None = None,
        tracking_tolerance: float = 0.0,
    ):
        self.thrusters = list(thrusters)
        if not np.isfinite(tracking_tolerance) or tracking_tolerance < 0.0:
            raise ValueError("tracking_tolerance must be finite and non-negative")
        self.tracking_tolerance = float(tracking_tolerance)
        self.thrust_weights = (
            np.ones(len(self.thrusters), dtype=float)
            if thrust_weights is None
            else np.asarray(thrust_weights, dtype=float).reshape(len(self.thrusters))
        )
        if (
            not np.all(np.isfinite(self.thrust_weights))
            or np.any(self.thrust_weights <= 0.0)
        ):
            raise ValueError("thrust_weights must contain positive finite values")

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
        u_now: np.ndarray | None = None,
        alpha_now: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return rate-feasible thrust and azimuth commands for this time step.

        ``tau_d`` is a six-component BODY-frame wrench; only surge, sway, and
        yaw are allocated. The returned arrays follow the input thruster order.
        """
        if not np.isfinite(t) or not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("t must be finite and dt must be finite and greater than zero")

        n = len(self.thrusters)
        if n == 0:
            return np.zeros(0), np.zeros(0)

        target_wrench = np.asarray(tau_d, dtype=float).reshape(6)[[0, 1, 5]]
        if not np.all(np.isfinite(target_wrench)):
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

        max_thrust = np.array(
            [thruster.u_max for thruster in self.thrusters], dtype=float
        )
        thrust_rate = np.array(
            [thruster.u_rate for thruster in self.thrusters], dtype=float
        )
        if np.any(max_thrust <= 0.0) or np.any(thrust_rate <= 0.0):
            raise ValueError("thrusters must have positive maximum thrust and thrust rate")

        low_thrust = np.maximum(-max_thrust, thrust_now - thrust_rate * dt)
        high_thrust = np.minimum(max_thrust, thrust_now + thrust_rate * dt)

        moving_thrusters: list[int] = []
        angle_bounds: list[tuple[float, float]] = []
        fixed_angles = np.array(
            [thruster.alpha0 for thruster in self.thrusters], dtype=float
        )
        for i, thruster in enumerate(self.thrusters):
            if thruster.kind.lower() == "tunnel" or thruster.rot_speed == 0.0:
                continue
            moving_thrusters.append(i)
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
        total_thrust = float(np.sum(max_thrust))
        lever_arms = np.maximum(
            1.0,
            [np.hypot(thruster.x, thruster.y) for thruster in self.thrusters],
        )
        wrench_scale = np.array(
            [
                max(total_thrust, 1.0),
                max(total_thrust, 1.0),
                max(float(np.sum(max_thrust * lever_arms)), 1.0),
            ]
        )
        thrust_scale = max_thrust
        angle_count = len(moving_thrusters)
        angle_index = {
            thruster_index: j for j, thruster_index in enumerate(moving_thrusters)
        }
        x0 = np.zeros(n + angle_count)
        x0[:n] = np.clip(thrust_now, low_thrust, high_thrust) / thrust_scale
        bounds = [
            (low_thrust[i] / thrust_scale[i], high_thrust[i] / thrust_scale[i])
            for i in range(n)
        ] + angle_bounds

        def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            thrust = x[:n] * thrust_scale
            angles = fixed_angles.copy()
            for j, i in enumerate(moving_thrusters):
                angles[i] = angle_now[i] + x[n + j]
            achieved = np.zeros(3)
            jacobian = np.zeros((3, n + angle_count))
            for i, thruster in enumerate(self.thrusters):
                column = self._column(thruster, angles[i])
                achieved += thrust[i] * column
                jacobian[:, i] = thrust_scale[i] * column
                if i in angle_index:
                    j = angle_index[i]
                    jacobian[:, n + j] = (
                        thrust[i] * self._column_derivative(thruster, angles[i])
                    )
            return (
                (achieved - target_wrench) / wrench_scale,
                jacobian / wrench_scale[:, None],
            )

        # Stage one finds the best achievable wrench before considering effort.
        def tracking_objective(x: np.ndarray) -> float:
            residual, _ = evaluate(x)
            return 0.5 * float(residual @ residual)

        def tracking_gradient(x: np.ndarray) -> np.ndarray:
            residual, jacobian = evaluate(x)
            return jacobian.T @ residual

        tracking_solution = minimize(
            tracking_objective,
            x0,
            jac=tracking_gradient,
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 60, "ftol": 1e-12},
        )
        if not tracking_solution.success or not np.all(np.isfinite(tracking_solution.x)):
            raise RuntimeError(
                f"thrust allocation tracking failed: {tracking_solution.message}"
            )

        # Stage two reduces weighted thrust while keeping the wrench error
        # within the configured tolerance of the stage-one result.
        tracking_residual, _ = evaluate(tracking_solution.x)

        force_scale = float(np.max(max_thrust))
        normalized_force_scale = max_thrust / force_scale
        l1_smoothing = 1e-4
        angle_regularization = 1e-8
        l2_tiebreaker = 1e-6

        def effort_objective(x: np.ndarray) -> float:
            normalized_thrust = normalized_force_scale * x[:n]
            smoothed_magnitude = np.sqrt(normalized_thrust**2 + l1_smoothing**2)
            return float(self.thrust_weights @ smoothed_magnitude) + (
                l2_tiebreaker * float(x[:n] @ x[:n])
            ) + 0.5 * angle_regularization * float(x[n:] @ x[n:])

        def effort_gradient(x: np.ndarray) -> np.ndarray:
            result = np.zeros_like(x)
            normalized_thrust = normalized_force_scale * x[:n]
            smoothed_magnitude = np.sqrt(normalized_thrust**2 + l1_smoothing**2)
            result[:n] = (
                self.thrust_weights
                * normalized_force_scale
                * normalized_thrust
                / smoothed_magnitude
                + 2.0 * l2_tiebreaker * x[:n]
            )
            result[n:] = angle_regularization * x[n:]
            return result

        def tracking_constraint(x: np.ndarray) -> float:
            residual, _ = evaluate(x)
            residual_limit = (
                float(np.linalg.norm(tracking_residual))
                + self.tracking_tolerance
                * float(np.linalg.norm(target_wrench / wrench_scale))
                + 1e-8
            )
            return residual_limit - float(np.linalg.norm(residual))

        def tracking_constraint_jacobian(x: np.ndarray) -> np.ndarray:
            residual, jacobian = evaluate(x)
            residual_norm = max(float(np.linalg.norm(residual)), 1e-12)
            return -jacobian.T @ residual / residual_norm

        _, tracking_jacobian = evaluate(tracking_solution.x)
        _, triangular, pivots = qr(tracking_jacobian.T, pivoting=True, mode="economic")
        diagonal = np.abs(np.diag(triangular))
        rank_tolerance = (
            np.finfo(float).eps * max(tracking_jacobian.shape) * diagonal.max()
            if diagonal.size
            else 0.0
        )
        constraint_rows = pivots[:int(np.count_nonzero(diagonal > rank_tolerance))]
        if constraint_rows.size:
            constraints = {
                "type": "eq",
                "fun": lambda x: evaluate(x)[0][constraint_rows]
                - tracking_residual[constraint_rows],
                "jac": lambda x: evaluate(x)[1][constraint_rows],
            }
        else:
            constraints = {
                "type": "ineq",
                "fun": tracking_constraint,
                "jac": tracking_constraint_jacobian,
            }

        effort_solution = minimize(
            effort_objective,
            tracking_solution.x,
            jac=effort_gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 60, "ftol": 1e-10},
        )
        if (
            effort_solution.success
            and np.all(np.isfinite(effort_solution.x))
            and tracking_constraint(effort_solution.x) >= -1e-8
        ):
            solution = effort_solution
        else:
            warnings.warn(
                "weighted thrust minimization failed; using the tracking-first "
                f"allocation instead: {effort_solution.message}",
                RuntimeWarning,
                stacklevel=2,
            )
            solution = tracking_solution

        thrust_cmd = np.clip(solution.x[:n] * thrust_scale, low_thrust, high_thrust)
        angle_cmd = fixed_angles.copy()
        for j, i in enumerate(moving_thrusters):
            angle_cmd[i] = wrap_angle_pi(angle_now[i] + solution.x[n + j])
        return thrust_cmd, angle_cmd
