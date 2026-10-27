"""Student observer templates and their common simulator interface.

The simulator calls, once per step (pose measurement only, no velocity):

    observer.step(t, dt, eta_measured, tau_est) -> ObserverEstimate(eta, nu, bias)

    eta_measured : (6,) measured NED pose [N, E, z, phi, theta, psi]
    tau_est      : (6,) desired controller wrench (before thruster dynamics)
                   of the PREVIOUS step (zero at the first step)
    eta, nu      : (6,) low-frequency NED pose and BODY velocity estimates
    bias         : (6,) slowly varying bias estimate (NED force) — may be zeros

A plain ``(eta, nu, bias)`` tuple is accepted as well.  Before every run the
engine calls ``reset(eta0)`` with the TRUE initial pose, so an observer can
start from the vessel's actual position.  Select an implementation with
``Part2SimConfig(observer_type=...)`` (Simulations 4-7 use the
``SELECTED_OBSERVER`` constant of ``run_case_part_2.py``); extra constructor
arguments go through ``Part2SimConfig(observer_kwargs=...)``.  Both classes
intentionally start as pass-through placeholders; students must implement and
tune them, keeping the tuned parameters as constructor defaults because the
checks call ``select_observer(kind)`` without arguments.
"""
from __future__ import annotations
from models.gunnerus_3dof import Gunnerus3DOF
from dataclasses import dataclass
import numpy as np


@dataclass
class ObserverEstimate:
    eta: np.ndarray
    nu: np.ndarray
    bias: np.ndarray


class Observer:
    """Interface shared by all Part 2 observers."""

    name = "observer"

    def reset(self, eta0: np.ndarray | None = None):
        pass

    def step(
        self, t: float, dt: float, eta_measured: np.ndarray, tau_est: np.ndarray
    ) -> ObserverEstimate:
        raise NotImplementedError

    @staticmethod
    def _placeholder(eta_measured: np.ndarray) -> ObserverEstimate:
        return ObserverEstimate(
            eta=np.asarray(eta_measured, dtype=float).reshape(6).copy(),
            nu=np.zeros(6),
            bias=np.zeros(6),
        )


class NonlinearPassiveObserver(Observer):
    """Template for a nonlinear passive wave-filtering observer.

    Suggested states are wave-frequency motion, low-frequency pose, BODY
    velocity, and slowly-varying bias.  Implement the correction and model
    propagation in :meth:`step`.
    """

    name = "nonlinear_passive"

    def __init__(
            self,
            w0: float = (2 * np.pi) / 8,
            zeta: float = 0.1,
            zeta_n: float = 1.0,
            wc: float = 1.2 * (2 * np.pi) / 8,
            T_bias: float = 1000.0,
            k3: float = 0.1,
            k4: float = 0.02,
    ):
        
        omega = np.diag([w0, w0, w0])
        ones = np.ones(3)
        vessel = Gunnerus3DOF()._v
        self.M = vessel._Mrb + vessel._Ma
        self.M_inv = np.linalg.inv(self.M)
        self.D = vessel._Dl.copy()
        Z = np.diag([zeta, zeta, zeta])
        self.A_w = np.block([[np.zeros((3, 3)), np.eye(3)],
                            [-omega**2, -2 * Z @ omega]])
        self.C_w = np.hstack([np.zeros((3, 3)), np.eye(3)])
        self.K1 = np.vstack([np.diag(-2.0 * (zeta_n - zeta) * wc / w0 * ones),
                            np.diag(2.0 * w0 * (zeta_n - zeta) * ones)])
        self.K2 = np.diag(wc * ones)
        self.K4 = np.diag(np.diag(self.M) * k4)
        self.K3 = k3 * self.K4
        self.T_inv = np.eye(3) / T_bias
        self.reset()
        

    def reset(self, eta0 = None):
        eta0 = np.zeros(6) if eta0 is None else np.asarray(eta0, dtype=float).reshape(6)
        self.eta_hat = eta0[[0, 1, 5]].copy()
        self.nu_hat = np.zeros(3)
        self.b_hat = np.zeros(3)
        self.xi_hat = np.zeros(6)

    @staticmethod
    def _wrap(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def step(self, t, dt, eta_measured, tau_est):
        y = np.asarray(eta_measured, dtype=float).reshape(6)[[0, 1, 5]]
        tau = np.asarray(tau_est, dtype=float).reshape(6)[[0, 1, 5]]

        y_hat = self.eta_hat + self.C_w @ self.xi_hat
        y_tilde = y - y_hat
        y_tilde[2] = self._wrap(y_tilde[2])

        psi = self.eta_hat[2]
        c, s = np.cos(psi), np.sin(psi)
        R = np.array([[c, -s, 0],
                      [s, c, 0],
                      [0, 0, 1]])

        xi_dot = self.A_w @ self.xi_hat + self.K1 @ y_tilde
        eta_dot = R @ self.nu_hat + self.K2 @y_tilde
        b_dot = - self.T_inv @self.b_hat + self.K3 @ y_tilde
        nu_dot = self.M_inv @ (-self.D @ self.nu_hat
                               + R.T @ self.b_hat
                               + tau
                               + R.T @ self.K4 @ y_tilde
                            )

        self.xi_hat = self.xi_hat + dt * xi_dot
        self.eta_hat = self.eta_hat + dt * eta_dot
        self.b_hat = self.b_hat + dt * b_dot
        self.nu_hat = self.nu_hat + dt * nu_dot
        self.eta_hat[2] = self._wrap(self.eta_hat[2])

        eta = np.zeros(6)
        nu = np.zeros(6)
        bias = np.zeros(6)
        eta[[0, 1, 5]] = self.eta_hat
        nu[[0, 1, 5]] = self.nu_hat
        bias[[0, 1, 5]] = self.b_hat
        return ObserverEstimate(eta=eta, nu=nu, bias=bias)


class KalmanFilterObserver(Observer):
    """Extended Kalman filter for low-frequency horizontal vessel motion.

    The state is ``[N, E, psi, N_dot, E_dot, psi_dot]``. Pose is measured;
    velocity is estimated. The constant-velocity model treats unmodeled loads
    as process noise, while the known previous desired wrench contributes a
    body-frame acceleration input.
    """

    name = "kalman"

    def __init__(
        self,
        *,
        q_accel: float = 5e-5,
        q_yaw_accel: float = 1e-8,
        r_position: float = 0.20**2,
        r_heading: float = np.deg2rad(0.20) ** 2,
        p_velocity: float = 0.5**2,
        surge_mass: float = 6.007e5,
        sway_mass: float = 7.067e5,
        yaw_inertia: float = 5.456e7,
    ):
        parameters = {
            "q_accel": q_accel,
            "q_yaw_accel": q_yaw_accel,
            "r_position": r_position,
            "r_heading": r_heading,
            "p_velocity": p_velocity,
            "surge_mass": surge_mass,
            "sway_mass": sway_mass,
            "yaw_inertia": yaw_inertia,
        }
        for name, value in parameters.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and greater than zero")

        self.q_accel = float(q_accel)
        self.q_yaw_accel = float(q_yaw_accel)
        self.r_position = float(r_position)
        self.r_heading = float(r_heading)
        self.p_velocity = float(p_velocity)
        self.surge_mass = float(surge_mass)
        self.sway_mass = float(sway_mass)
        self.yaw_inertia = float(yaw_inertia)
        self._measurement = np.zeros((3, 6))
        self._measurement[0, 0] = 1.0
        self._measurement[1, 1] = 1.0
        self._measurement[2, 2] = 1.0
        self._measurement_cov = np.diag(
            [self.r_position, self.r_position, self.r_heading]
        )
        self.reset()

    def reset(self, eta0: np.ndarray | None = None):
        eta = np.zeros(6) if eta0 is None else np.asarray(eta0, dtype=float).reshape(6)
        if not np.all(np.isfinite(eta)):
            raise ValueError("eta0 must contain only finite values")
        self._state = np.zeros(6)
        self._state[:3] = eta[[0, 1, 5]]
        self._state[2] = self._wrap(self._state[2])
        self._covariance = np.diag(
            [self.r_position, self.r_position, self.r_heading,
             self.p_velocity, self.p_velocity, self.p_velocity]
        )

    @staticmethod
    def _wrap(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _process_noise(self, dt: float) -> np.ndarray:
        q = np.zeros((6, 6))
        for position, velocity, spectral_density in (
            (0, 3, self.q_accel),
            (1, 4, self.q_accel),
            (2, 5, self.q_yaw_accel),
        ):
            q[position, position] = spectral_density * dt**3 / 3.0
            q[position, velocity] = q[velocity, position] = (
                spectral_density * dt**2 / 2.0
            )
            q[velocity, velocity] = spectral_density * dt
        return q

    def step(self, t, dt, eta_measured, tau_est):
        if not np.isfinite(t) or not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("t must be finite and dt must be finite and greater than zero")
        eta_measured = np.asarray(eta_measured, dtype=float).reshape(6)
        tau = np.asarray(tau_est, dtype=float).reshape(6)
        if not np.all(np.isfinite(eta_measured)) or not np.all(np.isfinite(tau)):
            raise ValueError("measurements and wrench must contain only finite values")

        # Predict with a constant-velocity model and the known prior body wrench.
        psi = self._state[2]
        c, s = np.cos(psi), np.sin(psi)
        fx, fy, mz = tau[0], tau[1], tau[5]
        acceleration_n = (c * fx - s * fy) / self.surge_mass
        acceleration_e = (s * fx + c * fy) / self.sway_mass
        acceleration_psi = mz / self.yaw_inertia
        predicted = self._state.copy()
        predicted[0] += dt * self._state[3]
        predicted[1] += dt * self._state[4]
        predicted[2] = self._wrap(self._state[2] + dt * self._state[5])
        predicted[3] += dt * acceleration_n
        predicted[4] += dt * acceleration_e
        predicted[5] += dt * acceleration_psi

        transition = np.eye(6)
        transition[0, 3] = dt
        transition[1, 4] = dt
        transition[2, 5] = dt
        transition[3, 2] = dt * (-s * fx - c * fy) / self.surge_mass
        transition[4, 2] = dt * (c * fx - s * fy) / self.sway_mass
        covariance = (
            transition @ self._covariance @ transition.T
            + self._process_noise(dt)
        )

        measured_pose = eta_measured[[0, 1, 5]]
        innovation = measured_pose - self._measurement @ predicted
        innovation[2] = self._wrap(innovation[2])
        innovation_cov = (
            self._measurement @ covariance @ self._measurement.T
            + self._measurement_cov
        )
        gain = np.linalg.solve(
            innovation_cov,
            (covariance @ self._measurement.T).T,
        ).T
        self._state = predicted + gain @ innovation
        self._state[2] = self._wrap(self._state[2])

        # Joseph form preserves covariance symmetry and positive semidefiniteness.
        correction = np.eye(6) - gain @ self._measurement
        self._covariance = (
            correction @ covariance @ correction.T
            + gain @ self._measurement_cov @ gain.T
        )

        eta = np.zeros(6)
        eta[[0, 1, 5]] = self._state[:3]
        c, s = np.cos(self._state[2]), np.sin(self._state[2])
        nu = np.zeros(6)
        nu[0] = c * self._state[3] + s * self._state[4]
        nu[1] = -s * self._state[3] + c * self._state[4]
        nu[5] = self._state[5]
        return ObserverEstimate(eta=eta, nu=nu, bias=np.zeros(6))


def select_observer(kind: str, **kwargs) -> Observer:
    """Construct one observer without changing simulator code."""
    choices = {
        "nonlinear_passive": NonlinearPassiveObserver,
        "kalman": KalmanFilterObserver,
    }
    try:
        return choices[kind](**kwargs)
    except KeyError as exc:
        raise ValueError(f"observer_type must be one of {tuple(choices)}") from exc
