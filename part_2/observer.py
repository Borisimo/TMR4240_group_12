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
    """Estimate wave motion, vessel motion, velocity, and environmental bias.

    The 15 states are ``[xi(6), eta(3), nu(3), b(3)]``. Pose measurements
    contain both low-frequency pose and wave motion: ``y = eta + C_w xi + v``.
    Each step predicts the state and covariance, then corrects them using the
    measured pose.

    The vessel model is ``eta_dot = R(psi) nu``,
    ``M nu_dot = -D nu + R(psi).T b + tau``, and
    ``b_dot = -T_bias^-1 b``. The prediction Jacobian includes the heading
    derivatives of ``R nu`` and ``R.T b``, plus the velocity, bias, and wave
    model terms. Measurement matrix ``H`` is ``[C_w, I, 0, 0]``.

    Process noise ``Q`` has separate blocks for wave excitation, unknown
    vessel acceleration, and changing environmental bias; measurement noise
    ``R`` is set from the pose sensor variances. On reset, pose comes from
    ``eta0`` and wave, velocity, and bias start at zero. Initial covariance
    is diagonal, with values set by the ``p_*`` parameters.
    """

    name = "kalman"
    WAVE = slice(0, 6)
    WAVE_RATE = slice(3, 6)
    POSE = slice(6, 9)
    VELOCITY = slice(9, 12)
    BIAS = slice(12, 15)
    HEADING_INDEX = 8

    def __init__(
        self,
        *,
        w0: float = (2 * np.pi) / 8,
        zeta: float = 0.1,
        T_bias: float = 1000.0,
        q_wave: float = 5e-3,
        q_wave_yaw: float = 1e-5,
        q_accel: float = 5e-5,
        q_yaw_accel: float = 1e-8,
        q_bias: float = 1e2,
        q_bias_yaw: float = 1e1,
        r_position: float = 0.20**2,
        r_heading: float = np.deg2rad(0.20) ** 2,
        p_wave: float = 0.5**2,
        p_wave_rate: float = 0.5**2,
        p_pose: float = 0.20**2,
        p_heading: float = np.deg2rad(0.20) ** 2,
        p_velocity: float = 0.5**2,
        p_yaw_rate: float = np.deg2rad(0.5) ** 2,
        p_bias: float = 1e8,
        p_bias_yaw: float = 1e12,
    ):
        parameters = {
            "w0": w0,
            "zeta": zeta,
            "T_bias": T_bias,
            "q_wave": q_wave,
            "q_wave_yaw": q_wave_yaw,
            "q_accel": q_accel,
            "q_yaw_accel": q_yaw_accel,
            "q_bias": q_bias,
            "q_bias_yaw": q_bias_yaw,
            "r_position": r_position,
            "r_heading": r_heading,
            "p_wave": p_wave,
            "p_wave_rate": p_wave_rate,
            "p_pose": p_pose,
            "p_heading": p_heading,
            "p_velocity": p_velocity,
            "p_yaw_rate": p_yaw_rate,
            "p_bias": p_bias,
            "p_bias_yaw": p_bias_yaw,
        }
        for name, value in parameters.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and greater than zero")

        vessel = Gunnerus3DOF()._v
        self.M = vessel._Mrb + vessel._Ma
        self.M_inv = np.linalg.inv(self.M)
        self.D = vessel._Dl.copy()
        omega = np.diag([w0, w0, w0])
        damping = np.diag([zeta, zeta, zeta])
        self.A_w = np.block([
            [np.zeros((3, 3)), np.eye(3)],
            [-omega**2, -2.0 * damping @ omega],
        ])
        self.C_w = np.hstack([np.zeros((3, 3)), np.eye(3)])
        self.T_inv = np.eye(3) / T_bias
        self.q_wave = float(q_wave)
        self.q_wave_yaw = float(q_wave_yaw)
        self.q_accel = float(q_accel)
        self.q_yaw_accel = float(q_yaw_accel)
        self.q_bias = float(q_bias)
        self.q_bias_yaw = float(q_bias_yaw)
        self.r_position = float(r_position)
        self.r_heading = float(r_heading)
        self.p_wave = float(p_wave)
        self.p_wave_rate = float(p_wave_rate)
        self.p_pose = float(p_pose)
        self.p_heading = float(p_heading)
        self.p_velocity = float(p_velocity)
        self.p_yaw_rate = float(p_yaw_rate)
        self.p_bias = float(p_bias)
        self.p_bias_yaw = float(p_bias_yaw)
        self._measurement = np.zeros((3, 15))
        self._measurement[:, self.WAVE] = self.C_w
        self._measurement[:, self.POSE] = np.eye(3)
        self._measurement_cov = np.diag(
            [self.r_position, self.r_position, self.r_heading]
        )
        self._process_noise_density = np.zeros((15, 15))
        self._process_noise_density[self.WAVE_RATE, self.WAVE_RATE] = np.diag(
            [self.q_wave, self.q_wave, self.q_wave_yaw]
        )
        self._process_noise_density[self.VELOCITY, self.VELOCITY] = np.diag(
            [self.q_accel, self.q_accel, self.q_yaw_accel]
        )
        self._process_noise_density[self.BIAS, self.BIAS] = np.diag(
            [self.q_bias, self.q_bias, self.q_bias_yaw]
        )
        self.reset()

    def reset(self, eta0: np.ndarray | None = None):
        eta = np.zeros(6) if eta0 is None else np.asarray(eta0, dtype=float).reshape(6)
        if not np.all(np.isfinite(eta)):
            raise ValueError("eta0 must contain only finite values")
        self._state = np.zeros(15)
        self._state[self.POSE] = eta[[0, 1, 5]]
        self._state[self.HEADING_INDEX] = self._wrap(
            self._state[self.HEADING_INDEX]
        )
        self._covariance = np.diag([
            self.p_wave, self.p_wave, self.p_heading,
            self.p_wave_rate, self.p_wave_rate, self.p_yaw_rate,
            self.p_pose, self.p_pose, self.p_heading,
            self.p_velocity, self.p_velocity, self.p_yaw_rate,
            self.p_bias, self.p_bias, self.p_bias_yaw,
        ])

    @staticmethod
    def _wrap(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    @staticmethod
    def _rotation(psi: float) -> np.ndarray:
        c, s = np.cos(psi), np.sin(psi)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    @staticmethod
    def _rotation_derivative(psi: float) -> np.ndarray:
        c, s = np.cos(psi), np.sin(psi)
        return np.array([[-s, -c, 0.0], [c, -s, 0.0], [0.0, 0.0, 0.0]])

    def _dynamics_and_jacobian(self, state: np.ndarray, tau: np.ndarray):
        xi = state[self.WAVE]
        eta = state[self.POSE]
        nu = state[self.VELOCITY]
        bias = state[self.BIAS]
        rotation = self._rotation(eta[2])
        rotation_t = rotation.T
        rotation_derivative_t = self._rotation_derivative(eta[2]).T

        derivative = np.zeros(15)
        derivative[self.WAVE] = self.A_w @ xi
        derivative[self.POSE] = rotation @ nu
        derivative[self.VELOCITY] = self.M_inv @ (
            -self.D @ nu + rotation_t @ bias + tau
        )
        derivative[self.BIAS] = -self.T_inv @ bias

        jacobian = np.zeros((15, 15))
        jacobian[self.WAVE, self.WAVE] = self.A_w
        jacobian[self.POSE, self.HEADING_INDEX] = (
            self._rotation_derivative(eta[2]) @ nu
        )
        jacobian[self.POSE, self.VELOCITY] = rotation
        jacobian[self.VELOCITY, self.HEADING_INDEX] = (
            self.M_inv @ rotation_derivative_t @ bias
        )
        jacobian[self.VELOCITY, self.VELOCITY] = -self.M_inv @ self.D
        jacobian[self.VELOCITY, self.BIAS] = self.M_inv @ rotation_t
        jacobian[self.BIAS, self.BIAS] = -self.T_inv
        return derivative, jacobian

    def _predict(self, dt: float, tau: np.ndarray):
        derivative, jacobian = self._dynamics_and_jacobian(self._state, tau)
        predicted_state = self._state + dt * derivative
        predicted_state[self.HEADING_INDEX] = self._wrap(
            predicted_state[self.HEADING_INDEX]
        )

        transition = np.eye(15) + dt * jacobian
        predicted_covariance = (
            transition @ self._covariance @ transition.T
            + dt * self._process_noise_density
        )
        return predicted_state, predicted_covariance

    def _correct(self, predicted_state, predicted_covariance, measured_pose):
        innovation = measured_pose - self._measurement @ predicted_state
        innovation[2] = self._wrap(innovation[2])
        innovation_covariance = (
            self._measurement @ predicted_covariance @ self._measurement.T
            + self._measurement_cov
        )
        gain = np.linalg.solve(
            innovation_covariance,
            (predicted_covariance @ self._measurement.T).T,
        ).T
        corrected_state = predicted_state + gain @ innovation
        corrected_state[self.HEADING_INDEX] = self._wrap(
            corrected_state[self.HEADING_INDEX]
        )

        # Joseph form keeps the covariance symmetric and positive semidefinite.
        correction = np.eye(15) - gain @ self._measurement
        corrected_covariance = (
            correction @ predicted_covariance @ correction.T
            + gain @ self._measurement_cov @ gain.T
        )
        return corrected_state, corrected_covariance

    def step(self, t, dt, eta_measured, tau_est):
        if not np.isfinite(t) or not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("t must be finite and dt must be finite and greater than zero")
        eta_measured = np.asarray(eta_measured, dtype=float).reshape(6)
        tau = np.asarray(tau_est, dtype=float).reshape(6)
        if not np.all(np.isfinite(eta_measured)) or not np.all(np.isfinite(tau)):
            raise ValueError("measurements and wrench must contain only finite values")

        tau_body = tau[[0, 1, 5]]
        measured_pose = eta_measured[[0, 1, 5]]
        predicted_state, predicted_covariance = self._predict(dt, tau_body)
        self._state, self._covariance = self._correct(
            predicted_state, predicted_covariance, measured_pose
        )

        eta = np.zeros(6)
        eta[[0, 1, 5]] = self._state[self.POSE]
        nu = np.zeros(6)
        nu[[0, 1, 5]] = self._state[self.VELOCITY]
        bias = np.zeros(6)
        bias[[0, 1, 5]] = self._state[self.BIAS]
        return ObserverEstimate(eta=eta, nu=nu, bias=bias)


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
