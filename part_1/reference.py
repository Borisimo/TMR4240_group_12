"""
Reference template

Students should filter or shape commanded setpoints before they are sent to
the controller. The simulator calls, once per step:

    ref.step(t, dt, eta_cmd) -> (eta_ref, nu_ref, acc_ref)

All generalized vectors are 6-DOF, ordered [surge, sway, heave, roll, pitch,
yaw]. The 3-DOF model uses indices [0, 1, 5]; leave the rest zero.

Inputs:
    t       : current simulation time [s]
    dt      : time step [s]
    eta_cmd : (6,) commanded setpoint
              (use N_cmd = eta_cmd[0], E_cmd = eta_cmd[1], psi_cmd = eta_cmd[5])

Outputs (all NED-frame, (6,) each):
    eta_ref : filtered reference
              (fill in N_ref = [0], E_ref = [1], psi_ref = [5])
    nu_ref  : reference velocities
              (fill in Ndot_ref = [0], Edot_ref = [1], psidot_ref = [5])
    acc_ref : reference accelerations
              (fill in Nddot_ref = [0], Eddot_ref = [1], psiddot_ref = [5])

The simulator forwards all three to the controller, so a smooth reference
model here directly enables velocity/acceleration feedforward there.
"""
from typing import Tuple
import numpy as np

# Per-axis tuning parameters live with the rest of the Part 1 configuration.
from part_1.config import RefAxisConfig


class ReferenceModel:
    """
    Template for student reference model.

    The default implementation is pass-through, so eta_ref = eta_cmd and the
    reference velocities/accelerations are zero.
    """

    def __init__(
        self,
        dt: float,
        cfg_xy: RefAxisConfig | None = None,
        cfg_psi: RefAxisConfig | None = None,
    ):
        self.dt = float(dt)
        self.cfg_xy = cfg_xy if cfg_xy is not None else RefAxisConfig()
        self.cfg_psi = cfg_psi if cfg_psi is not None else RefAxisConfig()
        self.eta_ref = np.zeros(6)
        self.nu_ref = np.zeros(6)
        self.acc_ref = np.zeros(6)

    def reset(self, eta0: np.ndarray) -> None:
        """Initialize the reference at the vessel's current (6,) state."""
        self.eta_ref = np.asarray(eta0, dtype=float).reshape(6).copy()
        self.nu_ref = np.zeros(6)
        self.acc_ref = np.zeros(6)

    def step(
        self, t: float, dt: float, eta_cmd: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # TODO: Replace this pass-through placeholder with your reference model.

        #implementation of the second order low pass filter  
        # eta_ddot + 2*zeta*w_r*eta_dot + w_r^2*eta = w_r^2*eta_cmd

        #filter coeff 
        wr_NE = self.cfg_xy.wn
        zeta_NE = self.cfg_xy.zeta
        wr_psi = self.cfg_psi.wn
        zeta_psi = self.cfg_psi.zeta

        #ref NE
        for i in (0,1):
            eta = self.eta_ref[i]
            eta_dot = self.nu_ref[i]
            cmd = eta_cmd[i]

            #compute eta_ddot from filter dyn
            eta_ddot = -2*zeta_NE* wr_NE * eta_dot + (cmd - eta)*wr_NE*wr_NE

            #euler intration
            self.acc_ref[i] = eta_ddot
            self.nu_ref[i] = eta_dot + dt*self.acc_ref[i]
            self.eta_ref[i] = eta + dt*self.nu_ref[i]

        #ref spi 
        psi = self.eta_ref[5]
        psi_dot = self.nu_ref[5]
        psi_sp = eta_cmd[5]

        #compute error with arctan2 
        err = np.arctan2(np.sin(psi_sp - psi), np.cos(psi_sp - psi))
        #filter dynamics
        psi_ddot = - 2 * zeta_psi * wr_psi * psi_dot + wr_psi**2 * err

        #euler integration
        self.nu_ref[5] = psi_dot + dt * psi_ddot
        self.eta_ref[5] = psi + dt * self.nu_ref[5]
        self.eta_ref[5] = np.arctan2(np.sin(self.eta_ref[5]), np.cos(self.eta_ref[5]))  # keep wrapped
        self.acc_ref[5] = psi_ddot

        # self.eta_ref = np.asarray(eta_cmd, dtype=float).reshape(6).copy()
        # self.nu_ref = np.zeros(6)
        # self.acc_ref = np.zeros(6)
        return self.eta_ref, self.nu_ref, self.acc_ref
