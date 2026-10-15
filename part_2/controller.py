"""
Controller template

Students should implement a controller that maps the vessel state and the
full reference to a body-frame wrench. The simulator calls, once per step:

    controller.compute(t, dt, eta, nu, eta_ref, nu_ref, acc_ref) -> tau_d

All generalized vectors are 6-DOF, ordered [surge, sway, heave, roll, pitch,
yaw]. The 3-DOF model uses indices [0, 1, 5]; the remaining components are
zero on input and ignored on output.

Inputs (full loop state and full reference):
    t       : current simulation time [s]
    dt      : time step [s]
    eta     : (6,) vessel NED state [N, E, z, phi, theta, psi]
              (use N = eta[0], E = eta[1], psi = eta[5])
    nu      : (6,) vessel BODY velocities [u, v, w, p, q, r]
              (use u = nu[0], v = nu[1], r = nu[5])
    eta_ref : (6,) NED reference state
              (use N_d = eta_ref[0], E_d = eta_ref[1], psi_d = eta_ref[5])
    nu_ref  : (6,) NED-frame reference velocities
              (use Ndot_d = nu_ref[0], Edot_d = nu_ref[1], psidot_d = nu_ref[5])
    acc_ref : (6,) NED-frame reference accelerations, same layout as nu_ref
              (use for model-based / inertia feedforward)

Output:
    tau_d   : (6,) desired BODY wrench [Fx, Fy, Fz, Mx, My, Mz] (N, Nm)
              (fill in Fx = tau_d[0], Fy = tau_d[1], Mz = tau_d[5];
               leave the other components zero)

Optional hooks the simulator will use IF you define them (safe to omit):
    reset()                                  — called before each run
    apply_external_aw(tau_applied, psi, dt)  — anti-windup with the (6,)
                                               wrench actually applied after
                                               allocation and the actuator
                                               model (ideal in Part 1)
    last_pid_body  : {"P","I","D"} -> (6,) BODY components   (logged)
    int_ned (2,), int_psi (float)            — integrator states (logged)

Constructor contract — the automated checks (``python check.py``, ``pytest``,
``notebooks/part_1_demo.ipynb``) construct your controller as
``DPController()`` with NO arguments, so your final tuned gains must be the
constructor defaults. Tuning only inside ``run_case_part1.py`` will pass your
own runs but fail the checks.
"""
import numpy as np
from simulation.utils import Rz, angle_diff
from part_1.config import PIDGains, M

class DPController:
    """
    Template for student DP controller.

    Students may implement any type of controller (PID, LQR, backstepping,
    ...). Only compute() is required; everything else is optional.
    """
    

    def __init__(self, *args, **kwargs):
        gains = PIDGains()
        self.Kp = np.diag(gains.Kp)
        self.Ki = np.diag(gains.Ki)
        self.Kd = np.diag(gains.Kd)
        self.int_ned = np.zeros(2)
        self.int_psi = 0.0

        # Diagonal mass/inertia for the inertia feedforward term below.
        # Matches the same diagonal-only simplification Kp/Kd already use.
        self.M_diag = np.array([M[0, 0], M[1, 1], M[2, 2]])

    def reset(self) -> None:
        """Optional: reset internal states (integrators, filters) before a run."""
        #pass
        self.int_ned = np.zeros(2)
        self.int_psi = 0.0

    def compute(
        self,
        t: float,
        dt: float,
        eta: np.ndarray,
        nu: np.ndarray,
        eta_ref: np.ndarray,
        nu_ref: np.ndarray | None = None,
        acc_ref: np.ndarray | None = None,
    ):

        if nu_ref is None:
            nu_ref = np.zeros(6)

        psi = eta[5]
        R = Rz(psi) # NED = R @ body (3x3) -> [N,E,psi]

        #position/heading error, NED -> BODY
        e_ned = np.array([eta_ref[0]-eta[0],
                          eta_ref[1]-eta[1],
                          angle_diff(eta_ref[5], eta[5])
                          ])
        e_body = R.T @ e_ned

        #velocity error: nu_ref is NED, rotate to BODY before comparing with nu
        nu_ref_body = R.T @ np.array([nu_ref[0], nu_ref[1], nu_ref[5]])
        e_nu = nu_ref_body - np.array([nu[0], nu[1], nu[5]])

        # inertia feedforward: acc_ref is NED, rotate to BODY, then M*a.
        # Anticipates the force needed to match the reference's acceleration
        # instead of only reacting once a tracking error has built up --
        # this is what actually closes the r-tracking lag, not more damping.
        acc_ref_body = R.T @ np.array([acc_ref[0], acc_ref[1], acc_ref[5]])
        tau_ff = self.M_diag * acc_ref_body

        # integral (kept in NED, then rotated)
        self._last_int_inc = e_ned * dt  # undone by apply_external_aw if saturated
        self.int_ned += e_ned[:2] * dt
        self.int_psi += e_ned[2] * dt
        i_ned = np.array([self.int_ned[0],
                          self.int_ned[1],
                          self.int_psi
                          ])
        i_body = R.T @ i_ned

        tau3 = self.Kp @ e_body + self.Kd @ e_nu + self.Ki @ i_body + tau_ff
        self._last_tau_cmd3 = tau3.copy()  # full commanded wrench, FF included

        tau_d = np.zeros(6)
        tau_d[0], tau_d[1], tau_d[5] = tau3

        self.last_pid_body = {
            "P": np.zeros(6), "I": np.zeros(6), "D": np.zeros(6), "FF": np.zeros(6)
        }
        self.last_pid_body["P"][[0,1,5]] = self.Kp @ e_body
        self.last_pid_body["I"][[0,1,5]] = self.Ki @ i_body
        self.last_pid_body["D"][[0,1,5]] = self.Kd @ e_nu
        self.last_pid_body["FF"][[0,1,5]] = tau_ff

        return tau_d
    
    def apply_external_aw(self, tau_applied, psi, dt):
        tau_applied3 = np.array([
            tau_applied[0], 
            tau_applied[1], 
            tau_applied[5]
            ])
        
        tau_cmd3 = self._last_tau_cmd3#np.array([
            #self.last_pid_body["P"][0]+self.last_pid_body["I"][0]+self.last_pid_body["D"][0],
            #self.last_pid_body["P"][1]+self.last_pid_body["I"][1]+self.last_pid_body["D"][1],
            #self.last_pid_body["P"][5]+self.last_pid_body["I"][5]+self.last_pid_body["D"][5]
            #])
        
        # Conditional integration: if the thrusters could not deliver the
        # command, undo this step's integration instead of back-calculating
        # (back-calculation drove I to cancel P after large steps).
        unachieved = tau_cmd3 - tau_applied3
        if np.linalg.norm(unachieved) > 1e-3 * max(np.linalg.norm(tau_cmd3), 1.0):
            self.int_ned -= self._last_int_inc[:2]
            self.int_psi -= self._last_int_inc[2]
