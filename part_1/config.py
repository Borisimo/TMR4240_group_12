# part_1/config.py
# -----------------------------------------------------------------------------
# TMR4240 Marine Control Systems I
# Project – Design of Dynamic Positioning System
#
# Copyright (C) 2026: NTNU, Trondheim
# License: GPL-3.0-or-later
# -----------------------------------------------------------------------------
"""
Project Part 1 configuration — all tunable parameters in one place.

This file belongs to YOU, not to the simulator engine. Nothing in
``simulation/`` hardcodes a parameter you are asked to tune: the engine only
wires your models together, and every knob it needs is passed in from here
via ``run_case_part1.py``.

TODO (students): put your own configuration dataclasses in this file —
controller gains, thrust-allocation weights, wind/current parameters, ... —
so that every simulation can be reconfigured by editing this one file and
``run_case_part1.py``, without touching your model implementations. Example:

    @dataclass
    class PIDGains:
        Kp: np.ndarray = ...
        Ki: np.ndarray = ...
        Kd: np.ndarray = ...
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from models.thruster_dynamics import ThrusterConfig

#import pickle
import pickle
from importlib.resources import files

@dataclass
class SimConfig:
    """Part 1 simulation clock and options (ideal thrusters by default)."""
    dt: float = 0.05
    T: float = 1000.0
    method: str = "Euler"
    use_reference: bool = True
    thruster_dynamics: bool = False  # Part 1: ideal actuators (no rate limits, no saturation)
    bypass_actuators: bool = False   # apply tau_d directly (debug)


@dataclass
class RefAxisConfig:
    """Reference-model configuration for one axis (see part_1/reference.py).

    Tune these per simulation and justify the values in the report.

    Note: the automated checks build the default ``ReferenceModel(dt)``, which
    uses the field defaults below — so keep your final tuned values as the
    defaults here (overriding them only in ``run_case_part1.py`` will not
    reach the checks).
    """
    # TODO (students): wn below is a placeholder, NOT a tuned value. Choose
    # the natural frequency yourself and justify it in the report (see the
    # project text, Reference Model section).
    wn: float = 0.022#0.03                     # natural frequency [rad/s] (to be tuned)
    zeta: float = 1.0                   # damping ratio [-]
    rate_limit: Optional[float] = None  # max |x_dot| (m/s or rad/s); None = off


            

pkl = files("mcsimpy.vessel_data.gunnerus") / "parV_RVG3DOF.pkl"
with open(str(pkl), "rb") as f:
    data = pickle.load(f)

M_RB, M_A = data["Mrb"], data["Ma"]
D_l, D_u, D_v, D_r = data["Dl"], data["Du"], data["Dv"], data["Dr"]
M = M_RB + M_A


@dataclass
class PIDGains:
    wn_N   = 0.123    #0.1150
    wn_E   = 0.123    #0.1170
    wn_psi = 0.132    #0.0860

    zeta_N   = 1.0  #1.4
    zeta_E   = 1.0  #1.4
    zeta_psi = 1.0  #1.4

    Kp: np.ndarray = field(init=False)
    Ki: np.ndarray = field(init=False)
    Kd: np.ndarray = field(init=False)

    def __post_init__(self):
        self.Kp = np.array([
            self.wn_N**2 * M[0,0], 
            self.wn_E**2 * M[1,1], 
            self.wn_psi**2 * M[2,2]
        ])
            
        self.Ki = np.array([
            (self.wn_N/15) * self.Kp[0], 
            (self.wn_E/15) * self.Kp[1], 
            (self.wn_psi/15) * self.Kp[2]
        ])
            
        self.Kd = np.array([
            2*self.zeta_N*self.wn_N * M[0,0] - D_l[0,0], 
            2*self.zeta_E*self.wn_E * M[1,1] - D_l[1,1], 
            2*self.zeta_psi*self.wn_psi * M[2,2] - D_l[2,2]
        ])



def default_thrusters_gunnerus3() -> list[ThrusterConfig]:
    """Three-thruster Gunnerus layout from the project description (Table 3)."""
    return [
        ThrusterConfig("Tunnel_Bow", "tunnel",  x=+12.0, y=0.0,
                       u_max=32000,  u_rate=4000,  rot_speed=0.0,    alpha0=np.pi / 2),
        ThrusterConfig("Azimuth_1",  "azimuth", x=-13.0, y=+3.0,
                       u_max=80000,  u_rate=10000, rot_speed=0.2094, alpha0=0.0),
        ThrusterConfig("Azimuth_2",  "azimuth", x=-13.0, y=-3.0,
                       u_max=80000,  u_rate=10000, rot_speed=0.2094, alpha0=0.0),
    ]
