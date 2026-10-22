"""Thrust utilisation for Project Part 2, Simulation 5 — student work.

Definitions (from the project text, Part 2, Simulation 5):

* **Thrust utilisation** at time t is the sum of the magnitudes of the
  individual thruster forces, as a percentage of the sum of the maximum
  nominal thrusts of all active thrusters,

      U_T(t) = 100 * sum_i |u_i(t)| / sum_i u_max,i .

* **Average thrust utilisation** is the time average of U_T(t) over the
  evaluation window — by default the second half of the run, so the
  transient is excluded.

``simulation/capability.py`` (provided) runs the directional sweep, computes
the deviation measures and draws the polar plots; it calls the two functions
below for every direction.  Part 2, Simulation 5 raises
``NotImplementedError`` until you implement them.  The public check
(``python check.py --part 2``) verifies them on a synthetic log.

Inputs: ``logs`` carries ``logs.t`` (n,) time [s] and ``logs.u``
(n, n_thrusters) ACTUAL thrusts [N] (one signed column per thruster);
``thrusters`` is the list of ``ThrusterConfig`` with ``u_max`` [N].
"""
from __future__ import annotations

import numpy as np


def thrust_utilization(logs, thrusters) -> np.ndarray:
    """Instantaneous thrust utilisation U_T(t) in percent, from ACTUAL thrusts.

    TODO (students): implement the definition above — the sum of the
    magnitudes of the individual actuator forces (``logs.u``, one column per
    thruster) as a percentage of the sum of the maximum nominal thrusts of
    all active thrusters (``ThrusterConfig.u_max``).  Return one value per
    time sample, shape ``(len(logs.t),)``.
    """
    u = np.asarray(logs.u, dtype=float) 
    u_max = np.array([th.u_max for th in thrusters], float)
    return 100.0 * np.abs(u).sum(axis=1) / u_max.sum()

    raise NotImplementedError(
        "implement thrust_utilization() in part_2/utilization.py — the "
        "definition is in the project text (Part 2, Simulation 5)")


def average_utilization(logs, thrusters, t_from: float | None = None) -> float:
    """Average thrust utilisation [%] over ``t >= t_from`` (default: the second
    half of the run, so the transient is excluded).

    TODO (students): implement — the time average of
    :func:`thrust_utilization` over the evaluation window.
    """
    
    t = np.asarray(logs.t, dtype=float)
    if t_from is None:
        t_from = t[0] + 0.5 * (t[-1] - t[0])      # second half of the run
    mask = t >= t_from
    U = thrust_utilization(logs, thrusters)
    return float(U[mask].mean())

    raise NotImplementedError(
        "implement average_utilization() in part_2/utilization.py — the "
        "definition is in the project text (Part 2, Simulation 5)")
