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

    The signed actual thrusts are converted to magnitudes before summing.
    """
    times = np.asarray(logs.t, dtype=float).reshape(-1)
    thrusts = np.asarray(logs.u, dtype=float)
    limits = np.asarray([thruster.u_max for thruster in thrusters], dtype=float)
    if times.size == 0:
        raise ValueError("logs.t must contain at least one sample")
    if thrusts.shape != (times.size, limits.size):
        raise ValueError(
            "logs.u must have shape (len(logs.t), len(thrusters)); "
            f"got {thrusts.shape}"
        )
    if limits.size == 0 or not np.all(np.isfinite(limits)) or np.any(limits <= 0.0):
        raise ValueError("thrusters must have positive finite maximum thrust")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(thrusts)):
        raise ValueError("logs.t and logs.u must contain only finite values")

    return 100.0 * np.sum(np.abs(thrusts), axis=1) / np.sum(limits)


def average_utilization(logs, thrusters, t_from: float | None = None) -> float:
    """Average thrust utilisation [%] over ``t >= t_from`` (default: the second
    half of the run, so the transient is excluded).

    """
    times = np.asarray(logs.t, dtype=float).reshape(-1)
    utilization = thrust_utilization(logs, thrusters)
    start = (times[0] + times[-1]) / 2.0 if t_from is None else float(t_from)
    if not np.isfinite(start):
        raise ValueError("t_from must be finite")
    selected = utilization[times >= start]
    if selected.size == 0:
        raise ValueError("the evaluation window contains no time samples")
    return float(np.mean(selected))
