"""Planar double pendulum (Lagrangian dynamics, SI units).

Model
-----
Two point masses ``m1``, ``m2`` on ideal massless rigid rods of length ``L1``,
``L2``; no friction or drag; uniform gravity ``g``. Angles ``theta1``,
``theta2`` are measured from the downward vertical (counter-clockwise
positive, with x to the right and y upward). With ``d = theta1 - theta2``::

    T = 1/2 (m1+m2) L1^2 w1^2 + 1/2 m2 L2^2 w2^2 + m2 L1 L2 w1 w2 cos d
    V = -(m1+m2) g L1 cos(theta1) - m2 g L2 cos(theta2)

The Euler-Lagrange equations give a 2x2 linear system for the angular
accelerations, solved directly at each right-hand-side evaluation::

    [ (m1+m2) L1^2        m2 L1 L2 cos d ] [a1]   [ -m2 L1 L2 w2^2 sin d - (m1+m2) g L1 sin(theta1) ]
    [ m2 L1 L2 cos d      m2 L2^2        ] [a2] = [  m2 L1 L2 w1^2 sin d - m2 g L2 sin(theta2)      ]

Integration: ``scipy.integrate.solve_ivp`` with DOP853 and tight tolerances.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh

DEFAULT_RTOL = 1e-12
DEFAULT_ATOL = 1e-12


@dataclass(frozen=True)
class PendulumParams:
    m1: float = 1.0  # kg
    m2: float = 1.0  # kg
    L1: float = 1.0  # m
    L2: float = 1.0  # m
    g: float = 9.81  # m/s^2

    def validate(self) -> None:
        for name in ("m1", "m2", "L1", "L2", "g"):
            v = getattr(self, name)
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be a positive, finite number.")


@dataclass(frozen=True)
class PendulumResult:
    t: np.ndarray  # (N,) s
    theta1: np.ndarray  # rad
    theta2: np.ndarray  # rad
    omega1: np.ndarray  # rad/s
    omega2: np.ndarray  # rad/s
    x1: np.ndarray  # m
    y1: np.ndarray  # m
    x2: np.ndarray  # m
    y2: np.ndarray  # m
    kinetic: np.ndarray  # J
    potential: np.ndarray  # J
    energy: np.ndarray  # J
    energy_drift: np.ndarray  # |E(t)-E(0)|/|E(0)| (see energy_scale)
    success: bool
    message: str


@dataclass(frozen=True)
class DivergenceResult:
    t: np.ndarray
    separation: np.ndarray  # sqrt(dtheta1^2 + dtheta2^2), rad
    growth_rate: float  # 1/s, ESTIMATE (see estimate_growth_rate)
    fit_window: tuple[float, float]  # (t_start, t_end) of the fit, s
    reference: PendulumResult
    perturbed: PendulumResult


def _mass_matrix_solve(y, p: PendulumParams):
    th1, th2, w1, w2 = y
    d = th1 - th2
    c, s = np.cos(d), np.sin(d)
    a11 = (p.m1 + p.m2) * p.L1**2
    a12 = p.m2 * p.L1 * p.L2 * c
    a22 = p.m2 * p.L2**2
    b1 = -p.m2 * p.L1 * p.L2 * w2**2 * s - (p.m1 + p.m2) * p.g * p.L1 * np.sin(th1)
    b2 = p.m2 * p.L1 * p.L2 * w1**2 * s - p.m2 * p.g * p.L2 * np.sin(th2)
    det = a11 * a22 - a12 * a12  # > 0 always since m1 > 0
    return (a22 * b1 - a12 * b2) / det, (a11 * b2 - a12 * b1) / det


def _rhs(_t, y, p: PendulumParams):
    a1, a2 = _mass_matrix_solve(y, p)
    return [y[2], y[3], a1, a2]


def energies(theta1, theta2, omega1, omega2, p: PendulumParams):
    """Return ``(kinetic, potential)`` for arrays/scalars of the state."""
    d = np.asarray(theta1) - np.asarray(theta2)
    kinetic = (
        0.5 * (p.m1 + p.m2) * p.L1**2 * np.square(omega1)
        + 0.5 * p.m2 * p.L2**2 * np.square(omega2)
        + p.m2 * p.L1 * p.L2 * np.multiply(omega1, omega2) * np.cos(d)
    )
    potential = -(p.m1 + p.m2) * p.g * p.L1 * np.cos(theta1) - p.m2 * p.g * p.L2 * np.cos(theta2)
    return kinetic, potential


def energy_scale(p: PendulumParams) -> float:
    """Characteristic energy ``(m1+m2) g L1 + m2 g L2`` (used if E(0) is ~0)."""
    return (p.m1 + p.m2) * p.g * p.L1 + p.m2 * p.g * p.L2


def simulate_double_pendulum(
    params: PendulumParams,
    theta1_0: float,
    theta2_0: float,
    omega1_0: float,
    omega2_0: float,
    t_end: float,
    n_points: int = 2000,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> PendulumResult:
    """Integrate the double pendulum from the given initial state (rad, rad/s)."""
    params.validate()
    state0 = np.array([theta1_0, theta2_0, omega1_0, omega2_0], dtype=float)
    if not np.all(np.isfinite(state0)):
        raise ValueError("Initial conditions must be finite.")
    if not np.isfinite(t_end) or t_end <= 0:
        raise ValueError("t_end must be positive and finite.")
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")

    sol = solve_ivp(
        _rhs,
        (0.0, t_end),
        state0,
        method="DOP853",
        t_eval=np.linspace(0.0, t_end, n_points),
        args=(params,),
        rtol=rtol,
        atol=atol,
    )
    th1, th2, w1, w2 = sol.y
    x1 = params.L1 * np.sin(th1)
    y1 = -params.L1 * np.cos(th1)
    x2 = x1 + params.L2 * np.sin(th2)
    y2 = y1 - params.L2 * np.cos(th2)
    ke, pe = energies(th1, th2, w1, w2, params)
    e = ke + pe
    scale = abs(e[0]) if abs(e[0]) > 1e-12 * energy_scale(params) else energy_scale(params)
    drift = np.abs(e - e[0]) / scale
    return PendulumResult(
        t=sol.t, theta1=th1, theta2=th2, omega1=w1, omega2=w2,
        x1=x1, y1=y1, x2=x2, y2=y2,
        kinetic=ke, potential=pe, energy=e, energy_drift=drift,
        success=bool(sol.success), message=str(sol.message),
    )


def small_angle_frequencies(params: PendulumParams):
    """Normal-mode angular frequencies (rad/s, ascending) and mode shapes.

    Solves ``K v = w^2 A v`` for the linearised system about the downward
    equilibrium. Returns ``(omegas, modes)``; ``modes[:, i]`` is the
    ``(theta1, theta2)`` shape of mode ``i``.
    """
    params.validate()
    p = params
    A = np.array([[(p.m1 + p.m2) * p.L1**2, p.m2 * p.L1 * p.L2],
                  [p.m2 * p.L1 * p.L2, p.m2 * p.L2**2]])
    K = np.diag([(p.m1 + p.m2) * p.g * p.L1, p.m2 * p.g * p.L2])
    w2, modes = eigh(K, A)
    return np.sqrt(w2), modes


def estimate_growth_rate(t, separation, sep_max: float = 1e-2, min_points: int = 5):
    """ESTIMATE of the early exponential growth rate of a separation curve.

    Least-squares slope of ``ln(separation)`` versus ``t`` over the initial
    stretch where the separation is still below ``sep_max`` (so the
    perturbation is small and the linearised picture is roughly valid).
    This is a finite-time, window-dependent estimate; it is NOT a rigorous
    Lyapunov exponent (no renormalisation, no long-time limit, no ensemble).

    Returns ``(rate, (t_start, t_end))``; rate is NaN if the window has fewer
    than ``min_points`` usable samples.
    """
    t = np.asarray(t, dtype=float)
    sep = np.asarray(separation, dtype=float)
    over = np.nonzero(sep > sep_max)[0]
    end = over[0] if over.size else sep.size
    sl = slice(0, end)
    ok = sep[sl] > 0
    if np.count_nonzero(ok) < min_points:
        return float("nan"), (float(t[0]), float(t[min(end, t.size) - 1]))
    tt, ss = t[sl][ok], np.log(sep[sl][ok])
    slope = np.polyfit(tt, ss, 1)[0]
    return float(slope), (float(tt[0]), float(tt[-1]))


def simulate_divergence(
    params: PendulumParams,
    theta1_0: float,
    theta2_0: float,
    omega1_0: float,
    omega2_0: float,
    t_end: float,
    n_points: int = 4000,
    delta_theta1: float = 1e-6,
    sep_max: float = 1e-2,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> DivergenceResult:
    """Integrate two nearly identical initial conditions and compare them.

    The second trajectory has ``theta1_0 + delta_theta1``. The separation is
    ``sqrt((dtheta1)^2 + (dtheta2)^2)`` in radians (angles not wrapped).
    ``growth_rate`` is an early-time estimate, see :func:`estimate_growth_rate`.
    """
    ref = simulate_double_pendulum(params, theta1_0, theta2_0, omega1_0, omega2_0, t_end, n_points, rtol, atol)
    per = simulate_double_pendulum(params, theta1_0 + delta_theta1, theta2_0, omega1_0, omega2_0, t_end, n_points, rtol, atol)
    sep = np.hypot(per.theta1 - ref.theta1, per.theta2 - ref.theta2)
    rate, window = estimate_growth_rate(ref.t, sep, sep_max=sep_max)
    return DivergenceResult(ref.t, sep, rate, window, ref, per)
