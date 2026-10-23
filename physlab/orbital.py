"""Two-body gravitational dynamics in the relative coordinate.

Model
-----
Two point masses ``m1`` and ``m2`` interact only through Newtonian gravity.
The centre of mass moves uniformly, so the problem reduces to a single body of
reduced mass ``m = m1*m2/(m1+m2)`` moving in the field of a fixed centre with
gravitational parameter ``mu = G*(m1+m2)``::

    r'' = -mu * r / |r|^3

All quantities are SI (m, s, kg, J, kg m^2/s). Motion is planar, so the state
is ``(x, y, vx, vy)``. Energies and angular momentum are those of the
relative (reduced-mass) motion, which equal the internal energy and angular
momentum of the two-body system in its centre-of-mass frame.

Integration uses ``scipy.integrate.solve_ivp`` with DOP853 and tight tolerances.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

G = 6.67430e-11  # m^3 kg^-1 s^-2 (CODATA 2018)

DEFAULT_RTOL = 1e-12
DEFAULT_ATOL = 1e-6  # metres / (m/s); small relative to orbital scales


@dataclass(frozen=True)
class OrbitResult:
    """Output of :func:`simulate_two_body` (relative coordinate, SI units)."""

    t: np.ndarray  # (N,) s
    position: np.ndarray  # (N, 2) m
    velocity: np.ndarray  # (N, 2) m/s
    kinetic: np.ndarray  # (N,) J
    potential: np.ndarray  # (N,) J
    energy: np.ndarray  # (N,) J
    angular_momentum: np.ndarray  # (N,) kg m^2/s
    energy_drift: np.ndarray  # (N,) |E(t)-E(0)|/|E(0)|, dimensionless
    mu: float
    reduced_mass: float
    success: bool
    message: str


def gravitational_parameter(m1: float, m2: float) -> float:
    """Return ``mu = G (m1 + m2)``."""
    return G * (m1 + m2)


def reduced_mass(m1: float, m2: float) -> float:
    """Return ``m1 m2 / (m1 + m2)``."""
    return m1 * m2 / (m1 + m2)


def circular_speed(m1: float, m2: float, r: float) -> float:
    """Speed of the relative coordinate for a circular orbit of radius ``r``."""
    return float(np.sqrt(gravitational_parameter(m1, m2) / r))


def escape_speed(m1: float, m2: float, r: float) -> float:
    """Relative speed at separation ``r`` giving zero total energy (parabolic)."""
    return float(np.sqrt(2.0 * gravitational_parameter(m1, m2) / r))


def kepler_period(m1: float, m2: float, a: float) -> float:
    """Kepler's third law (two-body form): ``T = 2 pi sqrt(a^3 / (G (m1+m2)))``."""
    return float(2.0 * np.pi * np.sqrt(a**3 / gravitational_parameter(m1, m2)))


def semi_major_axis(m1: float, m2: float, r: float, v: float) -> float:
    """Semi-major axis from the vis-viva equation (negative for hyperbolic)."""
    mu = gravitational_parameter(m1, m2)
    return float(1.0 / (2.0 / r - v**2 / mu))


def _rhs(_t: float, y: np.ndarray, mu: float) -> np.ndarray:
    x, yy, vx, vy = y
    r3 = (x * x + yy * yy) ** 1.5
    return np.array([vx, vy, -mu * x / r3, -mu * yy / r3])


def simulate_two_body(
    m1: float,
    m2: float,
    r0,
    v0,
    t_end: float,
    n_points: int = 2000,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> OrbitResult:
    """Integrate the relative two-body motion.

    Parameters
    ----------
    m1, m2 : masses in kg (> 0).
    r0, v0 : initial relative position (m) and velocity (m/s), length-2.
    t_end : final time in s (> 0).
    n_points : number of uniformly spaced output samples (>= 2).
    rtol, atol : solver tolerances (DOP853).
    """
    if m1 <= 0 or m2 <= 0:
        raise ValueError("Masses must be positive.")
    if t_end <= 0:
        raise ValueError("t_end must be positive.")
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    if r0.shape != (2,) or v0.shape != (2,):
        raise ValueError("r0 and v0 must be length-2 vectors.")
    if np.linalg.norm(r0) == 0:
        raise ValueError("Initial separation must be non-zero.")

    mu = gravitational_parameter(m1, m2)
    m = reduced_mass(m1, m2)
    t_eval = np.linspace(0.0, t_end, n_points)

    sol = solve_ivp(
        _rhs,
        (0.0, t_end),
        np.concatenate([r0, v0]),
        method="DOP853",
        t_eval=t_eval,
        args=(mu,),
        rtol=rtol,
        atol=atol,
    )

    pos = sol.y[:2].T
    vel = sol.y[2:].T
    r = np.linalg.norm(pos, axis=1)
    kinetic = 0.5 * m * np.sum(vel**2, axis=1)
    potential = -G * m1 * m2 / r
    energy = kinetic + potential
    ang = m * (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0])
    drift = np.abs(energy - energy[0]) / abs(energy[0]) if energy[0] != 0 else np.full_like(energy, np.nan)

    return OrbitResult(
        t=sol.t,
        position=pos,
        velocity=vel,
        kinetic=kinetic,
        potential=potential,
        energy=energy,
        angular_momentum=ang,
        energy_drift=drift,
        mu=mu,
        reduced_mass=m,
        success=bool(sol.success),
        message=str(sol.message),
    )


# ---------------------------------------------------------------------------
# Analytic orbit characterisation and UI helpers
# ---------------------------------------------------------------------------

PARABOLIC_TOL = 1e-9  # |E| / |U| below this is classed as parabolic
MAX_OUTPUT_POINTS = 20000
MAX_PERIODS = 200  # cap on simulated duration for bound orbits


@dataclass(frozen=True)
class OrbitElements:
    """Analytic orbit properties from the initial state (relative coordinate)."""

    kind: str  # "bound", "parabolic", or "unbound"
    specific_energy: float  # J/kg
    specific_angular_momentum: float  # m^2/s
    eccentricity: float
    semi_major_axis: float | None  # m; None for parabolic
    period: float | None  # s; bound only
    periapsis: float  # m
    apoapsis: float | None  # m; bound only


def orbit_elements(m1: float, m2: float, r0, v0) -> OrbitElements:
    """Classify the orbit and compute its elements from the initial state."""
    mu = gravitational_parameter(m1, m2)
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    r = float(np.linalg.norm(r0))
    v2 = float(v0 @ v0)
    eps = 0.5 * v2 - mu / r
    h = float(abs(r0[0] * v0[1] - r0[1] * v0[0]))
    e = float(np.sqrt(max(0.0, 1.0 + 2.0 * eps * h * h / mu**2)))

    if abs(eps) <= PARABOLIC_TOL * mu / r:
        kind, a = "parabolic", None
        peri = h * h / (2.0 * mu)
    else:
        a = -mu / (2.0 * eps)
        kind = "bound" if eps < 0 else "unbound"
        peri = a * (1.0 - e)
        if a < 0:  # hyperbola: a<0 and e>1 give positive periapsis
            peri = abs(a) * (e - 1.0)
    if kind == "bound":
        return OrbitElements(kind, eps, h, e, a, kepler_period(m1, m2, a), peri, a * (1.0 + e))
    return OrbitElements(kind, eps, h, e, a, None, peri, None)


def validate_inputs(m1, m2, r0, v0, t_end, n_points) -> list[str]:
    """Return human-readable problems with the inputs (empty list if valid)."""
    errors: list[str] = []
    for name, val in [
        ("Central mass", m1),
        ("Orbiting mass", m2),
        ("Initial distance", r0),
        ("Duration", t_end),
    ]:
        if val is None or not np.isfinite(val) or val <= 0:
            errors.append(f"{name} must be a positive, finite number.")
    if v0 is None or not np.isfinite(v0) or v0 < 0:
        errors.append("Initial speed must be a finite, non-negative number.")
    if not (2 <= n_points <= MAX_OUTPUT_POINTS):
        errors.append(f"Number of output points must be between 2 and {MAX_OUTPUT_POINTS}.")
    if errors:
        return errors

    if v0 == 0:
        errors.append("Initial speed of zero is a radial plunge (collision); choose a non-zero speed.")
        return errors
    el = orbit_elements(m1, m2, [r0, 0.0], [0.0, v0])
    if el.periapsis < 1e-3 * r0:
        errors.append(
            "Periapsis is below 0.1% of the initial distance: the orbit is nearly radial "
            "and the point-mass model becomes stiff. Increase the initial speed."
        )
    if el.kind == "bound" and el.period is not None and t_end > MAX_PERIODS * el.period:
        errors.append(
            f"Duration exceeds {MAX_PERIODS} orbital periods ({MAX_PERIODS * el.period:.4g} s); shorten it."
        )
    return errors


def result_to_dataframe(res: OrbitResult):
    """Tabulate an :class:`OrbitResult` (SI units) for display or CSV export."""
    import pandas as pd

    return pd.DataFrame(
        {
            "t_s": res.t,
            "x_m": res.position[:, 0],
            "y_m": res.position[:, 1],
            "vx_m_s": res.velocity[:, 0],
            "vy_m_s": res.velocity[:, 1],
            "kinetic_J": res.kinetic,
            "potential_J": res.potential,
            "total_energy_J": res.energy,
            "angular_momentum_kg_m2_s": res.angular_momentum,
            "relative_energy_drift": res.energy_drift,
        }
    )
