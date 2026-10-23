"""Electric field, potential and field lines of static 2D-positioned point charges.

Model
-----
Static point charges in vacuum, located at ``(x_i, y_i)`` in the plane z = 0.
The full 3D Coulomb law is used (field ~ 1/r^2, potential ~ 1/r) and evaluated
on that plane. No induced charges, conductors, dielectrics or motion::

    E(r) = sum_i k q_i (r - r_i) / |r - r_i|^3
    V(r) = sum_i k q_i / |r - r_i|,        k = 1 / (4 pi eps0)

Units: charges in coulombs, positions in metres, E in V/m, V in volts.
``microcoulombs_to_coulombs`` is provided for user-friendly input.

Singularity handling (honest note)
----------------------------------
E and V diverge at a charge. Grid/evaluation points closer than ``r_min`` to
*any* charge are returned as NaN (masked), not clipped or softened, because the
point-charge model has no valid value there. ``r_min`` is a presentation
choice; it does not alter values anywhere else.

Field lines
-----------
Lines are integrated along the normalised field, ``dr/ds = +/- E/|E|``, with
``solve_ivp`` (DOP853) in arclength ``s``. This gives line *shape* only;
line spacing in a plot is only a qualitative indicator of field strength.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

EPSILON_0 = 8.8541878128e-12  # F/m (CODATA 2018)
K_COULOMB = 1.0 / (4.0 * np.pi * EPSILON_0)  # N m^2 / C^2

MICROCOULOMB = 1e-6


def microcoulombs_to_coulombs(q_uC):
    """Convert microcoulombs to coulombs."""
    return np.asarray(q_uC, dtype=float) * MICROCOULOMB


def _check_charges(charges, positions):
    q = np.atleast_1d(np.asarray(charges, dtype=float))
    pos = np.asarray(positions, dtype=float)
    if q.ndim != 1 or q.size == 0:
        raise ValueError("charges must be a non-empty 1-D sequence.")
    if pos.shape != (q.size, 2):
        raise ValueError("positions must have shape (n_charges, 2).")
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(pos))):
        raise ValueError("charges and positions must be finite.")
    return q, pos


def _check_r_min(r_min):
    if not np.isfinite(r_min) or r_min < 0:
        raise ValueError("r_min must be a non-negative, finite number.")


def field_at_points(charges, positions, x, y, r_min: float = 0.0):
    """Ex, Ey (V/m) and V (V) at arbitrary points (arrays of equal shape).

    Points within ``r_min`` of any charge, or exactly on a charge (distance 0
    even if ``r_min`` is 0), are NaN in all three outputs. Loops over charges
    and is vectorised over points.
    """
    q, pos = _check_charges(charges, positions)
    _check_r_min(r_min)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")

    ex = np.zeros(x.shape)
    ey = np.zeros(x.shape)
    v = np.zeros(x.shape)
    masked = np.zeros(x.shape, dtype=bool)
    for qi, (xi, yi) in zip(q, pos):
        dx, dy = x - xi, y - yi
        r2 = dx * dx + dy * dy
        masked |= (r2 < r_min**2) | (r2 == 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.sqrt(r2)
            r3 = r2 * r
            ex += K_COULOMB * qi * dx / r3
            ey += K_COULOMB * qi * dy / r3
            v += K_COULOMB * qi / r
    ex[masked] = np.nan
    ey[masked] = np.nan
    v[masked] = np.nan
    return ex, ey, v


def field_on_grid(charges, positions, x_range, y_range, nx: int = 200, ny: int = 200, r_min: float = 0.0):
    """Evaluate Ex, Ey, V on a regular grid.

    Returns ``(X, Y, Ex, Ey, V)`` with shape ``(ny, nx)`` (``indexing='xy'``).
    ``x_range``/``y_range`` are ``(min, max)`` in metres.
    """
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must be at least 2.")
    for lo, hi in (x_range, y_range):
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            raise ValueError("Ranges must be finite with max > min.")
    X, Y = np.meshgrid(np.linspace(*x_range, nx), np.linspace(*y_range, ny))
    ex, ey, v = field_at_points(charges, positions, X, Y, r_min)
    return X, Y, ex, ey, v


@dataclass(frozen=True)
class FieldLine:
    x: np.ndarray  # m
    y: np.ndarray  # m
    start_charge: int  # index of the charge the line was started from
    end_reason: str  # "negative_charge" | "boundary" | "max_length" | "null_field" | "other"
    length: float  # arclength reached, m


def trace_field_lines(
    charges,
    positions,
    bounds,
    r_min: float = 0.0,
    lines_per_charge: int = 12,
    start_radius: float | None = None,
    stop_radius: float | None = None,
    max_length: float | None = None,
    points_per_line: int = 400,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> list[FieldLine]:
    """Trace electric field lines in the plane.

    Parameters
    ----------
    bounds : ``(xmin, xmax, ymin, ymax)`` plot window in metres; lines stop on leaving it.
    r_min : exclusion radius (m); also sets defaults below.
    lines_per_charge : number of lines for the largest-|q| source; others get a
        proportional number (at least 4).
    start_radius : radius (m) of the start circle around each source
        (default ``max(2*r_min, 1e-2 * window diagonal)``).
    stop_radius : lines stop when closer than this to a negative charge
        (default ``max(r_min, 0.5 * start_radius)``; must be < start_radius).
    max_length : maximum arclength (m) per line (default 4 x window diagonal).

    Lines start from positive charges and follow E. If there are no positive
    charges, they start from negative charges and follow -E (so a plot is not
    empty). Returns a list of :class:`FieldLine`.
    """
    q, pos = _check_charges(charges, positions)
    _check_r_min(r_min)
    xmin, xmax, ymin, ymax = bounds
    if not (xmax > xmin and ymax > ymin):
        raise ValueError("bounds must satisfy xmax > xmin and ymax > ymin.")
    if lines_per_charge < 1:
        raise ValueError("lines_per_charge must be at least 1.")
    diag = float(np.hypot(xmax - xmin, ymax - ymin))
    if start_radius is None:
        start_radius = max(2.0 * r_min, 1e-2 * diag)
    if start_radius <= 0 or start_radius <= r_min:
        raise ValueError("start_radius must be positive and larger than r_min.")
    if stop_radius is None:
        stop_radius = max(r_min, 0.5 * start_radius)
    if not (0 < stop_radius < start_radius):
        raise ValueError("stop_radius must satisfy 0 < stop_radius < start_radius.")
    if max_length is None:
        max_length = 4.0 * diag
    if max_length <= 0 or points_per_line < 2:
        raise ValueError("max_length must be positive and points_per_line >= 2.")

    positive = np.nonzero(q > 0)[0]
    if positive.size:
        sources, sign = positive, +1.0
    else:
        sources, sign = np.nonzero(q < 0)[0], -1.0
        if sources.size == 0:  # all charges zero: no field
            return []
    negatives = np.nonzero(q < 0)[0] if sign > 0 else np.array([], dtype=int)
    q_ref = np.max(np.abs(q[sources]))
    e_floor = 1e-9 * K_COULOMB * np.sum(np.abs(q)) / diag**2  # "null field" threshold, V/m

    def direction(_s, state):
        ex, ey, _ = field_at_points(q, pos, state[0], state[1], 0.0)
        mag = np.hypot(ex, ey)
        if not np.isfinite(mag) or mag < e_floor:
            return [0.0, 0.0]
        return [sign * ex / mag, sign * ey / mag]

    def hit_boundary(_s, state):
        return min(state[0] - xmin, xmax - state[0], state[1] - ymin, ymax - state[1])

    hit_boundary.terminal, hit_boundary.direction = True, -1

    def null_field(_s, state):
        ex, ey, _ = field_at_points(q, pos, state[0], state[1], 0.0)
        return np.hypot(ex, ey) - 1.5 * e_floor if np.isfinite(ex) else -1.0

    null_field.terminal, null_field.direction = True, -1

    def make_neg_event(i):
        def ev(_s, state):
            return float(np.hypot(state[0] - pos[i, 0], state[1] - pos[i, 1])) - stop_radius

        ev.terminal, ev.direction = True, -1
        return ev

    events = [hit_boundary, null_field] + [make_neg_event(i) for i in negatives]
    n_fixed = 2

    lines: list[FieldLine] = []
    for i in sources:
        n_lines = max(4, int(round(lines_per_charge * abs(q[i]) / q_ref)))
        for ang in 2 * np.pi * (np.arange(n_lines) + 0.5) / n_lines:
            x0 = pos[i, 0] + start_radius * np.cos(ang)
            y0 = pos[i, 1] + start_radius * np.sin(ang)
            if not (xmin < x0 < xmax and ymin < y0 < ymax):
                continue
            sol = solve_ivp(
                direction, (0.0, max_length), [x0, y0], method="DOP853",
                t_eval=np.linspace(0.0, max_length, points_per_line),
                events=events, rtol=rtol, atol=atol, max_step=diag / 200.0,
            )
            xs, ys = sol.y
            if sol.status == 1:
                if sol.t_events[0].size:
                    reason = "boundary"
                elif sol.t_events[1].size:
                    reason = "null_field"
                elif any(te.size for te in sol.t_events[n_fixed:]):
                    reason = "negative_charge"
                else:
                    reason = "other"
                # append the exact termination point so the line visibly ends there
                s_end = min(te[0] for te in sol.t_events if te.size)
                end_state = sol.y_events[[k for k, te in enumerate(sol.t_events) if te.size and te[0] == s_end][0]][0]
                xs = np.append(xs, end_state[0])
                ys = np.append(ys, end_state[1])
                length = float(s_end)
            elif sol.status == 0:
                reason, length = "max_length", float(sol.t[-1])
            else:
                reason, length = "other", float(sol.t[-1]) if sol.t.size else 0.0
            lines.append(FieldLine(xs, ys, int(i), reason, length))
    return lines
