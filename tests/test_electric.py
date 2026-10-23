"""Validation tests for physlab.electric.

Stated tolerances (measured values from development runs in brackets):
  * single-charge E and V vs kQ/r^2, kQ/r: 1e-12 relative [~1e-16]
  * superposition: 1e-12 relative (floating-point sums) [~1e-16]
  * dipole on the perpendicular bisector, exact two-charge expression:
    1e-12 relative; point-dipole far-field k p / y^3 at y = 100 d: 1e-4
    relative (the neglected term is ~ 1.5 (d/2y)^2 = 3.75e-5)
  * grad V vs -E, central differences on a 0.05 m grid, points >= 2 m from
    any charge: 2e-3 relative to local |E| [measured ~8.7e-4; error is
    O(h^2), checked by halving h]
  * field-line end points: within 1e-4 relative of the stated stop radius
"""

import numpy as np
import pytest

from physlab import electric as el

K = el.K_COULOMB


def rel(a, b):
    return np.max(np.abs(a - b) / np.abs(b))


def test_constants():
    assert el.EPSILON_0 == pytest.approx(8.8541878128e-12)
    assert K == pytest.approx(8.9875517923e9, rel=1e-9)
    assert el.microcoulombs_to_coulombs(2.5) == pytest.approx(2.5e-6)


@pytest.mark.parametrize("q", [3e-6, -3e-6])
def test_single_charge_field_magnitude_and_direction(q):
    pos = [[1.0, -2.0]]
    r = np.array([0.1, 0.5, 1.0, 3.0, 25.0])
    for ang in (0.0, 0.7, 2.0, -2.5):
        x = 1.0 + r * np.cos(ang)
        y = -2.0 + r * np.sin(ang)
        ex, ey, _ = el.field_at_points([q], pos, x, y)
        assert rel(np.hypot(ex, ey), K * abs(q) / r**2) < 1e-12
        s = np.sign(q)  # + : away from charge, - : toward it
        assert np.allclose(ex / np.hypot(ex, ey), s * np.cos(ang), atol=1e-12)
        assert np.allclose(ey / np.hypot(ex, ey), s * np.sin(ang), atol=1e-12)


def test_single_charge_potential():
    q = 2e-6
    r = np.array([0.2, 1.0, 4.0, 50.0])
    _, _, v = el.field_at_points([q], [[0.0, 0.0]], r, np.zeros_like(r))
    assert rel(v, K * q / r) < 1e-12
    _, _, vneg = el.field_at_points([-q], [[0.0, 0.0]], r, np.zeros_like(r))
    assert rel(vneg, -K * q / r) < 1e-12


def test_superposition():
    qs = [2e-6, -5e-6, 1e-6]
    ps = [[-1.0, 0.5], [2.0, 1.0], [0.0, -2.0]]
    rng = np.random.default_rng(0)
    x, y = rng.uniform(-6, 6, 200), rng.uniform(-6, 6, 200)
    ex, ey, v = el.field_at_points(qs, ps, x, y)
    sx = sy = sv = 0.0
    for qi, pi in zip(qs, ps):
        a, b, c = el.field_at_points([qi], [pi], x, y)
        sx, sy, sv = sx + a, sy + b, sv + c
    scale_e = np.hypot(sx, sy)
    assert np.max(np.hypot(ex - sx, ey - sy) / scale_e) < 1e-12
    assert np.max(np.abs(v - sv) / np.abs(sv)) < 1e-12


def test_dipole_perpendicular_bisector():
    q, d = 1e-6, 0.02
    ps = [[d / 2, 0.0], [-d / 2, 0.0]]
    y = np.array([0.5, 1.0, 2.0])
    ex, ey, v = el.field_at_points([q, -q], ps, np.zeros_like(y), y)
    # exact two-charge result: E = -x_hat * k q d / (y^2 + d^2/4)^(3/2)
    exact = -K * q * d / (y**2 + d**2 / 4) ** 1.5
    assert np.max(np.abs(ex - exact) / np.abs(exact)) < 1e-12
    assert np.max(np.abs(ey)) < 1e-12 * np.max(np.abs(ex))  # no y-component by symmetry
    assert np.max(np.abs(v)) < 1e-12 * K * q / y.min()  # V = 0 on the bisector
    # far-field point-dipole approximation, y = 100 d
    yf = 100 * d
    exf, _, _ = el.field_at_points([q, -q], ps, [0.0], [yf])
    approx = -K * (q * d) / yf**3
    assert abs(exf[0] - approx) / abs(approx) < 1e-4


def test_masked_points_are_nan_not_inf():
    qs, ps = [1e-6, -1e-6], [[0.0, 0.0], [1.0, 0.0]]
    # grid has points exactly on both charges (0,0) and (1,0)
    X, Y, ex, ey, v = el.field_on_grid(qs, ps, (-1.0, 2.0), (-1.0, 1.0), nx=31, ny=21, r_min=0.15)
    d = np.minimum(np.hypot(X, Y), np.hypot(X - 1.0, Y))
    near = d < 0.15
    assert near.any()
    for arr in (ex, ey, v):
        assert np.all(np.isnan(arr[near]))
        assert not np.any(np.isinf(arr))
        assert np.all(np.isfinite(arr[~near]))
    # even with r_min = 0 a point exactly on a charge is NaN, never inf
    ex0, ey0, v0 = el.field_at_points(qs, ps, [0.0], [0.0], r_min=0.0)
    assert np.isnan(ex0[0]) and np.isnan(ey0[0]) and np.isnan(v0[0])


def _gradient_error(h):
    qs, ps = [3e-6, -2e-6], [[-1.0, 0.0], [1.5, 0.5]]
    xs = np.arange(-6.0, 6.0 + h / 2, h)
    ys = np.arange(-5.0, 5.0 + h / 2, h)
    X, Y = np.meshgrid(xs, ys)
    ex, ey, v = el.field_at_points(qs, ps, X, Y, r_min=0.0)
    dVdy, dVdx = np.gradient(v, h)
    d_to_charge = np.min([np.hypot(X - p[0], Y - p[1]) for p in ps], axis=0)
    ok = d_to_charge >= 2.0
    ok[[0, -1], :] = False
    ok[:, [0, -1]] = False  # one-sided edge differences are only first-order
    emag = np.hypot(ex, ey)
    return (np.hypot(-dVdx - ex, -dVdy - ey)[ok] / emag[ok]).max()


def test_gradient_of_potential_matches_minus_field():
    # Central differences: error is O(h^2). Measured max relative error at
    # h = 0.05 m is ~8.7e-4 (points >= 2 m from any charge); tolerance 2e-3.
    e_fine = _gradient_error(0.05)
    assert e_fine < 2e-3
    # and it really scales ~h^2: halving h cuts the error by close to 4x
    e_coarse = _gradient_error(0.1)
    assert 3.0 < e_coarse / e_fine < 5.0


def test_invalid_inputs():
    with pytest.raises(ValueError):
        el.field_at_points([], np.zeros((0, 2)), [0.0], [0.0])
    with pytest.raises(ValueError):
        el.field_at_points([1e-6], [[0.0, 0.0, 0.0]], [0.0], [0.0])
    with pytest.raises(ValueError):
        el.field_at_points([np.nan], [[0.0, 0.0]], [0.0], [0.0])
    with pytest.raises(ValueError):
        el.field_at_points([1e-6], [[0.0, 0.0]], [1.0], [0.0], r_min=-1.0)
    with pytest.raises(ValueError):
        el.field_on_grid([1e-6], [[0.0, 0.0]], (1.0, 0.0), (0.0, 1.0))


# --- field lines -----------------------------------------------------------

BOUNDS = (-5.0, 5.0, -5.0, 5.0)


def test_lines_from_single_positive_charge_are_radial_and_reach_boundary():
    lines = el.trace_field_lines([1e-6], [[0.0, 0.0]], BOUNDS, r_min=0.1, lines_per_charge=8)
    assert len(lines) == 8
    for ln in lines:
        assert ln.end_reason == "boundary"
        ang = np.arctan2(ln.y[0], ln.x[0])
        # points stay on the ray through the start point
        cross = ln.x * np.sin(ang) - ln.y * np.cos(ang)
        assert np.max(np.abs(cross)) < 1e-6
        assert np.hypot(ln.x[-1], ln.y[-1]) > 4.9  # reached the window edge
        assert np.all(np.diff(np.hypot(ln.x, ln.y)) > 0)  # moves outward


def test_dipole_lines_end_on_negative_charge_or_boundary_and_follow_field():
    qs, ps = [1e-6, -1e-6], [[-1.0, 0.0], [1.0, 0.0]]
    lines = el.trace_field_lines(qs, ps, BOUNDS, r_min=0.05, lines_per_charge=16,
                                 start_radius=0.1, stop_radius=0.05)
    reasons = {ln.end_reason for ln in lines}
    assert "negative_charge" in reasons
    assert reasons <= {"negative_charge", "boundary", "max_length", "null_field"}
    for ln in lines:
        assert ln.start_charge == 0
        if ln.end_reason == "negative_charge":
            assert np.hypot(ln.x[-1] - 1.0, ln.y[-1]) == pytest.approx(0.05, rel=1e-4)
        # tangent to E: direction between consecutive points aligned with E at the midpoint
        mx, my = 0.5 * (ln.x[1:] + ln.x[:-1]), 0.5 * (ln.y[1:] + ln.y[:-1])
        ex, ey, _ = el.field_at_points(qs, ps, mx, my)
        tx, ty = np.diff(ln.x), np.diff(ln.y)
        cosang = (tx * ex + ty * ey) / (np.hypot(tx, ty) * np.hypot(ex, ey))
        keep = np.hypot(tx, ty) > 1e-9
        assert np.min(cosang[keep]) > 0.99


def test_max_length_respected_and_negative_only_traces_backward():
    lines = el.trace_field_lines([1e-6], [[0.0, 0.0]], BOUNDS, r_min=0.1, max_length=1.0, lines_per_charge=4)
    assert all(ln.end_reason == "max_length" and ln.length <= 1.0 + 1e-12 for ln in lines)
    neg = el.trace_field_lines([-1e-6], [[0.0, 0.0]], BOUNDS, r_min=0.1, lines_per_charge=4)
    assert len(neg) == 4 and all(ln.end_reason == "boundary" for ln in neg)
    assert all(np.hypot(ln.x[-1], ln.y[-1]) > np.hypot(ln.x[0], ln.y[0]) for ln in neg)


def test_trace_invalid_inputs():
    with pytest.raises(ValueError):
        el.trace_field_lines([1e-6], [[0.0, 0.0]], (1.0, 0.0, 0.0, 1.0))
    with pytest.raises(ValueError):
        el.trace_field_lines([1e-6], [[0.0, 0.0]], BOUNDS, r_min=0.5, start_radius=0.2)
