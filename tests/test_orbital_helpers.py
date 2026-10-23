"""Tests for orbit classification, input validation, and CSV tabulation helpers."""

import numpy as np
import pytest

from physlab import orbital as ob

M1, M2, R = 5.972e24, 1000.0, 7.0e6


def _el(factor):
    v = factor * ob.circular_speed(M1, M2, R)
    return ob.orbit_elements(M1, M2, [R, 0], [0, v])


def test_circular_elements():
    el = _el(1.0)
    assert el.kind == "bound"
    assert el.eccentricity < 1e-9
    assert el.semi_major_axis == pytest.approx(R, rel=1e-12)
    assert el.period == pytest.approx(ob.kepler_period(M1, M2, R), rel=1e-12)


def test_elliptical_elements():
    el = _el(1.2)
    assert el.kind == "bound"
    # starting at pericentre with v = 1.2 v_c: e = 1.2^2 - 1
    assert el.eccentricity == pytest.approx(0.44, rel=1e-9)
    assert el.periapsis == pytest.approx(R, rel=1e-9)
    assert el.apoapsis == pytest.approx(R * (1 + 0.44) / (1 - 0.44), rel=1e-9)


def test_escape_and_hyperbolic_elements():
    par = _el(np.sqrt(2.0))
    assert par.kind == "parabolic"
    assert par.eccentricity == pytest.approx(1.0, abs=1e-6)
    assert par.period is None
    hyp = _el(1.6)
    assert hyp.kind == "unbound"
    assert hyp.eccentricity > 1
    assert hyp.periapsis == pytest.approx(R, rel=1e-9)


def test_elements_match_simulation_apsides():
    el = _el(1.2)
    v = 1.2 * ob.circular_speed(M1, M2, R)
    res = ob.simulate_two_body(M1, M2, [R, 0], [0, v], el.period, 4000)
    r = np.linalg.norm(res.position, axis=1)
    assert r.max() == pytest.approx(el.apoapsis, rel=1e-4)
    assert r.min() == pytest.approx(el.periapsis, rel=1e-4)


@pytest.mark.parametrize(
    "args",
    [
        (0, M2, R, 7e3, 1e4, 100),
        (M1, -1, R, 7e3, 1e4, 100),
        (M1, M2, 0, 7e3, 1e4, 100),
        (M1, M2, R, -1, 1e4, 100),
        (M1, M2, R, 7e3, 0, 100),
        (M1, M2, R, float("nan"), 1e4, 100),
        (M1, M2, R, 7e3, 1e4, 1),
        (M1, M2, R, 0.0, 1e4, 100),  # radial plunge
        (M1, M2, R, 10.0, 1e4, 100),  # near-radial, tiny periapsis
        (M1, M2, R, 7.5e3, 1e12, 100),  # too many periods
    ],
)
def test_validate_rejects_bad_input(args):
    assert ob.validate_inputs(*args)


def test_validate_accepts_good_input():
    v = ob.circular_speed(M1, M2, R)
    assert ob.validate_inputs(M1, M2, R, v, 3600.0, 500) == []


def test_dataframe_columns_and_shape():
    v = ob.circular_speed(M1, M2, R)
    res = ob.simulate_two_body(M1, M2, [R, 0], [0, v], 1000.0, 50)
    df = ob.result_to_dataframe(res)
    assert len(df) == 50
    assert {"t_s", "x_m", "y_m", "total_energy_J", "relative_energy_drift"} <= set(df.columns)
    assert "total_energy_J" in df.to_csv(index=False)
