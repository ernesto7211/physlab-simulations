"""Validation tests for physlab.pendulum.

Stated tolerances (DOP853, rtol=atol=1e-12). Measured values from development
runs are given in brackets; each tolerance leaves margin above them.
  * small-angle normal-mode period vs linear theory: 1e-6 relative
    [measured <= ~1e-8; amplitude 1e-4 rad gives O(amp^2)=1e-8 nonlinearity]
  * energy conservation, moderate amplitude (20 s): drift < 1e-9
    [measured ~5e-11 to 1e-12]
  * rest state: angles stay exactly 0 (< 1e-15)
  * divergence: large-angle max separation > 1e3 x small-angle one
    [measured ratio ~3e7]
"""

import numpy as np
import pytest

from physlab import pendulum as pd

PARAM_SETS = [
    pd.PendulumParams(),  # equal masses and lengths
    pd.PendulumParams(m1=2.0, m2=0.5, L1=1.0, L2=1.7, g=9.81),  # unequal
]


def _measured_period(t, signal):
    """Mean spacing of upward-to-downward zero crossings (linear interpolation)."""
    idx = np.nonzero((signal[:-1] > 0) & (signal[1:] <= 0))[0]
    tc = t[idx] - signal[idx] * (t[idx + 1] - t[idx]) / (signal[idx + 1] - signal[idx])
    assert len(tc) >= 4
    return np.diff(tc).mean()


@pytest.mark.parametrize("p", PARAM_SETS)
@pytest.mark.parametrize("mode", [0, 1])
def test_small_angle_normal_mode_frequencies(p, mode):
    omegas, shapes = pd.small_angle_frequencies(p)
    amp = 1e-4  # rad
    v = shapes[:, mode] / np.abs(shapes[:, mode]).max() * amp
    T = 2 * np.pi / omegas[mode]
    res = pd.simulate_double_pendulum(p, v[0], v[1], 0.0, 0.0, 6 * T, 60001)
    assert res.success
    sig = res.theta1 if abs(v[0]) >= abs(v[1]) else res.theta2
    assert abs(_measured_period(res.t, sig) - T) / T < 1e-6


def test_small_angle_modes_closed_form_equal_case():
    # m1=m2=L1=L2=1: w^2 = (g)(2 -/+ sqrt(2)) -> check eigen-solver against it
    omegas, _ = pd.small_angle_frequencies(pd.PendulumParams())
    g = 9.81
    expected = np.sqrt(g * np.array([2 - np.sqrt(2), 2 + np.sqrt(2)]))
    assert np.allclose(omegas, expected, rtol=1e-12)


@pytest.mark.parametrize("p", PARAM_SETS)
def test_energy_conserved_moderate_amplitude(p):
    res = pd.simulate_double_pendulum(p, 0.5, -0.3, 0.0, 0.0, 20.0, 4000)
    assert res.success
    assert res.energy_drift[0] == 0
    assert np.max(res.energy_drift) < 1e-9
    # kinetic + potential really is the total
    assert np.allclose(res.kinetic + res.potential, res.energy)


def test_geometry_consistent_with_rod_lengths():
    p = PARAM_SETS[1]
    res = pd.simulate_double_pendulum(p, 1.0, -0.5, 0.2, 0.1, 5.0, 500)
    assert np.allclose(np.hypot(res.x1, res.y1), p.L1, rtol=1e-12)
    assert np.allclose(np.hypot(res.x2 - res.x1, res.y2 - res.y1), p.L2, rtol=1e-12)


def test_rest_at_equilibrium_stays_at_rest():
    res = pd.simulate_double_pendulum(pd.PendulumParams(), 0.0, 0.0, 0.0, 0.0, 10.0, 200)
    assert np.max(np.abs(res.theta1)) < 1e-15
    assert np.max(np.abs(res.theta2)) < 1e-15
    assert np.max(np.abs(res.omega1)) < 1e-15
    assert np.max(np.abs(res.omega2)) < 1e-15


def test_large_angle_diverges_far_more_than_small_angle():
    p = pd.PendulumParams()
    small = pd.simulate_divergence(p, 0.01, 0.01, 0.0, 0.0, 20.0)
    large = pd.simulate_divergence(p, 2.5, 1.5, 0.0, 0.0, 20.0)
    assert small.separation[0] == pytest.approx(1e-6, rel=1e-9)
    assert large.separation[0] == pytest.approx(1e-6, rel=1e-9)
    # small-angle: stays about the size of the initial perturbation
    assert small.separation.max() < 1e-4
    # large-angle: perturbation grows by many orders of magnitude
    assert large.separation.max() > 1.0
    assert large.separation.max() > 1e3 * small.separation.max()
    # growth-rate ESTIMATE: clearly positive for large angles, near zero for small
    assert large.growth_rate > 0.5
    assert abs(small.growth_rate) < 0.1


def test_growth_rate_estimator_recovers_known_exponential():
    t = np.linspace(0, 5, 500)
    sep = 1e-6 * np.exp(0.8 * t)
    rate, window = pd.estimate_growth_rate(t, sep, sep_max=1e-2)
    assert rate == pytest.approx(0.8, rel=1e-9)
    assert window[0] == 0.0


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        pd.simulate_double_pendulum(pd.PendulumParams(m1=-1.0), 0.1, 0.1, 0, 0, 1.0)
    with pytest.raises(ValueError):
        pd.simulate_double_pendulum(pd.PendulumParams(), float("nan"), 0.1, 0, 0, 1.0)
