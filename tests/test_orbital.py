"""Validation tests for physlab.orbital.

Stated tolerances (tight DOP853, rtol=1e-12):
  * circular radius constant to 1e-8 relative
  * period vs Kepler's third law to 1e-6 relative
  * energy drift < 1e-7, angular momentum drift < 1e-7 (relative; e=0.44
    orbit, 5 periods; observed energy drift peaks ~2e-8 near pericentre)
  * escape: |E| / |KE| < 1e-9 at start and end
"""

import numpy as np

from physlab import orbital as ob

M1 = 5.972e24  # kg (Earth-like primary)
M2 = 7.348e22  # kg (Moon-like secondary)
R0 = 3.844e8  # m


def _circular(n=4000, periods=3):
    v = ob.circular_speed(M1, M2, R0)
    T = ob.kepler_period(M1, M2, R0)
    return ob.simulate_two_body(M1, M2, [R0, 0], [0, v], periods * T, n), T


def test_circular_orbit_radius_constant():
    res, _ = _circular()
    assert res.success
    r = np.linalg.norm(res.position, axis=1)
    assert np.max(np.abs(r - R0)) / R0 < 1e-8


def test_period_matches_kepler_third_law():
    # Eccentric orbit: start at pericentre with 1.2x circular speed.
    r0 = 1.0e7
    v0 = 1.2 * ob.circular_speed(M1, M2, r0)
    a = ob.semi_major_axis(M1, M2, r0, v0)
    T = ob.kepler_period(M1, M2, a)
    res = ob.simulate_two_body(M1, M2, [r0, 0], [0, v0], 3 * T, 30000)
    theta = np.unwrap(np.arctan2(res.position[:, 1], res.position[:, 0]))
    # time at which swept angle first reaches 2*pi (linear interpolation)
    i = np.argmax(theta >= 2 * np.pi)
    frac = (2 * np.pi - theta[i - 1]) / (theta[i] - theta[i - 1])
    T_meas = res.t[i - 1] + frac * (res.t[i] - res.t[i - 1])
    assert abs(T_meas - T) / T < 1e-6


def test_conservation_of_energy_and_angular_momentum():
    r0 = 1.0e7
    v0 = 1.2 * ob.circular_speed(M1, M2, r0)
    T = ob.kepler_period(M1, M2, ob.semi_major_axis(M1, M2, r0, v0))
    res = ob.simulate_two_body(M1, M2, [r0, 0], [0, v0], 5 * T, 5000)
    assert res.energy_drift[0] == 0
    assert np.max(res.energy_drift) < 1e-7
    L_drift = np.abs(res.angular_momentum - res.angular_momentum[0]) / abs(res.angular_momentum[0])
    assert np.max(L_drift) < 1e-7


def test_escape_velocity_is_parabolic_and_unbound():
    r0 = 7.0e6
    v0 = ob.escape_speed(M1, M2, r0)
    res = ob.simulate_two_body(M1, M2, [r0, 0], [0, v0], 2.0e5, 3000)
    ke0 = res.kinetic[0]
    assert abs(res.energy[0]) / ke0 < 1e-9
    assert abs(res.energy[-1]) / ke0 < 1e-9
    r = np.linalg.norm(res.position, axis=1)
    assert np.all(np.diff(r) > 0)  # moving outward, never returns
    # Parabolic: speed decays toward zero, v = sqrt(2 mu / r) throughout
    v = np.linalg.norm(res.velocity, axis=1)
    assert np.allclose(v, np.sqrt(2 * res.mu / r), rtol=1e-8)
