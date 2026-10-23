"""Tests for pendulum UI helpers: validation, frame selection, CSV tabulation."""

import numpy as np
import pytest

from physlab import pendulum as pn

GOOD = dict(m1=1.0, m2=1.0, L1=1.0, L2=1.0, g=9.81, theta1_0=1.0, theta2_0=0.5,
            t_end=10.0, n_points=1000, delta=1e-6)


def test_validate_accepts_good_input():
    assert pn.validate_inputs(**GOOD) == []


@pytest.mark.parametrize(
    "change",
    [
        dict(m1=0.0), dict(m2=-1.0), dict(L1=0.0), dict(L2=float("inf")), dict(g=-9.81),
        dict(theta1_0=float("nan")), dict(theta2_0=None),
        dict(t_end=0.0), dict(t_end=pn.MAX_DURATION + 1), dict(t_end=float("nan")),
        dict(n_points=1), dict(n_points=pn.MAX_OUTPUT_POINTS + 1),
        dict(delta=0.0), dict(delta=1e-12), dict(delta=5.0), dict(delta=float("nan")),
    ],
)
def test_validate_rejects_bad_input(change):
    assert pn.validate_inputs(**{**GOOD, **change})


def test_validate_accepts_negative_perturbation():
    assert pn.validate_inputs(**{**GOOD, "delta": -1e-6}) == []


@pytest.mark.parametrize("n, m", [(1, 300), (2, 300), (299, 300), (300, 300), (4000, 300), (10000, 50)])
def test_frame_indices(n, m):
    idx = pn.animation_frame_indices(n, m)
    assert idx[0] == 0 and idx[-1] == n - 1
    assert len(idx) <= m
    assert np.all(np.diff(idx) > 0)
    if n <= m:
        assert len(idx) == n


def test_frame_indices_rejects_bad_args():
    with pytest.raises(ValueError):
        pn.animation_frame_indices(0, 10)
    with pytest.raises(ValueError):
        pn.animation_frame_indices(10, 1)


def test_dataframe_contents():
    p = pn.PendulumParams()
    div = pn.simulate_divergence(p, 1.0, 0.5, 0.0, 0.0, 2.0, 100, delta_theta1=1e-6)
    df = pn.divergence_to_dataframe(div)
    assert len(df) == 100
    for col in ("t_s", "theta1_ref_rad", "theta1_pert_rad", "x2_ref_m", "y2_pert_m",
                "total_energy_ref_J", "relative_energy_drift_pert", "angular_separation_rad"):
        assert col in df.columns
    assert df["angular_separation_rad"].iloc[0] == pytest.approx(1e-6, rel=1e-9)
    assert np.allclose(df["angular_separation_rad"],
                       np.hypot(df["theta1_pert_rad"] - df["theta1_ref_rad"],
                                df["theta2_pert_rad"] - df["theta2_ref_rad"]))
    assert "angular_separation_rad" in df.to_csv(index=False)
