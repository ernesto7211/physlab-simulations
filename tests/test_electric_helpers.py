"""Tests for electric-field UI helpers: validation, presets, levels, arrows, CSV."""

import numpy as np
import pytest

from physlab import electric as el

GOOD = dict(charges_uC=[1.0, -1.0], xs=[-1.0, 1.0], ys=[0.0, 0.0], half_width=5.0, r_min=0.2, n_grid=100)


def test_validate_accepts_good_input():
    assert el.validate_inputs(**GOOD) == []


@pytest.mark.parametrize(
    "change",
    [
        dict(charges_uC=[0.0, 0.0]),  # all zero
        dict(charges_uC=[], xs=[], ys=[]),  # no charges
        dict(charges_uC=[1.0] * 7, xs=[0.0] * 7, ys=list(range(7))),  # too many
        dict(charges_uC=[1.0, float("nan")]),
        dict(xs=[-1.0, float("inf")]),
        dict(charges_uC=[1.0, 5000.0]),  # too large
        dict(xs=[-1.0, 5000.0]),  # position out of range
        dict(xs=[1.0, 1.0], ys=[0.0, 0.0]),  # overlapping
        dict(xs=[1.0, 1.000001], ys=[0.0, 0.0]),  # almost overlapping
        dict(half_width=0.0), dict(half_width=float("nan")),
        dict(r_min=0.0), dict(r_min=-1.0), dict(r_min=2.0),  # > 25% of half-width
        dict(n_grid=5), dict(n_grid=10_000),
        dict(charges_uC=[1.0], xs=[0.0, 1.0], ys=[0.0]),  # shape mismatch
    ],
)
def test_validate_rejects_bad_input(change):
    assert el.validate_inputs(**{**GOOD, **change})


def test_zero_charge_can_overlap_a_real_charge():
    # q = 0 is ignored, so sitting on top of another charge is harmless
    args = {**GOOD, "charges_uC": [1.0, 0.0], "xs": [0.0, 0.0], "ys": [0.0, 0.0]}
    assert el.validate_inputs(**args) == []


def test_active_charges_drops_zeros_and_converts_units():
    q, pos = el.active_charges([2.0, 0.0, -3.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    assert np.allclose(q, [2e-6, -3e-6])
    assert np.array_equal(pos, [[1.0, 4.0], [3.0, 6.0]])


def test_charges_outside_window():
    assert el.charges_outside_window([0.0, 6.0, -2.0], [0.0, 0.0, -7.0], 5.0) == [1, 2]
    assert el.charges_outside_window([5.0], [-5.0], 5.0) == []


@pytest.mark.parametrize("name", list(el.PRESETS))
def test_presets_are_valid_and_expected_size(name):
    preset = el.PRESETS[name]
    assert 1 <= len(preset) <= el.MAX_CHARGES
    x, y, q = zip(*preset)
    assert el.validate_inputs(q, x, y, 5.0, 0.2, 100) == []
    assert all(abs(c) > 0 for c in q)


def test_preset_physics_signs():
    assert len(el.PRESETS["Single charge"]) == 1
    assert sum(q for *_, q in el.PRESETS["Dipole"]) == 0
    assert all(q > 0 for *_, q in el.PRESETS["Two like charges"])
    quad = el.PRESETS["Quadrupole"]
    assert len(quad) == 4 and sum(q for *_, q in quad) == 0
    # alternating signs around the square: net dipole moment is zero too
    assert sum(x * q for x, _, q in quad) == 0 and sum(y * q for _, y, q in quad) == 0


def test_equipotential_levels_are_nice_and_within_range():
    X, Y, ex, ey, v = el.field_on_grid([1e-6, -1e-6], [[-1.0, 0.0], [1.0, 0.0]], (-5, 5), (-5, 5), 120, 120, 0.2)
    start, end, size = el.equipotential_levels(v, n_levels=15)
    lo, hi = np.nanpercentile(v, [3, 97])
    assert lo <= start <= end <= hi
    assert size > 0
    mant = size / 10 ** np.floor(np.log10(size))
    assert min(abs(mant - m) for m in (1, 2, 5, 10)) < 1e-9
    n_lines = (end - start) / size
    assert 5 <= n_lines <= 30


def test_equipotential_levels_errors():
    with pytest.raises(ValueError):
        el.equipotential_levels(np.full((3, 3), np.nan))
    with pytest.raises(ValueError):
        el.equipotential_levels(np.ones((3, 3)))


def test_log_magnitude_bins():
    mag = np.array([1.0, 10.0, 100.0, 1000.0, np.nan, 0.0])
    idx, edges = el.log_magnitude_bins(mag, 3)
    assert list(idx[:4]) == [0, 0, 1, 2] or list(idx[:4]) == [0, 1, 2, 2]  # monotone in log10
    assert np.all(np.diff(idx[:4]) >= 0) and idx[0] == 0 and idx[3] == 2
    assert idx[4] == -1 and idx[5] == -1
    assert len(edges) == 4 and edges[0] == pytest.approx(1.0) and edges[-1] == pytest.approx(1000.0)
    assert el.log_magnitude_bins(np.array([np.nan]), 4)[1].size == 0


def test_arrow_segments_geometry():
    xs, ys = el.arrow_segments([0.0, 5.0], [0.0, 5.0], [3.0, np.nan], [4.0, 1.0], length=2.0)
    # the NaN-field arrow is skipped: exactly one arrow = 5 points + separator
    assert xs.shape == (6,) and np.isnan(xs[-1]) and np.isnan(ys[-1])
    tail, tip = np.array([xs[0], ys[0]]), np.array([xs[1], ys[1]])
    assert np.hypot(*(tip - tail)) == pytest.approx(2.0)  # normalised length
    assert np.allclose((tip - tail) / 2.0, [0.6, 0.8])  # direction of (3, 4)
    assert np.allclose((tip + tail) / 2.0, [0.0, 0.0])  # centred on the sample point
    for b in ((xs[2], ys[2]), (xs[4], ys[4])):  # barbs point back, shorter than the shaft
        d = np.hypot(b[0] - tip[0], b[1] - tip[1])
        assert d == pytest.approx(0.3 * 2.0)
        assert (np.array(b) - tip) @ (tip - tail) < 0


def test_arrow_segments_all_masked_is_empty():
    xs, ys = el.arrow_segments([0.0], [0.0], [np.nan], [np.nan], 1.0)
    assert xs.size == 0 and ys.size == 0


def test_grid_dataframe():
    X, Y, ex, ey, v = el.field_on_grid([1e-6], [[0.0, 0.0]], (-1, 1), (-1, 1), 21, 21, 0.15)
    df = el.grid_to_dataframe(X, Y, ex, ey, v)
    assert len(df) == 21 * 21
    assert {"x_m", "y_m", "Ex_V_per_m", "Ey_V_per_m", "E_magnitude_V_per_m", "V_volts", "masked"} <= set(df.columns)
    assert df["masked"].sum() == np.isnan(v).sum() > 0
    assert df.loc[df["masked"], "V_volts"].isna().all()
    csv = df.to_csv(index=False)
    assert "inf" not in csv.lower()
    # a finite row has a self-consistent magnitude
    row = df[~df["masked"]].iloc[0]
    assert row["E_magnitude_V_per_m"] == pytest.approx(np.hypot(row["Ex_V_per_m"], row["Ey_V_per_m"]))
