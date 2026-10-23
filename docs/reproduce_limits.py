"""Reproduce the numbers quoted in docs/numerical_limitations.md.

Run from the repository root:

    python docs/reproduce_limits.py

Everything printed here is computed by calling the library code in physlab/.
Results are for one run on one machine (wall-clock times in particular vary);
the physical/numerical quantities (drifts, separations, counts) are deterministic.
Takes roughly one to two minutes.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from physlab import orbital as ob, pendulum as pn, electric as el

M1, M2 = 5.972e24, 1000.0
print("=== ORBITAL ===")
r0 = 7.0e6
vc = ob.circular_speed(M1, M2, r0)
def run(factor, periods, rtol=ob.DEFAULT_RTOL, n=20000, r0=r0):
    v0 = factor * ob.circular_speed(M1, M2, r0)
    a = ob.semi_major_axis(M1, M2, r0, v0)
    T = ob.kepler_period(M1, M2, a)
    t0 = time.time()
    res = ob.simulate_two_body(M1, M2, [r0, 0], [0, v0], periods * T, n, rtol=rtol)
    el_ = ob.orbit_elements(M1, M2, [r0, 0], [0, v0])
    Ld = np.max(np.abs(res.angular_momentum - res.angular_momentum[0]) / abs(res.angular_momentum[0]))
    return res, el_, Ld, time.time() - t0
for f, p in [(1.0, 3), (1.0, 200), (1.2, 5), (1.2, 200), (1.35, 5)]:
    res, e, Ld, dt = run(f, p)
    print(f"v={f}*vc periods={p}: e={e.eccentricity:.4f} max E drift={res.energy_drift.max():.2e} max L drift={Ld:.2e} success={res.success} wall={dt:.2f}s")
# high eccentricity: start at periapsis with v = f vc; e = f^2-1 (f<sqrt2). e=0.9 -> f=sqrt(1.9)
for e_target in (0.9, 0.99, 0.999):
    f = np.sqrt(1 + e_target)
    res, e, Ld, dt = run(f, 5, n=40000)
    pr = e.periapsis / e.apoapsis
    errs = ob.validate_inputs(M1, M2, r0, f * vc, 5 * e.period, 1000)
    print(f"e={e.eccentricity:.4f} peri/apo={pr:.2e} max E drift={res.energy_drift.max():.2e} L drift={Ld:.2e} success={res.success} wall={dt:.2f}s validate_errors={errs}")
# tolerance sensitivity, e=0.44, 5 periods
for rt in (1e-6, 1e-8, 1e-10, 1e-12):
    res, e, Ld, dt = run(1.2, 5, rtol=rt)
    print(f"rtol={rt:g}: max E drift={res.energy_drift.max():.2e}, L drift={Ld:.2e}")
# validate threshold: periapsis ratio 1e-3 of r0 starting at apoapsis? We start at r0 tangentially; periapsis < r0 only if v<vc
for f in (0.1, 0.05, 0.03, 0.02):
    errs = ob.validate_inputs(M1, M2, r0, f * vc, 3600.0, 1000)
    e = ob.orbit_elements(M1, M2, [r0, 0], [0, f * vc])
    print(f"v={f}vc periapsis/r0={e.periapsis/r0:.2e} e={e.eccentricity:.5f} rejected={bool(errs)}")
# parabolic drift normalisation
v0 = ob.escape_speed(M1, M2, r0)
res = ob.simulate_two_body(M1, M2, [r0, 0], [0, v0], 6 * 3600.0, 3000)
d = np.abs(res.energy - res.energy[0]) / abs(res.potential[0])
print(f"parabolic: E0/U0={res.energy[0]/res.potential[0]:.2e} max |dE|/|U0|={d.max():.2e}")
print("G =", ob.G, "units: SI")

print("=== PENDULUM ===")
p = pn.PendulumParams()
t0 = time.time()
r = pn.simulate_double_pendulum(p, np.radians(120), np.radians(-10), 0, 0, 20.0, 4000)
print(f"default 120/-10 deg 20s: max drift={r.energy_drift.max():.2e} wall={time.time()-t0:.2f}s")
for T in (20, 60):
    r = pn.simulate_double_pendulum(p, np.radians(120), np.radians(-10), 0, 0, float(T), 8000)
    print(f"default {T}s: max drift={r.energy_drift.max():.2e}")
for th in [(10, 5), (60, 30), (120, -10), (170, 170), (179, 179)]:
    r = pn.simulate_double_pendulum(p, np.radians(th[0]), np.radians(th[1]), 0, 0, 60.0, 8000)
    print(f"theta=({th[0]},{th[1]}) deg 60s: max drift={r.energy_drift.max():.2e} success={r.success}")
# divergence timing
d = pn.simulate_divergence(p, np.radians(120), np.radians(-10), 0, 0, 20.0, 8000)
sep = d.separation
def first(th):
    i = np.argmax(sep > th)
    return d.t[i] if sep[i] > th else None
print(f"default perturbation 1e-6: first t sep>1e-3={first(1e-3)}, >1e-2={first(1e-2)}, >1e-1={first(1e-1)}, >1={first(1.0)}; growth_rate={d.growth_rate:.3f} window={d.fit_window}")
for sm in (1e-4, 1e-3, 1e-2, 1e-1):
    rate, w = pn.estimate_growth_rate(d.t, sep, sep_max=sm)
    print(f"  sep_max={sm:g}: rate={rate:.3f} window=({w[0]:.2f},{w[1]:.2f})")
# initial-condition dependence of rate
for th in [(120, -10), (90, 90), (150, 30), (30, 30), (10, 5)]:
    dd = pn.simulate_divergence(p, np.radians(th[0]), np.radians(th[1]), 0, 0, 20.0, 8000)
    print(f"  theta=({th[0]},{th[1]}) growth_rate={dd.growth_rate:.3f} max sep={dd.separation.max():.2e}")
# integrator tolerance vs chaotic trajectory: when does solution with looser tol depart from rtol=1e-12 one
ref = pn.simulate_double_pendulum(p, np.radians(120), np.radians(-10), 0, 0, 30.0, 6001, rtol=1e-13, atol=1e-13)
for rt in (1e-8, 1e-10, 1e-12):
    o = pn.simulate_double_pendulum(p, np.radians(120), np.radians(-10), 0, 0, 30.0, 6001, rtol=rt, atol=rt)
    sepi = np.hypot(o.theta1 - ref.theta1, o.theta2 - ref.theta2)
    def f(th):
        i = np.argmax(sepi > th)
        return round(float(o.t[i]), 2) if sepi[i] > th else None
    print(f"rtol=atol={rt:g} vs 1e-13 reference (30 s): first t sep>1e-3: {f(1e-3)}, >1e-1: {f(1e-1)}, final sep={sepi[-1]:.2e}, max E drift={o.energy_drift.max():.2e}")
# small-angle (non chaotic) comparison
ref = pn.simulate_double_pendulum(p, 0.1, 0.05, 0, 0, 30.0, 6001, rtol=1e-13, atol=1e-13)
o = pn.simulate_double_pendulum(p, 0.1, 0.05, 0, 0, 30.0, 6001, rtol=1e-8, atol=1e-8)
print(f"non-chaotic (0.1,0.05 rad) rtol 1e-8 vs 1e-13: max sep over 30 s = {np.hypot(o.theta1-ref.theta1,o.theta2-ref.theta2).max():.2e}")

print("=== ELECTRIC ===")
print(f"k={el.K_COULOMB:.6e} eps0={el.EPSILON_0:.10e}")
q, rmin = 1e-6, 0.2
print(f"1 uC at r_min=0.2 m: E={el.K_COULOMB*q/rmin**2:.3e} V/m, V={el.K_COULOMB*q/rmin:.3e} V")
for name, pre in el.PRESETS.items():
    x, y, qq = zip(*pre)
    qC, pos = el.active_charges(qq, x, y)
    for n in (160,):
        X, Y, ex, ey, v = el.field_on_grid(qC, pos, (-5, 5), (-5, 5), n, n, 0.2)
        masked = np.isnan(v).sum()
        lo, hi = np.nanpercentile(v, [3, 97])
        t0 = time.time()
        lines = el.trace_field_lines(qC, pos, (-5, 5, -5, 5), r_min=0.2, lines_per_charge=16)
        dt = time.time() - t0
        reasons = {}
        for l in lines:
            reasons[l.end_reason] = reasons.get(l.end_reason, 0) + 1
        print(f"{name}: grid {n}x{n} masked={masked} ({100*masked/v.size:.2f}%), V range clipped to [{lo:.3g},{hi:.3g}] V, "
              f"max|V| finite={np.nanmax(np.abs(v)):.3g}; lines={len(lines)} reasons={reasons} trace wall={dt:.2f}s")
# grid resolution effect on masking fraction
qC, pos = el.active_charges([1.0], [0.0], [0.0])
for n in (40, 80, 160, 300):
    X, Y, ex, ey, v = el.field_on_grid(qC, pos, (-5, 5), (-5, 5), n, n, 0.2)
    print(f"single charge n={n}: masked={np.isnan(v).sum()} expected~pi*r^2/dx^2={np.pi*0.04/((10/(n-1))**2):.1f}")
# gradient vs -E error at different h (reuse test geometry)
def graderr(h):
    qs, ps = [3e-6, -2e-6], [[-1.0, 0.0], [1.5, 0.5]]
    xs = np.arange(-6.0, 6.0 + h / 2, h); ys = np.arange(-5.0, 5.0 + h / 2, h)
    X, Y = np.meshgrid(xs, ys)
    ex, ey, v = el.field_at_points(qs, ps, X, Y, 0.0)
    dVdy, dVdx = np.gradient(v, h)
    dmin = np.min([np.hypot(X - p_[0], Y - p_[1]) for p_ in ps], axis=0)
    ok = dmin >= 2.0; ok[[0, -1], :] = False; ok[:, [0, -1]] = False
    return (np.hypot(-dVdx - ex, -dVdy - ey)[ok] / np.hypot(ex, ey)[ok]).max()
for h in (0.1, 0.05, 0.025):
    print(f"central-difference grad V vs -E, h={h}: max rel err={graderr(h):.2e}")
# near-charge gradient error (h=0.05, points 0.3 m from charge)
qs, ps = [3e-6], [[0.0, 0.0]]
h = 0.05
xs = np.arange(-2, 2 + h/2, h); X, Y = np.meshgrid(xs, xs)
ex, ey, v = el.field_at_points(qs, ps, X, Y, 0.0)
dVdy, dVdx = np.gradient(v, h)
dd = np.hypot(X, Y)
for rlo, rhi in ((0.2, 0.3), (0.5, 0.7), (1.0, 1.5)):
    m = (dd >= rlo) & (dd < rhi); m[[0, -1], :] = False; m[:, [0, -1]] = False
    e = np.hypot(-dVdx - ex, -dVdy - ey)[m] / np.hypot(ex, ey)[m]
    print(f"  near-charge central-difference error h=0.05, r in [{rlo},{rhi}): max rel err={e.max():.2e}")
# field-line endpoint accuracy vs rtol
qC, pos = el.active_charges([1.0, -1.0], [-1.0, 1.0], [0.0, 0.0])
for rt in (1e-5, 1e-7, 1e-9):
    lines = el.trace_field_lines(qC, pos, (-5, 5, -5, 5), r_min=0.05, lines_per_charge=16, start_radius=0.1, stop_radius=0.05, rtol=rt)
    ends = [np.hypot(l.x[-1] - 1.0, l.y[-1]) for l in lines if l.end_reason == "negative_charge"]
    print(f"rtol={rt:g}: {len(ends)} lines reach negative charge, end radius min/max={min(ends):.5f}/{max(ends):.5f} (stop radius 0.05)")
# field line lengths: dipole lines reaching negative vs total, and default max_length
lines = el.trace_field_lines(qC, pos, (-5, 5, -5, 5), r_min=0.2, lines_per_charge=16)
print("dipole default settings: lines", len(lines), "max length reached", max(l.length for l in lines), "max_length default", 4*np.hypot(10, 10))
# float64 magnitude: field at very large distance (no overflow) sanity
ex, ey, v = el.field_at_points([1e-6], [[0.0, 0.0]], [1e6], [0.0])
print("E at 1e6 m:", ex[0], "V:", v[0])

print()
print("=== EXTRA CHECKS ===")

qC, pos = el.active_charges([1.0, -1.0], [-1.0, 1.0], [0.0, 0.0])
kw = dict(r_min=0.2, lines_per_charge=16)
ref = el.trace_field_lines(qC, pos, (-5, 5, -5, 5), rtol=1e-12, atol=1e-14, **kw)
print("polyline sample spacing (default): ", 4 * np.hypot(10, 10) / 399, "m")
for rt in (1e-5, 1e-7, 1e-9):
    got = el.trace_field_lines(qC, pos, (-5, 5, -5, 5), rtol=rt, **kw)
    worst = 0.0
    for a, b in zip(ref, got):
        # compare the two polylines by distance from each point of b to the curve a (nearest point)
        pa = np.column_stack([a.x, a.y])
        for p in np.column_stack([b.x, b.y]):
            d = np.min(np.hypot(pa[:, 0] - p[0], pa[:, 1] - p[1]))
            worst = max(worst, d)
    print(f"rtol={rt:g} vs 1e-12: max nearest-point distance between polylines = {worst:.2e} m "
          f"(sample spacing makes this a loose upper bound ~ spacing/2 = {4*np.hypot(10,10)/399/2:.2e})")
    print("   same end reasons:", [l.end_reason for l in ref] == [l.end_reason for l in got])

# orbital: how much does output sampling hide? drift at n=500 vs n=20000, e=0.44, 5 periods
M1, M2, r0 = 5.972e24, 1000.0, 7.0e6
v0 = 1.2 * ob.circular_speed(M1, M2, r0)
T = ob.kepler_period(M1, M2, ob.semi_major_axis(M1, M2, r0, v0))
for n in (200, 3000, 20000, 200000):
    res = ob.simulate_two_body(M1, M2, [r0, 0], [0, v0], 5 * T, n)
    print(f"orbital e=0.44 5 periods n_out={n}: max E drift at samples={res.energy_drift.max():.2e}")
# orbital: atol effect: atol 1e-6 vs 1e-9 vs 1e-3 at rtol 1e-12
for at in (1e-3, 1e-6, 1e-9):
    res = ob.simulate_two_body(M1, M2, [r0, 0], [0, v0], 5 * T, 20000, atol=at)
    print(f"atol={at:g}: max E drift {res.energy_drift.max():.2e}")
# distance scale: Earth-Moon circular orbit drift over 200 periods
r = 3.844e8
vm = ob.circular_speed(M1, 7.348e22, r)
Tm = ob.kepler_period(M1, 7.348e22, r)
res = ob.simulate_two_body(M1, 7.348e22, [r, 0], [0, vm], 200 * Tm, 20000)
print(f"Earth-Moon-like circular 200 periods: max E drift {res.energy_drift.max():.2e}")
# page limits
print("MAX_PERIODS", ob.MAX_PERIODS, "PARABOLIC_TOL", ob.PARABOLIC_TOL)
