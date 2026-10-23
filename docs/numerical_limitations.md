# Numerical limitations of PhysLab

This document describes, for each simulation, where the numerical error comes
from, which tolerances the code uses, and where the results should not be
trusted. Every number below was produced by running the code in `physlab/`
(script: [`reproduce_limits.py`](reproduce_limits.py)). They come from **one run
on one machine** (Windows, Python 3.12.10, NumPy 2.5.4, SciPy 1.18.1). Drifts,
separations and counts are deterministic for that software stack; wall-clock times are not
and are only rough indications. Nothing here is compared with laboratory measurements: the
checks are against analytic formulas, conserved quantities, and runs with
tighter tolerances.

The models are standard textbook models; nothing here is claimed to be new.

---

## 1. Orbital mechanics (`physlab/orbital.py`)

### Model and solver settings

- Relative-coordinate two-body problem, `r'' = -mu r/|r|^3`, `mu = G(m1+m2)`, `G = 6.6743e-11` (CODATA 2018), SI units, planar.
- `scipy.integrate.solve_ivp`, method DOP853 (explicit Runge–Kutta, order 8(5,3)), `rtol = 1e-12`, `atol = 1e-6`.
  `atol` is applied to all four state components (metres for position, m/s for velocity).
- Output is sampled from the solver's dense output at `n_points` uniformly spaced times.
- Energy and angular momentum are those of the reduced-mass motion; relative energy drift is `|E(t) - E(0)| / |E(0)|`.

### Error sources

1. **Truncation error of the integrator.** Controlled by `rtol` and `atol`. The error-control test is local (per step), so the global error accumulates over many orbits.
2. **The absolute tolerance, not the relative one, limits accuracy in the default configuration.**
   With `rtol = 1e-12`, the allowed error per component is roughly `rtol*|y| + atol`.
   For the 7000 km Earth-mass example below the circular speed is about 7.5 km/s, so `rtol*|v|` is about 7.5e-9 m/s, which is smaller than `atol = 1e-6` m/s (for the position components, `rtol*|r|` is about 7e-6 m, larger than `atol`). The measurements below are consistent with this: lowering `atol` lowers the drift by orders of magnitude, while lowering `rtol` alone does not (see the sweeps).
   (This explanation is my inference from the sweeps and the tolerance formula; I did not instrument the step-size controller.)
3. **Floating-point round-off.** Present at the level of about 1e-12 and below in the circular case.
4. **Output sampling.** Drift is reported only at the output times; a peak between samples is not seen (see the sampling table).
5. **Model error.** Point masses, Newtonian gravity, two bodies only, no body radius, no relativity, no drag, planar motion. These are modelling choices, not numerical errors, but they limit what the results mean.

### Measured drift (Earth-mass primary, 1000 kg secondary, start at r0 = 7000 km, tangential velocity)

Maximum drift over the run, sampled at 20,000 output points unless noted.

| Case | Duration | Max E drift | Max L drift |
|---|---|---|---|
| Circular (v = v_c, e = 0) | 3 periods | 1.86e-12 | 9.29e-13 |
| Circular | 200 periods | 4.28e-12 | 2.14e-12 |
| e = 0.44 (v = 1.2 v_c) | 5 periods | 1.06e-8 | 7.48e-9 |
| e = 0.44 | 200 periods | 1.05e-8 | 1.41e-8 |
| e = 0.8225 (v = 1.35 v_c) | 5 periods | 1.09e-9 | 1.45e-10 |
| Earth–Moon-like circular (r = 3.844e8 m, m2 = 7.348e22 kg) | 200 periods | 3.53e-12 | not computed |

Highly eccentric orbits, starting at periapsis (40,000 output points, 5 periods):

| Eccentricity | periapsis / apoapsis | Max E drift | Max L drift |
|---|---|---|---|
| 0.9 | 5.26e-2 | 2.37e-9 | 1.37e-10 |
| 0.99 | 5.03e-3 | 8.51e-9 | 1.21e-10 |
| 0.999 | 5.00e-4 | 6.45e-8 | 1.37e-10 |

The 200-period rows have only 100 output samples per period, and as the sampling table below shows, sparse sampling can slightly under-report the peak drift.

Energy drift grows with eccentricity (stronger acceleration at periapsis), while angular-momentum drift stays near 1e-10 in these cases.
In the e = 0.44 case the drift at 200 periods is about the same as at 5 periods (1.05e-8 vs 1.06e-8), so no growth with orbit count was visible at this sampling.

### Sensitivity to the tolerances (e = 0.44, 5 periods)

`rtol` sweep with `atol = 1e-6`:

| rtol | Max E drift | Max L drift |
|---|---|---|
| 1e-6 | 2.08e-5 | 4.39e-6 |
| 1e-8 | 1.71e-7 | 9.19e-8 |
| 1e-10 | 5.68e-9 | 4.56e-9 |
| 1e-12 (default) | 1.06e-8 | 7.48e-9 |

The drift is **not monotonic** in `rtol`: tightening from 1e-10 to 1e-12 did not reduce it.

`atol` sweep with `rtol = 1e-12`:

| atol | Max E drift |
|---|---|
| 1e-3 | 1.72e-7 |
| 1e-6 (default) | 1.06e-8 |
| 1e-9 | 3.72e-11 |

So the default `atol` is a limiting factor for this kind of orbit. The default drift (about 1e-8) is small compared with anything visible in the plots and is within the tolerances used in the tests (1e-7 for the eccentric case), but it is not as small as "rtol = 1e-12" might suggest. I have not changed the code defaults.

### Output sampling (e = 0.44, 5 periods)

| Output points | Max E drift seen |
|---|---|
| 200 | 8.62e-9 |
| 3,000 | 1.04e-8 |
| 20,000 | 1.06e-8 |
| 200,000 | 1.06e-8 |

With few output points the maximum drift is slightly under-reported.
The page default is 3,000 points.

### Where results become unreliable

- **Close approaches.** The acceleration scales as `1/r^2`, so orbits that pass very close to the primary need very small steps. The page rejects inputs whose periapsis is below 0.1% of the *initial distance* (`validate_inputs`). In this check, `v = 0.05 v_c` (periapsis/r0 = 1.25e-3, e = 0.9975) was accepted and `v = 0.03 v_c` (periapsis/r0 = 4.5e-4, e = 0.9991) was rejected.
  This rule is relative to the starting distance, so an orbit that *starts at* periapsis and has a huge apoapsis is not rejected (the e = 0.999 row above was accepted, with a drift of 6.45e-8).
  The cut-off value is a judgement call, not derived from a measured failure of the solver.
- **Physically impossible periapsis.** Bodies are point masses with no radius, and nothing checks collision with a real body. For the Earth-mass example, `v = 0.1 v_c` from 7000 km gives a periapsis of about 35 km (5.03e-3 × 7000 km), far inside the real Earth. The simulation completes normally; the result is mathematically valid for point masses and physically meaningless for a real planet.
- **Parabolic orbits.** At exactly escape speed `E(0)` is zero (it came out as exactly 0 for the case tried), so `|E(t) - E(0)| / |E(0)|` is undefined. The orbital page normalises by `|U(0)|` in this case; the measured maximum was 9.78e-11 over 6 hours from 7000 km (3,000 samples). The classification uses `|E| <= 1e-9 * |U|` as "parabolic" (`PARABOLIC_TOL`), so a speed within about 5e-10 (relative) of escape speed is reported as parabolic.
- **Very long runs.** The page limits bound orbits to 200 periods (`MAX_PERIODS`); 200 periods of the circular case took about 5 seconds here. Beyond the tested range I have no drift data.
- **Inputs outside the tested range.** Mass ratios, distances and speeds other than those above (e.g. extreme mass ratios, speeds far above escape speed) were not systematically swept. The tests cover Earth-like and Earth–Moon-like parameters.

---

## 2. Double pendulum (`physlab/pendulum.py`)

### Model and solver settings

- Lagrangian equations of motion for two point masses on ideal massless rigid rods, no friction, planar. The 2×2 mass-matrix system is solved directly at each step (the determinant is positive for any `m1 > 0`).
- DOP853 with `rtol = atol = 1e-12`.
- Energy drift is `|E(t) - E(0)| / |E(0)|` (falls back to the characteristic scale `(m1+m2) g L1 + m2 g L2` if `E(0)` is essentially zero).
- Separation between two runs is `sqrt(dtheta1^2 + dtheta2^2)`; angles are not wrapped, so it can exceed π.
- The app limits the duration to 60 s and the perturbation to magnitudes between 1e-10 and 1 rad.

### Error sources

1. **Truncation error** of the integrator, as above.
2. **Round-off.**
3. **Chaos amplifies both of them.** For large initial angles, any perturbation, including the integrator's own error and round-off, grows roughly exponentially until it saturates at order 1 rad (see below). After that, a computed path is only one possible path consistent with the tolerance.
4. **Model error.** Real pendulums have friction, rod mass and flexibility; none is modelled.

### Measured energy drift (equal masses and lengths, g = 9.81, released from rest)

| Initial angles (θ1, θ2) | Duration | Max E drift |
|---|---|---|
| (10°, 5°) | 60 s | 1.96e-13 |
| (60°, 30°) | 60 s | 3.08e-12 |
| (120°, −10°) (page default) | 20 s | 2.22e-9 (4,000 points) |
| (120°, −10°) | 60 s | 7.17e-9 (8,000 points) |
| (170°, 170°) | 60 s | 7.06e-11 |
| (179°, 179°) | 60 s | 7.31e-11 |

Energy drift is small in all of these, but **small energy drift does not mean the trajectory is accurate.** A chaotic trajectory can conserve energy well and still be wrong in phase (next section).

### How long is the default case predictable?

Default case (120°, −10°), second pendulum perturbed by 1e-6 rad in θ1 (8,000 output points over 20 s).
Time at which the separation first exceeds a threshold:

| Separation threshold | First time |
|---|---|
| 1e-3 rad | 6.52 s |
| 1e-2 rad | 8.94 s |
| 1e-1 rad | 11.70 s |
| 1 rad | 14.28 s |

Effect of the solver tolerance on the same trajectory (30 s, compared against a run with `rtol = atol = 1e-13`; that reference is itself a numerical solution, not exact truth):

| Tolerance (rtol = atol) | Time sep > 1e-3 rad | Time sep > 0.1 rad | Final separation at 30 s | Max E drift |
|---|---|---|---|---|
| 1e-8 | 11.78 s | 16.39 s | 7.18 rad | 2.68e-5 |
| 1e-10 | 18.71 s | 22.44 s | 15.2 rad | 3.41e-7 |
| 1e-12 (app default) | 24.74 s | 28.73 s | 0.294 rad | 3.25e-9 |

So for this initial condition, with the app's default tolerance, the computed path departs from a tighter-tolerance path by 0.1 rad at about 28.7 s, and the app allows durations up to 60 s. **Beyond roughly 25 to 30 s, the plotted trajectory for this initial condition should not be treated as the true trajectory.**
That threshold is specific to this initial condition and to this comparison; other initial conditions will differ.

For a non-chaotic case (0.1, 0.05 rad), `rtol = 1e-8` vs `1e-13` differed by at most 5.60e-8 rad over 30 s.

### The growth-rate estimate

The page's "early growth rate" is the least-squares slope of `ln(separation)` versus time over the initial stretch while the separation is below `sep_max` (default 1e-2 rad), with at least 5 samples. It is **not** a Lyapunov exponent: it is finite-time, uses a single pair of trajectories, and has no renormalisation.

Dependence on the cut-off (default case):

| sep_max | Rate (1/s) | Fit window |
|---|---|---|
| 1e-4 | 0.772 | 0 to 5.88 s |
| 1e-3 | 0.877 | 0 to 6.51 s |
| 1e-2 (default) | 1.022 | 0 to 8.93 s |
| 1e-1 | 1.000 | 0 to 11.70 s |

Dependence on the initial condition (`sep_max = 1e-2`, 20 s):

| Initial (θ1, θ2) | Rate (1/s) | Max separation (rad) |
|---|---|---|
| (120°, −10°) | 1.022 | 15.0 |
| (90°, 90°) | 0.539 | 2.05 |
| (150°, 30°) | 1.401 | 51.6 |
| (30°, 30°) | 0.029 | 2.34e-6 |
| (10°, 5°) | 0.015 | 1.84e-6 |

The estimate varies by tens of percent with the window and by a factor of about two or more between initial conditions. For the near-linear small-angle cases it returns small positive values (0.015 to 0.029 per second) even though the separation there only grows to a few microradians, so a small positive rate on its own does not indicate chaos.

### Where results become unreliable

- Large initial angles (as in the default case), after the separation plot has climbed to order 1 rad: see the table above.
- Inputs near the upper duration limit (60 s) for chaotic cases.
- Perturbations at or below the level of the integrator error (the app rejects perturbations under 1e-10 rad; with tolerance 1e-12 the effective noise floor is not exactly known, and I did not measure it).
- Any use of the energy-drift curve as proof of accuracy.

---

## 3. Electric fields (`physlab/electric.py`)

### Model and settings

- Static point charges in vacuum at 2D positions, full 3D Coulomb law, evaluated in the plane z = 0. `k = 1/(4 pi eps0)`, `eps0 = 8.8541878128e-12 F/m` (CODATA 2018), giving `k = 8.987552e9 N m^2/C^2`.
- E and V are **closed-form sums** (no time stepping), so there is no integration error in them; the error sources are floating-point round-off, grid sampling, masking, and field-line tracing.
- Field lines: DOP853 on `dr/ds = ±E/|E|`, `rtol = 1e-9`, `atol = 1e-12`, `max_step = diagonal/200`, events for window exit, negative charges, and null field.

### Singularity masking

- Points closer than `r_min` to any charge (or exactly at a charge) are set to NaN for Ex, Ey, V. They are not clipped, softened or interpolated.
- `r_min` is a display parameter. The largest values left on the grid are bounded by `r_min`: for a 1 µC charge and the page default `r_min = 0.2 m`, `E = 2.247e5 V/m` and `V = 4.494e4 V` at the edge of the mask.
- The number of masked grid points depends on how the grid happens to sit relative to the charge. Single 1 µC charge at the origin, `r_min = 0.2 m`, window ±5 m:

| Grid points per side | Masked points | Disc area / cell area (estimate) |
|---|---|---|
| 40 | 4 | 1.9 |
| 80 | 4 | 7.8 |
| 160 | 32 | 31.8 |
| 300 | 112 | 112.3 |

At coarse grids the mask can differ noticeably from the true disc (the 80-point grid masks only 4 points where the area estimate says about 8).

Masked fraction on the default 160 × 160 grid (window ±5 m, `r_min = 0.2 m`) for the page presets: single charge 32 points (0.12%), dipole 64 (0.25%), two like charges 64 (0.25%), quadrupole 128 (0.50%).

Nothing inside the masked region is evaluated, so the page cannot say anything about the field at distances below `r_min`. Charges that are close together (closer than `2 r_min`) have overlapping masks; the page does not reject this.

### Contour range clipping

Because V diverges at charges, the page contours only the range between the 3rd and 97th percentile of the finite grid values. By construction, about 6% of the finite grid values lie outside the contoured range and get no contour line. Example ranges on the default grid (volts): single charge [1.43e3, 8.99e3] (maximum |V| on the grid 4.04e4); dipole [−6.01e3, 6.01e3] (3.9e4); two like charges [2.91e3, 1.55e4] (4.54e4); quadrupole [−8.6e3, 8.6e3] (3.89e4).
Contour lines are drawn from grid values by Plotly (piecewise-linear interpolation), so features smaller than a grid cell are not resolved.

### Grid derivative error (resolution check)

The page does not differentiate V on the grid (E is computed analytically), but the test `test_gradient_of_potential_matches_minus_field` uses central differences of V on a grid as a consistency check, and the error indicates how well a grid of a given spacing resolves V.

Two charges (3 µC at (−1, 0), −2 µC at (1.5, 0.5)), points at least 2 m from any charge, maximum relative error of `-grad V` vs `E`:

| Grid spacing h | Max relative error |
|---|---|
| 0.1 m | 3.48e-3 |
| 0.05 m | 8.69e-4 |
| 0.025 m | 2.17e-4 |

The error falls by about 4× per halving, as expected for second-order central differences.
Near a single 3 µC charge (h = 0.05 m), the error grows rapidly as you approach it: 2.51e-3 for r in [1.0, 1.5) m, 1.01e-2 for r in [0.5, 0.7) m, 6.67e-2 for r in [0.2, 0.3) m.
So the grid does not resolve V well within a few tenths of a metre of a charge at the default window size.

### Field-line tracing

- **Termination.** For the four presets, with the default window (±5 m), `r_min = 0.2 m`, and 16 lines for the largest charge:

| Preset | Lines | Ended at negative charge | Ended at window edge | Other reasons |
|---|---|---|---|---|
| Single charge | 16 | 0 | 16 | 0 |
| Dipole | 16 | 10 | 6 | 0 |
| Two like charges | 32 | 0 | 32 | 0 |
| Quadrupole | 32 | 24 | 8 | 0 |

  Those counts depend on the window size: lines that would reach the negative charge but leave the window first are stopped at the edge. No line in these runs ended by the maximum length or the null-field rule. The null-field threshold is `1e-9 k sum|q| / diag^2`; for the dipole example that is about 9.0e-8 V/m.
- **Tolerance.** For a dipole with 16 lines, the paths computed at `rtol = 1e-5`, `1e-7` and `1e-9` stayed within about 1.7e-12 m (nearest-point distance) of a path computed at `rtol = 1e-12`, and ended for the same reasons. Line end points at negative charges landed at the stop radius (0.05 m) to the printed precision. This is because the forced `max_step` (about 0.07 m) keeps steps short, so the tolerance has little influence in this setting. The nearest-point measure ignores error *along* the curve, which does not matter for the line shape. I only tested this dipole case.
- **Sampling.** Each line is stored at 400 points uniformly spaced in arclength over `max_length` (default 4 × window diagonal = 56.57 m), i.e. every 0.142 m regardless of the length actually used. Plotted lines are polylines through those samples. As a geometric estimate (not measured), a chord of 0.142 m on a curve of radius 1 m deviates by about `s^2/(8R)` = 2.5e-3 m, so strongly curved portions near charges are drawn coarsely.
- **Line density is qualitative.** The number of lines starting at each positive charge is proportional to its |q| with a minimum of 4. Spacing in the plot is not a measurement of field strength.
- **Null points.** Near a point where the field vanishes (for example between two like charges) the direction E/|E| is ill-defined. Lines can end early there ("null_field") or bend sharply. No such terminations occurred for the presets above.
- **Negative-only systems.** Lines are traced backward from the negative charges, which is a display convenience.
- **Time.** On the machine used, tracing the default presets took about 1.7 to 5.5 seconds in one run (dipole 1.72 s, single 2.04 s, two like charges 4.15 s, quadrupole 5.51 s). With six charges and many lines it will take longer; I did not time that.

### Where results become unreliable

- Inside the masked discs (no values by design) and within a few tenths of a metre of a charge at coarse grids.
- Contour lines outside the 3rd–97th percentile range (not drawn).
- Field-line spacing used as a measure of field strength.
- Near null points.
- Large coordinates or charges: the page limits positions to ±1000 m and charges to ±1000 µC; I did not test the numerical behaviour at those limits other than a basic check that E at 10^6 m from a 1 µC charge evaluates without overflow (8.99e-9 V/m).

---

## 4. What is not covered

- The pages (`views/`) have no automated tests in the repository. I exercised them manually with Streamlit's headless test runner and a browser at a 390 px emulated width; those scripts are not saved in the repo.
- The tolerances above describe the numerical code, not the physical models; no experimental data were used anywhere.
- Performance (wall-clock) numbers are from a single run on one machine and are not benchmarks.
