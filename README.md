notepad README.md
# PhysLab: Interactive Numerical Physics Simulations

PhysLab is a [Streamlit](https://streamlit.io) app with three interactive physics simulations, each backed by a
plain-Python numerical module and an automated test suite:

| Simulation | What it does | Core module | Page |
|---|---|---|---|
| **Orbital Mechanics** | Two-body gravitational orbit in the relative coordinate, with energy, angular momentum, orbit type, eccentricity and period | `physlab/orbital.py` | `views/orbital_page.py` |
| **Double Pendulum** | Planar double pendulum next to a nearly identical copy, showing how the two diverge | `physlab/pendulum.py` | `views/pendulum_page.py` |
| **Electric Fields** | Electric field, potential, equipotentials and field lines of up to six static point charges | `physlab/electric.py` | `views/electric_page.py` |

The simulations are standard textbook models solved with NumPy and SciPy and drawn with Plotly. Nothing here is claimed
to be new. The tests compare the numerical results with analytic formulas and conserved quantities; **no experimental
data are used anywhere in the project.**

- **Live app:** https://physlab-simulations-3xkjkebznidep2hwetjrtk.streamlit.app/
- **Repository:** https://github.com/ernesto7211/physlab-simulations
- **Numerical limitations (with measured numbers):** [`docs/numerical_limitations.md`](docs/numerical_limitations.md)


## The physics

All formulas below are the ones implemented in the code.

### 1. Orbital mechanics

Two point masses `m1`, `m2` interact only through Newtonian gravity. The centre of mass moves uniformly, so the
motion reduces to the relative coordinate `r = r2 - r1` (planar, state `(x, y, vx, vy)`, SI units):

$$\ddot{\mathbf r} = -\mu\,\frac{\mathbf r}{|\mathbf r|^{3}},\qquad \mu = G\,(m_1+m_2),\qquad m = \frac{m_1 m_2}{m_1+m_2}$$

with `G = 6.6743e-11 m^3 kg^-1 s^-2` (CODATA 2018). Reported quantities use the reduced mass `m`:

$$T = \tfrac12 m v^2,\qquad U = -\frac{G m_1 m_2}{r},\qquad E = T + U,\qquad L = m\,(x v_y - y v_x)$$

Orbit classification from the initial state (specific energy `eps = v^2/2 - mu/r`, specific angular momentum `h = |x vy - y vx|`):

$$a = -\frac{\mu}{2\varepsilon},\qquad e = \sqrt{1 + \frac{2\varepsilon h^{2}}{\mu^{2}}},\qquad T_\text{orb} = 2\pi\sqrt{\frac{a^{3}}{\mu}}$$

`eps < 0`: bound ellipse; `|eps| <= 1e-9 * mu/r`: treated as parabolic (escape speed `sqrt(2 mu / r)`); `eps > 0`: unbound.
Relative energy drift is `|E(t) - E(0)| / |E(0)|`.

**Assumptions:** point masses, Newtonian gravity, two bodies only, planar motion; no relativity, drag, body radii or
third bodies. On the page the initial velocity is tangential.

### 2. Double pendulum

Two point masses `m1`, `m2` on ideal massless rigid rods of lengths `L1`, `L2`; no friction; uniform gravity `g`;
angles `theta1`, `theta2` measured from the downward vertical. With `delta = theta1 - theta2`:

$$T = \tfrac12 (m_1+m_2)L_1^2\dot\theta_1^2 + \tfrac12 m_2 L_2^2\dot\theta_2^2 + m_2 L_1 L_2\,\dot\theta_1\dot\theta_2\cos\delta$$

$$V = -(m_1+m_2)\,g L_1\cos\theta_1 - m_2\, g L_2\cos\theta_2$$

The Euler–Lagrange equations are a 2×2 linear system for the angular accelerations (solved directly at each step):

$$\begin{pmatrix}(m_1+m_2)L_1^2 & m_2L_1L_2\cos\delta\\ m_2L_1L_2\cos\delta & m_2L_2^2\end{pmatrix}
\begin{pmatrix}\ddot\theta_1\\ \ddot\theta_2\end{pmatrix}=
\begin{pmatrix}-m_2L_1L_2\dot\theta_2^2\sin\delta-(m_1+m_2)gL_1\sin\theta_1\\ m_2L_1L_2\dot\theta_1^2\sin\delta-m_2gL_2\sin\theta_2\end{pmatrix}$$

Small-angle normal modes (used in the tests) solve `K v = w^2 A v` with `A` the mass matrix at `delta = 0` and
`K = diag((m1+m2) g L1, m2 g L2)`.

**Divergence.** Two runs differing by a small `delta_theta1` (default 1e-6 rad) are compared through
`sep(t) = sqrt(dtheta1^2 + dtheta2^2)` (angles not wrapped). The reported "early growth rate" is the least-squares slope of
`ln(sep)` vs `t` while `sep < 1e-2 rad`. **It is a finite-time estimate, not a rigorous Lyapunov exponent.**

### 3. Electric fields

Static point charges `q_i` at `(x_i, y_i)` in vacuum, with the full 3D Coulomb law evaluated in the plane z = 0
(field ∝ 1/r², potential ∝ 1/r):

$$\mathbf E(\mathbf r)=\sum_i k\,q_i\,\frac{\mathbf r-\mathbf r_i}{|\mathbf r-\mathbf r_i|^{3}},\qquad
V(\mathbf r)=\sum_i \frac{k\,q_i}{|\mathbf r-\mathbf r_i|},\qquad k=\frac{1}{4\pi\varepsilon_0}$$

with `eps0 = 8.8541878128e-12 F/m` (CODATA 2018). Superposition is exact (sums over charges). Field lines are curves
tangent to `E`, integrated along `dr/ds = ±E/|E|`.

**Assumptions:** static charges, vacuum, no conductors, dielectrics or induced charges.

---

## Numerical methods

| | Orbital | Pendulum | Electric |
|---|---|---|---|
| Method | `solve_ivp`, DOP853 | `solve_ivp`, DOP853 | Closed-form sums (NumPy, vectorised over grid points, loop over charges); field lines with `solve_ivp`, DOP853 |
| Tolerances | `rtol = 1e-12`, `atol = 1e-6` | `rtol = atol = 1e-12` | Field lines: `rtol = 1e-9`, `atol = 1e-12`, `max_step = window diagonal / 200` |
| Output | `n_points` uniform samples (page: 200–10,000, default 3,000) | `n_points` uniform samples (page: 500–10,000, default 4,000) | Regular grid (page: 40–300 points per side, default 160) |

**Safeguards**

- **Orbital:** inputs are validated (positive masses, distance and duration; finite non-negative speed; zero speed rejected as a radial plunge). Orbits with periapsis below 0.1% of the initial distance are rejected as too stiff, bound orbits are limited to 200 periods, and at most 20,000 output points are allowed. If `E(0)` is essentially zero (parabolic) the page normalises drift by `|U(0)|`.
- **Pendulum:** positive finite masses, lengths and gravity; finite angles; duration limited to 60 s; perturbation magnitude between 1e-10 and 1 rad. Drift is normalised by `|E(0)|` (or a characteristic energy scale if `E(0)` is about zero).
- **Electric:**
  - **Singularity masking:** grid points closer than `r_min` to any charge (or exactly on a charge) are set to **NaN**, not clipped or softened, and shown as grey discs.
  - **Input limits:** 1–6 charges, at least one non-zero, |q| ≤ 1000 µC, positions within ±1000 m, `r_min` ≤ 25% of the plot half-width, non-overlapping charges.
  - **Field lines:** they stop near negative charges, at the window edge, at a maximum length, or at a null field. They start around positive charges (around negative charges, traced backward, if there are no positive ones).
  - **Contours:** contour levels cover the 3rd–97th percentile of V because V diverges near charges.
- **All pages:** invalid input shows an error message instead of crashing, results are computed behind spinners, and solver failures are caught and reported.

Measured numerical errors, sensitivity to tolerances and where results become unreliable are in
[`docs/numerical_limitations.md`](docs/numerical_limitations.md).

---

## Tests

**109 automated tests** (`python -m pytest --collect-only -q`), all passing at the time of writing. The homepage test table is checked against this count by `tests/test_theme.py`.
Tolerances are stated in each test file's docstring.

**`tests/test_orbital.py` (4)** — numerics vs analytic results
- `test_circular_orbit_radius_constant` — radius stays constant to 1e-8 relative
- `test_period_matches_kepler_third_law` — measured period of an e = 0.44 orbit matches Kepler's third law to 1e-6
- `test_conservation_of_energy_and_angular_momentum` — drifts below 1e-7 over 5 periods
- `test_escape_velocity_is_parabolic_and_unbound` — energy ≈ 0 (1e-9 of kinetic energy), radius increases monotonically, `v = sqrt(2 mu / r)` throughout

**`tests/test_orbital_helpers.py` (16)**
- `test_circular_elements`, `test_elliptical_elements`, `test_escape_and_hyperbolic_elements` — classification, eccentricity, period, apsides vs analytic values
- `test_elements_match_simulation_apsides` — apsides vs the simulated orbit
- `test_validate_rejects_bad_input` (10 cases), `test_validate_accepts_good_input` — input validation
- `test_dataframe_columns_and_shape` — CSV table

**`tests/test_pendulum.py` (12)**
- `test_small_angle_normal_mode_frequencies` (4 cases) — periods vs linear theory to 1e-6, two parameter sets, both modes
- `test_small_angle_modes_closed_form_equal_case` — eigen-solver vs closed form
- `test_energy_conserved_moderate_amplitude` (2 cases) — drift below 1e-9 over 20 s
- `test_geometry_consistent_with_rod_lengths`
- `test_rest_at_equilibrium_stays_at_rest`
- `test_large_angle_diverges_far_more_than_small_angle` — large-angle pair diverges more than 1000× further; growth-rate estimate clearly positive for large and near zero for small angles
- `test_growth_rate_estimator_recovers_known_exponential` — synthetic exponential
- `test_invalid_parameters_rejected`

**`tests/test_pendulum_helpers.py` (26)**
- `test_validate_accepts_good_input`, `test_validate_rejects_bad_input` (16 cases), `test_validate_accepts_negative_perturbation`
- `test_frame_indices` (6 cases), `test_frame_indices_rejects_bad_args` — animation frame selection
- `test_dataframe_contents` — CSV table

**`tests/test_electric.py` (13)**
- `test_constants`
- `test_single_charge_field_magnitude_and_direction` (2 cases) — `kQ/r^2`, radial direction, both signs, 1e-12
- `test_single_charge_potential` — `kQ/r`
- `test_superposition`
- `test_dipole_perpendicular_bisector` — exact two-charge expression (1e-12) and far-field dipole approximation (1e-4)
- `test_masked_points_are_nan_not_inf`
- `test_gradient_of_potential_matches_minus_field` — 2e-3 at h = 0.05 m, with h² scaling checked
- `test_invalid_inputs`
- `test_lines_from_single_positive_charge_are_radial_and_reach_boundary`
- `test_dipole_lines_end_on_negative_charge_or_boundary_and_follow_field`
- `test_max_length_respected_and_negative_only_traces_backward`
- `test_trace_invalid_inputs`

**`tests/test_electric_helpers.py` (32)**
- `test_validate_accepts_good_input`, `test_validate_rejects_bad_input` (17 cases), `test_zero_charge_can_overlap_a_real_charge`
- `test_active_charges_drops_zeros_and_converts_units`, `test_charges_outside_window`
- `test_presets_are_valid_and_expected_size` (4 cases), `test_preset_physics_signs`
- `test_equipotential_levels_are_nice_and_within_range`, `test_equipotential_levels_errors`
- `test_log_magnitude_bins`, `test_arrow_segments_geometry`, `test_arrow_segments_all_masked_is_empty`
- `test_grid_dataframe`

**`tests/test_theme.py` (6)**
- `test_palette_is_valid_unique_okabe_ito`, `test_page_metadata_is_complete_and_unique`, `test_page_files_exist_and_are_registered_in_app`, `test_github_url_format`
- `test_no_matplotlib_anywhere_in_project_code`
- `test_homepage_test_counts_match_pytest_collection`

**What is not tested automatically:** the Streamlit pages themselves (layout, plots, buttons, downloads). I checked them
manually (see below). Passing tests show the code reproduces the stated formulas and conservation laws within the stated
tolerances; they do not validate the physical models against measurements.

---

## Project structure

```
app.py                      Homepage (title, cards, validation table, limitations, footer) + st.navigation
views/                      Streamlit simulation pages (named views/, not pages/, so that Streamlit's
  orbital_page.py           legacy automatic navigation is not triggered alongside st.navigation)
  pendulum_page.py
  electric_page.py
physlab/                    Numerical core and helpers
  orbital.py                Two-body solver, orbit elements, validation, CSV table
  pendulum.py               Double-pendulum solver, divergence and growth-rate estimate, validation, CSV table
  electric.py               Field/potential on a grid, field-line tracer, presets, validation, contour/arrow helpers
  theme.py                  Page metadata, Okabe–Ito palette, test table shown on the homepage
  ui.py                     Shared page heading (the only module in physlab/ that imports Streamlit)
tests/                      pytest suite (109 tests)
docs/
  numerical_limitations.md  Error sources, tolerances, where results are unreliable
  reproduce_limits.py       Script that recomputes the numbers quoted in that document
pytest.ini                  pythonpath = . ; testpaths = tests
requirements.txt            streamlit, numpy, scipy, plotly, pandas (unpinned)
requirements-dev.txt        -r requirements.txt, pytest
LICENSE                     MIT
```

Plots use Plotly only (no Matplotlib) with the Okabe–Ito colour-blind-safe palette.

---

## Setup (Windows, Python 3.12)

Versions the project was developed and tested with: Python 3.12.10, NumPy 2.5.4, SciPy 1.18.1, Plotly 7.1.0,
pandas 3.1.0, Streamlit 1.66.1, pytest 9.1.1. The requirements files are not pinned.

In PowerShell, from the repository folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

If activation is blocked by the execution policy, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that window first (or call `.\.venv\Scripts\python.exe` directly).

### Run the tests

```powershell
python -m pytest
```

(The full suite takes about 10 to 30 seconds here; `tests/test_theme.py` starts a second `pytest --collect-only` to check the test counts shown on the homepage.)

### Run the app locally

```powershell
python -m streamlit run app.py --server.port 8501
```

Then open http://localhost:8501.

### Reproduce the numbers in the limitations document

```powershell
python docs\reproduce_limits.py
```

(about one to two minutes).

---

## Limitations and known issues

See [`docs/numerical_limitations.md`](docs/numerical_limitations.md) for details and numbers. In short:

- **Models, not reality.** Point masses and Newtonian gravity (orbits); ideal rods and no friction (pendulum); static point charges in vacuum (fields). No collision detection or body radii: the orbit page will happily compute an orbit that passes inside the real primary.
- **Orbital accuracy is limited by the absolute tolerance.** With the defaults (`rtol = 1e-12`, `atol = 1e-6`) the measured energy drift for an e = 0.44 orbit is about 1e-8; lowering `atol` to 1e-9 reduced it to about 4e-11, whereas tightening `rtol` alone did not help. I have not changed the defaults.
- **The double pendulum is chaotic.** For the default case (120°, −10°) a run at the app's default tolerance departs from a tighter-tolerance run by 0.1 rad at about 28.7 s, but the app allows durations up to 60 s. Trajectories beyond about 25 to 30 s for such cases should not be treated as the true path. Small energy drift does not imply an accurate trajectory.
- **The growth rate is only an estimate.** It varied from 0.77 to 1.02 per second with the fit cut-off, and by a factor of about two between initial conditions.
- **Electric fields are masked, not regularised.** Nothing is shown within `r_min` of a charge; the number of masked grid points depends on grid alignment; contour levels are clipped to the 3rd–97th percentile; field-line spacing is qualitative and lines are stored at 0.142 m arclength spacing by default.
- **Energy drift is reported at output samples only,** so very sparse sampling can under-report a peak.
- **Not covered by automated tests:** the Streamlit pages. I checked them manually with Streamlit's headless test runner and a browser at an emulated 390 px width; those checks are not saved in the repo. The Play/Pause animation, Plotly hover read-outs and CSV downloads were not tested by automation.
- **Not verified:** the deployed app (the URL above is a placeholder), other browsers, and real phones.
- **Dependencies are unpinned,** so a future library release could change behaviour.
- **Performance.** Tracing field lines with the default settings took roughly 2 to 6 seconds for the four presets on my machine in one run; larger configurations will be slower.

---


## References

1. E. Hairer, S. P. Nørsett, G. Wanner, *Solving Ordinary Differential Equations I: Nonstiff Problems*, 2nd ed., Springer (1993). (The DOP853 method.)
2. J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta formulae", *J. Comput. Appl. Math.* **6**, 19–26 (1980).
3. SciPy documentation: [`scipy.integrate.solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
4. E. Tiesinga, P. J. Mohr, D. B. Newell, B. N. Taylor, "CODATA recommended values of the fundamental physical constants: 2018", *Rev. Mod. Phys.* **93**, 025010 (2021).
5. H. Goldstein, C. Poole, J. Safko, *Classical Mechanics*, 3rd ed., Addison-Wesley (2002). (Two-body problem, Lagrangian mechanics.)
6. J. D. Jackson, *Classical Electrodynamics*, 3rd ed., Wiley (1999). (Coulomb's law, potential, superposition.)
7. T. Shinbrot, C. Grebogi, J. Wisdom, J. A. Yorke, "Chaos in a double pendulum", *Am. J. Phys.* **60**, 491–499 (1992).
8. S. H. Strogatz, *Nonlinear Dynamics and Chaos*, 2nd ed., Westview Press (2015). (Sensitive dependence on initial conditions, Lyapunov exponents.)
9. M. Okabe, K. Ito, "Color Universal Design (CUD): how to make figures and presentations that are friendly to colorblind people" (2008), https://jfly.uni-koeln.de/color/. (Colour palette.)
10. Streamlit documentation, https://docs.streamlit.io; Plotly Python documentation, https://plotly.com/python/.



