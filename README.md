# PhysLab: Interactive Numerical Physics Simulations

Streamlit app with three independently tested simulations (NumPy, SciPy, Plotly):

| Simulation | Module | Page | Status |
|---|---|---|---|
| Two-body orbital mechanics | `physlab/orbital.py` | `views/orbital_page.py` | Solver, page, tests |
| Double pendulum | `physlab/pendulum.py` | `views/pendulum_page.py` | Solver, page, tests |
| Electric fields from point charges | `physlab/electric.py` | `views/electric_page.py` | Solver, page, tests |

## Structure

```
app.py            homepage + st.navigation
views/            Streamlit simulation pages
physlab/          numerical core (no Streamlit dependency)
tests/            pytest validation
```

The pages folder is called `views/`, not `pages/`: a folder named `pages/`
makes Streamlit add its own automatic navigation alongside `st.navigation`.

## Models

- **Orbital:** Newtonian point masses reduced to the relative coordinate,
  `r'' = -mu r/|r|^3`, `mu = G(m1+m2)`, SI units, planar. DOP853, rtol 1e-12.
- **Pendulum:** planar double pendulum from the Lagrangian equations of
  motion (ideal massless rods, point masses, no friction). DOP853, rtol 1e-12.
  The divergence growth rate is an early-time estimate, not a rigorous Lyapunov exponent.
- **Electric:** static point charges in vacuum, 3D Coulomb law evaluated on a
  plane (CODATA 2018 constants). Grid points within a user-set radius of a
  charge are masked (NaN) rather than softened. Field lines show direction
  only; their spacing is qualitative.

Tolerances used in the tests are stated in each test module's docstring.

## Run

```
pip install -r requirements-dev.txt
pytest
streamlit run app.py
```
