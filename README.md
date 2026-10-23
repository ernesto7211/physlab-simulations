# PhysLab: Interactive Numerical Physics Simulations

Streamlit app with three independently validated simulations (NumPy, SciPy, Plotly):

| Simulation | Module | Status |
|---|---|---|
| Two-body orbital mechanics | `physlab/orbital.py` | Solver + tests done; UI page pending |
| Double pendulum | `physlab/pendulum.py` | Stub |
| Electric fields from point charges | `physlab/electric.py` | Stub |

## Structure

```
app.py            homepage + st.navigation (pages added per simulation)
pages/            reserved for simulation pages
physlab/          numerical core (no Streamlit dependency)
tests/            pytest validation
```

## Two-body model

Newtonian point masses reduced to the relative coordinate, `r'' = -mu r/|r|^3`,
`mu = G(m1+m2)`, in SI units, planar. Integrated with SciPy `solve_ivp`
(DOP853, rtol 1e-12). Outputs positions, velocities, times, kinetic, potential
and total energy, angular momentum, and relative energy drift
`|E(t)-E(0)|/|E(0)|`. Assumes point masses, no relativity, no drag.

Tests check circular-orbit radius, Kepler's third law, conservation of energy
and angular momentum, and parabolic escape at escape speed.

## Run

```
pip install -r requirements-dev.txt
pytest
streamlit run app.py
```
