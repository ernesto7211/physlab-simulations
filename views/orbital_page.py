"""Streamlit page: two-body orbital mechanics."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from physlab import orbital as ob

# Earth-like defaults (SI internally; UI uses kg, km, km/s, hours)
M_EARTH = 5.972e24
KM = 1e3
HOUR = 3600.0

PRESETS = {
    "Circular orbit": {"r_km": 7000.0, "factor": 1.0, "hours": 3.0},
    "Elliptical orbit": {"r_km": 7000.0, "factor": 1.2, "hours": 6.0},
    "Escape trajectory": {"r_km": 7000.0, "factor": float(np.sqrt(2.0)), "hours": 6.0},
    "Custom": None,
}


def _apply_preset() -> None:
    p = PRESETS[st.session_state["preset"]]
    if p is None:
        return
    m1 = st.session_state["m1"]
    m2 = st.session_state["m2"]
    r = p["r_km"] * KM
    if m1 > 0 and m2 > 0:
        st.session_state["r0_km"] = p["r_km"]
        st.session_state["v0_kms"] = p["factor"] * ob.circular_speed(m1, m2, r) / KM
        st.session_state["hours"] = p["hours"]


def _init_state() -> None:
    if "m1" not in st.session_state:
        st.session_state.update(m1=M_EARTH, m2=1000.0, preset="Circular orbit")
        _apply_preset()


st.title("Two-Body Orbital Mechanics")
_init_state()

with st.sidebar:
    st.header("Orbit parameters")
    st.selectbox("Preset", list(PRESETS), key="preset", on_change=_apply_preset)
    st.number_input("Central mass (kg)", min_value=0.0, format="%.4e", key="m1", step=1e23)
    st.number_input("Orbiting body mass (kg)", min_value=0.0, format="%.4e", key="m2", step=100.0)
    st.number_input("Initial distance (km)", min_value=0.0, format="%.3f", key="r0_km", step=100.0)
    st.number_input("Initial speed, tangential (km/s)", min_value=0.0, format="%.5f", key="v0_kms", step=0.1)
    st.number_input("Duration (hours)", min_value=0.0, format="%.3f", key="hours", step=0.5)
    n_points = st.slider("Output points", 200, 10000, 3000, step=100)
    st.caption("Changing a value manually does not change the preset; pick one to reload its values.")

m1, m2 = st.session_state["m1"], st.session_state["m2"]
r0 = st.session_state["r0_km"] * KM
v0 = st.session_state["v0_kms"] * KM
t_end = st.session_state["hours"] * HOUR

errors = ob.validate_inputs(m1, m2, r0, v0, t_end, n_points)
if errors:
    for msg in errors:
        st.error(msg)
    st.stop()


@st.cache_data(show_spinner="Integrating...")
def run(m1, m2, r0, v0, t_end, n_points):
    return ob.simulate_two_body(m1, m2, [r0, 0.0], [0.0, v0], t_end, n_points)


try:
    res = run(m1, m2, r0, v0, t_end, n_points)
except Exception as exc:  # solver failure must not crash the page
    st.error(f"Simulation failed: {exc}")
    st.stop()

if not res.success:
    st.error(f"Integrator did not finish: {res.message}")
    st.stop()

el = ob.orbit_elements(m1, m2, [r0, 0.0], [0.0, v0])

# --- summary ---------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Orbit type", {"bound": "Bound (ellipse)", "parabolic": "Parabolic (E ≈ 0)", "unbound": "Unbound (hyperbola)"}[el.kind])
c2.metric("Eccentricity", f"{el.eccentricity:.5f}")
c3.metric("Period", f"{el.period / HOUR:.4f} h" if el.period else "n/a (not bound)")
c4.metric("Periapsis", f"{el.periapsis / KM:,.1f} km")
if el.kind == "bound":
    st.caption(
        f"Semi-major axis {el.semi_major_axis / KM:,.1f} km, apoapsis {el.apoapsis / KM:,.1f} km. "
        f"Simulated {t_end / el.period:.2f} periods."
    )
if el.kind == "bound" and el.period and t_end < el.period:
    st.info("Duration is shorter than one period; only part of the orbit is shown.")

# --- orbit path ------------------------------------------------------------
x, y = res.position[:, 0] / KM, res.position[:, 1] / KM
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Orbit path", line=dict(color="#1f77b4")))
fig.add_trace(go.Scatter(x=[x[0]], y=[y[0]], mode="markers", name="Start", marker=dict(color="green", size=10)))
fig.add_trace(go.Scatter(x=[x[-1]], y=[y[-1]], mode="markers", name="End", marker=dict(color="red", size=10, symbol="square")))
fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name="Central body", marker=dict(color="orange", size=16, symbol="circle")))
fig.update_layout(
    title="Orbit in the relative coordinate (central body at origin)",
    xaxis_title="x (km)",
    yaxis_title="y (km)",
    yaxis=dict(scaleanchor="x", scaleratio=1),
    height=560,
)
st.plotly_chart(fig, width="stretch")

# --- energy & drift --------------------------------------------------------
t_h = res.t / HOUR
drift = res.energy_drift
drift_label = "Relative energy drift |E(t)−E(0)|/|E(0)|"
if el.kind == "parabolic":
    # E(0) ≈ 0 makes the usual normalisation meaningless; use |U(0)| instead.
    drift = np.abs(res.energy - res.energy[0]) / abs(res.potential[0])
    drift_label = "Energy drift |E(t)−E(0)|/|U(0)| (E≈0)"
drift_plot = np.where(drift > 0, drift, np.nan)  # log axis cannot show exact zeros

efig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                     subplot_titles=("Energy", drift_label))
efig.add_trace(go.Scatter(x=t_h, y=res.kinetic, name="Kinetic"), row=1, col=1)
efig.add_trace(go.Scatter(x=t_h, y=res.potential, name="Potential"), row=1, col=1)
efig.add_trace(go.Scatter(x=t_h, y=res.energy, name="Total", line=dict(width=3, color="black")), row=1, col=1)
efig.add_trace(go.Scatter(x=t_h, y=drift_plot, name="Drift", line=dict(color="crimson")), row=2, col=1)
efig.update_yaxes(title_text="Energy (J)", row=1, col=1)
efig.update_yaxes(title_text="Relative drift", type="log", exponentformat="power", row=2, col=1)
efig.update_xaxes(title_text="Time (hours)", row=2, col=1)
efig.update_layout(height=620)
st.plotly_chart(efig, width="stretch")
st.caption(f"Maximum drift over the run: {np.nanmax(drift):.2e}")

# --- download --------------------------------------------------------------
df = ob.result_to_dataframe(res)
st.download_button("Download results (CSV)", df.to_csv(index=False).encode("utf-8"),
                   file_name="orbit_results.csv", mime="text/csv")

# --- theory ----------------------------------------------------------------
with st.expander("Equations and numerical limitations"):
    st.markdown("**Newton's law of gravitation**")
    st.latex(r"\mathbf F = -\frac{G m_1 m_2}{r^2}\,\hat{\mathbf r}")
    st.markdown("The two-body problem reduces to the relative coordinate, with "
                r"$\mu = G(m_1+m_2)$ and reduced mass $m = m_1 m_2/(m_1+m_2)$:")
    st.latex(r"\ddot{\mathbf r} = -\mu\,\frac{\mathbf r}{|\mathbf r|^3}")
    st.markdown("**Conserved quantities**")
    st.latex(r"E = \tfrac12 m v^2 - \frac{G m_1 m_2}{r},\qquad \mathbf L = m\,\mathbf r\times\mathbf v")
    st.latex(r"a = -\frac{\mu}{2\varepsilon},\quad e=\sqrt{1+\frac{2\varepsilon h^2}{\mu^2}},\quad T = 2\pi\sqrt{\frac{a^3}{\mu}}")
    st.markdown(r"$E<0$: bound ellipse. $E=0$: parabolic (escape speed $\sqrt{2\mu/r}$). $E>0$: hyperbolic.")
    st.markdown(
        "**Numerical limitations**\n"
        "- Integrated with SciPy DOP853, `rtol=1e-12`, `atol=1e-6`. Error is not zero: "
        "drift grows with the number of orbits and peaks near periapsis.\n"
        "- Close approaches are stiff: the acceleration scales as $1/r^2$, so very eccentric "
        "orbits need tiny steps. Orbits with periapsis below 0.1% of the start distance are rejected.\n"
        "- Point masses only: no body radii (so no collision detection), no relativity, no drag, "
        "no third bodies. Motion is planar; initial velocity is tangential.\n"
        "- For a parabolic orbit E(0)≈0, so drift is normalised by |U(0)| instead.\n"
        "- Output points are samples of the solver's continuous solution; they do not affect accuracy."
    )
