"""Streamlit page: double pendulum and sensitivity to initial conditions."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from physlab import pendulum as pn

DEFAULTS = dict(
    th1_deg=120.0, th2_deg=-10.0, L1=1.0, L2=1.0, m1=1.0, m2=1.0, g=9.81,
    duration=20.0, delta=1e-6,
)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

st.title("Double Pendulum")
st.caption("Ideal massless rods, point masses, no friction. Angles from the downward vertical.")

with st.sidebar:
    st.header("Pendulum parameters")
    st.number_input("Initial angle θ₁ (degrees)", format="%.3f", step=5.0, key="th1_deg")
    st.number_input("Initial angle θ₂ (degrees)", format="%.3f", step=5.0, key="th2_deg")
    st.number_input("Rod length L₁ (m)", min_value=0.0, format="%.3f", step=0.1, key="L1")
    st.number_input("Rod length L₂ (m)", min_value=0.0, format="%.3f", step=0.1, key="L2")
    st.number_input("Mass m₁ (kg)", min_value=0.0, format="%.3f", step=0.1, key="m1")
    st.number_input("Mass m₂ (kg)", min_value=0.0, format="%.3f", step=0.1, key="m2")
    st.number_input("Gravity g (m/s²)", min_value=0.0, format="%.3f", step=0.1, key="g")
    st.number_input("Duration (s)", min_value=0.0, format="%.2f", step=1.0, key="duration")
    st.number_input("Perturbation of θ₁ for 2nd pendulum (rad)", format="%.2e", step=1e-6, key="delta")
    n_points = st.slider("Output points", 500, 10000, 4000, step=500)
    if st.button("Reset to defaults"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()

s = st.session_state
th1, th2 = np.radians(s["th1_deg"]), np.radians(s["th2_deg"])
errors = pn.validate_inputs(s["m1"], s["m2"], s["L1"], s["L2"], s["g"], th1, th2,
                            s["duration"], n_points, s["delta"])
if errors:
    for msg in errors:
        st.error(msg)
    st.stop()

params = pn.PendulumParams(s["m1"], s["m2"], s["L1"], s["L2"], s["g"])


@st.cache_data(show_spinner="Integrating...")
def run(params, th1, th2, t_end, n_points, delta):
    return pn.simulate_divergence(params, th1, th2, 0.0, 0.0, t_end, n_points, delta_theta1=delta)


try:
    div = run(params, th1, th2, s["duration"], n_points, s["delta"])
except Exception as exc:  # never crash the page on solver problems
    st.error(f"Simulation failed: {exc}")
    st.stop()

ref, per = div.reference, div.perturbed
if not (ref.success and per.success):
    st.error(f"Integrator did not finish: {ref.message if not ref.success else per.message}")
    st.stop()

# --- summary ---------------------------------------------------------------
c1, c2 = st.columns(2)
c1.metric("Final angular separation", f"{div.separation[-1]:.3e} rad")
c2.metric("Max angular separation", f"{div.separation.max():.3e} rad")
c3, c4 = st.columns(2)
c3.metric("Max energy drift (original)", f"{ref.energy_drift.max():.2e}")
c4.metric(
    "Early growth rate (estimate)",
    "n/a" if not np.isfinite(div.growth_rate) else f"{div.growth_rate:.3f} 1/s",
    help="Slope of ln(separation) vs t while separation < 1e-2 rad. A rough finite-time "
         "estimate, not a rigorous Lyapunov exponent.",
)

# --- animation -------------------------------------------------------------
reach = (s["L1"] + s["L2"]) * 1.08
idx = pn.animation_frame_indices(ref.t.size, 300)
dt_ms = int(np.clip(1000 * s["duration"] / len(idx), 20, 200))

def frame_traces(k):
    i = idx[k]
    return [
        go.Scatter(x=ref.x2[idx[: k + 1]], y=ref.y2[idx[: k + 1]]),
        go.Scatter(x=[0, ref.x1[i], ref.x2[i]], y=[0, ref.y1[i], ref.y2[i]]),
        go.Scatter(x=[ref.x1[i], ref.x2[i]], y=[ref.y1[i], ref.y2[i]]),
    ]

fig = go.Figure(
    data=[
        go.Scatter(x=ref.x2[:1], y=ref.y2[:1], mode="lines", name="Bob 2 trail",
                   line=dict(color="crimson", width=1.5)),
        go.Scatter(x=[0, ref.x1[0], ref.x2[0]], y=[0, ref.y1[0], ref.y2[0]], mode="lines",
                   name="Rods", line=dict(color="black", width=3)),
        go.Scatter(x=[ref.x1[0], ref.x2[0]], y=[ref.y1[0], ref.y2[0]], mode="markers",
                   name="Bobs", marker=dict(color=["#1f77b4", "#d62728"], size=14)),
    ],
    frames=[go.Frame(data=frame_traces(k), traces=[0, 1, 2], name=str(k)) for k in range(len(idx))],
)
anim_args = dict(frame=dict(duration=dt_ms, redraw=True), transition=dict(duration=0), fromcurrent=True)
st.subheader("Animation (original pendulum)")
fig.update_layout(
    xaxis=dict(range=[-reach, reach], title="x (m)", constrain="domain"),
    yaxis=dict(range=[-reach, reach], title="y (m)", scaleanchor="x", scaleratio=1),
    height=520,
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation="h", y=-0.12),
    updatemenus=[dict(
        type="buttons", direction="left", x=0.0, y=1.0, xanchor="left", yanchor="bottom", pad=dict(t=0, r=6),
        buttons=[
            dict(label="▶ Play", method="animate", args=[None, anim_args]),
            dict(label="⏸ Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))]),
        ],
    )],
    sliders=[dict(
        active=0, x=0.0, len=1.0, pad=dict(t=30, b=0),
        currentvalue=dict(prefix="t = ", suffix=" s", visible=True),
        steps=[dict(method="animate", label=f"{ref.t[idx[k]]:.2f}",
                    args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                         transition=dict(duration=0))])
               for k in range(len(idx))],
    )],
)
st.plotly_chart(fig, width="stretch")

# --- overlaid paths --------------------------------------------------------
pf = go.Figure()
pf.add_trace(go.Scatter(x=ref.x2, y=ref.y2, mode="lines", name="Original", line=dict(color="#1f77b4", width=1.5)))
pf.add_trace(go.Scatter(x=per.x2, y=per.y2, mode="lines", name="Perturbed", line=dict(color="#ff7f0e", width=1.5)))
pf.update_layout(
    title="Second-bob paths",
    xaxis_title="x (m)", yaxis_title="y (m)",
    yaxis=dict(scaleanchor="x", scaleratio=1),
    height=480, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.2),
)
st.plotly_chart(pf, width="stretch")

# --- separation (log) ------------------------------------------------------
sep = np.where(div.separation > 0, div.separation, np.nan)
sf = go.Figure(go.Scatter(x=div.t, y=sep, mode="lines", name="Separation", line=dict(color="crimson")))
sf.update_yaxes(type="log", exponentformat="power", title="√(Δθ₁²+Δθ₂²) (rad)")
sf.update_xaxes(title="Time (s)")
sf.update_layout(title="Angular separation of the two pendulums", height=380,
                 margin=dict(l=10, r=10, t=50, b=10))
if np.isfinite(div.growth_rate):
    sf.add_vrect(x0=div.fit_window[0], x1=div.fit_window[1], fillcolor="gray", opacity=0.15, line_width=0,
                 annotation_text="growth-rate fit window", annotation_position="top left")
st.plotly_chart(sf, width="stretch")

# --- energy drift ----------------------------------------------------------
ef = go.Figure()
for name, res, color in (("Original", ref, "#1f77b4"), ("Perturbed", per, "#ff7f0e")):
    ef.add_trace(go.Scatter(x=res.t, y=np.where(res.energy_drift > 0, res.energy_drift, np.nan),
                            mode="lines", name=name, line=dict(color=color)))
ef.update_yaxes(type="log", exponentformat="power", title="|E(t)−E(0)|/|E(0)|")
ef.update_xaxes(title="Time (s)")
ef.update_layout(title="Relative energy drift", height=380,
                 margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.25))
st.plotly_chart(ef, width="stretch")

# --- download --------------------------------------------------------------
df = pn.divergence_to_dataframe(div)
st.download_button("Download results (CSV)", df.to_csv(index=False).encode("utf-8"),
                   file_name="double_pendulum_results.csv", mime="text/csv")

# --- theory ----------------------------------------------------------------
with st.expander("Equations and numerical limitations"):
    st.markdown("**Energies** (with $\\delta=\\theta_1-\\theta_2$)")
    st.latex(r"T=\tfrac12(m_1+m_2)L_1^2\dot\theta_1^2+\tfrac12 m_2L_2^2\dot\theta_2^2+m_2L_1L_2\dot\theta_1\dot\theta_2\cos\delta")
    st.latex(r"V=-(m_1+m_2)gL_1\cos\theta_1-m_2gL_2\cos\theta_2")
    st.markdown("**Equations of motion** (Euler–Lagrange, solved as a 2×2 linear system for the accelerations)")
    st.latex(
        r"\begin{pmatrix}(m_1+m_2)L_1^2 & m_2L_1L_2\cos\delta\\ m_2L_1L_2\cos\delta & m_2L_2^2\end{pmatrix}"
        r"\begin{pmatrix}\ddot\theta_1\\ \ddot\theta_2\end{pmatrix}="
        r"\begin{pmatrix}-m_2L_1L_2\dot\theta_2^2\sin\delta-(m_1+m_2)gL_1\sin\theta_1\\"
        r" m_2L_1L_2\dot\theta_1^2\sin\delta-m_2gL_2\sin\theta_2\end{pmatrix}"
    )
    st.markdown(
        "**Assumptions and limitations**\n"
        "- Ideal massless rigid rods, point masses, no friction or drag, planar motion, both pendulums "
        "start at rest.\n"
        "- Integrated with SciPy DOP853, `rtol=atol=1e-12`. Energy drift is a check on numerical error, "
        "not a proof of correctness.\n"
        "- **Chaos:** for large angles, nearby trajectories separate roughly exponentially until the "
        "separation saturates at order 1 rad. After that, long-time prediction of the actual path is not "
        "meaningful. The same applies to the integrator's own rounding and truncation error: it is amplified "
        "like any other perturbation, so a trajectory computed beyond the point where the separation plot "
        "saturates should be treated as one plausible path, not *the* path.\n"
        "- Integrator error grows near sensitive trajectories (for example close to the inverted "
        "configuration), so energy drift can rise when the motion is violent.\n"
        "- The perturbation is applied to θ₁ only. Separation is √(Δθ₁²+Δθ₂²) with angles not wrapped, "
        "so it can exceed π.\n"
        "- The growth rate is a finite-time, window-dependent estimate (slope of ln separation while it is "
        "below 10⁻² rad), not a rigorous Lyapunov exponent; it is shown as `n/a` when too few points fall "
        "in the window.\n"
        f"- Duration is limited to {pn.MAX_DURATION:g} s and perturbations to "
        f"[{pn.MIN_PERTURBATION:g}, {pn.MAX_PERTURBATION:g}] rad; smaller perturbations are below what the "
        "solver tolerance can meaningfully resolve."
    )
