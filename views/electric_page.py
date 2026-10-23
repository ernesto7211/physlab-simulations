"""Streamlit page: electric field and potential of static point charges."""

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
import streamlit as st

from physlab import electric as el
from physlab import theme as th
from physlab.ui import page_header

DEFAULT_HW = 5.0
DEFAULT_RMIN = 0.2


def _apply_preset() -> None:
    preset = el.PRESETS[st.session_state["preset"]]
    st.session_state["n_charges"] = len(preset)
    for i, (x, y, q) in enumerate(preset):
        st.session_state[f"x{i}"], st.session_state[f"y{i}"], st.session_state[f"q{i}"] = x, y, q


if "n_charges" not in st.session_state:
    for i in range(el.MAX_CHARGES):  # defaults for unused slots
        st.session_state.setdefault(f"x{i}", 0.0)
        st.session_state.setdefault(f"y{i}", 0.0)
        st.session_state.setdefault(f"q{i}", 1.0)
    st.session_state["preset"] = "Dipole"
    st.session_state.setdefault("half_width", DEFAULT_HW)
    st.session_state.setdefault("r_min", DEFAULT_RMIN)
    _apply_preset()

page_header("electric", "Static point charges in vacuum, in the plane z = 0 (3D Coulomb law). No induced charges.")

with st.sidebar:
    st.header("Charges")
    st.selectbox("Preset", list(el.PRESETS), key="preset", on_change=_apply_preset)
    st.number_input("Number of charges", min_value=1, max_value=el.MAX_CHARGES, step=1, key="n_charges")
    n = int(st.session_state["n_charges"])
    for i in range(n):
        with st.expander(f"Charge {i + 1}", expanded=(n <= 3)):
            st.number_input("x (m)", format="%.3f", step=0.25, key=f"x{i}")
            st.number_input("y (m)", format="%.3f", step=0.25, key=f"y{i}")
            st.number_input("q (µC, sign = polarity)", format="%.3f", step=0.5, key=f"q{i}")
    st.header("Display")
    show_eq = st.checkbox("Equipotentials", value=True)
    show_lines = st.checkbox("Field lines", value=True)
    show_vec = st.checkbox("Field vectors", value=False)
    st.header("Window and numerics")
    st.number_input("Plot half-width (m)", min_value=0.0, format="%.2f", step=1.0, key="half_width")
    st.number_input("Exclusion radius r_min (m)", min_value=0.0, format="%.3f", step=0.05, key="r_min")
    n_grid = st.slider("Grid points per side", 40, 300, 160, step=20)
    lines_per_charge = st.slider("Field lines (largest charge)", 4, 32, 16, step=2)

s = st.session_state
qs = [s[f"q{i}"] for i in range(n)]
xs = [s[f"x{i}"] for i in range(n)]
ys = [s[f"y{i}"] for i in range(n)]
hw, r_min = s["half_width"], s["r_min"]

errors = el.validate_inputs(qs, xs, ys, hw, r_min, n_grid)
if errors:
    for msg in errors:
        st.error(msg)
    st.stop()
for i in el.charges_outside_window(xs, ys, hw):
    st.warning(f"Charge {i + 1} lies outside the plot window; its field is still included.")
if any(q == 0 for q in qs):
    st.info("Charges with q = 0 are ignored in the calculation.")

q_C, pos = el.active_charges(qs, xs, ys)


@st.cache_data(show_spinner="Computing field...")
def compute(q_C, pos, hw, r_min, n_grid):
    X, Y, ex, ey, v = el.field_on_grid(q_C, pos, (-hw, hw), (-hw, hw), n_grid, n_grid, r_min)
    return X, Y, ex, ey, v


@st.cache_data(show_spinner="Tracing field lines...")
def trace(q_C, pos, hw, r_min, per_charge):
    return el.trace_field_lines(q_C, pos, (-hw, hw, -hw, hw), r_min=r_min, lines_per_charge=per_charge)


try:
    X, Y, ex, ey, v = compute(q_C, pos, hw, r_min, n_grid)
except Exception as exc:  # never crash the page on numerical problems
    st.error(f"Calculation failed: {exc}")
    st.stop()
emag = np.hypot(ex, ey)

# --- summary ---------------------------------------------------------------
c1, c2 = st.columns(2)
c1.metric("Net charge", f"{sum(qs):+.3f} µC")
c2.metric("Masked grid points", f"{int(np.isnan(v).sum()):,} of {v.size:,}")

# --- figure ----------------------------------------------------------------
fig = go.Figure()
colorbars = 0

if show_eq:
    try:
        start, end, size = el.equipotential_levels(v)
        fig.add_trace(go.Contour(
            x=X[0], y=Y[:, 0], z=v, customdata=emag, name="V",
            contours=dict(start=start, end=end, size=size, coloring="lines", showlabels=True,
                          labelfont=dict(size=10)),
            line=dict(width=1.5), colorscale=th.SCALE_DIVERGING, zmin=start, zmax=end,
            colorbar=dict(title=dict(text="V (volts)", side="bottom"), orientation="h",
                          x=0.5, xanchor="center", y=-0.16, yanchor="top", len=0.9, thickness=12),
            hovertemplate="x=%{x:.3f} m<br>y=%{y:.3f} m<br>V=%{z:.4g} V<br>|E|=%{customdata:.4g} V/m<extra></extra>",
        ))
        colorbars += 1
    except ValueError as exc:
        st.info(f"Equipotentials not drawn: {exc}")

if show_vec:
    nv = 21
    vx, vy = np.meshgrid(np.linspace(-hw, hw, nv), np.linspace(-hw, hw, nv))
    vex, vey, _ = el.field_at_points(q_C, pos, vx, vy, r_min)
    vmag = np.hypot(vex, vey)
    bins, edges = el.log_magnitude_bins(vmag, 6)
    palette = pc.sample_colorscale(th.SCALE_SEQUENTIAL, np.linspace(0.05, 0.95, 6))
    length = 0.7 * (2 * hw / (nv - 1))
    for b in range(6):
        sel = bins == b
        if not sel.any():
            continue
        ax_, ay_ = el.arrow_segments(vx[sel], vy[sel], vex[sel], vey[sel], length)
        fig.add_trace(go.Scatter(x=ax_, y=ay_, mode="lines", line=dict(color=palette[b], width=1.8),
                                 hoverinfo="skip", showlegend=False))
    if edges.size:  # dummy trace carries the |E| colour bar (binned log scale)
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", hoverinfo="skip", showlegend=False,
            marker=dict(colorscale=th.SCALE_SEQUENTIAL, cmin=np.log10(edges[0]), cmax=np.log10(edges[-1]),
                        color=[np.log10(edges[0])], showscale=True,
                        colorbar=dict(title=dict(text="log10 |E| (V/m), arrows normalised", side="bottom"),
                                      orientation="h", x=0.5, xanchor="center",
                                      y=-0.16 - 0.2 * colorbars, yanchor="top", len=0.9, thickness=12)),
        ))
        colorbars += 1

if show_lines:
    try:
        lines = trace(q_C, pos, hw, r_min, lines_per_charge)
    except Exception as exc:
        lines = []
        st.error(f"Field-line tracing failed: {exc}")
    if lines:
        lx, ly, mx, my, msize = [], [], [], [], []
        for ln in lines:
            lx += list(ln.x) + [np.nan]
            ly += list(ln.y) + [np.nan]
            k = max(1, int(0.5 * (len(ln.x) - 1)))  # direction marker at mid-line
            mx += [ln.x[k - 1], ln.x[k], np.nan]
            my += [ln.y[k - 1], ln.y[k], np.nan]
            msize += [0, 9, 0]
        fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", name="Field lines",
                                 line=dict(color="rgba(40,40,40,0.8)", width=1.2), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=mx, y=my, mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(symbol="arrow", size=msize, angleref="previous",
                                             color="rgba(40,40,40,0.9)")))

# charge markers (all charges, including q = 0 shown in grey)
for sign, name, color, symbol in (
    (1, "Positive charge (+)", th.VERMILLION, "circle"),
    (-1, "Negative charge (-)", th.BLUE, "square"),
    (0, "Zero charge", "grey", "diamond"),
):
    idx = [i for i, q in enumerate(qs) if np.sign(q) == sign]
    if idx:
        fig.add_trace(go.Scatter(
            x=[xs[i] for i in idx], y=[ys[i] for i in idx], mode="markers+text", name=name,
            text=[f"{qs[i]:+g} µC" for i in idx], textposition="top center",
            marker=dict(color=color, size=10, symbol=symbol, line=dict(color="black", width=1.5)),
            hovertemplate="charge %{text}<br>x=%{x:.3f} m, y=%{y:.3f} m<extra></extra>",
        ))

# masked (excluded) regions around non-zero charges
for (cx, cy) in pos:
    fig.add_shape(type="circle", x0=cx - r_min, x1=cx + r_min, y0=cy - r_min, y1=cy + r_min,
                  fillcolor="rgba(120,120,120,0.35)", line=dict(color="gray", width=1, dash="dot"), layer="above")

fig.update_layout(
    xaxis=dict(range=[-hw, hw], title="x (m)", constrain="domain"),
    yaxis=dict(range=[-hw, hw], title="y (m)", scaleanchor="x", scaleratio=1),
    height=560 + 90 * colorbars,
    margin=dict(l=10, r=10, t=20, b=10 + 70 * colorbars),
    legend=dict(orientation="h", y=1.0, yanchor="bottom"),
)
st.plotly_chart(fig, width="stretch")
st.caption(f"Grey discs: points within r_min = {r_min:g} m of a charge are masked (NaN), not evaluated.")

# --- download --------------------------------------------------------------
df = el.grid_to_dataframe(X, Y, ex, ey, v)
st.download_button("Download grid data (CSV)", df.to_csv(index=False).encode("utf-8"),
                   file_name="electric_field_grid.csv", mime="text/csv")

# --- theory ----------------------------------------------------------------
with st.expander("Equations and numerical limitations"):
    st.markdown("**Coulomb's law** with $k = 1/(4\\pi\\varepsilon_0)$ (CODATA 2018 $\\varepsilon_0$):")
    st.latex(r"\mathbf E(\mathbf r)=\sum_i k\,q_i\,\frac{\mathbf r-\mathbf r_i}{|\mathbf r-\mathbf r_i|^3},\qquad "
             r"V(\mathbf r)=\sum_i \frac{k\,q_i}{|\mathbf r-\mathbf r_i|}")
    st.markdown(
        "**Superposition:** the field and potential of several charges are the plain sums of the "
        "single-charge contributions. The potential is a scalar sum, the field a vector sum, and "
        r"$\mathbf E=-\nabla V$." "\n\n"
        "**Equipotentials** are curves of constant $V$. They are everywhere perpendicular to field lines, "
        "and the field points from high to low potential. Contour labels are in volts.\n\n"
        "**Field lines** are curves tangent to $\\mathbf E$, drawn from positive charges to negative charges "
        "(or to the window edge). Their direction is exact; their spacing is only a rough, qualitative "
        "indicator of field strength. Arrows in the vector view are normalised, so only colour shows magnitude.\n\n"
        "**Numerical limitations**\n"
        "- **Singularity masking:** $E$ and $V$ diverge at a point charge. Grid points closer than "
        "$r_\\mathrm{min}$ to any charge are masked (NaN, shown as grey discs). This is a display choice, "
        "not physics; nothing is evaluated inside the discs.\n"
        "- **Finite grid:** contours and the CSV come from a regular grid, so features smaller than a grid "
        "cell are not resolved and contour lines are piecewise-linear interpolations.\n"
        "- **Contour range:** the potential diverges near charges, so contour levels span only the 3rd to "
        "97th percentile of $V$ on the grid; values beyond that range are not contoured.\n"
        "- **Field-line tracing:** lines are integrated with DOP853 (rtol 1e-9) with unit-speed arclength. "
        "They stop within a small radius of a negative charge, at the window edge, at a maximum length, or "
        "where the field is essentially zero. Near null points the direction is ill-defined and lines may end "
        "early; tracing error accumulates with line length.\n"
        "- **Model:** point charges, vacuum, 3D Coulomb law restricted to the plane z = 0; no conductors, "
        "dielectrics, induced charges or motion."
    )
