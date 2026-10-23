"""PhysLab: Interactive Numerical Physics Simulations (homepage + navigation)."""

import streamlit as st

from physlab import theme

st.set_page_config(page_title=theme.APP_NAME, page_icon="🔭", layout="wide", initial_sidebar_state="auto")


def home() -> None:
    st.title(f"🔭 {theme.APP_TITLE}")
    st.write(
        "PhysLab is a set of three interactive physics simulations: two-body gravitational orbits, a chaotic "
        "double pendulum, and the electric field of point charges. Each one solves the underlying equations "
        "numerically with NumPy and SciPy, draws the result with Plotly, and is checked by automated tests "
        "against analytic formulas or conserved quantities. Pick a simulation below, change the parameters in "
        "the sidebar, and download the computed data as CSV."
    )

    st.header("Simulations")
    cols = st.columns(len(theme.SIMULATION_PAGES))
    for col, info, page in zip(cols, theme.SIMULATION_PAGES, PAGE_OBJECTS):
        with col, st.container(border=True):
            st.subheader(f"{info.icon} {info.title}")
            st.write(info.summary)
            st.page_link(page, label=f"Open {info.title}", icon=info.icon)

    st.header("How the numbers are validated")
    st.write(
        f"The repository contains **{theme.TEST_TOTAL} automated tests** (counted by `pytest --collect-only`; a "
        "test fails if this list goes stale). They compare the numerical results with analytic formulas and "
        "conserved quantities; tolerances are stated in each test file's docstring. No experimental data are used."
    )
    st.table(
        {
            "Test file": [f"`{name.split('/')[-1]}`" for name, _, _ in theme.TEST_SUITES],
            "Tests": [n for _, n, _ in theme.TEST_SUITES],
            "What it checks": [desc for *_, desc in theme.TEST_SUITES],
        }
    )

    st.header("Limitations")
    st.markdown(
        "- **Models, not reality.** Orbits use two point masses with Newtonian gravity (no relativity, drag or "
        "third bodies) and planar motion. The pendulum has ideal massless rods, point masses and no friction. "
        "The electric field is for static point charges in vacuum (no conductors, dielectrics or induced charges).\n"
        "- **Numerical error is small but not zero.** Orbits and pendulums are integrated with SciPy's DOP853 "
        "(relative tolerance 1e-12); energy drift is plotted on each page so you can see it. Errors grow near "
        "close approaches and in chaotic motion.\n"
        "- **Chaos limits prediction.** For large angles the double pendulum amplifies tiny differences, "
        "including the integrator's own error, so long-time paths are not meaningful. The reported growth rate "
        "is an early-time estimate, not a rigorous Lyapunov exponent.\n"
        "- **Singularities are masked.** Electric field and potential grid points very close to a charge are "
        "left blank (NaN), controlled by a user-set radius. Field-line spacing is qualitative, and contour "
        "ranges are clipped to readable percentiles.\n"
        "- **Tests check numerics, not experiments.** Passing tests show the code reproduces the stated formulas "
        "and conservation laws within the stated tolerances; they do not validate the physical models against measurements."
    )

    st.divider()
    st.caption(f"Source code and issue tracker: [{theme.GITHUB_URL}]({theme.GITHUB_URL})")


home_page = st.Page(home, title="Home", icon=theme.HOME_ICON, default=True)
PAGE_OBJECTS = [st.Page(p.path, title=p.title, icon=p.icon) for p in theme.SIMULATION_PAGES]

pg = st.navigation([home_page, *PAGE_OBJECTS])
pg.run()
