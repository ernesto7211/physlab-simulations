"""PhysLab: Interactive Numerical Physics Simulations (homepage + navigation)."""

import streamlit as st

st.set_page_config(page_title="PhysLab", page_icon="🔭", layout="wide")


def home() -> None:
    st.title("PhysLab: Interactive Numerical Physics Simulations")
    st.write(
        "Three numerically validated simulations, each checked against "
        "analytic results or conserved quantities."
    )
    st.subheader("Simulations")
    st.markdown(
        "- **Two-body orbital mechanics** — interactive page available.\n"
        "- **Double pendulum** — interactive page available.\n"
        "- **Electric fields from point charges** — interactive page available."
    )


# Later steps add st.Page entries for the remaining simulations here.
pg = st.navigation(
    [
        st.Page(home, title="Home", icon="🏠", default=True),
        st.Page("views/orbital_page.py", title="Orbital Mechanics", icon="🪐"),
        st.Page("views/pendulum_page.py", title="Double Pendulum", icon="🌀"),
        st.Page("views/electric_page.py", title="Electric Fields", icon="⚡"),
    ]
)
pg.run()
