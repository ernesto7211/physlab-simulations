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
        "- **Two-body orbital mechanics** — core solver implemented and tested; "
        "interactive page coming in a later step.\n"
        "- **Double pendulum** — planned.\n"
        "- **Electric fields from point charges** — planned."
    )


# Later steps add st.Page entries for each simulation here.
pg = st.navigation([st.Page(home, title="Home", icon="🏠", default=True)])
pg.run()
