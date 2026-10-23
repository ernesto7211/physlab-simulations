"""Small Streamlit helpers shared by the pages (imports Streamlit; core modules do not)."""

import streamlit as st

from physlab.theme import PAGE_BY_KEY


def page_header(key: str, subtitle: str | None = None) -> None:
    """Consistent page heading (icon + navigation title) and a sidebar hint for small screens."""
    info = PAGE_BY_KEY[key]
    st.title(f"{info.icon} {info.title}")
    if subtitle:
        st.caption(subtitle)
    st.caption("Controls are in the sidebar. On a phone, tap the » arrow at the top left to open it.")
