"""Stand-in for examples not implemented yet. See lectures.py: placeholder_for()."""

import streamlit as st


def render(title):
    st.header(title)
    st.info(
        "Not implemented yet.\n\n"
        "To build this example: copy examples/_template.py to a new file, "
        "implement render(), then in lectures.py replace this entry's "
        "\"render\": placeholder_for(...) with the real function."
    )
