"""
Template for a new example module.

Contract:
  - expose a zero-argument `render()` function.
  - do NOT draw your own "back" button or page title -- app.py already
    shows the breadcrumb and the back button above your content.
  - use the full page width: put controls in a row of st.columns (or
    an st.expander) ABOVE the plot, not in a narrow left sidebar
    column next to it. That's what keeps the final dashboard from
    losing space to an empty-looking left column.

Copy this file, rename it, implement render(), then register it in
lectures.py.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def render():
    st.header("Example title")
    st.write("One or two sentences on what this example illustrates.")

    # --- controls: a row across the full width, not a side column ---
    c1, c2, c3 = st.columns(3)
    with c1:
        param_a = st.slider("Parameter A", 0.0, 10.0, 5.0)
    with c2:
        param_b = st.slider("Parameter B", 0.0, 10.0, 5.0)
    with c3:
        n = st.slider("n", 10, 1000, 200)

    # --- computation ---
    x = np.linspace(0, 10, n)
    y = param_a * np.sin(x) + param_b

    # --- full-width plot ---
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(x, y)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    st.pyplot(fig)
