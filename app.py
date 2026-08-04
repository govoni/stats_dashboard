"""
Course dashboard entry point.

Navigation model (kept entirely in st.session_state, no URL routing needed):

    "lectures"  -> grid of the 12 lecture buttons
    "examples"  -> grid of example buttons for the selected lecture
    "demo"      -> the selected example's own full-screen dashboard

Each example is a plain module under examples/ that exposes a
zero-argument render() function (see examples/_template.py for the
contract). app.py never draws example content itself -- it only draws
the "back" button and then calls render(). This keeps this file stable
as you add lectures/examples: you never edit app.py again, only
lectures.py and examples/*.py.

Run with:
    streamlit run app.py
"""

import streamlit as st
from lectures import LECTURES

import plotly.io as pio
import plotly.graph_objects as go

my_template = go.layout.Template(
    layout=go.Layout(
        font=dict(
            family="Arial",
            size=16,
            color="black"
        ),
        title=dict(
            font=dict(size=24, color="black")
        ),
        xaxis=dict(
            title=dict(font=dict(size=20, color="black")),
            tickfont=dict(size=16, color="black")
        ),
        yaxis=dict(
            title=dict(font=dict(size=20, color="black")),
            tickfont=dict(size=16, color="black")
        )
    )
)

pio.templates["my_framework"] = my_template
pio.templates.default = "my_framework"

st.set_page_config(page_title="Probability and Statistics", layout="wide")

if "view" not in st.session_state:
    st.session_state.view = "lectures"
    st.session_state.lecture_idx = None
    st.session_state.example_id = None


def go_to(view, lecture_idx=None, example_id=None):
    st.session_state.view = view
    if lecture_idx is not None:
        st.session_state.lecture_idx = lecture_idx
    if example_id is not None:
        st.session_state.example_id = example_id
    st.rerun()


# ------------------------------------------------------------------
# Screen 1: pick a lecture
# ------------------------------------------------------------------
if st.session_state.view == "lectures":
    st.title("Probability and Statistics — Examples")
    st.caption("Laboratorio di Calcolo e Statistica")

    n_cols = 3
    cols = st.columns(n_cols)
    for i, lec in enumerate(LECTURES):
        with cols[i % n_cols]:
            n_ex = len(lec["examples"])
            # label = f"{lec['title']}\n\n{n_ex} example{'s' if n_ex != 1 else ''}"
            label = f"{lec['title']}"
            if st.button(label, key=f"lec_{i}", use_container_width=True):
                go_to("examples", lecture_idx=i)

# ------------------------------------------------------------------
# Screen 2: pick an example within the chosen lecture
# ------------------------------------------------------------------
elif st.session_state.view == "examples":
    lec = LECTURES[st.session_state.lecture_idx]

    if st.button("← Back to lectures"):
        go_to("lectures")

    st.title(lec["title"])
    # st.caption("Pick an example to open its dashboard.")

    n_cols = 3
    cols = st.columns(n_cols)
    for i, ex in enumerate(lec["examples"]):
        with cols[i % n_cols]:
            if st.button(ex["title"], key=f"ex_{ex['id']}", use_container_width=True):
                go_to("demo", example_id=ex["id"])

# ------------------------------------------------------------------
# Screen 3: the example's own full-screen dashboard
# ------------------------------------------------------------------
elif st.session_state.view == "demo":
    lec = LECTURES[st.session_state.lecture_idx]
    ex = next(e for e in lec["examples"] if e["id"] == st.session_state.example_id)

    top_left, top_right = st.columns([1, 1])
    with top_left:
        if st.button("← Back to examples"):
            go_to("examples")
    with top_right:
        st.caption(f"{lec['title']}  ›  {ex['title']}")

    st.divider()
    ex["render"]()  # <- all example-specific code lives in examples/*.py
