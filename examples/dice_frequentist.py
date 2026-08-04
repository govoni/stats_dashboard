"""
Frequentist probability via repeated dice rolls. See lectures.py: lecture 1.

Illustrates P(outcome) = (# successes) / (# tosses) for each of the six
faces, and how that ratio converges to 1/6 as the number of tosses grows
(law of large numbers).
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

_MAX_N = 100_000
# log-spaced toss counts -> lets students see convergence without a
# custom log-slider widget
_N_OPTIONS = [1, 2, 5, 10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000]


def render():
    st.header("Rolling a die")

    # background-color: #fff9c4; /* yellow */
    st.markdown ("""
    <style>
    div[class*="st-key-prob_box"] {
        background-color: #ADD8E6; /* blue */
        padding: 36px;
        border-radius: 8px;
        text-align: center;
    }
    div[class*="st-key-prob_box"] p {
        font-size: 22px;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.container (key="prob_box"):
        st.markdown ("**Probability definition:**")
        st.latex (r"P(\text{outcome}) = \lim_{n \to \infty} \frac{n_{\text{successes}}}{n_{\text{tosses}}}")    

    # st.write(
    #     "The frequentist definition: P(outcome) is estimated as the "
    #     "number of times that outcome occurs divided by the total "
    #     "number of trials. Below, a fair six-sided die is rolled "
    #     "repeatedly and the ratio (successes / tosses) is tracked for "
    #     "every face."
    # )
    # st.markdown(
    #     r"""
    #     <div style='background-color:#fff9c4; padding:16px; border-radius:8px;
    #                 text-align:center; font-size:22px;'>
    #     <b>Probability definition:</b><br>
    #     $$P(\text{outcome}) = \lim_{n \to \infty} \frac{n_{\text{successes}}}{n_{\text{tosses}}}$$
    #     </div>
    #     """,
    #     unsafe_allow_html=True,
    # )

    # st.markdown(
    #     r":yellow-background[**Frequentist definition:** "
    #     r"$P(\text{outcome}) = \displaystyle\lim_{n \to \infty} \frac{n_{\text{successes}}}{n_{\text{tosses}}}$]"
    # )

    # st.markdown(r"$$\hat p = \frac{1}{n}\sum_{i=1}^n \mathbb{1}[x_i = k]$$")
    # st.markdown(r"As $n \to \infty$, the frequency converges to $P = 1/6$.")

    if "dice_seed" not in st.session_state:
        st.session_state.dice_seed = 0

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        n_tosses = st.select_slider("Number of tosses", options=_N_OPTIONS,
                                     value=2)
    with c2:
        highlight_face = st.selectbox("Face to highlight below", list(range(1, 7)))
    with c3:
        st.write("")  # vertical spacer to align button with the sliders
        st.write("")
        if st.button("🎲 New sequence of rolls"):
            st.session_state.dice_seed += 1

    # Fixed sequence for the current seed; slicing it as n_tosses grows
    # keeps the convergence plot consistent (adding rolls, not redrawing
    # a fresh independent sample each time).
    rng = np.random.default_rng(st.session_state.dice_seed)
    full_sequence = rng.integers(1, 7, size=_MAX_N)
    rolls = full_sequence[:n_tosses]

    counts = np.array([(rolls == face).sum() for face in range(1, 7)])
    freqs = counts / n_tosses

    # ---- bar chart: relative frequency of every outcome ----
    fig1, ax1 = plt.subplots(figsize=(11, 4))
    faces = np.arange(1, 7)
    ax1.bar(faces, freqs, color="steelblue", alpha=0.8, label="observed frequency")
    ax1.axhline(1 / 6, color="red", ls="--", lw=1.5, label="true probability = 1/6")
    ax1.set_xticks(faces)
    ax1.set_xlabel("outcome (face value)")
    ax1.set_ylabel("relative frequency")
    ax1.set_ylim(0, max(freqs.max(), 1 / 6) * 1.3)
    ax1.set_title(f"{n_tosses} tosses")
    ax1.legend()
    st.pyplot(fig1)

    # ---- table ----
    st.write("Successes / tosses for each outcome:")
    cols = st.columns(6)
    for i, face in enumerate(faces):
        with cols[i]:
            st.metric(f"Face {face}", f"{freqs[i]:.4f}",
                      f"{counts[i]} / {n_tosses}")

    # ---- convergence of one face as n grows (law of large numbers) ----
    st.subheader(f"Convergence for face {highlight_face} as tosses increase")
    is_success = (full_sequence[:n_tosses] == highlight_face).astype(float)
    cum_freq = np.cumsum(is_success) / np.arange(1, n_tosses + 1)

    # subsample points (log-spaced) so the plot stays fast even at n=100000
    n_points = min(400, n_tosses)
    idx = np.unique(np.geomspace(1, n_tosses, n_points).astype(int) - 1)

    fig2, ax2 = plt.subplots(figsize=(11, 4))
    ax2.plot(idx + 1, cum_freq[idx], color="darkorange", lw=1.5,
             label=f"running frequency of face {highlight_face}")
    ax2.axhline(1 / 6, color="red", ls="--", lw=1.5, label="true probability = 1/6")
    ax2.set_xscale("log")
    ax2.set_xlabel("number of tosses (log scale)")
    ax2.set_ylabel("relative frequency")
    ax2.legend()
    st.pyplot(fig2)

    # st.info(
    #     "Click 'New sequence of rolls' to see a different random "
    #     "realization -- at small n the frequencies fluctuate a lot "
    #     "and can be far from 1/6; the orange curve settles down as "
    #     "the number of tosses grows, which is exactly the intuition "
    #     "the frequentist definition of probability relies on."
    # )
