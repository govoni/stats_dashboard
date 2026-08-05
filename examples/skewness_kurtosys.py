"""
Skewness & kurtosis demo.

A standard Gaussian Z is passed through the sinh-arcsinh transform
(Jones & Pewsey, 2009):

    Y = sinh( (asinh(Z) + epsilon) / delta )

epsilon controls skewness (0 = symmetric), delta controls tail weight /
kurtosis (1 = same tails as Gaussian, <1 = heavier tails, >1 = lighter
tails). epsilon = 0, delta = 1 recovers the Gaussian exactly. This gives
two independent, physically intuitive knobs starting from a Gaussian,
which is exactly what's wanted for teaching the two moments separately
before combining them.
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats

_rng = np.random.default_rng(0)

DEFAULT_sl_epsilon = 0.
DEFAULT_sl_delta = 1.

# 2. Initialize the slider states if they don't exist yet
if "sl_epsilon" not in st.session_state:
    st.session_state.sl_epsilon = DEFAULT_sl_epsilon

if "sl_delta" not in st.session_state:
    st.session_state.sl_delta = DEFAULT_sl_delta

# 3. Create a callback function to reset session state values
def reset_sliders():
    st.session_state.sl_epsilon = DEFAULT_sl_epsilon
    st.session_state.sl_delta = DEFAULT_sl_delta




def render():
    st.header("Skewness and kurtosis: morphing a Gaussian")

    # ---- highlighted formula box (yellow, centered, larger font) ----
    st.markdown(
        """
        <style>
        div[class*="st-key-skew_kurt_box"] {
            background-color: #fff9c4;
            padding: 16px;
            border-radius: 8px;
            text-align: center;
        }
        div[class*="st-key-skew_kurt_box"] p {
            font-size: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # with st.container(key="skew_kurt_box"):
    #     st.markdown("**Third and fourth standardized moments**")
    #     st.latex(
    #         r"\text{skewness} = E\!\left[\left(\frac{X-\mu}{\sigma}\right)^{3}\right]"
    #         r"\qquad"
    #         r"\text{excess kurtosis} = E\!\left[\left(\frac{X-\mu}{\sigma}\right)^{4}\right] - 3"
    #     )

    # st.write("Start from a standard Gaussian Z and apply the sinh-arcsinh transform:")
    # st.latex(r"Y = \sinh\!\left(\frac{\operatorname{asinh}(Z) + \epsilon}{\delta}\right)")
    # st.write(
    #     "**epsilon** shifts the distribution away from symmetry (skewness). "
    #     "**delta** controls tail weight (kurtosis) while keeping it symmetric "
    #     "when epsilon = 0. epsilon = 0, delta = 1 recovers the Gaussian exactly."
    # )

    c1, c2, c3 = st.columns(3)
    with c1:
        epsilon = st.slider("epsilon (skewness knob)", -2.0, 2.0, 0.0, step=0.1, key='sl_epsilon')
    with c2:
        delta = st.slider("delta (tail-weight knob)", 0.3, 3.0, 1.0, step=0.05, key='sl_delta')
    with c3:
        n_samples = st.slider("Sample size", 1000, 50000, 10000, step=1000)

    z = _rng.standard_normal(n_samples)
    y = np.sinh((np.arcsinh(z) + epsilon) / delta)
    y = (y - y.mean()) / y.std()  # standardize -> fair overlay vs standard normal

    skew = stats.skew(y, bias=False)
    exkurt = stats.kurtosis(y, fisher=True, bias=False)  # 0 for a Gaussian

    m1, m2, m3 = st.columns(3)
    m1.metric("Sample skewness", f"{skew:.3f}", "0 = symmetric")
    m2.metric("Sample excess kurtosis", f"{exkurt:.3f}", "0 = Gaussian-like tails")
    with m3 :
        st.button("Reset Sliders to Default", on_click=reset_sliders)

 
    lo, hi = np.percentile(y, [0.5, 99.5])
    x_ref = np.linspace(lo, hi, 400)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    axes[0].hist(y, bins=80, range=(lo, hi), density=True, color="steelblue",
                 alpha=0.7, label="morphed distribution")
    axes[0].plot(x_ref, stats.norm.pdf(x_ref), "r--", lw=1.5, label="standard normal")
    axes[0].set_title("linear scale")
    axes[0].set_xlabel("y (standardized)")
    axes[0].set_ylabel("density")
    axes[0].legend()

    axes[1].hist(y, bins=80, range=(lo, hi), density=True, color="steelblue", alpha=0.7)
    axes[1].plot(x_ref, stats.norm.pdf(x_ref), "r--", lw=1.5)
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-4, 1)
    axes[1].set_title("log scale (tails)")
    axes[1].set_xlabel("y (standardized)")
    axes[1].set_ylabel("density (log)")

    plt.tight_layout()
    st.pyplot(fig)

    # st.info(
    #     "Increase |epsilon| with delta = 1 to see one tail grow longer "
    #     "than the other (skewness != 0). Reset epsilon = 0 and move "
    #     "delta below 1 to see both tails thicken symmetrically "
    #     "(positive excess kurtosis); delta above 1 pulls them in "
    #     "(negative excess kurtosis). The log-scale panel on the right "
    #     "makes tail differences visible in a way the linear histogram "
    #     "can't."
    # )