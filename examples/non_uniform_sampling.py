import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.stats import skewnorm

DEFAULT_sl_n_bins = 25
DEFAULT_sl_scale = 1.2

# 2. Initialize the slider states if they don't exist yet
if "sl_n_bins" not in st.session_state:
    st.session_state.sl_n_bins = DEFAULT_sl_n_bins

if "sl_scle" not in st.session_state:
    st.session_state.sl_scale = DEFAULT_sl_scale

# 3. Create a callback function to reset session state values
def reset_sliders():
    st.session_state.sl_n_bins = DEFAULT_sl_n_bins
    st.session_state.sl_scale = DEFAULT_sl_scale


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def get_marker_style(n_samples: int) -> tuple[float, float]:
    """Returns (marker_size, opacity) based on sample size."""
    if n_samples <= 50:
        return 8.0, 0.9
    else:
        return 5.0, 0.7


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def render ():
    st.header("Non-uniform sampling in 1D")

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

    # --- Controls ---
    col1, col2 = st.columns(2)
    a, b = -2., 5.

    with col1:

        if "log_n_val" not in st.session_state:
            st.session_state["log_n_val"] = 1.0

        current_n = int(round(10 ** st.session_state["log_n_val"]))

        st.info(f"**Number of Tosses (N):** `{current_n:,}`")
        log_n = st.slider(
            "Number of Tosses (N)",
            min_value=1.0,
            max_value=5.0,
            value=st.session_state["log_n_val"],
            step=0.01,
            key="log_n_val",
            label_visibility="collapsed",
        )
        n_samples = int(round(10**log_n))

    with col2:

        seed = st.number_input("Random Seed (0 for off)", min_value=0, value=0)

    # --- Data Generation ---
    if seed != 0:
        np.random.seed(seed)

    skew = 5.0  # Positive skew (right-skewed bell curve)
    x = np.linspace(-2, 6, 1000)
    pdf = skewnorm.pdf(x, skew, loc=0, scale=1.5)
    samples = skewnorm.rvs(skew, loc=0, scale=1.5, size=n_samples)

    marker_size, opacity = get_marker_style(n_samples)

    # Bin edges and x-axis limits
    x_lims = (a - 1.5, b + 1.5)

    # ==========================================
    # Empirical CDF built with Heaviside Steps \Theta(x - X_i)
    # ==========================================

    fig4, ax4 = plt.subplots(figsize=(10, 3.5))

    # Sort the sample to evaluate the empirical step function
    x_sorted = np.sort(samples)

    # 1. Empirical CDF using Heaviside Step Function: \hat{F}_N(x) = (1/N) * \sum \Theta(x - X_i)
    # Matplotlib's step plot with `where='post'` draws horizontal lines followed by jumps,
    # which exactly matches the sum of right-continuous \Theta(x - X_i) step functions.
    x_ecdf = np.concatenate([[x_lims[0]], x_sorted, [x_lims[1]]])
    y_ecdf = np.concatenate([[0], np.arange(1, n_samples + 1) / n_samples, [1]])

    ax4.step(
        x_ecdf,
        y_ecdf,
        where="post",
        color="#1f77b4",
        linewidth=1.8,
        label=r"cumulative probability",
    )

    ax4.axvline (x=a, color = 'tomato', linestyle = '--', label = 'x min')
    ax4.axvline (x=b, color = 'tomato', linestyle = '--', label = 'x max')
    ax4.axhline (y=-0.05, color='gray', linestyle='--', linewidth=1)
    
    y_jitter = np.full (n_samples, -0.05)

    ax4.scatter(
        samples,
        y_jitter,
        s=marker_size**2,
        # alpha=opacity,
        color="#002b80",
        edgecolors="none",
        label=r"sample",        
    )

    st.divider()
    st.pyplot(fig4)

    # ==========================================
    # HISTOGRAMS
    # ==========================================

    st.divider()
    st.header("Histogram representation")

    col3, col4 = st.columns(2)

    with col3 :

        n_bins = st.slider(
            "Number of Histogram Bins",
            min_value = 1,
            max_value = 1000,
            # value     = 25,
            step      = 1,
            key       = 'sl_n_bins',
        )
        bins = np.linspace(a, b, n_bins + 1)

        st.button("Reset Sliders to Default", on_click=reset_sliders)

    with col4 :
 
        scale = st.slider(
            f"Horizontal axis size (with respect to {b-a})",
            min_value = 0.1,
            max_value = 50.,
            # value     = 1.2,
            step      = 0.1,
            key       = 'sl_scale',
        )

    mean = 0.5 * (a+b)
    size = b-a
    x_lims = (mean - 0.5 * size * scale, mean + 0.5 * size * scale)
 
    # ==========================================
    # Unnormalized Histogram (Raw Counts)
    # ==========================================

    fig2, ax2 = plt.subplots(figsize=(10, 3.5))

    ax2.hist(
        samples,
        bins=bins,
        density=False,
        color="#B0C4DE",
        edgecolor="#4682B4",
        alpha=0.8,
    )
    ax2.set_xlim(x_lims)
    ax2.set_xlabel("Value (X)")
    ax2.set_ylabel("Count")
    ax2.set_title(
        "Raw Counts Histogram (Not Normalized)",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()

    # ==========================================
    # Normalized Density Histogram (density=True)
    # ==========================================

    fig3, ax3 = plt.subplots(figsize=(10, 3.5))

    # Matplotlib's density=True handles total area normalization reliably
    ax3.hist(
        samples,
        bins=bins,
        density=True,
        color="#ADD8E6",
        edgecolor="#1f77b4",
        alpha=0.8,
        label="Histogram Density",
    )

    ax3.set_xlim(x_lims)
    ax3.set_xlabel("Value (X)")
    ax3.set_ylabel("Density")
    ax3.set_title(
        "Density Histogram (Normalized, density=True)",
        fontsize=11,
        fontweight="bold",
    )
    ax3.legend(loc="upper right")
    plt.tight_layout()

    # ==========================================
    # DISPLAY (Direct Streamlit Pyplot Calls)
    # ==========================================

    st.pyplot(fig2)
    st.divider()
    st.pyplot(fig3)

    # ==========================================
    # HISTOGRAM CUMULATIVE
    # ==========================================

    fig5, ax5 = plt.subplots(figsize=(10, 3.5))

    # Matplotlib's density=True handles total area normalization reliably
    ax5.hist(
        samples,
        bins=bins,
        density=True,
        cumulative=True,
        histtype='step',
        color="#ADD8E6",
        edgecolor="#1f77b4",
        alpha=0.8,
        label="cumulative histogram",
    )

    ax5.step(
        x_ecdf,
        y_ecdf,
        where="post",
        color="#1f77b4",
        linewidth=1.8,
        label=r"cumulative probability",
    )

    ax5.set_xlim(x_lims)
    ax5.set_xlabel("Value (X)")
    ax5.set_ylabel("Probability")
    ax5.set_title(
        "Cumulative",
        fontsize=11,
        fontweight="bold",
    )
    ax5.legend(loc="upper right")
    plt.tight_layout()

    st.pyplot(fig5)
    st.divider()



if __name__ == "__main__":
    st.set_page_config(page_title="Uniform Sampling Demo", layout="wide")
    render ()