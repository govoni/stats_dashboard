import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.stats import skewnorm


def sturges (N_events) :
    return int(np.ceil( 1 + np.log2 (N_events) ))


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
        n_samples = st.slider(
            "Number of Tosses (N)",
            min_value = 1,
            max_value = 500,
            value     = 5,
            step      = 2,
        )

    with col2:

        seed = st.number_input("Random Seed (0 for off)", min_value=0, value=0)

    # --- Data Generation ---
    if seed != 0:
        np.random.seed(seed)

    skew = 5.0  # Positive skew (right-skewed bell curve)
    x = np.linspace(-2, 6, 1000)
    pdf = skewnorm.pdf(x, skew, loc=0, scale=1.5)
    samples = skewnorm.rvs(skew, loc=0, scale=1.5, size=n_samples)
    true_mean = skewnorm.mean(skew, loc=0, scale=1.5)
    true_sigma = skewnorm.std(skew, loc=0, scale=1.5)    


    marker_size, opacity = get_marker_style(n_samples)

    # Bin edges and x-axis limits
    x_lims = (a - 1.5, b + 1.5)
    n_bins = sturges (n_samples)
    bins = np.linspace(a, b, n_bins + 1)

    # ==========================================
    # SAMPLE, HISTOGRAM, MEAN AND SIGMA
    # ==========================================

    fig4, ax4 = plt.subplots(figsize=(10, 3.5))

    # Sort the sample to evaluate the empirical step function
    x_sorted = np.sort(samples)

    # 1. Empirical CDF using Heaviside Step Function: \hat{F}_N(x) = (1/N) * \sum \Theta(x - X_i)
    # Matplotlib's step plot with `where='post'` draws horizontal lines followed by jumps,
    # which exactly matches the sum of right-continuous \Theta(x - X_i) step functions.
    x_ecdf = np.concatenate([[x_lims[0]], x_sorted, [x_lims[1]]])
    y_ecdf = np.concatenate([[0], np.arange(1, n_samples + 1) / n_samples, [1]])

    ax4.hist(
        samples,
        bins=bins,
        density=True,
        color="#ADD8E6",
        edgecolor="#1f77b4",
        alpha=0.8,
        label="Histogram Density",
    )

    ax4.set_xlim(x_lims)
    ax4.set_xlabel("Value (X)")
    ax4.set_ylabel("Density")
    ax4.set_title(
        "Density Histogram (Normalized, density=True)",
        fontsize=11,
        fontweight="bold",
    )

    mean = np.mean (samples)
    sigma = np.std (samples)

    ax4.axvline (x=mean, color = 'red', linestyle = '-', label = '$\mu$')
    ax4.axvline (x=mean-sigma, color = 'sienna', linestyle = '-', label = '$\mu-\sigma$')
    ax4.axvline (x=mean+sigma, color = 'sienna', linestyle = '-', label = '$\mu+\sigma$')

    ax4.axvline (x=true_mean, color = 'red', alpha = 0.5, linestyle = '--', label = '$\mu_t$')
    ax4.axvline (x=true_mean-true_sigma, color = 'sienna', alpha = 0.5, linestyle = '--', label = '$\mu_t-\sigma_t$')
    ax4.axvline (x=true_mean+true_sigma, color = 'sienna', alpha = 0.5, linestyle = '--', label = '$\mu_t+\sigma_t$')

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

    ax4.set_xlim(x_lims)
    ax4.set_xlabel("Value (X)")
    ax4.set_ylabel(r"$F(x) = P(X \leq x)$")
    ax4.set_title(
        "Sample probability",
        fontsize=11,
        fontweight="bold",
    )
    ax4.legend(loc="upper right")

    plt.tight_layout()
    st.pyplot(fig4)


if __name__ == "__main__":
    st.set_page_config(page_title="Uniform Sampling Demo", layout="wide")
    render ()