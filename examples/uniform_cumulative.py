import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def get_marker_style(n_samples: int) -> tuple[float, float]:
    """Returns (marker_size, opacity) based on sample size."""
    if n_samples <= 10:
        return 8.0, 0.9
    elif n_samples <= 100:
        return 5.0, 0.7
    else:
        return 2.0, 0.35


def render ():
    st.header("Cumulative 1D Uniform Sampling")

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
        st.latex(
            r"""
            \hat{F}_N(x) = \frac{1}{N} \sum \Theta(x - X_i)
        """
        )

    # --- Controls ---
    col1, col2 = st.columns(2)
    a, b = 0., 1.

    with col1:

        if "log_n_val" not in st.session_state:
            st.session_state["log_n_val"] = 1.0

        current_n = int(round(10 ** st.session_state["log_n_val"]))

        st.info (f"**Number of Tosses (N):** `{current_n:,}`")
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
        show_vlines = st.checkbox ("Show Vertical Sample Drop Lines", value=True)
        seed = st.number_input("Random Seed (0 for off)", min_value=0, value=0)


    # --- Data Generation ---
    if seed != 0:
        np.random.seed(seed)

    samples = np.random.uniform(low=a, high=b, size=n_samples)
    marker_size, opacity = get_marker_style(n_samples)
    marker_size = 8

    x_lims = (a - 1.5, b + 1.5)

    # ==========================================
    # DRAWING: Empirical CDF built with Heaviside Steps \Theta(x - X_i)
    # ==========================================

    fig4, ax4 = plt.subplots(figsize=(10, 3.5))

    # Sort the sample to evaluate the empirical step function
    x_sorted = np.sort(samples)

    # 1. Empirical CDF using Heaviside Step Function: \hat{F}_N(x) = (1/N) * \sum \Theta(x - X_i)
    # Matplotlib's step plot with `where='post'` draws horizontal lines followed by jumps,
    # which exactly matches the sum of right-continuous \Theta(x - X_i) step functions.
    x_ecdf = np.concatenate([[x_lims[0]], x_sorted, [x_lims[1]]])
    y_ecdf = np.concatenate([[0], np.arange(1, n_samples + 1) / n_samples, [1]])

    if show_vlines :
        x_sorted = np.sort(samples)
        y_ecdf_tops = np.arange(1, n_samples + 1) / n_samples  # ECDF value after step i
        ax4.vlines(
            x=x_sorted,
            ymin=-0.05,
            ymax=y_ecdf_tops,
            color="#ff704d",
            linestyle="-",  # Dotted lines : (use '-' for solid)
            linewidth=0.8, # if n_samples > 50 else 1.2,
            alpha=0.5 if n_samples > 50 else 0.8,
            # label="Sample Locations $X_i$",
        )

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


    # 2. Theoretical CDF F(x) = (x - a) / (b - a) for a <= x <= b
    # if b > a:
    #     x_cdf_theory = [x_lims[0], a, b, x_lims[1]]
    #     y_cdf_theory = [0, 0, 1, 1]
    #     ax4.plot(
    #         x_cdf_theory,
    #         y_cdf_theory,
    #         color="#D32F2F",
    #         linestyle="--",
    #         linewidth=2,
    #         label=r"Theoretical CDF $F(x)$",
    #     )

    ax4.set_xlim(x_lims)
    ax4.set_ylim(-0.1, 1.05)
    ax4.set_xlabel("Value (X)")
    ax4.set_ylabel(r"$F(x) = P(X \leq x)$")
    ax4.set_title(
        "Cumulative Distribution Function (ECDF via Step Function $\Theta$)",
        fontsize=11,
        fontweight="bold",
    )
    ax4.legend(loc="upper left")
    plt.tight_layout()

    # ==========================================
    # DISPLAY (Sequential Stack)
    # ==========================================
    # st.pyplot(fig1)
    # st.divider()
    st.pyplot(fig4)


if __name__ == "__main__":
    st.set_page_config(page_title="Uniform Sampling Demo", layout="wide")
    render_uniform_distribution_example()