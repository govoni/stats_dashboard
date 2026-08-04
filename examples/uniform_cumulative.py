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
            X \sim \mathcal{U}(a, b) \implies f(x) = 
            \begin{cases} 
              \frac{1}{b - a} & \text{for } a \le x \le b \\[8pt]
              0 & \text{otherwise} 
            \end{cases}
        """
        )

    # --- Controls ---
    col1, col2 = st.columns(2)

    with col1:
        bounds = st.slider(
            "Uniform Range [a, b]",
            min_value=-10.0,
            max_value=10.0,
            value=(-2.0, 5.0),
            step=0.5,
        )
        a, b = bounds

        if "log_n_val" not in st.session_state:
            st.session_state["log_n_val"] = 1.0

        current_n = int(round(10 ** st.session_state["log_n_val"]))

        st.write(f"**Number of Tosses (N):** `{current_n:,}`")
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
        n_bins = st.slider(
            "Number of Histogram Bins",
            min_value=5,
            max_value=100,
            value=25,
            step=1,
        )

        seed = st.number_input("Random Seed (0 for off)", min_value=0, value=0)

    # --- Data Generation ---
    if seed != 0:
        np.random.seed(seed)

    samples = np.random.uniform(low=a, high=b, size=n_samples)
    marker_size, opacity = get_marker_style(n_samples)

    # Bin edges and x-axis limits
    bins = np.linspace(a, b, n_bins + 1)
    x_lims = (a - 1.5, b + 1.5)

    # ==========================================
    # DRAWING 1: 1D Scatter Points
    # ==========================================
    fig1, ax1 = plt.subplots(figsize=(10, 1.8))
    # y_jitter = np.random.uniform(-0.1, 0.1, size=n_samples)
    y_jitter = np.zeros (n_samples)

    ax1.scatter(
        samples,
        y_jitter,
        s=marker_size**2,
        alpha=opacity,
        color="#1f77b4",
        edgecolors="none",
    )
    ax1.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax1.set_xlim(x_lims)
    ax1.set_ylim(-0.25, 0.25)
    ax1.set_yticks([])
    ax1.set_title("Drawing 1: 1D Tossed Samples", fontsize=11, fontweight="bold")
    plt.tight_layout()

    # ==========================================
    # DRAWING 2: Unnormalized Histogram (Raw Counts)
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
        "Drawing 2: Raw Counts Histogram (Not Normalized)",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()

    # ==========================================
    # DRAWING 3: Normalized Density Histogram (density=True)
    # ==========================================
    fig3, ax3 = plt.subplots(figsize=(10, 3.5))

    ax3.hist(
        samples,
        bins=bins,
        density=True,
        color="#ADD8E6",
        edgecolor="#1f77b4",
        alpha=0.8,
        label="Empirical Density",
    )

    if b > a:
        pdf_height = 1.0 / (b - a)
        x_pdf = [x_lims[0], a, a, b, b, x_lims[1]]
        y_pdf = [0, 0, pdf_height, pdf_height, 0, 0]
        # ax3.plot(
        #     x_pdf,
        #     y_pdf,
        #     color="#D32F2F",
        #     linestyle="--",
        #     linewidth=2,
        #     label="Theoretical PDF $f(x)$",
        # )
        ax3.set_ylim(0, pdf_height * 1.25)

    ax3.set_xlim(x_lims)
    ax3.set_xlabel("Value (X)")
    ax3.set_ylabel("Density")
    ax3.set_title(
        "Drawing 3: Density Histogram (Normalized, density=True)",
        fontsize=11,
        fontweight="bold",
    )
    ax3.legend(loc="upper right")
    plt.tight_layout()

    # ==========================================
    # DRAWING 4: Empirical CDF built with Heaviside Steps \Theta(x - X_i)
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
        label=r"Empirical CDF $\hat{F}_N(x) = \frac{1}{N} \sum \Theta(x - X_i)$",
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
    ax4.set_ylim(-0.05, 1.05)
    ax4.set_xlabel("Value (X)")
    ax4.set_ylabel(r"$F(x) = P(X \leq x)$")
    ax4.set_title(
        "Drawing 4: Cumulative Distribution Function (ECDF via Step Function $\Theta$)",
        fontsize=11,
        fontweight="bold",
    )
    ax4.legend(loc="upper left")
    plt.tight_layout()

    # ==========================================
    # DISPLAY (Sequential Stack)
    # ==========================================
    st.pyplot(fig1)
    st.divider()
    st.pyplot(fig4)


if __name__ == "__main__":
    st.set_page_config(page_title="Uniform Sampling Demo", layout="wide")
    render_uniform_distribution_example()