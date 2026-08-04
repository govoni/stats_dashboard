import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def get_marker_style(n_samples: int) -> tuple[float, float]:
    """Returns (marker_size, opacity) based on sample size."""
    if n_samples <= 50:
        return 8.0, 0.9
    else:
        return 5.0, 0.7


def render ():
    st.header("Uniform sampling in 1D")

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
    ax1.set_title(f"{n_samples} Tossed Samples", fontsize=11, fontweight="bold")
    ax1.axvline (x=a, color = 'tomato', linestyle = '--', label = 'a')
    ax1.axvline (x=b, color = 'tomato', linestyle = '--', label = 'b')
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
        "Raw Counts Histogram (Not Normalized)",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()

    # ==========================================
    # DRAWING 3: Normalized Density Histogram (density=True)
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
        label="Empirical Density",
    )

    # Theoretical Uniform PDF step plot
    if b > a:
        pdf_height = 1.0 / (b - a)
        x_pdf = [x_lims[0], a, a, b, b, x_lims[1]]
        y_pdf = [0, 0, pdf_height, pdf_height, 0, 0]
        ax3.plot(
            x_pdf,
            y_pdf,
            color="#D32F2F",
            linestyle="--",
            linewidth=2,
            label="Theoretical PDF $f(x)$",
        )
        # ax3.set_ylim(0, pdf_height * 1.25)

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

    # with st.container (key="prob_box_2"):
    #     st.markdown (f'**number of samples: {n_samples}**')

    st.divider()
    st.pyplot(fig1)
    st.divider()
    st.header("Histogram representation")
    st.pyplot(fig2)
    st.divider()
    st.pyplot(fig3)

    with st.container (key="prob_box"):
        # st.markdown ("**Probability definition:**")
        st.latex(
            r"""
            X \sim \mathcal{U}(a, b) \implies f(x) = 
            \begin{cases} 
              \frac{1}{b - a} & \text{for } a \le x \le b \\[8pt]
              0 & \text{otherwise} 
            \end{cases}
        """
        )


if __name__ == "__main__":
    st.set_page_config(page_title="Uniform Sampling Demo", layout="wide")
    render_uniform_distribution_example()