import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def get_marker_style(n_samples: int) -> tuple[float, float]:
    """Returns (marker_size, opacity) based on sample size for 2D scatter."""
    if n_samples <= 50:
        return 12.0, 0.85
    elif n_samples <= 500:
        return 6.0, 0.65
    elif n_samples <= 5000:
        return 2.5, 0.45
    else:
        return 1.0, 0.25


def render ():
    st.header("2D Uniform Sampling")

    # Domain fixed at [-2, 2] x [-2, 2]
    x_min, x_max = -2.0, 2.0
    y_min, y_max = -2.0, 2.0
    area = (x_max - x_min) * (y_max - y_min)  # Area = 16
    theoretical_pdf = 1.0 / area  # f(x, y) = 1/16 = 0.0625

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
            (X, Y) \sim \mathcal{U}([x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}]) 
            \implies f(x, y) = 
            \begin{cases} 
              \frac{1}{\Delta x ~\Delta y} & \text{for } x, y \in \text{Domain} \\[8pt]
              0 & \text{otherwise} 
            \end{cases}
        """
        )

    # --- Controls (Only N, Bins, and Seed) ---
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Fixed Domain:** `[{x_min}, {x_max}] × [{y_min}, {y_max}]`")

        if "log_n_val_2d" not in st.session_state:
            st.session_state["log_n_val_2d"] = 2.0

        current_n = int(round(10 ** st.session_state["log_n_val_2d"]))
        st.write(f"**Number of Events (N):** `{current_n:,}`")

        log_n = st.slider(
            "Number of Events (N)",
            min_value=1.0,
            max_value=6.0,
            value=st.session_state["log_n_val_2d"],
            step=0.01,
            key="log_n_val_2d",
            label_visibility="collapsed",
        )
        n_samples = int(round(10**log_n))

    with col2:
        n_bins = st.slider(
            "Grid Bins (per axis)",
            min_value=5,
            max_value=50,
            value=20,
            step=1,
        )

        seed = st.number_input("Random Seed (0 for off)", min_value=0, value=0)

    # --- Data Generation ---
    if seed != 0:
        np.random.seed(seed)

    x_samples = np.random.uniform(low=x_min, high=x_max, size=n_samples)
    y_samples = np.random.uniform(low=y_min, high=y_max, size=n_samples)

    marker_size, opacity = get_marker_style(n_samples)

    # Grid Edges & Display Limits
    x_bins = np.linspace(x_min, x_max, n_bins + 1)
    y_bins = np.linspace(y_min, y_max, n_bins + 1)

    padding = 0.5
    x_lims = (x_min - padding, x_max + padding)
    y_lims = (y_min - padding, y_max + padding)

    # ==========================================
    # DRAWING 1: 2D Scatter Points
    # ==========================================
    fig1, ax1 = plt.subplots(figsize=(8, 6.5))

    ax1.scatter(
        x_samples,
        y_samples,
        s=marker_size**2,
        alpha=opacity,
        color="#1f77b4",
        edgecolors="none",
    )

    # Domain boundary box
    rect = plt.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        fill=False,
        edgecolor="#D32F2F",
        linestyle="--",
        linewidth=1.8,
        label="Sampling Domain",
    )
    ax1.add_patch(rect)

    ax1.set_xlim(x_lims)
    ax1.set_ylim(y_lims)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title("2D Tossed Samples", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right")
    plt.tight_layout()

    # ==========================================
    # DRAWING 2: Unnormalized 2D Histogram (Raw Counts)
    # ==========================================
    fig2, ax2 = plt.subplots(figsize=(8, 6.5))

    counts, _, _, im2 = ax2.hist2d(
        x_samples,
        y_samples,
        bins=[x_bins, y_bins],
        cmap="Blues",
        density=False,
    )

    cbar2 = fig2.colorbar(im2, ax=ax2)
    cbar2.set_label("Raw Event Count per Cell")

    ax2.set_xlim(x_lims)
    ax2.set_ylim(y_lims)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title(
        "2D Raw Counts Histogram (Not Normalized)",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()

    # ==========================================
    # DRAWING 3: Normalized 2D Density Histogram (density=True)
    # ==========================================
    fig3, ax3 = plt.subplots(figsize=(8, 6.5))

    # density=True normalizes so volume under 2D histogram integral = 1
    density, _, _, im3 = ax3.hist2d(
        x_samples,
        y_samples,
        bins=[x_bins, y_bins],
        # cmap="viridis",
        cmap="Blues",
        density=True,
        vmin=0.0,  # Force colorbar minimum to 0
        vmax=theoretical_pdf * 1.5,
    )

    cbar3 = fig3.colorbar(im3, ax=ax3)
    cbar3.set_label("Probability Density $f(x, y)$")
    cbar3.ax.axhline(
        y=theoretical_pdf,
        color="red",
        linewidth=4,
        linestyle="-",
        zorder=5,  # Keep line on top
    )

    ax3.set_xlim(x_lims)
    ax3.set_ylim(y_lims)
    ax3.set_aspect("equal", adjustable="box")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_title(
        f"2D Density Histogram (Normalized, Theoretical $f(x,y)={theoretical_pdf:.4f}$)",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()

    # ==========================================
    # DRAWING 4: 2D Empirical CDF using 2D Step Functions \Theta(x - X_i)\Theta(y - Y_i)
    # ==========================================
    fig4, ax4 = plt.subplots(figsize=(8, 6.5))

    # Evaluate 2D ECDF over an evaluation grid
    grid_size = 100
    grid_x = np.linspace(x_lims[0], x_lims[1], grid_size)
    grid_y = np.linspace(y_lims[0], y_lims[1], grid_size)
    GX, GY = np.meshgrid(grid_x, grid_y)

    # Vectorized calculation of \hat{F}_N(x, y) = (1/N) * \sum \Theta(x - X_i) * \Theta(y - Y_i)
    # Theta(x - X_i) is 1 if x >= X_i else 0
    ecdf_2d = np.mean(
        (GX[:, :, None] >= x_samples[None, None, :])
        & (GY[:, :, None] >= y_samples[None, None, :]),
        axis=2,
    )

    contour = ax4.contourf(GX, GY, ecdf_2d, levels=20, cmap="magma")
    cbar4 = fig4.colorbar(contour, ax=ax4)
    cbar4.set_label(r"$\hat{F}_N(x, y) = P(X \leq x, Y \leq y)$")

    # Overlay theoretical 2D CDF contours inside domain
    # F(x, y) = ((x - x_min) / (x_max - x_min)) * ((y - y_min) / (y_max - y_min))
    X_domain = np.clip(GX, x_min, x_max)
    Y_domain = np.clip(GY, y_min, y_max)
    F_theory = (
        ((X_domain - x_min) / (x_max - x_min))
        * ((Y_domain - y_min) / (y_max - y_min))
        * (GX >= x_min)
        * (GY >= y_min)
    )

    lines = ax4.contour(
        GX,
        GY,
        F_theory,
        levels=[0.2, 0.4, 0.6, 0.8],
        colors="white",
        linestyles="--",
        linewidths=1.2,
    )
    ax4.clabel(lines, inline=True, fontsize=8, fmt="F=%.1f")

    ax4.set_xlim(x_lims)
    ax4.set_ylim(y_lims)
    ax4.set_aspect("equal", adjustable="box")
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    ax4.set_title(
        r"2D Empirical CDF $\hat{F}_N(x,y) = \frac{1}{N}\sum \Theta(x-X_i)\Theta(y-Y_i)$",
        fontsize=10,
        fontweight="bold",
    )
    plt.tight_layout()

    # ==========================================
    # DISPLAY (Sequential Stack)
    # ==========================================


    with st.container (key="prob_box_2"):
        st.markdown (f'**number of samples: {n_samples}**')

    col3, col4 = st.columns([1.0, 1.18])

    with col3:
        st.pyplot (fig1)
    with col4:
        st.pyplot (fig3)

    # st.pyplot(fig2)
    # st.divider()
    # st.divider()
    # st.pyplot(fig4)


if __name__ == "__main__":
    st.set_page_config(page_title="2D Uniform Sampling Demo", layout="wide")
    render_2d_uniform_sampling()