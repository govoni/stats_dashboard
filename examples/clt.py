import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import streamlit as st


# ------------------------------------------------------------------
# 1. Theoretical Moments & Samplers
# ------------------------------------------------------------------
def get_dist_properties(dist_type):
    """Returns (sampler_func, mean, variance) for the selected distribution."""
    if dist_type == 'Uniform':
        mean = 0.5
        var = 1.0 / 12.0
        sampler = lambda size: np.random.uniform(0, 1, size)

    elif dist_type == 'Triangular (Symmetric)':
        mean = 0.5
        var = 1.0 / 24.0
        sampler = lambda size: np.random.triangular(0, 0.5, 1, size)

    elif dist_type == 'Triangular (Asymmetric)':
        c = 0.1  # Peak at 0.1 (Right-skewed)
        mean = (1.0 + c) / 3.0
        var = (1.0 - c + c**2) / 18.0
        sampler = lambda size: np.random.triangular(0, c, 1, size)

    elif dist_type == 'Parabolic (U-shape)':
        mean = 0.5
        var = 3.0 / 20.0
        def sampler(size):
            u = np.random.uniform(0, 1, size)
            return 0.5 + np.cbrt((u - 0.5) / 4.0)
            
    return sampler, mean, var

@st.cache_data
def generate_clt_samples(dist_type, N, num_samples=30000):
    """Generates normalized sums Z_N for a given N."""
    sampler, mean, var = get_dist_properties(dist_type)
    draws = sampler((num_samples, N))
    sum_N = np.sum(draws, axis=1)
    
    # Exact CLT Normalization to N(0, 1)
    Z_N = (sum_N - N * mean) / np.sqrt(N * var)
    return Z_N


def render():
    st.header("Central Limit Theorem")

    c1, c2, c3 = st.columns(3)

    with c1 :
        dist_choice = st.selectbox(
            "Select Input Distribution:",
            ['Uniform', 'Triangular (Symmetric)', 'Triangular (Asymmetric)', 'Parabolic (U-shape)']
        )

    with c2 :
        N_events = st.slider(
            "Number of Added Events (N):",
            min_value=1,
            max_value=20,
            value=1,
            step=1
        )

    with c3 :
        num_trials = st.slider(
            "Number of Monte Carlo Trials:",
            min_value=5000,
            max_value=500000,
            value=25000,
            step=5000
        )

    # ------------------------------------------------------------------
    # 3. Calculations & Data Generation
    # ------------------------------------------------------------------

    Z_data = generate_clt_samples(dist_choice, N_events, num_samples=num_trials)

    # Calculate metrics for the selected N
    current_skew = stats.skew(Z_data)
    current_kurt = stats.kurtosis(Z_data)  # Fisher kurtosis (Normal = 0)
    current_std = np.std(Z_data, ddof=1)

    # Display Key Metrics in Streamlit Columns
    col1, col2 = st.columns(2)
    col1.metric("Skewness (Target: 0.0)", f"{current_skew:.4f}")
    col2.metric("Excess Kurtosis (Target: 0.0)", f"{current_kurt:.4f}")

    # ------------------------------------------------------------------
    # 4. Matplotlib Plot Rendering
    # ------------------------------------------------------------------

    fig, ax_hist = plt.subplots(figsize=(10, 3.5))

    ax_hist.hist(
        Z_data, bins=60, density=True, alpha=0.6, color='skyblue', 
        edgecolor='black', label=f'Empirical Sum ($N={N_events}$)'
    )

    x_grid = np.linspace(-4, 4, 300)
    gaussian_pdf = stats.norm.pdf(x_grid, loc=0, scale=1)
    ax_hist.plot(x_grid, gaussian_pdf, 'r--', linewidth=2.5, label='Standard Normal $\\mathcal{N}(0, 1)$')

    ax_hist.set_title(
        f"Distribution of Normalized Sums for {dist_choice} ($N = {N_events}$)",
        fontsize=11,
        fontweight="bold",
    )

    ax_hist.set_xlabel("Normalized Value $Z_N$")
    ax_hist.set_ylabel("Probability Density")
    ax_hist.set_xlim(-4, 4)
    ax_hist.legend(loc='upper right')
    ax_hist.grid(True, linestyle=':', alpha=0.6)

    # --- Bottom Plot: Convergence of Moments ---
    n_range = np.arange(1, 51)
    skewness_list, kurtosis_list, std_list = [], [], []

    # Compute metrics over range for convergence curve
    for n in n_range:
        Z_n = generate_clt_samples(dist_choice, n, num_samples=5000)
        skewness_list.append(stats.skew(Z_n))
        kurtosis_list.append(stats.kurtosis(Z_n))
        std_list.append(np.std(Z_n, ddof=1))

    plt.tight_layout()

    # ------------------------------------------------------------------
    # 4. Convergence
    # ------------------------------------------------------------------

    fig2, ax_convergence = plt.subplots(figsize=(10, 3.5))

    ax_convergence.plot(n_range, skewness_list, 'o-', color='darkorange', markersize=4, label='Skewness $\\rightarrow 0$')
    ax_convergence.plot(n_range, kurtosis_list, 's-', color='purple', markersize=4, label='Excess Kurtosis $\\rightarrow 0$')
    # ax_convergence.plot(n_range, std_list, '^--', color='green', markersize=4, label='Sample $\\sigma \\rightarrow 1.0$')

    # Reference Lines
    ax_convergence.axhline(0, color='black', linestyle=':', linewidth=1)
    # ax_convergence.axhline(1, color='green', linestyle=':', linewidth=1)
    ax_convergence.axvline(N_events, color='red', linestyle='--', alpha=0.7, label=f'Current $N = {N_events}$')

    ax_convergence.set_title(
        "Convergence of Sample Moments vs $N$",
        fontsize=11,
        fontweight="bold",
    )

    ax_convergence.set_xlabel("Number of Added Events ($N$)")
    ax_convergence.set_ylabel("Metric Value")
    ax_convergence.set_xlim(1, 50)
    ax_convergence.set_ylim(-1.5, 2.0)
    ax_convergence.legend(loc='upper right', ncol=2, fontsize='small')
    ax_convergence.grid(True, linestyle=':', alpha=0.6)

    # Render Plot inside Streamlit
    plt.tight_layout()

    # ------------------------------------------------------------------
    # DRAWING
    # ------------------------------------------------------------------


    st.pyplot(fig)
    st.divider()
    st.pyplot(fig2)
