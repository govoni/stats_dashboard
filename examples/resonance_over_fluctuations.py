import numpy as np
import plotly.graph_objects as go
import streamlit as st


def render():
    st.header("Resonance Discovery & Signal Significance ($S/\\sqrt{B}$)")

    # Session state seed tracker for independent pseudo-experiments
    if "exp_seed" not in st.session_state:
        st.session_state["exp_seed"] = np.random.randint(0, 1_000_000)

    # --- Sidebar / Column Controls ---
    col_ctrl, col_plot = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Event Yields")
        n_bg = st.slider("Background Events ($N_B$)", 500, 50000, 10000, 500)
        n_sig = st.slider("Signal Events ($N_S$)", 0, 1000, 200, 10)

        st.markdown("---")
        st.subheader("Resonance Shape")
        mu = st.slider("Peak Position $\\mu$", 110.0, 140.0, 125.0, 0.5)
        sigma = st.slider("Resolution $\\sigma$", 0.5, 5.0, 2.0, 0.1)

        st.markdown("---")
        if st.button("Run New Pseudo-Experiment"):
            st.session_state["exp_seed"] = np.random.randint(0, 1_000_000)

    # --- Data Generation & Binning ---
    rng = np.random.default_rng(st.session_state["exp_seed"])

    mass_min, mass_max = 100.0, 150.0
    n_bins = 50
    bin_edges = np.linspace(mass_min, mass_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    # Sample raw counts
    bg_events = rng.uniform(mass_min, mass_max, size=n_bg)
    sig_events = rng.normal(mu, sigma, size=n_sig)
    sig_events = sig_events[
        (sig_events >= mass_min) & (sig_events <= mass_max)
    ]

    # Histogram binning
    bg_counts, _ = np.histogram(bg_events, bins=bin_edges)
    sig_counts, _ = np.histogram(sig_events, bins=bin_edges)

    # Significance metric calculated inside the signal window (± 2σ)
    peak_mask = (bin_centers >= mu - 2 * sigma) & (bin_centers <= mu + 2 * sigma)
    s_peak = np.sum(sig_counts[peak_mask])
    b_peak = np.sum(bg_counts[peak_mask])
    significance = s_peak / np.sqrt(b_peak) if b_peak > 0 else 0.0

    # --- Stacked Histogram Plot ---
    fig = go.Figure()

    # Bottom layer: Background
    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=bg_counts,
            width=bin_width,
            name="Uniform Background",
            marker_color="#4A5568",  # Slate Gray
            opacity=0.85,
        )
    )

    # Top layer: Signal stacked on top
    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=sig_counts,
            width=bin_width,
            name="Gaussian Signal",
            marker_color="#E53E3E",  # Red
            opacity=0.9,
        )
    )

    fig.update_layout(
        barmode="stack",
        xaxis_title="Value (X)",
        yaxis_title="Events / Bin",
        bargap=0.0,
        hovermode="x unified",
        margin=dict(l=10, r=10, b=10, t=20),
        legend=dict(x=0.02, y=0.95),
    )

    with col_plot:
        st.plotly_chart(fig, use_container_width=True)

        # --- Significance Display ---
        m1, m2, m3 = st.columns(3)
        m1.metric("Signal in Peak ($S_{2\\sigma}$)", f"{s_peak} events")
        m2.metric("Background in Peak ($B_{2\\sigma}$)", f"{b_peak} events")
        m3.metric("Significance ($S/\\sqrt{B}$)", f"{significance:.2f} σ")


if __name__ == "__main__":
    render()