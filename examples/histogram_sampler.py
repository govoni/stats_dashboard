"""
Reusable Streamlit component: repeated sampling with an accumulating
overlaid histogram, plus per-bin event-count distributions.

Usage from the host app:

    import histogram_sampler
    histogram_sampler.render()

Everything (state, algorithms, widgets) is self-contained in this
module; call render() from wherever it should appear in the page.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

MIN_BINS = 5
MAX_BINS = 21
N_CENTRAL_BINS = 5  # how many of the most central bins get a per-bin count plot
STATE_KEY = "histogram_sampler"  # namespace to avoid clashing with host app state


def _draw_samples(dist: str, low: float, high: float, mean: float, std: float, n: int) -> np.ndarray:
    """Generate N samples from the chosen distribution."""
    if dist == "Gaussian":
        return np.random.normal(loc=mean, scale=std, size=n)
    return np.random.uniform(low=low, high=high, size=n)


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def _bin_edges(low: float, high: float, n_bins: int) -> np.ndarray:
    return np.linspace(low, high, n_bins + 1)


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def _central_bin_indices(n_bins: int, n_central: int) -> range:
    """Indices (into a length-n_bins array) of the n_central most central bins."""
    n_central = min(n_central, n_bins)
    start = (n_bins - n_central) // 2
    return range(start, start + n_central)


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def _init_state(config):
    st.session_state[STATE_KEY] = {
        "config": config,
        "trials": [],       # list of raw sample arrays, one per draw
        "bin_counts": [],   # list of length-N_BINS count arrays, one per draw
    }


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def _main_histogram_figure (trials, bin_edges, low, high):
    fig, ax = plt.subplots (figsize=(10, 3.5))
    if not trials:
        ax.set_xlim (low, high)
        ax.text (0.5, 0.5, "Press 'Generate' to draw a sample",
                ha="center", va="center", transform=ax.transAxes, color="gray")
    else:
        # all but the most recent draw: fixed reference color/style
        for samples in trials[:-1]:
            ax.hist(samples, bins=bin_edges, histtype='step',
                    color='mediumblue', alpha=0.5, linewidth=1.2)
        # most recent draw: highlighted in red
        ax.hist(trials[-1], bins=bin_edges,# histtype='step',
                color='lightcoral', alpha=0.5, linewidth=1.5)
        ax.hist(trials[-1], bins=bin_edges, histtype='step',
                edgecolor='red', color='lightcoral', linewidth=1.5)
        for edge in bin_edges:
            ax.axvline(edge, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Value (X)")
    ax.set_ylabel("count")
    return fig
 

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def _bin_count_figure (per_bin_series):
    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    min_c = int (per_bin_series.min())
    max_c = int (per_bin_series.max())
    step = 1
    if (max_c > 20) : step = 2
    if (max_c > 50) : step = 3
    if (max_c > 100) : step = 5
    if (max_c > 1000) : step = 20
    # integer-valued counts -> unit-width bins centered on each integer,
    # spanning just the observed range (not necessarily starting at 0)
    edges = np.arange(min_c - 8, max_c + 8, step) - 0.5
    ax.hist(per_bin_series, bins=edges, color="steelblue",
            edgecolor="black", linewidth=0.5)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel("count in bin")
    ax.set_ylabel("# draws")
    return fig
 

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

 
def _bin_count_figure_old (per_bin_series):
    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    max_c = per_bin_series.max()
    edges = np.arange(0, max_c + 2) - 0.5
    ax.hist(per_bin_series, bins=edges, color="steelblue",
            edgecolor="black", linewidth=0.5)
    ax.set_xlabel("count in bin")
    ax.set_ylabel("# draws")
    return fig


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def render():
    """Render the sampler widget (controls + plots) at the call site."""

    with st.container():
        st.header ("Histogram bin sampling")

        col_1, col_2, col_3 = st.columns(3)
        with col_1 :
            dist = st.selectbox("Distribution", ["Gaussian", "Uniform"], key=f"{STATE_KEY}_dist")
            if dist == "Gaussian":
                mean = 0.
                std  = 1.
                low, high = mean - 3 * std, mean + 3 * std
            else:
                mean, std = 0.0, 1.0
                low = 0.
                high = 1. 
        with col_2 :
            n = st.number_input("N (samples per draw)", value=200, min_value=1,
                                 step=50, key=f"{STATE_KEY}_n")
        with col_3 :
            n_bins = st.slider("Number of bins", min_value=MIN_BINS, max_value=MAX_BINS,
                                value=MIN_BINS, step=2, key=f"{STATE_KEY}_nbins")

        col_a, col_b, col_c, col_d = st.columns(4)
        generate = col_a.button("Generate", use_container_width=True, key=f"{STATE_KEY}_gen")
        generate_10 = col_b.button("Generate 10", use_container_width=True, key=f"{STATE_KEY}_gen_10")
        generate_100 = col_c.button("Generate 100", use_container_width=True, key=f"{STATE_KEY}_gen_100")
        reset = col_d.button("Reset", use_container_width=True, key=f"{STATE_KEY}_reset")

        config = (dist, round(low, 6), round(high, 6), int(n), int(n_bins))

        if STATE_KEY not in st.session_state or st.session_state[STATE_KEY]["config"] != config:
            _init_state(config)

        state = st.session_state[STATE_KEY]
        bin_edges = _bin_edges(low, high, n_bins)

        if reset:
            state["trials"] = []
            state["bin_counts"] = []

        def generate_sample () :
            samples = _draw_samples(dist, low, high, mean, std, int(n))
            counts, _ = np.histogram(samples, bins=bin_edges)
            state["trials"].append(samples)
            state["bin_counts"].append(counts)

        if generate : generate_sample () 

        if generate_10 : 
            for i in range (10) : generate_sample () 

        if generate_100 : 
            for i in range (100) : generate_sample () 

        n_trials = len(state["trials"])
        st.caption(f"Draws so far: {n_trials}")

        fig = _main_histogram_figure(state["trials"], bin_edges, low, high)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        central_indices = _central_bin_indices(n_bins, N_CENTRAL_BINS)
        st.header("**Distribution of per-bin event counts, across draws**")
        st.markdown(f"for the {len(central_indices)} most central bins")

        cols = st.columns (len (central_indices))

        if n_trials == 0:
            for c, i in zip(cols, central_indices):
                with c:
                    st.markdown(f"Bin {i + 1}: [{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})")
                    st.caption("no data yet")
        else:
            bin_counts_arr = np.array(state["bin_counts"])  # shape (n_trials, n_bins)
            for c, i in zip(cols, central_indices):
                with c:
                    st.markdown(f"Bin {i + 1}: [{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})")
                    fig_b = _bin_count_figure(bin_counts_arr[:, i])
                    st.pyplot(fig_b)
                    plt.close(fig_b)
