"""
gaussian_mle_demo.py

A self-contained Streamlit "example" module showing, graphically, that the
maximum-likelihood estimates of (mu, sigma) for a Gaussian are the values
that maximize the likelihood of a small observed sample.

Usage in a multi-file dashboard: import this module and call render()
from your main app, e.g.:

    import gaussian_mle_demo
    gaussian_mle_demo.render()

Requires: streamlit, numpy, matplotlib, scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import streamlit as st

# ----------------------------------------------------------------------
# Fixed "ground truth" used to generate the sample. Feel free to expose
# these as sliders too, but keeping them fixed makes the pedagogical
# point (data are FIXED, parameters are what we vary) clearer.
# ----------------------------------------------------------------------
TRUE_MU = 0.0
TRUE_SIGMA = 1.0
N_POINTS = 5


def _get_sample(seed: int) -> np.ndarray:
    """Draw (and cache in session_state) the 5 data points.

    A separate "resample counter" is mixed into the effective seed so that
    clicking "draw a new sample" actually changes the draw even when the
    user-facing seed number hasn't changed (otherwise re-seeding with the
    same `seed` deterministically reproduces the same 5 points).
    """
    counter = st.session_state.get("mle_demo_resample_counter", 0)
    effective_seed = int(seed) * 100_003 + counter  # arbitrary mixing
    key = "mle_demo_sample"
    cache_tag = st.session_state.get("mle_demo_sample_tag")
    if key not in st.session_state or cache_tag != effective_seed:
        rng = np.random.default_rng(effective_seed)
        st.session_state[key] = rng.normal(TRUE_MU, TRUE_SIGMA, N_POINTS)
        st.session_state["mle_demo_sample_tag"] = effective_seed
    return st.session_state[key]


def _log_likelihood(x: np.ndarray, mu: float, sigma: float) -> float:
    return float(np.sum(norm.logpdf(x, loc=mu, scale=sigma)))


def render():
    st.header("Maximum Likelihood Estimation: fitting a Gaussian")

#     st.markdown(
#         r"""
# We draw **5 points** from a Gaussian with unknown $(\mu,\sigma)$.
# Move the sliders below to try different candidate values $(\mu,\sigma)$ and watch:

# - **Left panel:** the candidate Gaussian curve and the (fixed) data points, evaluated on that curve.
# - **Right panel:** the log-likelihood surface
#   $\ell(\mu,\sigma)=\sum_{i=1}^{5}\ln f(x_i;\mu,\sigma)$
#   over the $(\mu,\sigma)$ plane, with your current choice marked.

# The likelihood is largest exactly at the sample mean and sample standard deviation —
# that's the maximum likelihood estimate (MLE).
# """
#     )

    # --- controls -------------------------------------------------
    mu = st.slider(r"Hypothetical mean $\mu$", -3.0, 3.0, 0.0, 0.05)
    sigma = st.slider(r"Hypothetical std. dev. $\sigma$", 0.15, 3.0, 1.0, 0.05)

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 1])
    with col_ctrl1:
        seed = st.number_input("Random seed for the sample", min_value=0, value=42, step=1)
    with col_ctrl2:
        if st.button("Draw a new sample"):
            st.session_state["mle_demo_resample_counter"] = (
                st.session_state.get("mle_demo_resample_counter", 0) + 1
            )
    with col_ctrl3:
        show_mle = st.checkbox("Show analytic MLE", value=True)

    x_data = _get_sample(seed)

    # Analytic MLE for a Gaussian: sample mean, and the *biased* (1/n) std.
    mu_hat = float(np.mean(x_data))
    sigma_hat = float(np.std(x_data, ddof=0))  # MLE uses 1/n, not 1/(n-1)

    ll_current = _log_likelihood(x_data, mu, sigma)
    ll_at_mle = _log_likelihood(x_data, mu_hat, sigma_hat)

    # --- left panel: pdf + data points -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(5, 4.2))
        xs = np.linspace(min(x_data.min(), mu - 4 * sigma),
                          max(x_data.max(), mu + 4 * sigma), 400)
        ax1.plot(xs, norm.pdf(xs, mu, sigma), color="#1f77b4", lw=2,
                  label=fr"$\mathcal{{N}}(\mu={mu:.2f}, \sigma={sigma:.2f})$")

        # data points marked on the x-axis and lifted to the curve
        y_at_points = norm.pdf(x_data, mu, sigma)
        ax1.scatter(x_data, np.zeros_like(x_data), color="black", zorder=5,
                    label="data (on x-axis)")
        ax1.scatter(x_data, y_at_points, color="crimson", zorder=5,
                    label="pdf value at data")
        for xi, yi in zip(x_data, y_at_points):
            ax1.vlines(xi, 0, yi, color="crimson", linestyle="--", lw=1, alpha=0.7)

        ax1.set_xlabel("x")
        ax1.set_ylabel("probability density")
        ax1.set_title("Candidate Gaussian and the 5 data points")
        ax1.legend(loc="upper right", fontsize=8)
        st.pyplot(fig1)
        plt.close(fig1)

        st.metric("Log-likelihood at current (μ, σ)", f"{ll_current:.3f}")
        if show_mle:
            st.metric("Log-likelihood at the MLE", f"{ll_at_mle:.3f}")
            st.caption(
                f"MLE: μ̂ = {mu_hat:.3f} (sample mean),  "
                f"σ̂ = {sigma_hat:.3f} (sample std, 1/n convention)"
            )

    # --- right panel: 2D log-likelihood surface --------------------
    with col2:
        mu_grid = np.linspace(-3.0, 3.0, 150)
        sigma_grid = np.linspace(0.15, 3.0, 150)
        MU, SIGMA = np.meshgrid(mu_grid, sigma_grid)

        # log L(mu, sigma) = sum_i log f(x_i; mu, sigma), vectorized
        LL = np.zeros_like(MU)
        for xi in x_data:
            LL += norm.logpdf(xi, loc=MU, scale=SIGMA)

        fig2, ax2 = plt.subplots(figsize=(5, 4.2))
        # clip very negative values so the colormap isn't dominated by
        # the (uninteresting) tails
        floor = np.percentile(LL, 5)
        LL_clipped = np.clip(LL, floor, None)
        cf = ax2.contourf(MU, SIGMA, LL_clipped, levels=30, cmap="viridis")
        ax2.contour(MU, SIGMA, LL_clipped, levels=10, colors="white",
                    linewidths=0.4, alpha=0.5)
        fig2.colorbar(cf, ax=ax2, label="log-likelihood")

        ax2.scatter([mu], [sigma], color="red", s=90, marker="o",
                    edgecolor="white", zorder=5, label="current (μ, σ)")
        if show_mle:
            ax2.scatter([mu_hat], [sigma_hat], color="gold", s=140, marker="*",
                        edgecolor="black", zorder=6, label="MLE (μ̂, σ̂)")
        ax2.scatter([TRUE_MU], [TRUE_SIGMA], color="cyan", s=90, marker="^",
                    edgecolor="black", zorder=5, label="true (μ, σ)")

        ax2.set_xlabel(r"$\mu$")
        ax2.set_ylabel(r"$\sigma$")
        ax2.set_title("Log-likelihood surface")
        ax2.legend(loc="upper right", fontsize=8)
        st.pyplot(fig2)
        plt.close(fig2)

#     st.markdown(
#         r"""
# **Take-away for students:** the height of the surface on the right at any point
# $(\mu,\sigma)$ is *exactly* the sum of the log heights of the crimson dashed lines
# on the left, for a Gaussian centered/scaled at that $(\mu,\sigma)$. Dragging the
# sliders away from the sample mean/std makes at least one point fall in a low-density
# region of the curve, and the total log-likelihood drops — this is the geometric
# content of "maximum likelihood."
# """
#     )


if __name__ == "__main__":
    # allows running this file standalone with `streamlit run gaussian_mle_demo.py`
    render()
