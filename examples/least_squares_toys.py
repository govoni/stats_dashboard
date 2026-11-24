"""
Least-squares fit with a single free parameter.

Model: y = a * x   (straight line through the origin, no intercept)

The OLS estimator for a single parameter is closed-form:
    â = Σ(xi * yi) / Σ(xi²)

This is the simplest possible least-squares problem and illustrates
the core ideas: the estimator, its distribution over many toys, and
convergence to the true value.

Layout rules (from README):
  - Controls in a row of st.columns() ABOVE the plots.
  - No title / back button (app.py handles that).
  - Only implement render().
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# ---------------------------------------------------------------------------
# True model parameters
# ---------------------------------------------------------------------------
TRUE_A   = 2.5          # the single parameter we estimate
N_POINTS = 20           # data points per toy
X_MIN    = 0.0
X_MAX    = 5.0
SIGMA    = 1.5          # additive Gaussian noise std-dev (assumed known)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_toy(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(X_MIN, X_MAX, N_POINTS)
    y = TRUE_A * x + rng.normal(0.0, SIGMA, size=N_POINTS)
    return x, y


def _fit(x: np.ndarray, y: np.ndarray) -> float:
    """OLS estimator for y = a*x:  â = Σ(xi*yi) / Σ(xi²)"""
    return float(np.dot(x, y) / np.dot(x, x))


def _theoretical_std() -> float:
    """Var(â) = σ² / Σ(xi²)  →  std(â) = σ / sqrt(Σ(xi²))"""
    x = np.linspace(X_MIN, X_MAX, N_POINTS)
    return float(SIGMA / np.sqrt(np.dot(x, x)))


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def _one_toy(rng):
    x, y = _generate_toy(rng)
    return x, y, _fit(x, y)


def _init_state():
    if "lsq_rng" not in st.session_state:
        st.session_state.lsq_rng = np.random.default_rng(42)
    if "lsq_estimates" not in st.session_state:
        x, y, a = _one_toy(st.session_state.lsq_rng)
        st.session_state.lsq_estimates = [a]
        st.session_state.lsq_last_x    = x
        st.session_state.lsq_last_y    = y


def _add_toys(n: int):
    rng = st.session_state.lsq_rng
    for _ in range(n):
        x, y, a = _one_toy(rng)
        st.session_state.lsq_estimates.append(a)
        st.session_state.lsq_last_x = x
        st.session_state.lsq_last_y = y


def _reset():
    st.session_state.lsq_rng = np.random.default_rng(42)
    x, y, a = _one_toy(st.session_state.lsq_rng)
    st.session_state.lsq_estimates = [a]
    st.session_state.lsq_last_x    = x
    st.session_state.lsq_last_y    = y


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_fit(x, y):

    Y_MIN = TRUE_A * X_MIN
    Y_MAX = TRUE_A * X_MAX

    a_hat = _fit(x, y)
    x_line = np.array([0.0, X_MAX])

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    ax.scatter(x, y, color="#4fc3f7", s=30, zorder=3, label="data")
    ax.plot(x_line, a_hat * x_line,
            color="#ff7043", lw=2, label=f"fit  (â = {a_hat:.3f})")
    ax.plot(x_line, TRUE_A * x_line,
            color="#aed581", lw=1.5, ls="--", label=f"true (a = {TRUE_A})")

    delta_x = (X_MAX - X_MIN) * 0.05
    ax.set_xlim(X_MIN - delta_x, X_MAX + delta_x)
    # ax.set_xlim(0, X_MAX * 1.05)
    ax.set_ylim(Y_MIN - 2 * SIGMA, Y_MAX + 2 * SIGMA)
    ax.set_xlabel("x", color="white")
    ax.set_ylabel("y", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(fontsize=8, facecolor="#1e2127", labelcolor="white",
              edgecolor="#444")
    fig.tight_layout()
    return fig


def _plot_estimator(estimates):
    ests = np.array(estimates)
    n    = len(ests)

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    n_bins = max(10, min(50, n // 3)) // 2
    ax.hist(ests, bins=n_bins, color="#4fc3f7", alpha=0.75,
            edgecolor="#0e1117", label=f"toys (n = {n})")

    ax.axvline(TRUE_A, color="#aed581", lw=2, ls="--",
               label=f"true a = {TRUE_A}")

    ymax = ax.get_ylim()[1]
    ax.plot(ests[-1], ymax * 0.05, marker="v", color="#ff7043",
            markersize=10, zorder=5, label=f"latest â = {ests[-1]:.3f}")

    ax.axvline(ests.mean(), color="#ffb74d", lw=1.5, ls=":",
               label=f"mean â = {ests.mean():.3f}")

    ax.set_xlabel("estimated slope â", color="white")
    ax.set_ylabel("toys", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(fontsize=8, facecolor="#1e2127", labelcolor="white",
              edgecolor="#444")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render():
    _init_state()

    st.title ("Least Squares with one parameter")

    # ── Controls row ────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])

    with c1:
        st.markdown(
            f"#### **Model:** &nbsp; y = a·x",
            # f"**Model:** &nbsp; y = a·x &nbsp;|&nbsp; ",
            # f"true a = {TRUE_A}, σ = {SIGMA}, N = {N_POINTS} pts/toy",
            unsafe_allow_html=True,
        )

    with c2:
        one_toy = st.button("▶ One more toy", use_container_width=True)

    with c3:
        hundred_toys = st.button("▶▶ +100 toys", use_container_width=True)

    with c4:
        thousand_toys = st.button("▶▶▶ +1000 toys", use_container_width=True)

    with c5:
        reset = st.button("↺ Reset", use_container_width=True)

    # with c6:
    #     n_done = len(st.session_state.lsq_estimates)
    #     st.markdown(
    #         f"<div style='text-align:right; padding-top:6px; color:#aaa'>"
    #         f"Toys accumulated: <b style='color:white'>{n_done}</b></div>",
    #         unsafe_allow_html=True,
    #     )

    if reset:
        _reset()
    elif one_toy:
        _add_toys(1)
    elif hundred_toys:
        _add_toys(100)
    elif thousand_toys:
        _add_toys(1000)

    # ── Plots ────────────────────────────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.markdown("#### Latest toy fit")
        fig_fit = _plot_fit(
            st.session_state.lsq_last_x,
            st.session_state.lsq_last_y,
        )
        st.pyplot(fig_fit, use_container_width=True)
        plt.close(fig_fit)

    with right:
        st.markdown("#### Distribution of the estimator â")
        fig_est = _plot_estimator(st.session_state.lsq_estimates)
        st.pyplot(fig_est, use_container_width=True)
        plt.close(fig_est)

    # ── Annotation ───────────────────────────────────────────────────────────
    ests  = np.array(st.session_state.lsq_estimates)
    bias  = ests.mean() - TRUE_A
    std   = ests.std(ddof=1) if len(ests) > 1 else float("nan")
    th_std = _theoretical_std()

    # st.markdown(
    #     f"**Empirical mean â:** {ests.mean():.4f} &nbsp;|&nbsp; "
    #     f"**Bias:** {bias:+.4f} &nbsp;|&nbsp; "
    #     f"**Empirical std:** {std:.4f} &nbsp;|&nbsp; "
    #     f"**Theoretical std (σ/√Σxᵢ²):** {th_std:.4f}",
    #     unsafe_allow_html=True,
    # )
