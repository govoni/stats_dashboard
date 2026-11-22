"""
χ²_min distribution from toy experiments.

Model: y = a * x   (single parameter, line through the origin)

For each toy the minimum chi-squared at the best-fit value is:
    Q² = χ²_min = Σ[(yi - â*xi)² / σ²]

Under the assumption that the model is correct and σ is known,
Q² follows a chi-squared distribution with k = N - 1 degrees of
freedom (N data points, 1 fitted parameter).

The render shows:
  Left  : latest toy — data + best-fit line + residuals panel
  Right : histogram of Q² values accumulated over all toys,
          overlaid with the theoretical χ²(k) pdf and its mean (k)
          and the ±1σ band (k ± √2k).

Layout rules (from README):
  - Controls in a row of st.columns() ABOVE the plots.
  - No title / back button (app.py handles that).
  - Only implement render().
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2 as chi2_dist
import streamlit as st


# ---------------------------------------------------------------------------
# True model parameters
# ---------------------------------------------------------------------------
TRUE_A   = 2.5
N_POINTS = 20
X_MIN    = 0.5
X_MAX    = 5.0
SIGMA    = 1.5     # noise std-dev, assumed known

NDF = N_POINTS - 1   # degrees of freedom


# ---------------------------------------------------------------------------
# Core maths
# ---------------------------------------------------------------------------

def _x_fixed() -> np.ndarray:
    return np.linspace(X_MIN, X_MAX, N_POINTS)


def _generate_toy(rng: np.random.Generator):
    x = _x_fixed()
    y = TRUE_A * x + rng.normal(0.0, SIGMA, size=N_POINTS)
    return x, y


def _fit_and_q2(x, y) -> tuple[float, float]:
    """Returns (â, Q²) where Q² = χ²_min."""
    a_hat = float(np.dot(x, y) / np.dot(x, x))
    residuals = y - a_hat * x
    q2 = float(np.sum(residuals**2) / SIGMA**2)
    return a_hat, q2


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def _one_toy(rng):
    x, y = _generate_toy(rng)
    a_hat, q2 = _fit_and_q2(x, y)
    return x, y, a_hat, q2


def _init_state():
    if "q2_rng" not in st.session_state:
        st.session_state.q2_rng = np.random.default_rng(42)
    if "q2_values" not in st.session_state:
        x, y, a_hat, q2 = _one_toy(st.session_state.q2_rng)
        st.session_state.q2_values  = [q2]
        st.session_state.q2_last_x  = x
        st.session_state.q2_last_y  = y
        st.session_state.q2_last_a  = a_hat


def _add_toys(n: int):
    rng = st.session_state.q2_rng
    for _ in range(n):
        x, y, a_hat, q2 = _one_toy(rng)
        st.session_state.q2_values.append(q2)
        st.session_state.q2_last_x = x
        st.session_state.q2_last_y = y
        st.session_state.q2_last_a = a_hat


def _reset():
    st.session_state.q2_rng = np.random.default_rng(42)
    x, y, a_hat, q2 = _one_toy(st.session_state.q2_rng)
    st.session_state.q2_values  = [q2]
    st.session_state.q2_last_x  = x
    st.session_state.q2_last_y  = y
    st.session_state.q2_last_a  = a_hat


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_fit(x, y, a_hat, q2):
    """Data + best-fit line (top) and residuals (bottom)."""
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(5, 4.5),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
        sharex=True,
    )
    for ax in (ax_top, ax_bot):
        ax.set_facecolor("#0e1117")
    fig.patch.set_facecolor("#0e1117")

    x_line = np.array([0.0, X_MAX])

    # Top: data + fit + true
    ax_top.scatter(x, y, color="#4fc3f7", s=25, zorder=3, label="data")
    ax_top.plot(x_line, a_hat * x_line, color="#ff7043", lw=2,
                label=f"fit  â = {a_hat:.3f}")
    ax_top.plot(x_line, TRUE_A * x_line, color="#aed581", lw=1.5, ls="--",
                label=f"true a = {TRUE_A}")
    ax_top.set_ylabel("y", color="white")
    ax_top.tick_params(colors="white", labelbottom=False)
    ax_top.set_xlim(0, X_MAX * 1.05)
    for spine in ax_top.spines.values():
        spine.set_edgecolor("#444")
    ax_top.legend(fontsize=8, facecolor="#1e2127", labelcolor="white",
                  edgecolor="#444")

    # Bottom: residuals
    residuals = y - a_hat * x
    ax_bot.bar(x, residuals, width=(X_MAX - X_MIN) / N_POINTS * 0.8,
               color="#4fc3f7", alpha=0.7)
    ax_bot.axhline(0, color="#888", lw=0.8)
    ax_bot.axhline( SIGMA, color="#ffb74d", lw=1, ls=":")
    ax_bot.axhline(-SIGMA, color="#ffb74d", lw=1, ls=":",
                   label="±σ")
    ax_bot.set_ylabel("res.", color="white", fontsize=8)
    ax_bot.set_xlabel("x", color="white")
    ax_bot.tick_params(colors="white")
    for spine in ax_bot.spines.values():
        spine.set_edgecolor("#444")
    ax_bot.legend(fontsize=7, facecolor="#1e2127", labelcolor="white",
                  edgecolor="#444", loc="upper right")

    ax_top.set_title(f"Q² = χ²_min = {q2:.2f}   (ndf = {NDF},"
                     f"  Q²/ndf = {q2/NDF:.2f})",
                     color="white", fontsize=9, pad=4)
    fig.tight_layout()
    return fig


def _plot_q2_hist(q2_values, q2_latest):
    q2_arr = np.array(q2_values)
    n      = len(q2_arr)

    # x range: 0 to a bit beyond the theoretical 99th percentile
    x_max = chi2_dist.ppf(0.999, NDF) * 1.15
    x_vals = np.linspace(0, x_max, 500)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    # Histogram — normalised to density
    n_bins = max(10, min(60, n // 3))
    ax.hist(q2_arr, bins=n_bins, range=(0, x_max),
            density=True, color="#4fc3f7", alpha=0.75,
            edgecolor="#0e1117", label=f"toys (n = {n})")

    # Theoretical χ²(NDF) pdf
    pdf_vals = chi2_dist.pdf(x_vals, NDF)
    ax.plot(x_vals, pdf_vals, color="#ff7043", lw=2,
            label=f"χ²(k={NDF}) pdf")

    # Mean = k
    ax.axvline(NDF, color="#aed581", lw=1.5, ls="--",
               label=f"mean = k = {NDF}")

    # ±1σ band  (std of χ²(k) = √2k)
    std_th = np.sqrt(2 * NDF)
    ax.axvspan(NDF - std_th, NDF + std_th,
               color="#aed581", alpha=0.08, label=f"±√2k = ±{std_th:.1f}")

    # Latest toy marker
    ymax = ax.get_ylim()[1]
    ax.plot(q2_latest, ymax * 0.04, marker="v", color="#ffb74d",
            markersize=10, zorder=5,
            label=f"latest Q² = {q2_latest:.2f}")

    # Empirical mean
    ax.axvline(q2_arr.mean(), color="#ce93d8", lw=1.5, ls=":",
               label=f"emp. mean = {q2_arr.mean():.2f}")

    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Q² = χ²_min", color="white")
    ax.set_ylabel("probability density", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(fontsize=7.5, facecolor="#1e2127", labelcolor="white",
              edgecolor="#444")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render():

    _init_state()

    st.title ("Least Squares and chisquare")

    # ── Controls row ────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])

    with c1:
        st.markdown(
            f"**Model:** &nbsp; y = a·x &nbsp;|&nbsp; "
            f"true a = {TRUE_A}, σ = {SIGMA}, "
            f"N = {N_POINTS} pts/toy, ndf = {NDF}",
            unsafe_allow_html=True,
        )

    with c2:
        one_toy = st.button("▶ One more toy", use_container_width=True,
                            key="q2_one")
    with c3:
        hundred_toys = st.button("▶▶ +100 toys", use_container_width=True,
                                 key="q2_100")
    with c4:
        thousand_toys = st.button("▶▶▶ +1000 toys", use_container_width=True,
                                  key="q2_1000")
    with c5:
        reset = st.button("↺ Reset", use_container_width=True,
                          key="q2_reset")
    # with c6:
    #     n_done = len(st.session_state.q2_values)
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
        st.markdown("#### Latest toy fit & residuals")
        fig_fit = _plot_fit(
            st.session_state.q2_last_x,
            st.session_state.q2_last_y,
            st.session_state.q2_last_a,
            st.session_state.q2_values[-1],
        )
        st.pyplot(fig_fit, use_container_width=True)
        plt.close(fig_fit)

    with right:
        st.markdown(f"#### Q² distribution vs χ²(k = {NDF})")
        fig_q2 = _plot_q2_hist(
            st.session_state.q2_values,
            st.session_state.q2_values[-1],
        )
        st.pyplot(fig_q2, use_container_width=True)
        plt.close(fig_q2)

    # ── Annotation ───────────────────────────────────────────────────────────
    q2_arr = np.array(st.session_state.q2_values)
    std_th = np.sqrt(2 * NDF)
    p_val  = float(chi2_dist.sf(q2_arr[-1], NDF))   # p-value of latest toy

    # st.markdown(
    #     f"**Theoretical:** mean = {NDF},  std = √(2·{NDF}) = {std_th:.2f} &nbsp;|&nbsp; "
    #     f"**Empirical:** mean = {q2_arr.mean():.3f},  "
    #     f"std = {q2_arr.std(ddof=1):.3f} &nbsp;|&nbsp; "
    #     f"**p-value of latest toy:** {p_val:.3f}",
    #     unsafe_allow_html=True,
    # )
