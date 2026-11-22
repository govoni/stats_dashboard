"""
Graphic method for uncertainty estimation and coverage.

Model: y = a * x   (single parameter, line through the origin)

For each toy experiment:
  1. Fit: find â = Σ(xi*yi) / Σ(xi²)  (OLS, closed form)
  2. Build χ²(a) = Σ[(yi - a*xi)² / σ²]  as a function of a
  3. Find the 1-sigma interval graphically: solve χ²(a) = χ²_min + 1
     → two roots a_lo, a_hi  (also closed-form for this linear model)
  4. Check coverage: does [a_lo, a_hi] contain the true value?

The Δχ² = 1 rule comes from the likelihood-ratio / profile-likelihood
argument: for a Gaussian likelihood with one free parameter,
-2 ln(L/L_max) = χ²(a) - χ²_min is chi-squared distributed with 1 d.o.f.,
and the 68.3% confidence interval is defined by Δχ² ≤ 1.

Layout:
  Top-left  : latest toy — data + best-fit line
  Top-right : Δχ²(a) curve, Δχ²=1 line, interval endpoints, true value
  Bottom    : growing bar chart of intervals; green = covers truth, red = misses

Layout rules (from README):
  - Controls in a row of st.columns() ABOVE the plots.
  - No title / back button (app.py handles that).
  - Only implement render().
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st


# ---------------------------------------------------------------------------
# True model parameters
# ---------------------------------------------------------------------------
TRUE_A   = 2.5
N_POINTS = 20
X_MIN    = 0.5     # avoid x=0 so the model is well-defined
X_MAX    = 5.0
SIGMA    = 1.5     # noise std-dev, assumed known

MAX_BARS = 80      # max number of intervals shown in the coverage bar chart


# ---------------------------------------------------------------------------
# Core maths  (all closed-form for y = a*x)
# ---------------------------------------------------------------------------

def _x_fixed() -> np.ndarray:
    return np.linspace(X_MIN, X_MAX, N_POINTS)


def _generate_toy(rng: np.random.Generator):
    x = _x_fixed()
    y = TRUE_A * x + rng.normal(0.0, SIGMA, size=N_POINTS)
    return x, y


def _fit(x, y) -> float:
    """â = Σ(xi yi) / Σ(xi²)"""
    return float(np.dot(x, y) / np.dot(x, x))


def _chi2(a: np.ndarray, x, y) -> np.ndarray:
    """χ²(a) = Σ[(yi - a xi)² / σ²]  — a may be an array."""
    a = np.asarray(a)
    residuals = y[None, :] - a[:, None] * x[None, :]   # (n_a, N)
    return np.sum(residuals**2, axis=1) / SIGMA**2


def _interval(x, y) -> tuple[float, float, float]:
    """
    Returns (a_hat, a_lo, a_hi) from the Δχ² = 1 condition.

    χ²(a) = Σ(yi - a xi)²/σ²
           = [Σxi² a² - 2 Σxiyi a + Σyi²] / σ²

    χ²(a) - χ²_min = 1  is a quadratic in a with known solution:
        Σxi²/σ² (a - â)² = 1
        a = â ± σ / sqrt(Σxi²)

    This is exactly the theoretical std of â, confirming the two methods agree.
    """
    sx2 = float(np.dot(x, x))
    a_hat = float(np.dot(x, y) / sx2)
    delta = SIGMA / np.sqrt(sx2)
    return a_hat, a_hat - delta, a_hat + delta


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def _one_toy(rng):
    x, y = _generate_toy(rng)
    a_hat, a_lo, a_hi = _interval(x, y)
    covers = (a_lo <= TRUE_A <= a_hi)
    return x, y, a_hat, a_lo, a_hi, covers


def _init_state():
    if "cov_rng" not in st.session_state:
        st.session_state.cov_rng = np.random.default_rng(42)
    if "cov_results" not in st.session_state:
        x, y, a_hat, a_lo, a_hi, covers = _one_toy(st.session_state.cov_rng)
        st.session_state.cov_results = [(a_hat, a_lo, a_hi, covers)]
        st.session_state.cov_last_x  = x
        st.session_state.cov_last_y  = y


def _add_toys(n: int):
    rng = st.session_state.cov_rng
    for _ in range(n):
        x, y, a_hat, a_lo, a_hi, covers = _one_toy(rng)
        st.session_state.cov_results.append((a_hat, a_lo, a_hi, covers))
        st.session_state.cov_last_x = x
        st.session_state.cov_last_y = y


def _reset():
    st.session_state.cov_rng = np.random.default_rng(42)
    x, y, a_hat, a_lo, a_hi, covers = _one_toy(st.session_state.cov_rng)
    st.session_state.cov_results = [(a_hat, a_lo, a_hi, covers)]
    st.session_state.cov_last_x  = x
    st.session_state.cov_last_y  = y


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_fit(x, y):
    a_hat, a_lo, a_hi = _interval(x, y)
    x_line = np.array([0.0, X_MAX])

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    ax.scatter(x, y, color="#4fc3f7", s=30, zorder=3, label="data")
    ax.plot(x_line, a_hat * x_line, color="#ff7043", lw=2,
            label=f"fit  â = {a_hat:.3f}")
    ax.plot(x_line, TRUE_A * x_line, color="#aed581", lw=1.5, ls="--",
            label=f"true a = {TRUE_A}")
    # Show the band spanned by [a_lo, a_hi]
    ax.fill_between(x_line,
                    a_lo * x_line, a_hi * x_line,
                    color="#ff7043", alpha=0.15, label=f"1σ band")

    ax.set_xlim(0, X_MAX * 1.05)
    ax.set_xlabel("x", color="white")
    ax.set_ylabel("y", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(fontsize=8, facecolor="#1e2127", labelcolor="white",
              edgecolor="#444")
    fig.tight_layout()
    return fig


def _plot_chi2(x, y):
    a_hat, a_lo, a_hi = _interval(x, y)
    # Fixed axis centred on the true value; half-width = 2 × theoretical sigma_a.
    # This keeps the true-value line stationary across toys, and the tighter
    # range places the Δχ²=1 intersects well up the parabola.
    sigma_a    = SIGMA / np.sqrt(np.dot(x, x))
    half_width = 2.0 * sigma_a
    a_vals = np.linspace(TRUE_A - half_width, TRUE_A + half_width, 500)

    chi2_vals = _chi2(a_vals, x, y)
    chi2_min  = float(_chi2(np.array([a_hat]), x, y)[0])
    dchi2     = chi2_vals - chi2_min

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    ax.plot(a_vals, dchi2, color="#4fc3f7", lw=2, label="Δχ²(a)")

    # Δχ² = 1 horizontal line
    ax.axhline(1.0, color="#ffb74d", lw=1.5, ls="--", label="Δχ² = 1")

    # Interval endpoints
    for val, lbl in [(a_lo, "a_lo"), (a_hi, "a_hi")]:
        ax.axvline(val, color="#ff7043", lw=1.2, ls=":")
        ax.plot(val, 1.0, "o", color="#ff7043", ms=6, zorder=5)

    # Shade the interval
    mask = (a_vals >= a_lo) & (a_vals <= a_hi)
    ax.fill_between(a_vals[mask], 0, dchi2[mask],
                    color="#ff7043", alpha=0.15)

    # True value
    ax.axvline(TRUE_A, color="#aed581", lw=1.5, ls="--",
               label=f"true a = {TRUE_A}")

    # Bracket annotation
    ax.annotate("", xy=(a_hi, 1.0), xytext=(a_lo, 1.0),
                arrowprops=dict(arrowstyle="<->", color="#ff7043", lw=1.5))
    ax.text((a_lo + a_hi) / 2, 1.15,
            f"[{a_lo:.2f}, {a_hi:.2f}]",
            ha="center", va="bottom", fontsize=7, color="#ff7043")

    ax.set_ylim(bottom=0)
    ax.set_xlabel("a", color="white")
    ax.set_ylabel("Δχ²(a) = χ²(a) − χ²_min", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(fontsize=8, facecolor="#1e2127", labelcolor="white",
              edgecolor="#444")
    fig.tight_layout()
    return fig


def _plot_coverage(results):
    n_total  = len(results)
    n_shown  = min(n_total, MAX_BARS)
    shown    = results[-n_shown:]          # most recent toys
    n_covers = sum(r[3] for r in results)  # count over ALL toys
    coverage = n_covers / n_total

    fig, ax = plt.subplots(figsize=(10, max(2.5, n_shown * 0.18 + 0.6)))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    for i, (a_hat, a_lo, a_hi, covers) in enumerate(shown):
        color = "#66bb6a" if covers else "#ef5350"
        ax.barh(i, a_hi - a_lo, left=a_lo,
                height=0.7, color=color, alpha=0.8)
        ax.plot(a_hat, i, "|", color="white", ms=5, mew=1.2)

    ax.axvline(TRUE_A, color="#aed581", lw=2, ls="--",
               label=f"true a = {TRUE_A}")

    # Legend patches
    green_patch = mpatches.Patch(color="#66bb6a", alpha=0.8, label="covers truth")
    red_patch   = mpatches.Patch(color="#ef5350", alpha=0.8, label="misses truth")
    ax.legend(handles=[green_patch, red_patch], fontsize=8,
              facecolor="#1e2127", labelcolor="white", edgecolor="#444",
              loc="upper right")

    tail = f"(last {n_shown})" if n_total > n_shown else ""
    ax.set_title(
        f"Coverage: {n_covers}/{n_total} = {coverage:.1%}  {tail}"
        f"   —   expected 68.3%",
        color="white", fontsize=9, pad=4,
    )
    ax.set_xlabel("a", color="white")
    ax.set_ylabel(f"toy index {tail}", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render():
    _init_state()

    st.title ("Least Squares and coverage")

    # ── Controls row ────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns([2, 1, 1, 1, 1, 1])

    with c1:
        st.markdown(
            f"**Model:** &nbsp; y = a·x &nbsp;|&nbsp; "
            f"true a = {TRUE_A}, σ = {SIGMA}, N = {N_POINTS} pts/toy",
            unsafe_allow_html=True,
        )

    with c2:
        one_toy = st.button("▶ One more toy", use_container_width=True,
                            key="cov_one")

    with c3:
        hundred_toys = st.button("▶▶ +100 toys", use_container_width=True,
                                 key="cov_100")

    with c4:
        thousand_toys = st.button("▶▶▶ +1000 toys", use_container_width=True,
                                  key="cov_1000")

    with c5:
        reset = st.button("↺ Reset", use_container_width=True,
                          key="cov_reset")

    with c6:
        n_done = len(st.session_state.cov_results)
        st.markdown(
            f"<div style='text-align:right; padding-top:6px; color:#aaa'>"
            f"Toys accumulated: <b style='color:white'>{n_done}</b></div>",
            unsafe_allow_html=True,
        )

    if reset:
        _reset()
    elif one_toy:
        _add_toys(1)
    elif hundred_toys:
        _add_toys(100)
    elif thousand_toys:
        _add_toys(1000)

    # ── Top row: fit  |  Δχ² curve ──────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.markdown("#### Latest toy fit")
        fig_fit = _plot_fit(
            st.session_state.cov_last_x,
            st.session_state.cov_last_y,
        )
        st.pyplot(fig_fit, use_container_width=True)
        plt.close(fig_fit)

    with right:
        st.markdown("#### Δχ²(a) — graphic uncertainty method")
        fig_chi2 = _plot_chi2(
            st.session_state.cov_last_x,
            st.session_state.cov_last_y,
        )
        st.pyplot(fig_chi2, use_container_width=True)
        plt.close(fig_chi2)

    # ── Bottom: coverage bar chart ───────────────────────────────────────────
    st.markdown("#### Coverage — does the interval contain the true value?")
    fig_cov = _plot_coverage(st.session_state.cov_results)
    st.pyplot(fig_cov, use_container_width=True)
    plt.close(fig_cov)
