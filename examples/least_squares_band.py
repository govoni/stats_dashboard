"""
Uncertainty band on the fitted line.

Model: y = a*x + b   (two parameters: slope + intercept)

After fitting with OLS the predicted value at any x is
    ŷ(x) = â*x + b̂

Its variance is obtained by error propagation through the
covariance matrix of the estimator Cov(θ̂) = σ²(XᵀX)⁻¹:

    Var(ŷ(x)) = f(x)ᵀ Cov(θ̂) f(x)
              = σ² [x, 1] (XᵀX)⁻¹ [x, 1]ᵀ

where f(x) = [x, 1] is the gradient of ŷ with respect to θ = (a, b).

The 1-σ band is  ŷ(x) ± √Var(ŷ(x)).

The band is narrowest at the centroid x̄ of the data
(where the two parameters are most constrained jointly) and
widens symmetrically towards the edges — a hyperbolic shape.

Layout:
  Left  : latest toy — data, best-fit line, 1-σ and 2-σ bands, true line
  Right : same plot but accumulated over many toys: each toy contributes
          its own band (drawn faintly), so the envelope of all bands
          builds up and the student can see where the coverage sits.

Layout rules (from README):
  - Controls in a row of st.columns() ABOVE the plots.
  - No title / back button (app.py handles that).
  - Only implement render().


In the plot on the right, the band is the standard deviation of the fitted line 
at each x — i.e. the half-width of the uncertainty band of a single fit.
So the green band is centred on the true line but has the same width 
as the uncertainty band of one fit.  
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# ---------------------------------------------------------------------------
# True model parameters
# ---------------------------------------------------------------------------
TRUE_A   = 2.5
TRUE_B   = 1.0
N_POINTS = 4
X_MIN    = 1.0
X_MAX    = 4.0
X_PLOT_MIN = 0.0    # wider axis for plotting only
X_PLOT_MAX = 5.0
SIGMA    = 1.5


# ---------------------------------------------------------------------------
# Core maths
# ---------------------------------------------------------------------------

def _x_fixed() -> np.ndarray:
    return np.linspace(X_MIN, X_MAX, N_POINTS)


def _generate_toy(rng: np.random.Generator):
    x = _x_fixed()
    y = TRUE_A * x + TRUE_B + rng.normal(0.0, SIGMA, size=N_POINTS)
    return x, y


def _fit(x, y) -> tuple[float, float]:
    """OLS: θ̂ = (XᵀX)⁻¹ Xᵀy  →  (â, b̂)."""
    X = np.column_stack([x, np.ones_like(x)])
    theta = np.linalg.solve(X.T @ X, X.T @ y)
    return float(theta[0]), float(theta[1])


def _band(x_data, x_fine) -> np.ndarray:
    """
    1-σ half-width of ŷ at each point in x_fine.

    σ_ŷ(x) = σ · sqrt( [x,1] (XᵀX)⁻¹ [x,1]ᵀ )
    """
    X = np.column_stack([x_data, np.ones_like(x_data)])
    XtXinv = np.linalg.inv(X.T @ X)
    # For each x in x_fine compute f = [x, 1] and fᵀ (XᵀX)⁻¹ f
    F = np.column_stack([x_fine, np.ones_like(x_fine)])  # (M, 2)
    var = np.einsum("mi,ij,mj->m", F, XtXinv, F)        # (M,)
    return SIGMA * np.sqrt(var)


def _theoretical_band(x_fine) -> np.ndarray:
    """Band computed from the fixed design (same x grid every toy)."""
    return _band(_x_fixed(), x_fine)


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def _one_toy(rng):
    x, y = _generate_toy(rng)
    a, b = _fit(x, y)
    return x, y, a, b

def _init_state():
    if "band_rng" not in st.session_state:
        st.session_state.band_rng = np.random.default_rng(42)
    if "band_fits" not in st.session_state:
        x, y, a, b = _one_toy(st.session_state.band_rng)
        st.session_state.band_fits   = [(a, b)]
        st.session_state.band_last_x = x
        st.session_state.band_last_y = y


def _add_toys(n: int):
    rng = st.session_state.band_rng
    for _ in range(n):
        x, y, a, b = _one_toy(rng)
        st.session_state.band_fits.append((a, b))
        st.session_state.band_last_x = x
        st.session_state.band_last_y = y


def _reset():
    st.session_state.band_rng = np.random.default_rng(42)
    x, y, a, b = _one_toy(st.session_state.band_rng)
    st.session_state.band_fits   = [(a, b)]
    st.session_state.band_last_x = x
    st.session_state.band_last_y = y


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

X_FINE = np.linspace(X_PLOT_MIN, X_PLOT_MAX, 400)
TH_BAND = None   # computed once on first plot call


def _get_th_band():
    global TH_BAND
    if TH_BAND is None:
        TH_BAND = _theoretical_band(X_FINE)
    return TH_BAND


def _plot_single(x, y, a_hat, b_hat):
    """Left panel: one toy with 1-σ and 2-σ bands."""
    sigma_y = _band(x, X_FINE)
    y_fit   = a_hat * X_FINE + b_hat
    y_true  = TRUE_A * X_FINE + TRUE_B

    Y_PLOT_MIN = TRUE_A * X_PLOT_MIN + TRUE_B - 2 * SIGMA
    Y_PLOT_MAX = TRUE_A * X_PLOT_MAX + TRUE_B + 2 * SIGMA

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    # 2-σ band
    ax.fill_between(X_FINE,
                    y_fit - 2 * sigma_y,
                    y_fit + 2 * sigma_y,
                    color="#ff7043", alpha=0.12, label=r"±2σ band")
    # 1-σ band
    ax.fill_between(X_FINE,
                    y_fit - sigma_y,
                    y_fit + sigma_y,
                    color="#ff7043", alpha=0.25, label=r"±1σ band")

    ax.scatter(x, y, color="#4fc3f7", s=25, zorder=4, label="data")
    ax.plot(X_FINE, y_fit,  color="#ff7043", lw=2,
            label=fr"fit  (â={a_hat:.2f}, b̂={b_hat:.2f})")
    ax.plot(X_FINE, y_true, color="#aed581", lw=1.5, ls="--",
            label=fr"true (a={TRUE_A}, b={TRUE_B})")

    # Mark the centroid x̄ where the band is narrowest
    x_bar = x.mean()
    ax.axvline(x_bar, color="#888", lw=0.8, ls=":")
    ax.text(x_bar + 0.05, ax.get_ylim()[0] * 0.98 if ax.get_ylim()[0] > 0
            else 1.0, r"x̄", color="#888", fontsize=8, va="bottom")

    ax.set_xlim(X_PLOT_MIN, X_PLOT_MAX)
    ax.set_ylim(Y_PLOT_MIN, Y_PLOT_MAX)
    ax.set_xlabel("x", color="white")
    ax.set_ylabel("y", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(fontsize=7.5, facecolor="#1e2127", labelcolor="white",
              edgecolor="#444")
    fig.tight_layout()
    return fig


def _plot_accumulated(fits):
    """Right panel: faint bands from all toys + theoretical band."""
    n = len(fits)
    th_band = _get_th_band()
    y_true  = TRUE_A * X_FINE + TRUE_B

    Y_PLOT_MIN = TRUE_A * X_PLOT_MIN + TRUE_B - 2 * SIGMA
    Y_PLOT_MAX = TRUE_A * X_PLOT_MAX + TRUE_B + 2 * SIGMA

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    # Draw each toy's 1-σ band very faintly
    alpha_each = max(0.01, min(0.15, 1.5 / n))
    for a, b in fits:
        y_fit  = a * X_FINE + b
        sigma_y = th_band   # same design → same band shape, shifted by fit
        ax.fill_between(X_FINE,
                        y_fit - sigma_y,
                        y_fit + sigma_y,
                        color="#4fc3f7", alpha=alpha_each, linewidth=0)

    # Theoretical band around the true line
    ax.fill_between(X_FINE,
                    y_true - 2 * th_band,
                    y_true + 2 * th_band,
                    color="#aed581", alpha=0.10, label=r"true ±2σ")
    ax.fill_between(X_FINE,
                    y_true - th_band,
                    y_true + th_band,
                    color="#aed581", alpha=0.20, label=r"true ±1σ")
    ax.plot(X_FINE, y_true, color="#aed581", lw=1.5, ls="--",
            label=f"true line")

    # Latest toy line on top
    a_last, b_last = fits[-1]
    ax.plot(X_FINE, a_last * X_FINE + b_last,
            color="#ff7043", lw=1.5, label=f"latest fit")

    # Narrowest-point marker
    x_bar = _x_fixed().mean()
    ax.axvline(x_bar, color="#888", lw=0.8, ls=":")

    ax.set_xlim(X_PLOT_MIN, X_PLOT_MAX)
    ax.set_ylim(Y_PLOT_MIN, Y_PLOT_MAX)
    ax.set_xlabel("x", color="white")
    ax.set_ylabel("y", color="white")
    ax.set_title(f"{n} toy band{'s' if n != 1 else ''} accumulated",
                 color="white", fontsize=9, pad=4)
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

    st.title ("Least Squares and extrapolation uncertrainty")

    # ── Controls row ────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])

    with c1:
        st.markdown(
            r"#### **Model:** $y = a \cdot x + b$",
            # r"**Model:** $y = a \cdot x + b$ &nbsp;|&nbsp; "
            # fr"true $a$ = {TRUE_A}, $b$ = {TRUE_B}, "
            # fr"$\sigma$ = {SIGMA}, $N$ = {N_POINTS} pts/toy",
            unsafe_allow_html=True,
        )

    with c2:
        one_toy = st.button("▶ One more toy", use_container_width=True,
                            key="band_one")
    with c3:
        hundred_toys = st.button("▶▶ +100 toys", use_container_width=True,
                                 key="band_100")
    with c4:
        thousand_toys = st.button("▶▶▶ +1000 toys", use_container_width=True,
                                  key="band_1000")
    with c5:
        reset = st.button("↺ Reset", use_container_width=True,
                          key="band_reset")
    # with c6:
    #     n_done = len(st.session_state.band_fits)
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

    x  = st.session_state.band_last_x
    y  = st.session_state.band_last_y
    a_hat, b_hat = _fit(x, y)

    with left:
        st.markdown(r"#### Latest toy — fit $\pm$ uncertainty band")
        fig_single = _plot_single(x, y, a_hat, b_hat)
        st.pyplot(fig_single, use_container_width=True)
        plt.close(fig_single)

    with right:
        st.markdown("#### All toys — band envelope")
        fig_accum = _plot_accumulated(st.session_state.band_fits)
        st.pyplot(fig_accum, use_container_width=True)
        plt.close(fig_accum)

    # ── Annotation ───────────────────────────────────────────────────────────
    th_band = _get_th_band()
    x_bar   = _x_fixed().mean()
    # band half-width at x̄ (minimum) and at the edges
    band_at_mean  = float(_theoretical_band(np.array([x_bar]))[0])
    band_at_start = float(_theoretical_band(np.array([X_PLOT_MIN]))[0])
    band_at_end   = float(_theoretical_band(np.array([X_PLOT_MAX]))[0])

    # st.markdown(
    #     r"**Uncertainty band** $\sigma_{\hat{y}}(x) = \sigma\,\sqrt{\mathbf{f}(x)^\top (X^\top X)^{-1} \mathbf{f}(x)}$"
    #     r" with $\mathbf{f}(x) = [x,\,1]^\top$ &nbsp;|&nbsp; "
    #     fr"min at $\bar{{x}}$ = {x_bar:.2f}: &nbsp; {band_at_mean:.3f} &nbsp;|&nbsp; "
    #     fr"at $x$ = {X_PLOT_MIN}: &nbsp; {band_at_start:.3f} &nbsp;|&nbsp; "
    #     fr"at $x$ = {X_PLOT_MAX}: &nbsp; {band_at_end:.3f}",
    #     unsafe_allow_html=True,
    # )
