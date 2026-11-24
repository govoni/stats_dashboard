"""
Least-squares fit with two free parameters.

Model: y = a * x + b   (slope + intercept)

OLS normal equations give closed-form estimators:
    â = [Σxi² Σyi - Σxi Σxiyi] / D    (wait — standard derivation)
    â = [N Σxiyi - Σxi Σyi] / D
    b̂ = [Σxi² Σyi - Σxi Σxiyi] / D
    D  = N Σxi² - (Σxi)²

The key teaching point is the CORRELATION between â and b̂:
the analytical covariance matrix of the OLS estimator is
    Cov = σ² (XᵀX)⁻¹
which is non-diagonal whenever the design matrix X has non-zero
mean (i.e. x is not centred), producing a visible tilted ellipse
in the (â, b̂) scatter plot.

Layout:
  - Left  : latest toy fit
  - Right : 2D scatter of (â, b̂) over all toys, with
            the 1-σ and 2-σ theoretical error ellipses and
            the true parameter point marked.

Layout rules (from README):
  - Controls in a row of st.columns() ABOVE the plots.
  - No title / back button (app.py handles that).
  - Only implement render().
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import streamlit as st


# ---------------------------------------------------------------------------
# True model parameters
# ---------------------------------------------------------------------------
TRUE_A   = 2.5     # slope
TRUE_B   = 1.0     # intercept
N_POINTS = 20
X_MIN    = 0.0
X_MAX    = 5.0
SIGMA    = 1.5     # additive Gaussian noise std-dev (assumed known)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _x_fixed() -> np.ndarray:
    """The same x grid is used for every toy."""
    return np.linspace(X_MIN, X_MAX, N_POINTS)


def _design_matrix(x: np.ndarray) -> np.ndarray:
    """X = [x | 1],  shape (N, 2)."""
    return np.column_stack([x, np.ones_like(x)])


def _generate_toy(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x = _x_fixed()
    y = TRUE_A * x + TRUE_B + rng.normal(0.0, SIGMA, size=N_POINTS)
    return x, y


def _fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    OLS: θ̂ = (XᵀX)⁻¹ Xᵀy   →  returns (â, b̂).
    """
    X = _design_matrix(x)
    theta = np.linalg.solve(X.T @ X, X.T @ y)
    return float(theta[0]), float(theta[1])


def _covariance_matrix(x: np.ndarray) -> np.ndarray:
    """Cov(θ̂) = σ² (XᵀX)⁻¹,  shape (2,2)."""
    X = _design_matrix(x)
    return SIGMA**2 * np.linalg.inv(X.T @ X)


def _error_ellipse(cov: np.ndarray, center: tuple,
                   n_sigma: float, n_pts: int = 300):
    """
    Return (ex, ey) arrays tracing the n_sigma error ellipse of a 2D Gaussian
    with covariance matrix cov centred at center.
    """
    vals, vecs = np.linalg.eigh(cov)
    # Semi-axes scaled by n_sigma
    a_ax = n_sigma * np.sqrt(vals[1])   # larger eigenvalue → major axis
    b_ax = n_sigma * np.sqrt(vals[0])
    angle = np.arctan2(vecs[1, 1], vecs[0, 1])  # rotation of major axis

    t = np.linspace(0, 2 * np.pi, n_pts)
    # Ellipse in principal-axis frame
    ex_local = a_ax * np.cos(t)
    ey_local = b_ax * np.sin(t)
    # Rotate back
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    ex = center[0] + cos_a * ex_local - sin_a * ey_local
    ey = center[1] + sin_a * ex_local + cos_a * ey_local
    return ex, ey


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def _one_toy(rng):
    x, y = _generate_toy(rng)
    a, b = _fit(x, y)
    return x, y, a, b


def _init_state():
    if "lsq2_rng" not in st.session_state:
        st.session_state.lsq2_rng = np.random.default_rng(42)
    if "lsq2_as" not in st.session_state:
        x, y, a, b = _one_toy(st.session_state.lsq2_rng)
        st.session_state.lsq2_as     = [a]
        st.session_state.lsq2_bs     = [b]
        st.session_state.lsq2_last_x = x
        st.session_state.lsq2_last_y = y


def _add_toys(n: int):
    rng = st.session_state.lsq2_rng
    for _ in range(n):
        x, y, a, b = _one_toy(rng)
        st.session_state.lsq2_as.append(a)
        st.session_state.lsq2_bs.append(b)
        st.session_state.lsq2_last_x = x
        st.session_state.lsq2_last_y = y


def _reset():
    st.session_state.lsq2_rng = np.random.default_rng(42)
    x, y, a, b = _one_toy(st.session_state.lsq2_rng)
    st.session_state.lsq2_as     = [a]
    st.session_state.lsq2_bs     = [b]
    st.session_state.lsq2_last_x = x
    st.session_state.lsq2_last_y = y


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_fit(x, y):
    a_hat, b_hat = _fit(x, y)
    x_line = np.array([X_MIN, X_MAX])

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    ax.scatter(x, y, color="#4fc3f7", s=30, zorder=3, label="data")
    ax.plot(x_line, a_hat * x_line + b_hat,
            color="#ff7043", lw=2,
            label=f"fit  (â={a_hat:.3f}, b̂={b_hat:.3f})")
    ax.plot(x_line, TRUE_A * x_line + TRUE_B,
            color="#aed581", lw=1.5, ls="--",
            label=f"true (a={TRUE_A}, b={TRUE_B})")

    delta_x = (X_MAX - X_MIN) * 0.05
    ax.set_xlim(X_MIN - delta_x, X_MAX + delta_x)
    # ax.set_xlim(X_MIN, X_MAX * 1.05)
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


def _plot_2d(as_, bs_):
    as_ = np.array(as_)
    bs_ = np.array(bs_)
    n   = len(as_)

    cov    = _covariance_matrix(_x_fixed())
    center = (TRUE_A, TRUE_B)

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    # Scatter of toy estimates
    ax.scatter(as_, bs_, color="#4fc3f7", s=8, alpha=0.5,
               zorder=2, label=f"toys (n={n})")

    # Latest toy highlighted
    ax.scatter(as_[-1], bs_[-1], color="#ff7043", s=60,
               zorder=4, label=f"latest")

    # True parameter point
    ax.scatter(*center, color="#aed581", s=80, marker="*",
               zorder=5, label=f"true (a={TRUE_A}, b={TRUE_B})")

    # Theoretical error ellipses (1σ and 2σ)
    for n_sig, lw, ls, label in [
        (1, 1.8, "-",  "1σ ellipse"),
        (2, 1.2, "--", "2σ ellipse"),
    ]:
        ex, ey = _error_ellipse(cov, center, n_sig)
        ax.plot(ex, ey, color="#ffb74d", lw=lw, ls=ls, zorder=3,
                label=label)

    # Empirical covariance ellipse (only once we have enough points)
    if n >= 5:
        emp_cov = np.cov(np.vstack([as_, bs_]))
        emp_center = (as_.mean(), bs_.mean())
        ex, ey = _error_ellipse(emp_cov, emp_center, 1)
        ax.plot(ex, ey, color="#ce93d8", lw=1.5, ls=":", zorder=3,
                label="empirical 1σ")

    # Correlation annotation
    rho = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ax.set_title(f"theoretical ρ(â,b̂) = {rho:.3f}", color="#ffb74d",
                 fontsize=9, pad=4)

    ax.set_xlabel("slope â", color="white")
    ax.set_ylabel("intercept b̂", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(fontsize=7, facecolor="#1e2127", labelcolor="white",
              edgecolor="#444", loc="upper left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render():
    _init_state()

    st.title ("Least Squares with two parameters")

    # ── Controls row ────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])

    with c1:
        st.markdown(
            f"#### **Model:** &nbsp; y = a·x + b",
            # f"**Model:** &nbsp; y = a·x + b &nbsp;|&nbsp; "
            # f"true a={TRUE_A}, b={TRUE_B}, σ={SIGMA}, N={N_POINTS} pts/toy",
            unsafe_allow_html=True,
        )

    with c2:
        one_toy = st.button("▶ One more toy", use_container_width=True,
                            key="lsq2_one")

    with c3:
        hundred_toys = st.button("▶▶ +100 toys", use_container_width=True,
                                 key="lsq2_100")

    with c4:
        thousand_toys = st.button("▶▶▶ +1000 toys", use_container_width=True,
                                  key="lsq2_1000")

    with c5:
        reset = st.button("↺ Reset", use_container_width=True,
                          key="lsq2_reset")

    # with c6:
    #     n_done = len(st.session_state.lsq2_as)
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
            st.session_state.lsq2_last_x,
            st.session_state.lsq2_last_y,
        )
        st.pyplot(fig_fit, use_container_width=True)
        plt.close(fig_fit)

    with right:
        st.markdown("#### Joint distribution of (â, b̂)")
        fig_2d = _plot_2d(
            st.session_state.lsq2_as,
            st.session_state.lsq2_bs,
        )
        st.pyplot(fig_2d, use_container_width=True)
        plt.close(fig_2d)

    # ── Annotation ───────────────────────────────────────────────────────────
    as_ = np.array(st.session_state.lsq2_as)
    bs_ = np.array(st.session_state.lsq2_bs)
    cov = _covariance_matrix(_x_fixed())
    rho_th = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    if len(as_) > 1:
        emp_cov = np.cov(np.vstack([as_, bs_]))
        rho_emp = emp_cov[0, 1] / np.sqrt(emp_cov[0, 0] * emp_cov[1, 1])
        rho_str = f"**Empirical ρ:** {rho_emp:.3f} &nbsp;|&nbsp; "
    else:
        rho_str = ""

    # st.markdown(
    #     f"**mean â:** {as_.mean():.4f} &nbsp;|&nbsp; "
    #     f"**mean b̂:** {bs_.mean():.4f} &nbsp;|&nbsp; "
    #     f"{rho_str}"
    #     f"**Theoretical ρ(â,b̂):** {rho_th:.3f}",
    #     unsafe_allow_html=True,
    # )
