"""
Neyman belt construction for a Gaussian with proportional sigma.

Observable: x ~ N(mu, (k*mu)^2),  single measurement
Parameter:  mu  (unknown mean, mu > 0)
Known:      k   (relative resolution, fixed)

Acceptance interval (central, equal tails):
    x_lo(mu) = mu * (1 - z*k)
    x_hi(mu) = mu * (1 + z*k)
where z = norm.ppf(1 - alpha/2).

Both boundaries are straight lines through the origin with slopes
(1 - z*k) and (1 + z*k) — the belt fans out from the origin.

Inverting at x_obs (vertical slice through the belt):
    mu_lo = x_obs / (1 + z*k)
    mu_hi = x_obs / (1 - z*k)

The CI is asymmetric around x_obs: it is wider on the high-mu side.
Its relative half-width is constant regardless of x_obs:
    mu_hi / mu_lo = (1 + z*k) / (1 - z*k)

The pdf drawn sideways at mu_true is a Gaussian whose width grows
with mu — visually wider at higher mu values.

Real-life example: calorimeter energy measurement at high energy,
where the constant term of the resolution dominates: sigma_E = b*E.

Layout:
  Left   : Neyman belt (fanning lines), pdf at mu_true drawn sideways,
           x_obs vertical line, resulting CI on mu.
  Right  : coverage plot over toys.

Layout rules (from README):
  - Controls in a row of st.columns() ABOVE the plots.
  - No title / back button (app.py handles that).
  - Only implement render().
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm
import streamlit as st


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
MU_TRUE  = 3.0      # true mean (must be > 0)
K        = 0.20     # relative resolution: sigma = k * mu
ALPHA    = 0.32     # 1 - CL  (0.32 → 68%, 0.05 → 95%)
CL       = 1 - ALPHA
Z        = float(norm.ppf(1 - ALPHA / 2))

# Plot ranges  (mu > 0 required)
MU_MIN, MU_MAX = 0.0, 7.0
X_MIN,  X_MAX  = 0.0, 7.0

MAX_BARS = 80


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _sigma(mu: float) -> float:
    return K * mu


def _acceptance_interval(mu: float) -> tuple[float, float]:
    return mu * (1 - Z * K), mu * (1 + Z * K)


def _confidence_interval(x_obs: float) -> tuple[float, float]:
    """
    Invert the belt at x_obs.
    mu_lo = x_obs / (1 + z*k),   mu_hi = x_obs / (1 - z*k)
    Valid only when z*k < 1 (guaranteed for reasonable k and CL).
    """
    mu_lo = x_obs / (1 + Z * K)
    mu_hi = x_obs / (1 - Z * K)
    return float(mu_lo), float(mu_hi)


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def _one_toy(rng) -> tuple[float, float, float, bool]:
    x_obs  = float(rng.normal(MU_TRUE, _sigma(MU_TRUE)))
    # guard: x_obs must be positive for the inversion to make sense
    x_obs  = max(x_obs, 1e-6)
    mu_lo, mu_hi = _confidence_interval(x_obs)
    covers = (mu_lo <= MU_TRUE <= mu_hi)
    return x_obs, mu_lo, mu_hi, covers


def _init_state():
    if "npg_rng" not in st.session_state:
        st.session_state.npg_rng = np.random.default_rng(42)
    if "npg_toys" not in st.session_state:
        x_obs, mu_lo, mu_hi, covers = _one_toy(st.session_state.npg_rng)
        st.session_state.npg_toys   = [(x_obs, mu_lo, mu_hi, covers)]
        st.session_state.npg_last_x = x_obs


def _add_toys(n: int):
    rng = st.session_state.npg_rng
    for _ in range(n):
        x_obs, mu_lo, mu_hi, covers = _one_toy(rng)
        st.session_state.npg_toys.append((x_obs, mu_lo, mu_hi, covers))
        st.session_state.npg_last_x = x_obs


def _reset():
    st.session_state.npg_rng = np.random.default_rng(42)
    x_obs, mu_lo, mu_hi, covers = _one_toy(st.session_state.npg_rng)
    st.session_state.npg_toys   = [(x_obs, mu_lo, mu_hi, covers)]
    st.session_state.npg_last_x = x_obs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_belt(x_obs):
    mu_lo_ci, mu_hi_ci = _confidence_interval(x_obs)

    mu_grid = np.linspace(0.01, MU_MAX, 400)
    x_lo    = mu_grid * (1 - Z * K)
    x_hi    = mu_grid * (1 + Z * K)

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    # ── Belt ────────────────────────────────────────────────────────────────
    ax.fill_betweenx(mu_grid, x_lo, x_hi,
                     color="#4fc3f7", alpha=0.18, linewidth=0,
                     label=f"{CL:.0%} acceptance belt")
    ax.plot(x_lo, mu_grid, color="#4fc3f7", lw=1.0, alpha=0.7)
    ax.plot(x_hi, mu_grid, color="#4fc3f7", lw=1.0, alpha=0.7)

    # ── Diagonal reference (belt would be this if sigma were constant) ───────
    ax.plot([X_MIN, X_MAX], [MU_MIN, MU_MAX],
            color="#555", lw=0.8, ls=":", zorder=1,
            label=r"$x = \mu$ (reference)")

    # ── Gaussian pdf drawn sideways at mu_true ───────────────────────────────
    # Width at mu_true: sigma = k * mu_true
    sigma_true = _sigma(MU_TRUE)
    x_pdf  = np.linspace(X_MIN, X_MAX, 400)
    pdf    = norm.pdf(x_pdf, MU_TRUE, sigma_true)
    scale  = 1.5 / pdf.max()
    ax.plot(x_pdf, MU_TRUE + pdf * scale,
            color="#aed581", lw=1.2, alpha=0.6)
    ax.fill_between(x_pdf, MU_TRUE, MU_TRUE + pdf * scale,
                    color="#aed581", alpha=0.08)
    # Shade acceptance region
    x_lo_true, x_hi_true = _acceptance_interval(MU_TRUE)
    mask = (x_pdf >= x_lo_true) & (x_pdf <= x_hi_true)
    ax.fill_between(x_pdf[mask], MU_TRUE, MU_TRUE + pdf[mask] * scale,
                    color="#aed581", alpha=0.22,
                    label=fr"pdf at $\mu_{{\rm true}}$ (shaded = {CL:.0%})")

    # ── Asymmetry annotation: show that acceptance interval is asymmetric ────
    ax.annotate("", xy=(x_hi_true, MU_TRUE - 0.25),
                xytext=(MU_TRUE, MU_TRUE - 0.25),
                arrowprops=dict(arrowstyle="<->", color="#aed581", lw=1.2))
    ax.annotate("", xy=(MU_TRUE, MU_TRUE - 0.25),
                xytext=(x_lo_true, MU_TRUE - 0.25),
                arrowprops=dict(arrowstyle="<->", color="#aed581", lw=1.2))
    ax.text((MU_TRUE + x_hi_true) / 2, MU_TRUE - 0.45,
            f"+{(x_hi_true-MU_TRUE):.2f}", ha="center",
            fontsize=7, color="#aed581")
    ax.text((x_lo_true + MU_TRUE) / 2, MU_TRUE - 0.45,
            f"−{(MU_TRUE-x_lo_true):.2f}", ha="center",
            fontsize=7, color="#aed581")

    # ── True mu line ─────────────────────────────────────────────────────────
    ax.axhline(MU_TRUE, color="#aed581", lw=1.2, ls="--",
               label=fr"$\mu_{{\rm true}}$ = {MU_TRUE}")

    # ── Observed x_obs ───────────────────────────────────────────────────────
    ax.axvline(x_obs, color="#ff7043", lw=2, zorder=4,
               label=fr"$x_{{\rm obs}}$ = {x_obs:.3f}")

    # ── Resulting CI on mu ───────────────────────────────────────────────────
    ax.plot([x_obs, x_obs], [mu_lo_ci, mu_hi_ci],
            color="#ffb74d", lw=5, solid_capstyle="round",
            zorder=5,
            label=fr"CI: [{mu_lo_ci:.2f}, {mu_hi_ci:.2f}]")
    ax.plot(x_obs, mu_lo_ci, "^", color="#ffb74d", ms=7, zorder=6)
    ax.plot(x_obs, mu_hi_ci, "v", color="#ffb74d", ms=7, zorder=6)
    ax.axhline(mu_lo_ci, color="#ffb74d", lw=0.8, ls=":", alpha=0.5)
    ax.axhline(mu_hi_ci, color="#ffb74d", lw=0.8, ls=":", alpha=0.5)

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(MU_MIN, MU_MAX)
    ax.set_xlabel("observable $x$", color="white")
    ax.set_ylabel("parameter $\\mu$", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(fontsize=7.5, facecolor="#1e2127", labelcolor="white",
              edgecolor="#444", loc="upper left")
    fig.tight_layout()
    return fig


def _plot_coverage(toys):
    n_total  = len(toys)
    n_shown  = min(n_total, MAX_BARS)
    shown    = toys[-n_shown:]
    n_covers = sum(t[3] for t in toys)
    coverage = n_covers / n_total

    fig, ax = plt.subplots(figsize=(5, max(2.5, n_shown * 0.18 + 0.6)))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    for i, (x_obs, mu_lo, mu_hi, covers) in enumerate(shown):
        color = "#66bb6a" if covers else "#ef5350"
        ax.barh(i, mu_hi - mu_lo, left=mu_lo,
                height=0.7, color=color, alpha=0.85)
        # mark x_obs and the centre of the CI to show asymmetry
        ci_centre = (mu_lo + mu_hi) / 2
        ax.plot(x_obs,     i, "|", color="white",   ms=5, mew=1.0)
        ax.plot(ci_centre, i, ".", color="#ffb74d",  ms=4, zorder=4)

    ax.axvline(MU_TRUE, color="#aed581", lw=2, ls="--",
               label=fr"$\mu_{{\rm true}}$ = {MU_TRUE}")

    green_p = mpatches.Patch(color="#66bb6a", alpha=0.85,
                             label="covers $\\mu_{\\rm true}$")
    red_p   = mpatches.Patch(color="#ef5350", alpha=0.85,
                             label="misses $\\mu_{\\rm true}$")
    ax.legend(handles=[green_p, red_p], fontsize=8,
              facecolor="#1e2127", labelcolor="white", edgecolor="#444",
              loc="upper right")

    tail = f" (last {n_shown})" if n_total > n_shown else ""
    ax.set_title(
        f"Coverage: {n_covers}/{n_total} = {coverage:.1%}{tail}"
        f"   —   expected {CL:.1%}",
        color="white", fontsize=9, pad=4,
    )
    ax.set_xlabel("$\\mu$", color="white")
    ax.set_ylabel(f"toy index{tail}", color="white")
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

    # ── Layout: controls column | belt | coverage ────────────────────────────
    col_ctrl, col_belt, col_cov = st.columns([1, 2, 2])

    with col_ctrl:
        st.markdown(
            fr"**Model**"
            fr"<br>$x \sim \mathcal{{N}}(\mu,\,(k\mu)^2)$"
            fr"<br>$k$ = {K}"
            fr"<br>$\mu_{{\rm true}}$ = {MU_TRUE}"
            fr"<br>CL = {CL:.0%}",
            unsafe_allow_html=True,
        )
        st.markdown("")
        one_toy       = st.button("▶ One more toy",   use_container_width=True, key="npg_one")
        hundred_toys  = st.button("▶▶ +100 toys",     use_container_width=True, key="npg_100")
        thousand_toys = st.button("▶▶▶ +1000 toys",   use_container_width=True, key="npg_1000")
        reset         = st.button("↺ Reset",           use_container_width=True, key="npg_reset")

    if reset:
        _reset()
    elif one_toy:
        _add_toys(1)
    elif hundred_toys:
        _add_toys(100)
    elif thousand_toys:
        _add_toys(1000)

    x_obs        = st.session_state.npg_last_x
    mu_lo, mu_hi = _confidence_interval(x_obs)

    with col_belt:
        st.markdown(r"#### Neyman belt — $\sigma = k\mu$")
        fig_belt = _plot_belt(x_obs)
        st.pyplot(fig_belt, use_container_width=True)
        plt.close(fig_belt)

    with col_cov:
        st.markdown("#### Coverage over toy experiments")
        fig_cov = _plot_coverage(st.session_state.npg_toys)
        st.pyplot(fig_cov, use_container_width=True)
        plt.close(fig_cov)

    # ── Annotation ───────────────────────────────────────────────────────────
    toys     = st.session_state.npg_toys
    n_covers = sum(t[3] for t in toys)
    coverage = n_covers / len(toys)
    asym     = (mu_hi - x_obs) / (x_obs - mu_lo) if x_obs > mu_lo else float("nan")

    st.markdown(
        fr"**Latest:** $x_{{\rm obs}}$ = {x_obs:.3f}, "
        fr"CI = [{mu_lo:.3f}, {mu_hi:.3f}] &nbsp;|&nbsp; "
        fr"asymmetry $(+/-)$ = {mu_hi - x_obs:.3f} / {x_obs - mu_lo:.3f} "
        fr"= {asym:.2f} &nbsp;|&nbsp; "
        fr"**Coverage:** {n_covers}/{len(toys)} = {coverage:.1%} "
        fr"(expected {CL:.1%})",
        unsafe_allow_html=True,
    )
