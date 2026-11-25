"""
Neyman belt construction for a Gaussian observable with known width.

Observable: x ~ N(mu, sigma^2),  single measurement
Parameter:  mu  (unknown mean)
Known:      sigma  (fixed width)

Central interval construction:
  For each mu, the acceptance interval [x_lo, x_hi] satisfies
      P(x_lo <= x <= x_hi | mu) = CL
  with equal tails:
      x_lo(mu) = mu - z * sigma
      x_hi(mu) = mu + z * sigma
  where z = norm.ppf(1 - alpha/2).

The belt is therefore a straight diagonal band of constant width 2*z*sigma,
with slopes dx_lo/dmu = dx_hi/dmu = 1.

Inverting at x_obs (reading the vertical slice through the belt):
      mu_lo = x_obs - z * sigma
      mu_hi = x_obs + z * sigma

The CI has the same width as the acceptance interval, shifted by x_obs.
Coverage is exactly CL for every mu — no overcoverage — because the
distribution is continuous.

For n measurements the sufficient statistic is the sample mean x_bar,
which is N(mu, sigma^2/n), so the construction is identical with
sigma -> sigma/sqrt(n).

Layout:
  Left   : Neyman belt, with the Gaussian pdf drawn at mu_true (and
           optionally at a second mu to show how the interval shifts),
           the observed x_obs, and the resulting CI on mu.
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
MU_TRUE  = 2.0      # true mean used for toy generation
SIGMA    = 1.0      # known fixed width
N_MEAS   = 1        # number of measurements per toy (uses sample mean)
ALPHA    = 0.32     # 1 - CL  (0.32 → 68%, 0.05 → 95%)
CL       = 1 - ALPHA

# z-score for the chosen CL
Z        = float(norm.ppf(1 - ALPHA / 2))
SIGMA_EFF = SIGMA / np.sqrt(N_MEAS)   # std of sample mean

# Plot ranges
MU_MIN, MU_MAX = -1.0,  5.0
X_MIN,  X_MAX  = -1.0,  5.0

MAX_BARS = 80


# ---------------------------------------------------------------------------
# Core: acceptance interval and CI  (both trivially closed-form here)
# ---------------------------------------------------------------------------

def _acceptance_interval(mu: float) -> tuple[float, float]:
    return mu - Z * SIGMA_EFF, mu + Z * SIGMA_EFF


def _confidence_interval(x_obs: float) -> tuple[float, float]:
    """Inversion of the belt: same formula, centred on x_obs."""
    return x_obs - Z * SIGMA_EFF, x_obs + Z * SIGMA_EFF


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def _one_toy(rng) -> tuple[float, float, float, bool]:
    x_obs  = float(rng.normal(MU_TRUE, SIGMA_EFF))
    mu_lo, mu_hi = _confidence_interval(x_obs)
    covers = (mu_lo <= MU_TRUE <= mu_hi)
    return x_obs, mu_lo, mu_hi, covers


def _init_state():
    if "neyg_rng" not in st.session_state:
        st.session_state.neyg_rng = np.random.default_rng(42)
    if "neyg_toys" not in st.session_state:
        x_obs, mu_lo, mu_hi, covers = _one_toy(st.session_state.neyg_rng)
        st.session_state.neyg_toys   = [(x_obs, mu_lo, mu_hi, covers)]
        st.session_state.neyg_last_x = x_obs


def _add_toys(n: int):
    rng = st.session_state.neyg_rng
    for _ in range(n):
        x_obs, mu_lo, mu_hi, covers = _one_toy(rng)
        st.session_state.neyg_toys.append((x_obs, mu_lo, mu_hi, covers))
        st.session_state.neyg_last_x = x_obs


def _reset():
    st.session_state.neyg_rng = np.random.default_rng(42)
    x_obs, mu_lo, mu_hi, covers = _one_toy(st.session_state.neyg_rng)
    st.session_state.neyg_toys   = [(x_obs, mu_lo, mu_hi, covers)]
    st.session_state.neyg_last_x = x_obs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_belt(x_obs):
    mu_lo, mu_hi = _confidence_interval(x_obs)

    mu_grid = np.linspace(MU_MIN, MU_MAX, 400)
    x_lo    = mu_grid - Z * SIGMA_EFF
    x_hi    = mu_grid + Z * SIGMA_EFF

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    # ── Belt ────────────────────────────────────────────────────────────────
    ax.fill_betweenx(mu_grid, x_lo, x_hi,
                     color="#4fc3f7", alpha=0.18, linewidth=0,
                     label=f"{CL:.0%} acceptance belt")
    ax.plot(x_lo, mu_grid, color="#4fc3f7", lw=1.0, alpha=0.7)
    ax.plot(x_hi, mu_grid, color="#4fc3f7", lw=1.0, alpha=0.7)

    # ── Gaussian pdf drawn sideways at mu_true ───────────────────────────────
    # Shows the probability density the belt is built from
    x_pdf  = np.linspace(X_MIN, X_MAX, 400)
    pdf    = norm.pdf(x_pdf, MU_TRUE, SIGMA_EFF)
    # Scale to occupy ~1.5 mu-units in height, centred at mu_true
    scale  = 1.5 / pdf.max()
    ax.plot(x_pdf, MU_TRUE + pdf * scale,
            color="#aed581", lw=1.2, ls="-", alpha=0.6)
    ax.fill_between(x_pdf, MU_TRUE, MU_TRUE + pdf * scale,
                    color="#aed581", alpha=0.08)
    # Shade the acceptance region of the pdf
    mask = (x_pdf >= MU_TRUE - Z * SIGMA_EFF) & \
           (x_pdf <= MU_TRUE + Z * SIGMA_EFF)
    ax.fill_between(x_pdf[mask], MU_TRUE,
                    MU_TRUE + pdf[mask] * scale,
                    color="#aed581", alpha=0.22,
                    label=fr"pdf at $\mu_{{true}}$ (shaded = {CL:.0%})")

    # ── True mu line ─────────────────────────────────────────────────────────
    ax.axhline(MU_TRUE, color="#aed581", lw=1.2, ls="--",
               label=fr"$\mu_{{\rm true}}$ = {MU_TRUE}")

    # ── Observed x_obs ───────────────────────────────────────────────────────
    ax.axvline(x_obs, color="#ff7043", lw=2, zorder=4,
               label=fr"$x_{{\rm obs}}$ = {x_obs:.3f}")

    # ── Resulting CI on mu ───────────────────────────────────────────────────
    ax.plot([x_obs, x_obs], [mu_lo, mu_hi],
            color="#ffb74d", lw=5, solid_capstyle="round",
            zorder=5, label=fr"CI: [{mu_lo:.2f}, {mu_hi:.2f}]")
    ax.plot(x_obs, mu_lo, "^", color="#ffb74d", ms=7, zorder=6)
    ax.plot(x_obs, mu_hi, "v", color="#ffb74d", ms=7, zorder=6)
    ax.axhline(mu_lo, color="#ffb74d", lw=0.8, ls=":", alpha=0.5)
    ax.axhline(mu_hi, color="#ffb74d", lw=0.8, ls=":", alpha=0.5)

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
        ax.plot(x_obs, i, "|", color="white", ms=5, mew=1.0)

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
            fr"<br>$x \sim \mathcal{{N}}(\mu,\, \sigma^2)$"
            fr"<br>$\sigma$ = {SIGMA}, $N$ = {N_MEAS}"
            fr"<br>$\mu_{{\rm true}}$ = {MU_TRUE}"
            fr"<br>CL = {CL:.0%}",
            unsafe_allow_html=True,
        )
        st.markdown("")
        one_toy       = st.button("▶ One more toy",   use_container_width=True, key="neyg_one")
        hundred_toys  = st.button("▶▶ +100 toys",     use_container_width=True, key="neyg_100")
        thousand_toys = st.button("▶▶▶ +1000 toys",   use_container_width=True, key="neyg_1000")
        reset         = st.button("↺ Reset",           use_container_width=True, key="neyg_reset")

    if reset:
        _reset()
    elif one_toy:
        _add_toys(1)
    elif hundred_toys:
        _add_toys(100)
    elif thousand_toys:
        _add_toys(1000)

    x_obs        = st.session_state.neyg_last_x
    mu_lo, mu_hi = _confidence_interval(x_obs)

    with col_belt:
        st.markdown("#### Neyman belt construction")
        fig_belt = _plot_belt(x_obs)
        st.pyplot(fig_belt, use_container_width=True)
        plt.close(fig_belt)

    with col_cov:
        st.markdown("#### Coverage over toy experiments")
        fig_cov = _plot_coverage(st.session_state.neyg_toys)
        st.pyplot(fig_cov, use_container_width=True)
        plt.close(fig_cov)

    # ── Annotation ───────────────────────────────────────────────────────────
    toys     = st.session_state.neyg_toys
    n_covers = sum(t[3] for t in toys)
    coverage = n_covers / len(toys)

    st.markdown(
        fr"**Latest:** $x_{{\rm obs}}$ = {x_obs:.3f}, "
        fr"CI = [{mu_lo:.3f}, {mu_hi:.3f}], "
        fr"half-width = $z\,\sigma$ = {Z:.3f} $\times$ {SIGMA_EFF:.3f} = {Z*SIGMA_EFF:.3f}"
        fr" &nbsp;|&nbsp; "
        fr"**Coverage:** {n_covers}/{len(toys)} = {coverage:.1%} "
        fr"(expected {CL:.1%})",
        unsafe_allow_html=True,
    )
