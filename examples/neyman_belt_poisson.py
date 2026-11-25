"""
Neyman construction of a confidence belt for a Poisson mean.

Observed: n ~ Poisson(mu)
Unknown parameter: mu  (the true mean)
Goal: build a confidence belt at level 1-alpha.

Construction (Feldman-Cousins ordering is NOT used here; we use the
classical central interval ordering to keep the logic transparent):

  For each mu on a fine grid:
    - find n_lo(mu), n_hi(mu) such that
        P(n <= n_hi | mu) >= 1 - alpha/2   (upper tail)
        P(n >= n_lo | mu) >= 1 - alpha/2   (lower tail)
      i.e. the central interval in n that contains at least (1-alpha)
      of the Poisson probability.
    - draw the horizontal segment [n_lo(mu), n_hi(mu)] at height mu.

The belt is the union of all those segments.

Given an observed n_obs, the confidence interval [mu_lo, mu_hi] is the
vertical projection of the belt onto the mu axis at x = n_obs:
    mu_lo = smallest mu whose interval reaches n_obs from below
    mu_hi = largest  mu whose interval reaches n_obs from above

Exact Poisson inversion gives the well-known formulas:
    mu_lo = 0.5 * chi2.ppf(alpha/2,   2*n_obs)       (0 if n_obs=0)
    mu_hi = 0.5 * chi2.ppf(1-alpha/2, 2*(n_obs+1))

Layout:
  Left   : the Neyman belt, the observed n_obs (vertical line),
           and the resulting confidence interval on mu (horizontal segment)
  Right  : growing coverage plot — one bar per toy, green if the
           interval contains mu_true, red if it misses; coverage fraction
           shown in the title.

Layout rules (from README):
  - Controls in a row of st.columns() ABOVE the plots.
  - No title / back button (app.py handles that).
  - Only implement render().
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import poisson, chi2 as chi2_dist
import streamlit as st


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
MU_TRUE  = 4.5      # true Poisson mean used for toy experiments
ALPHA    = 0.32     # 1 - confidence level  (0.32 → 68% CL, 0.05 → 95% CL)
CL       = 1 - ALPHA

MU_MAX   = 15.0     # upper edge of mu axis in the belt plot
N_MAX    = 20       # upper edge of n axis

# Fine mu grid for drawing the belt
MU_GRID  = np.linspace(0.001, MU_MAX, 600)

MAX_BARS = 80       # max intervals shown in the coverage panel


# ---------------------------------------------------------------------------
# Core: Poisson central interval in n, and CI on mu
# ---------------------------------------------------------------------------

def _acceptance_interval(mu: float) -> tuple[int, int]:
    """
    Central acceptance interval [n_lo, n_hi] in n such that
    P(n_lo <= n <= n_hi | mu) >= CL.
    Uses the smallest such interval symmetric in tail probability.
    """
    # lower tail: largest n_lo s.t. P(n < n_lo | mu) <= ALPHA/2
    n_lo = int(poisson.ppf(ALPHA / 2, mu))
    # upper tail: smallest n_hi s.t. P(n > n_hi | mu) <= ALPHA/2
    n_hi = int(poisson.ppf(1 - ALPHA / 2, mu))
    # ppf can be off by one for discrete distributions; verify
    while poisson.cdf(n_lo - 1, mu) > ALPHA / 2 and n_lo > 0:
        n_lo -= 1
    while 1 - poisson.cdf(n_hi, mu) > ALPHA / 2:
        n_hi += 1
    return n_lo, n_hi


def _confidence_interval(n_obs: int) -> tuple[float, float]:
    """
    Exact Poisson CI for the mean given observed count n_obs.
    Uses chi-squared quantiles (classical result).
    """
    if n_obs == 0:
        mu_lo = 0.0
    else:
        mu_lo = 0.5 * chi2_dist.ppf(ALPHA / 2, 2 * n_obs)
    mu_hi = 0.5 * chi2_dist.ppf(1 - ALPHA / 2, 2 * (n_obs + 1))
    return float(mu_lo), float(mu_hi)


# ---------------------------------------------------------------------------
# Belt: precompute once per session (depends only on ALPHA, MU_GRID)
# ---------------------------------------------------------------------------

def _build_belt():
    """Returns list of (mu, n_lo, n_hi) for each mu in MU_GRID."""
    belt = []
    for mu in MU_GRID:
        n_lo, n_hi = _acceptance_interval(mu)
        belt.append((mu, n_lo, n_hi))
    return belt


def _get_belt():
    if "neyman_belt" not in st.session_state:
        st.session_state.neyman_belt = _build_belt()
    return st.session_state.neyman_belt


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def _one_toy(rng) -> tuple[int, float, float, bool]:
    n_obs = int(rng.poisson(MU_TRUE))
    mu_lo, mu_hi = _confidence_interval(n_obs)
    covers = (mu_lo <= MU_TRUE <= mu_hi)
    return n_obs, mu_lo, mu_hi, covers


def _init_state():
    if "neyman_rng" not in st.session_state:
        st.session_state.neyman_rng = np.random.default_rng(42)
    if "neyman_toys" not in st.session_state:
        n_obs, mu_lo, mu_hi, covers = _one_toy(st.session_state.neyman_rng)
        st.session_state.neyman_toys    = [(n_obs, mu_lo, mu_hi, covers)]
        st.session_state.neyman_last_n  = n_obs


def _add_toys(n: int):
    rng = st.session_state.neyman_rng
    for _ in range(n):
        n_obs, mu_lo, mu_hi, covers = _one_toy(rng)
        st.session_state.neyman_toys.append((n_obs, mu_lo, mu_hi, covers))
        st.session_state.neyman_last_n = n_obs


def _reset():
    # Keep the belt — it doesn't change
    st.session_state.neyman_rng = np.random.default_rng(42)
    n_obs, mu_lo, mu_hi, covers = _one_toy(st.session_state.neyman_rng)
    st.session_state.neyman_toys   = [(n_obs, mu_lo, mu_hi, covers)]
    st.session_state.neyman_last_n = n_obs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_belt(belt, n_obs):
    """Left panel: the Neyman belt + Poisson pmf at MU_TRUE + observed n + CI."""
    mu_lo, mu_hi = _confidence_interval(n_obs)

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    # ── Belt ────────────────────────────────────────────────────────────────
    mus   = np.array([b[0] for b in belt])
    n_los = np.array([b[1] for b in belt], dtype=float)
    n_his = np.array([b[2] for b in belt], dtype=float)

    ax.fill_betweenx(mus, n_los, n_his,
                     color="#4fc3f7", alpha=0.18, linewidth=0,
                     label=f"{CL:.0%} acceptance belt")
    ax.plot(n_los, mus, color="#4fc3f7", lw=0.8, alpha=0.6)
    ax.plot(n_his, mus, color="#4fc3f7", lw=0.8, alpha=0.6)

    # ── Poisson pmf drawn sideways at MU_TRUE ───────────────────────────────
    # Each integer n is a Rectangle:
    #   x in [n - 0.35, n + 0.35]  (bar centred on the integer tick)
    #   y in [MU_TRUE, MU_TRUE + p * scale]  (upward in the mu direction)
    # Bars inside the acceptance interval are highlighted.
    n_lo_true, n_hi_true = _acceptance_interval(MU_TRUE)
    n_vals = np.arange(0, N_MAX + 1)
    pmf    = poisson.pmf(n_vals, MU_TRUE)
    scale  = 1.8 / pmf.max()   # tallest bar = 1.8 mu-units tall

    for n_val, p in zip(n_vals, pmf):
        if p < 1e-4:
            continue
        inside  = (n_lo_true <= n_val <= n_hi_true)
        color_b = "#aed581" if inside else "#4a6741"
        alpha_b = 0.60      if inside else 0.25
        ax.add_patch(plt.Rectangle(
            (n_val - 0.35, MU_TRUE), 0.70, p * scale,
            color=color_b, alpha=alpha_b, zorder=3,
        ))

    # Legend proxy for the pmf
    ax.add_patch(plt.Rectangle((0, 0), 0, 0, color="#aed581", alpha=0.60,
                 label=fr"pmf at $\mu_{{\rm true}}$ (green = acceptance)"))

    # ── True mu line ─────────────────────────────────────────────────────────
    ax.axhline(MU_TRUE, color="#aed581", lw=1.2, ls="--",
               label=fr"$\mu_{{\rm true}}$ = {MU_TRUE}")

    # ── Observed n_obs ───────────────────────────────────────────────────────
    ax.axvline(n_obs, color="#ff7043", lw=2, zorder=4,
               label=f"$n_{{\\rm obs}}$ = {n_obs}")

    # ── Resulting CI on mu ───────────────────────────────────────────────────
    ax.plot([n_obs, n_obs], [mu_lo, mu_hi],
            color="#ffb74d", lw=4, solid_capstyle="round",
            zorder=5, label=fr"CI: [{mu_lo:.2f}, {mu_hi:.2f}]")
    ax.plot(n_obs, mu_lo, "^", color="#ffb74d", ms=7, zorder=6)
    ax.plot(n_obs, mu_hi, "v", color="#ffb74d", ms=7, zorder=6)
    ax.axhline(mu_lo, color="#ffb74d", lw=0.8, ls=":", alpha=0.6)
    ax.axhline(mu_hi, color="#ffb74d", lw=0.8, ls=":", alpha=0.6)

    ax.set_xlim(-0.5, N_MAX)
    ax.set_ylim(0, MU_MAX)
    ax.set_xlabel("observed $n$", color="white")
    ax.set_ylabel("parameter $\\mu$", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(fontsize=8, facecolor="#1e2127", labelcolor="white",
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

    for i, (n_obs, mu_lo, mu_hi, covers) in enumerate(shown):
        color = "#66bb6a" if covers else "#ef5350"
        ax.barh(i, mu_hi - mu_lo, left=mu_lo,
                height=0.7, color=color, alpha=0.85)
        ax.plot(n_obs, i, "|", color="white", ms=5, mew=1.0)

    ax.axvline(MU_TRUE, color="#aed581", lw=2, ls="--",
               label=fr"$\mu_{{\rm true}}$ = {MU_TRUE}")

    green_p = mpatches.Patch(color="#66bb6a", alpha=0.85, label="covers $\\mu_{\\rm true}$")
    red_p   = mpatches.Patch(color="#ef5350", alpha=0.85, label="misses $\\mu_{\\rm true}$")
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
    belt = _get_belt()

    # ── Layout: controls column | belt | coverage ────────────────────────────
    col_ctrl, col_belt, col_cov = st.columns([1, 2, 2])

    with col_ctrl:
        st.markdown(
            fr"**Model** &nbsp;  "
            fr"$n \sim \text{{Poisson}}(\mu)$"
            fr"<br>$\mu_{{\rm true}}$ = {MU_TRUE}"
            fr"<br>CL = {CL:.0%} &nbsp; ($\alpha$ = {ALPHA})",
            unsafe_allow_html=True,
        )
        st.markdown("")
        one_toy       = st.button("▶ One more toy",   use_container_width=True, key="ney_one")
        hundred_toys  = st.button("▶▶ +100 toys",     use_container_width=True, key="ney_100")
        thousand_toys = st.button("▶▶▶ +1000 toys",   use_container_width=True, key="ney_1000")
        reset         = st.button("↺ Reset",           use_container_width=True, key="ney_reset")

    if reset:
        _reset()
    elif one_toy:
        _add_toys(1)
    elif hundred_toys:
        _add_toys(100)
    elif thousand_toys:
        _add_toys(1000)

    n_obs = st.session_state.neyman_last_n
    mu_lo, mu_hi = _confidence_interval(n_obs)

    with col_belt:
        st.markdown("#### Neyman belt construction")
        fig_belt = _plot_belt(belt, n_obs)
        st.pyplot(fig_belt, use_container_width=True)
        plt.close(fig_belt)

    with col_cov:
        st.markdown("#### Coverage over toy experiments")
        fig_cov = _plot_coverage(st.session_state.neyman_toys)
        st.pyplot(fig_cov, use_container_width=True)
        plt.close(fig_cov)

    # ── Annotation ───────────────────────────────────────────────────────────
    toys     = st.session_state.neyman_toys
    n_covers = sum(t[3] for t in toys)
    coverage = n_covers / len(toys)

    st.markdown(
        fr"**Latest:** $n_{{\rm obs}}$ = {n_obs}, "
        fr"CI = [{mu_lo:.3f}, {mu_hi:.3f}] &nbsp;|&nbsp; "
        fr"**Coverage:** {n_covers}/{len(toys)} = {coverage:.1%} "
        fr"(expected {CL:.1%})",
        unsafe_allow_html=True,
    )
