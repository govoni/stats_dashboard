"""
Neyman belt construction — four distributions in one render.

A selectbox in the left column lets the student switch between:
  1. Gaussian (fixed σ)        — symmetric, parallel belt, exact coverage
  2. Gaussian (σ = k·μ)        — symmetric pdf, fanning belt, asymmetric CI
  3. Poisson                   — discrete, staircase belt, overcoverage
  4. Log-normal                — asymmetric pdf, curved belt, asymmetric CI

Each distribution is encapsulated in a Dist dataclass that provides:
  - sample(rng)          → one toy observable
  - ci(x_obs)            → (mu_lo, mu_hi) confidence interval
  - belt_curves(mu_grid) → (x_lo_arr, x_hi_arr) acceptance boundaries
  - pdf_at_mu(mu, x_arr) → probability density (or pmf) at that mu
  - is_discrete          → True for Poisson
  - x_label, mu_label    → axis titles
  - x_range, mu_range    → (min, max) for plot axes
  - info_md              → short markdown description for the left panel

State is kept per-distribution under keys  nb_<name>_rng / nb_<name>_toys
so switching distribution never loses accumulated toys.

Layout rules (from README):
  - Controls in a row of st.columns() ABOVE the plots.
  - No title / back button (app.py handles that).
  - Only implement render().
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass, field
from typing import Callable
from scipy.stats import norm, poisson, chi2 as chi2_dist, lognorm
import streamlit as st


# ============================================================================
# Shared constants
# ============================================================================
ALPHA   = 0.32          # 1 - CL
CL      = 1 - ALPHA
Z       = float(norm.ppf(1 - ALPHA / 2))
MAX_BARS = 80


# ============================================================================
# Distribution definitions
# ============================================================================

@dataclass
class Dist:
    name       : str
    mu_true    : float
    x_range    : tuple
    mu_range   : tuple
    is_discrete: bool
    x_label    : str
    info_md    : str

    # callables set after __init__
    _sample    : Callable = field(repr=False, default=None)
    _ci        : Callable = field(repr=False, default=None)
    _belt      : Callable = field(repr=False, default=None)
    _pdf_at_mu : Callable = field(repr=False, default=None)

    def sample(self, rng):               return self._sample(rng)
    def ci(self, x_obs):                 return self._ci(x_obs)
    def belt_curves(self, mu_grid):      return self._belt(mu_grid)
    def pdf_at_mu(self, mu, x_arr):      return self._pdf_at_mu(mu, x_arr)


# ----------------------------------------------------------------------------
# 1. Gaussian fixed σ
# ----------------------------------------------------------------------------
def _make_gauss_fixed():
    MU, SIG = 2.0, 1.0
    xlo, xhi = -1.0, 5.0
    mlo, mhi = -1.0, 5.0

    def sample(rng):
        x = float(rng.normal(MU, SIG))
        return x

    def ci(x_obs):
        return x_obs - Z * SIG, x_obs + Z * SIG

    def belt(mu_grid):
        return mu_grid - Z * SIG, mu_grid + Z * SIG

    def pdf(mu, x_arr):
        return norm.pdf(x_arr, mu, SIG)

    return Dist(
        name="Gaussian (fixed σ)", mu_true=MU,
        x_range=(xlo, xhi), mu_range=(mlo, mhi),
        is_discrete=False, x_label="observable $x$",
        info_md=(
            fr"$x \sim \mathcal{{N}}(\mu, \sigma^2)$"
            fr"<br>$\sigma$ = {SIG} (known)"
            fr"<br>$\mu_{{\rm true}}$ = {MU}"
            "<br><br>Belt: parallel band."
            "<br>CI symmetric, constant width."
            "<br>Coverage exact."
        ),
        _sample=sample, _ci=ci, _belt=belt, _pdf_at_mu=pdf,
    )


# ----------------------------------------------------------------------------
# 2. Gaussian σ = k·μ
# ----------------------------------------------------------------------------
def _make_gauss_prop():
    MU, K = 3.0, 0.20
    xlo, xhi = 0.0, 7.0
    mlo, mhi = 0.0, 7.0

    def sample(rng):
        return float(max(rng.normal(MU, K * MU), 1e-6))

    def ci(x_obs):
        return x_obs / (1 + Z * K), x_obs / (1 - Z * K)

    def belt(mu_grid):
        return mu_grid * (1 - Z * K), mu_grid * (1 + Z * K)

    def pdf(mu, x_arr):
        return norm.pdf(x_arr, mu, K * mu)

    return Dist(
        name="Gaussian (σ = k·μ)", mu_true=MU,
        x_range=(xlo, xhi), mu_range=(mlo, mhi),
        is_discrete=False, x_label="observable $x$",
        info_md=(
            fr"$x \sim \mathcal{{N}}(\mu,\,(k\mu)^2)$"
            fr"<br>$k$ = {K} (relative resolution)"
            fr"<br>$\mu_{{\rm true}}$ = {MU}"
            "<br><br>Belt: fanning lines through origin."
            "<br>CI asymmetric, constant relative width."
            "<br>Coverage exact."
        ),
        _sample=sample, _ci=ci, _belt=belt, _pdf_at_mu=pdf,
    )


# ----------------------------------------------------------------------------
# 3. Poisson
# ----------------------------------------------------------------------------
def _make_poisson():
    MU   = 4.5
    NMAX = 20
    xlo, xhi = -0.5, float(NMAX)
    mlo, mhi = 0.0, 15.0
    MU_GRID  = np.linspace(0.001, mhi, 600)

    def _acc(mu):
        n_lo = int(poisson.ppf(ALPHA / 2, mu))
        n_hi = int(poisson.ppf(1 - ALPHA / 2, mu))
        while poisson.cdf(n_lo - 1, mu) > ALPHA / 2 and n_lo > 0:
            n_lo -= 1
        while 1 - poisson.cdf(n_hi, mu) > ALPHA / 2:
            n_hi += 1
        return n_lo, n_hi

    # precompute belt (called once, cached in session state)
    _belt_cache = {}

    def belt(mu_grid):
        key = id(mu_grid)
        if key not in _belt_cache:
            los, his = [], []
            for mu in mu_grid:
                lo, hi = _acc(mu)
                los.append(lo); his.append(hi)
            _belt_cache[key] = np.array(los, float), np.array(his, float)
        return _belt_cache[key]

    def sample(rng):
        return float(int(rng.poisson(MU)))

    def ci(x_obs):
        n = int(x_obs)
        lo = 0.0 if n == 0 else 0.5 * chi2_dist.ppf(ALPHA / 2, 2 * n)
        hi = 0.5 * chi2_dist.ppf(1 - ALPHA / 2, 2 * (n + 1))
        return float(lo), float(hi)

    def pdf(mu, x_arr):
        # returns pmf at integer positions; x_arr is used only for range
        ns = np.arange(0, NMAX + 1)
        return ns, poisson.pmf(ns, mu)   # special: return (ns, pmf)

    return Dist(
        name="Poisson", mu_true=MU,
        x_range=(xlo, xhi), mu_range=(mlo, mhi),
        is_discrete=True, x_label="observed count $n$",
        info_md=(
            fr"$n \sim \text{{Poisson}}(\mu)$"
            fr"<br>$\mu_{{\rm true}}$ = {MU}"
            "<br><br>Belt: staircase boundaries."
            "<br>CI asymmetric, variable width."
            "<br>Coverage ≥ CL (overcoverage)."
        ),
        _sample=sample, _ci=ci, _belt=belt, _pdf_at_mu=pdf,
    )


# ----------------------------------------------------------------------------
# 4. Log-normal  (asymmetric)
#    Observable: x ~ LogNormal(log(mu), s^2)
#    Parameter:  mu = median of x  (= exp of the location parameter)
#    s = known shape parameter (controls asymmetry)
# ----------------------------------------------------------------------------
def _make_lognormal():
    MU, S = 3.0, 0.40   # true median, shape (log-std)
    xlo, xhi = 0.0, 10.0
    mlo, mhi = 0.0,  8.0

    # Acceptance interval: central CL interval of LogNormal(log(mu), s^2)
    # x_lo = exp(log(mu) - z*s) = mu * exp(-z*s)
    # x_hi = exp(log(mu) + z*s) = mu * exp(+z*s)
    exp_lo = np.exp(-Z * S)
    exp_hi = np.exp(+Z * S)

    def sample(rng):
        return float(rng.lognormal(np.log(MU), S))

    def ci(x_obs):
        # invert: mu_lo = x_obs / exp(+z*s), mu_hi = x_obs / exp(-z*s)
        return x_obs / exp_hi, x_obs / exp_lo

    def belt(mu_grid):
        return mu_grid * exp_lo, mu_grid * exp_hi

    def pdf(mu, x_arr):
        # lognorm in scipy: scale=mu (median), s=S (shape)
        return lognorm.pdf(x_arr, s=S, scale=mu)

    return Dist(
        name="Log-normal (asymmetric)", mu_true=MU,
        x_range=(xlo, xhi), mu_range=(mlo, mhi),
        is_discrete=False, x_label="observable $x$",
        info_md=(
            fr"$x \sim \text{{LogNormal}}(\ln\mu,\, s^2)$"
            fr"<br>$s$ = {S} (shape, known)"
            fr"<br>$\mu$ = median of $x$"
            fr"<br>$\mu_{{\rm true}}$ = {MU}"
            "<br><br>Belt: curved, asymmetric."
            "<br>CI asymmetric, skewed high."
            "<br>Coverage exact."
        ),
        _sample=sample, _ci=ci, _belt=belt, _pdf_at_mu=pdf,
    )


# Registry — order determines selectbox order
DISTS = {
    d.name: d for d in [
        _make_gauss_fixed(),
        _make_gauss_prop(),
        _make_poisson(),
        _make_lognormal(),
    ]
}
DIST_NAMES = list(DISTS.keys())


# ============================================================================
# Session-state helpers  (namespaced per distribution)
# ============================================================================

def _key(dist, suffix):
    return f"nb_{dist.name.replace(' ', '_')}_{suffix}"


def _run_toy(dist, rng):
    x_obs = dist.sample(rng)
    mu_lo, mu_hi = dist.ci(x_obs)
    covers = (mu_lo <= dist.mu_true <= mu_hi)
    return x_obs, mu_lo, mu_hi, covers


def _init_dist(dist):
    rk, tk, lk = _key(dist, "rng"), _key(dist, "toys"), _key(dist, "last")
    if rk not in st.session_state:
        st.session_state[rk] = np.random.default_rng(42)
    if tk not in st.session_state:
        x_obs, mu_lo, mu_hi, covers = _run_toy(dist, st.session_state[rk])
        st.session_state[tk] = [(x_obs, mu_lo, mu_hi, covers)]
        st.session_state[lk] = x_obs


def _add_toys(dist, n):
    rk, tk, lk = _key(dist, "rng"), _key(dist, "toys"), _key(dist, "last")
    for _ in range(n):
        x_obs, mu_lo, mu_hi, covers = _run_toy(dist, st.session_state[rk])
        st.session_state[tk].append((x_obs, mu_lo, mu_hi, covers))
        st.session_state[lk] = x_obs


def _reset_dist(dist):
    rk, tk, lk = _key(dist, "rng"), _key(dist, "toys"), _key(dist, "last")
    st.session_state[rk] = np.random.default_rng(42)
    x_obs, mu_lo, mu_hi, covers = _run_toy(dist, st.session_state[rk])
    st.session_state[tk] = [(x_obs, mu_lo, mu_hi, covers)]
    st.session_state[lk] = x_obs


# ============================================================================
# Plotting helpers
# ============================================================================

def _ax_style(ax):
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")


def _plot_belt(dist, x_obs):
    mu_lo_ci, mu_hi_ci = dist.ci(x_obs)
    mu_min, mu_max = dist.mu_range
    x_min,  x_max  = dist.x_range

    # mu grid for belt curves (avoid 0 for log-normal / Poisson)
    mu_grid = np.linspace(max(mu_min, 0.01), mu_max, 500)
    x_lo_b, x_hi_b = dist.belt_curves(mu_grid)

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor("#0e1117")
    _ax_style(ax)

    # ── Belt ────────────────────────────────────────────────────────────────
    ax.fill_betweenx(mu_grid, x_lo_b, x_hi_b,
                     color="#4fc3f7", alpha=0.18, linewidth=0,
                     label=f"{CL:.0%} acceptance belt")
    ax.plot(x_lo_b, mu_grid, color="#4fc3f7", lw=1.0, alpha=0.7)
    ax.plot(x_hi_b, mu_grid, color="#4fc3f7", lw=1.0, alpha=0.7)

    # ── Distribution drawn sideways at mu_true ───────────────────────────────
    mu_t = dist.mu_true
    scale_h = (mu_max - mu_min) * 0.18   # bars/pdf occupy this many mu-units

    if dist.is_discrete:
        # Poisson: rectangle bars
        ns, pmf = dist.pdf_at_mu(mu_t, None)
        x_lo_t, x_hi_t = dist.belt_curves(np.array([mu_t]))
        n_lo_t, n_hi_t = int(round(x_lo_t[0])), int(round(x_hi_t[0]))
        bar_scale = scale_h / pmf.max()
        for n_val, p in zip(ns, pmf):
            if p < 1e-4: continue
            inside  = (n_lo_t <= n_val <= n_hi_t)
            ax.add_patch(plt.Rectangle(
                (n_val - 0.35, mu_t), 0.70, p * bar_scale,
                color="#aed581" if inside else "#4a6741",
                alpha=0.60 if inside else 0.25, zorder=3,
            ))
        ax.add_patch(plt.Rectangle((0, 0), 0, 0, color="#aed581", alpha=0.60,
                     label=fr"pmf at $\mu_{{\rm true}}$ (green = acceptance)"))
    else:
        # Continuous: sideways pdf curve
        x_pdf = np.linspace(x_min, x_max, 400)
        pdf   = dist.pdf_at_mu(mu_t, x_pdf)
        pdf_scale = scale_h / pdf.max()
        ax.plot(x_pdf, mu_t + pdf * pdf_scale,
                color="#aed581", lw=1.2, alpha=0.6)
        ax.fill_between(x_pdf, mu_t, mu_t + pdf * pdf_scale,
                        color="#aed581", alpha=0.08)
        # Shade acceptance region
        x_lo_t = float(dist.belt_curves(np.array([mu_t]))[0][0])
        x_hi_t = float(dist.belt_curves(np.array([mu_t]))[1][0])
        mask = (x_pdf >= x_lo_t) & (x_pdf <= x_hi_t)
        ax.fill_between(x_pdf[mask], mu_t, mu_t + pdf[mask] * pdf_scale,
                        color="#aed581", alpha=0.25,
                        label=fr"pdf at $\mu_{{\rm true}}$ (shaded = {CL:.0%})")

    # ── True mu line ─────────────────────────────────────────────────────────
    ax.axhline(mu_t, color="#aed581", lw=1.2, ls="--",
               label=fr"$\mu_{{\rm true}}$ = {mu_t}")

    # ── Observed x_obs ───────────────────────────────────────────────────────
    ax.axvline(x_obs, color="#ff7043", lw=2, zorder=4,
               label=fr"$x_{{\rm obs}}$ = {x_obs:.2f}")

    # ── Resulting CI on mu ───────────────────────────────────────────────────
    ax.plot([x_obs, x_obs], [mu_lo_ci, mu_hi_ci],
            color="#ffb74d", lw=5, solid_capstyle="round",
            zorder=5, label=fr"CI: [{mu_lo_ci:.2f}, {mu_hi_ci:.2f}]")
    ax.plot(x_obs, mu_lo_ci, "^", color="#ffb74d", ms=7, zorder=6)
    ax.plot(x_obs, mu_hi_ci, "v", color="#ffb74d", ms=7, zorder=6)
    ax.axhline(mu_lo_ci, color="#ffb74d", lw=0.8, ls=":", alpha=0.5)
    ax.axhline(mu_hi_ci, color="#ffb74d", lw=0.8, ls=":", alpha=0.5)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(mu_min, mu_max)
    ax.set_xlabel(dist.x_label, color="white")
    ax.set_ylabel("parameter $\\mu$", color="white")
    ax.legend(fontsize=7.5, facecolor="#1e2127", labelcolor="white",
              edgecolor="#444", loc="upper left")
    fig.tight_layout()
    return fig


def _plot_coverage(dist, toys):
    mu_t    = dist.mu_true
    n_total = len(toys)
    n_shown = min(n_total, MAX_BARS)
    shown   = toys[-n_shown:]
    n_covers = sum(t[3] for t in toys)
    coverage = n_covers / n_total

    fig, ax = plt.subplots(figsize=(5, max(2.5, n_shown * 0.18 + 0.6)))
    fig.patch.set_facecolor("#0e1117")
    _ax_style(ax)

    for i, (x_obs, mu_lo, mu_hi, covers) in enumerate(shown):
        ax.barh(i, mu_hi - mu_lo, left=mu_lo, height=0.7,
                color="#66bb6a" if covers else "#ef5350",
                alpha=0.85)
        ax.plot(x_obs, i, "|", color="white", ms=5, mew=1.0)

    ax.axvline(mu_t, color="#aed581", lw=2, ls="--",
               label=fr"$\mu_{{\rm true}}$ = {mu_t}")
    ax.legend(
        handles=[
            mpatches.Patch(color="#66bb6a", alpha=0.85, label="covers $\\mu_{\\rm true}$"),
            mpatches.Patch(color="#ef5350", alpha=0.85, label="misses $\\mu_{\\rm true}$"),
        ],
        fontsize=8, facecolor="#1e2127", labelcolor="white",
        edgecolor="#444", loc="upper right",
    )
    tail = f" (last {n_shown})" if n_total > n_shown else ""
    ax.set_title(
        f"Coverage: {n_covers}/{n_total} = {coverage:.1%}{tail}"
        f"   —   expected {CL:.1%}",
        color="white", fontsize=9, pad=4,
    )
    ax.set_xlabel("$\\mu$", color="white")
    ax.set_ylabel(f"toy index{tail}", color="white")
    fig.tight_layout()
    return fig


# ============================================================================
# Public entry point
# ============================================================================

def render():
    # ── Layout ───────────────────────────────────────────────────────────────

    st.title ("The Neyman belt construction")

    col_ctrl, col_belt, col_cov = st.columns([1, 2, 2])

    with col_ctrl:
        choice = st.selectbox("Distribution", DIST_NAMES, key="nb_choice")
        dist   = DISTS[choice]
        _init_dist(dist)

        # st.markdown(dist.info_md + f"<br>CL = {CL:.0%}",
        #             unsafe_allow_html=True)
        # st.markdown("")

        one_toy       = st.button("▶ One more toy",   use_container_width=True, key="nb_one")
        hundred_toys  = st.button("▶▶ +100 toys",     use_container_width=True, key="nb_100")
        thousand_toys = st.button("▶▶▶ +1000 toys",   use_container_width=True, key="nb_1000")
        reset         = st.button("↺ Reset",           use_container_width=True, key="nb_reset")

    # handle buttons outside col_ctrl so Streamlit doesn't nest re-runs
    if reset:
        _reset_dist(dist)
    elif one_toy:
        _add_toys(dist, 1)
    elif hundred_toys:
        _add_toys(dist, 100)
    elif thousand_toys:
        _add_toys(dist, 1000)

    toys  = st.session_state[_key(dist, "toys")]
    x_obs = st.session_state[_key(dist, "last")]
    mu_lo, mu_hi = dist.ci(x_obs)

    with col_belt:
        st.markdown("#### Neyman belt construction")
        fig_belt = _plot_belt(dist, x_obs)
        st.pyplot(fig_belt, use_container_width=True)
        plt.close(fig_belt)

    with col_cov:
        st.markdown("#### Coverage over toy experiments")
        fig_cov = _plot_coverage(dist, toys)
        st.pyplot(fig_cov, use_container_width=True)
        plt.close(fig_cov)

    # ── Annotation ───────────────────────────────────────────────────────────
    # n_covers = sum(t[3] for t in toys)
    # coverage = n_covers / len(toys)
    # st.markdown(
    #     fr"**Latest:** $x_{{\rm obs}}$ = {x_obs:.3f}, "
    #     fr"CI = [{mu_lo:.3f}, {mu_hi:.3f}] &nbsp;|&nbsp; "
    #     fr"**Coverage:** {n_covers}/{len(toys)} = {coverage:.1%} "
    #     fr"(expected {CL:.1%})",
    #     unsafe_allow_html=True,
    # )
