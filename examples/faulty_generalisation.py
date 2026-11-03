"""
examples/faulty_generalisation.py
──────────────────────────────────
Interactive illustration of the "Law of Faulty Generalisation" fallacy:
drawing a universal conclusion from a sample that is too small, biased,
or unrepresentative of the full population.

Three scenarios the student can explore:
  1. Small sample  – flip a coin N times; watch the empirical frequency
                     wander far from 0.5 and then converge as N grows.
  2. Biased sample – estimate the mean height of a population by sampling
                     only from the tail (e.g. basketball players).
  3. Survivorship  – estimate average company lifetime from companies that
                     are *currently* alive (classic survivorship bias).

Controls are in a single row of st.columns() above the plots,
as required by the dashboard shell.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── reproducible RNG ---------------------------------------------------------
RNG = np.random.default_rng(42)


# ── helper: clear, styled figure ---------------------------------------------
def _fig(figsize=(10, 4)):
    fig = plt.figure(figsize=figsize, facecolor="none")
    return fig


# ── Scenario 1: small-sample coin flipping -----------------------------------
def _coin_flip(n_flips: int):
    """Show empirical P(heads) vs true 0.5 as sample size varies."""
    outcomes = RNG.integers(0, 2, size=n_flips)          # 0=tails, 1=heads
    cumulative_freq = np.cumsum(outcomes) / np.arange(1, n_flips + 1)

    fig = _fig()
    ax = fig.add_subplot(111)
    ax.plot(cumulative_freq, color="#e07b39", lw=1.5, label="Observed P(heads)")
    ax.axhline(0.5, color="#3a7abf", lw=1.5, ls="--", label="True P = 0.5")
    ax.axvline(n_flips, color="gray", lw=0.7, ls=":")

    # shade the "hasty conclusion" zone
    early = max(1, n_flips // 10)
    ax.axvspan(0, early, alpha=0.12, color="red",
               label=f"'Hasty' zone (first {early} flips)")
    early_val = cumulative_freq[early - 1]
    ax.annotate(
        f"After {early} flips:\nP̂ = {early_val:.2f}  ← faulty conclusion",
        xy=(early, early_val), xytext=(early + n_flips * 0.05, early_val + 0.08),
        fontsize=8, color="darkred",
        arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8),
    )

    ax.set_xlabel("Number of flips")
    ax.set_ylabel("Empirical P(heads)")
    ax.set_ylim(0, 1)
    ax.set_xlim(1, n_flips)
    ax.legend(fontsize=8)
    ax.set_title(
        f"Coin flip: empirical frequency after {n_flips} flips  "
        f"(final P̂ = {cumulative_freq[-1]:.3f})"
    )
    fig.tight_layout()
    return fig


# ── Scenario 2: biased sample ------------------------------------------------
def _biased_sample(bias_pct: int):
    """
    True population: heights ~ N(170, 10).
    Biased sample: draw `bias_pct`% from the top decile (≥185 cm).
    Show how the sample mean overshoots the true mean.
    """
    n_total = 400
    n_biased = int(n_total * bias_pct / 100)
    n_random = n_total - n_biased

    pop_mean, pop_std = 170, 10
    population = RNG.normal(pop_mean, pop_std, 5000)

    # Random part of the sample
    random_part = RNG.normal(pop_mean, pop_std, n_random)
    # Biased part: only tall people (top ≈10 %)
    tall_pool = population[population >= 185]
    biased_part = RNG.choice(tall_pool, size=min(n_biased, len(tall_pool)),
                             replace=True)
    sample = np.concatenate([random_part, biased_part])
    sample_mean = sample.mean()

    fig = _fig()
    ax = fig.add_subplot(111)

    xs = np.linspace(population.min(), population.max(), 300)
    from scipy.stats import norm
    ax.fill_between(xs, norm.pdf(xs, pop_mean, pop_std),
                    alpha=0.25, color="#3a7abf", label="True population")
    ax.hist(sample, bins=30, density=True, alpha=0.55,
            color="#e07b39", label=f"Biased sample (n={n_total})")
    ax.axvline(pop_mean, color="#3a7abf", lw=2, ls="--",
               label=f"True mean = {pop_mean} cm")
    ax.axvline(sample_mean, color="#e07b39", lw=2,
               label=f"Sample mean = {sample_mean:.1f} cm")

    ax.set_xlabel("Height (cm)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.set_title(
        f"Height estimation: {bias_pct}% of sample drawn from tall subgroup  "
        f"(bias = {sample_mean - pop_mean:+.1f} cm)"
    )
    fig.tight_layout()
    return fig


# ── Scenario 3: survivorship bias --------------------------------------------
def _survivorship(visibility_years: int):
    """
    Generate company lifetimes ~ Exp(1/10 years).
    'Observed' = only companies still alive at `visibility_years`.
    Compare true vs observed mean lifetime.
    """
    n_companies = 2000
    true_mean_lifetime = 10   # years
    lifetimes = RNG.exponential(true_mean_lifetime, n_companies)

    survivors = lifetimes[lifetimes > visibility_years]
    true_mean = lifetimes.mean()
    obs_mean  = survivors.mean() if len(survivors) > 0 else float("nan")
    survival_rate = len(survivors) / n_companies * 100

    fig = _fig()
    ax = fig.add_subplot(111)

    bins = np.linspace(0, lifetimes.max() + 5, 50)
    bin_width = bins[1] - bins[0]

    # Both histograms normalised to the *total* population count so that
    # the survivor bar heights are fractions of the full distribution and
    # the orange bars sit visibly inside the blue ones.
    ax.hist(lifetimes, bins=bins,
            weights=np.ones(len(lifetimes)) / (len(lifetimes) * bin_width),
            alpha=0.35, color="#3a7abf", label="All companies (ground truth)")
    ax.hist(survivors, bins=bins,
            weights=np.ones(len(survivors)) / (len(lifetimes) * bin_width),
            alpha=0.6, color="#e07b39",
            label=f"Survivors (seen after {visibility_years} yr)")

    ax.axvline(true_mean, color="#3a7abf", lw=2, ls="--",
               label=f"True mean = {true_mean:.1f} yr")
    if not np.isnan(obs_mean):
        ax.axvline(obs_mean, color="#e07b39", lw=2,
                   label=f"Observed mean = {obs_mean:.1f} yr")

    ax.set_xlabel("Company lifetime (years)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.set_title(
        f"Survivorship bias: {survival_rate:.0f}% of companies still alive "
        f"at year {visibility_years}  "
        f"(observed − true = {obs_mean - true_mean:+.1f} yr)"
    )
    fig.tight_layout()
    return fig


# ── main render() called by the dashboard shell -----------------------------
def render():
    st.markdown(
        """
        ## Law of Faulty Generalisation
        Drawing a universal conclusion from a sample that is **too small,
        biased, or unrepresentative** of the full population.
        Choose a scenario and adjust the controls to see how the fallacy arises.
        """
    )

    # ── scenario picker + controls in one row ────────────────────────────────
    col_scene, col_ctrl1, col_ctrl2 = st.columns([2, 2, 2])

    with col_scene:
        scenario = st.selectbox(
            "Scenario",
            [
                "1 – Small sample (coin flip)",
                "2 – Biased sample (height survey)",
                "3 – Survivorship bias (company lifetimes)",
            ],
            key="fg_scenario",
        )

    # ── render chosen scenario ───────────────────────────────────────────────
    if scenario.startswith("1"):
        with col_ctrl1:
            n_flips = st.slider("Total flips", 10, 2000, 200,
                                step=10, key="fg_nflips")
        with col_ctrl2:
            st.markdown("&nbsp;", unsafe_allow_html=True)   # spacer

        fig = _coin_flip(n_flips)
        st.pyplot(fig)
        plt.close(fig)

        # st.info(
        #     "**What to notice:** with only a handful of flips the empirical "
        #     "frequency can sit far from 0.5 — it would be a faulty generalisation "
        #     "to declare the coin biased on that evidence. The frequency converges "
        #     "only as the sample grows (Law of Large Numbers)."
        # )

    elif scenario.startswith("2"):
        with col_ctrl1:
            bias_pct = st.slider("% of sample from 'tall' subgroup",
                                 0, 80, 30, step=5, key="fg_bias")
        with col_ctrl2:
            st.markdown("&nbsp;", unsafe_allow_html=True)

        fig = _biased_sample(bias_pct)
        st.pyplot(fig)
        plt.close(fig)

        # st.info(
        #     "**What to notice:** if we survey a group that over-represents tall "
        #     "people (e.g. a basketball crowd), our sample mean overshoots the "
        #     "population mean. Any generalisation — 'the average person is "
        #     f"{170 + bias_pct // 5:.0f} cm tall' — is faulty."
        # )

    else:
        with col_ctrl1:
            vis_years = st.slider("Observation starts at year",
                                  0, 25, 5, step=1, key="fg_surv")
        with col_ctrl2:
            st.markdown("&nbsp;", unsafe_allow_html=True)

        fig = _survivorship(vis_years)
        st.pyplot(fig)
        plt.close(fig)

        # st.info(
        #     "**What to notice:** if we only observe companies that are *still* "
        #     "alive at year *t*, we never see the short-lived ones. Our estimated "
        #     "mean lifetime is therefore inflated — a classic survivorship bias. "
        #     "The later we start observing, the worse the distortion."
        # )

    # # ── shared take-away box ─────────────────────────────────────────────────
    # st.markdown("---")
    # st.markdown(
    #     """
    #     **Frequentist take-away**

    #     A valid inference requires:

    #     * **Large enough n** – so that sampling variability is small relative to the effect.
    #     * **Random / representative sampling** – the sample must mirror the
    #       population, not just the easily-observed sub-group.
    #     * **No selection on the outcome** – do not restrict the sample to units
    #       that survived, succeeded, or were detected.

    #     Violating any of these conditions can make the *sample frequency* a poor
    #     estimate of the *population frequency* — the core of the faulty-generalisation fallacy.
    #     """
    # )
