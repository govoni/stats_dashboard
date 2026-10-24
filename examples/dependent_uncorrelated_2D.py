"""
examples/dependent_uncorrelated_2D.py
======================================
Dependent but uncorrelated 2D distributions.

Shows scatter plots + marginal histograms for several joint distributions
where X and Y are clearly statistically dependent, yet their Pearson
linear correlation coefficient is zero (or vanishingly small).

The key teaching point: ρ = 0 does NOT imply independence.

Layout: controls on the left column, figure on the right column.
The figure is kept compact (height ≤ screen height when projected).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st
from scipy import stats


# ── distribution generators ─────────────────────────────────────────────────

def gen_ring(n, rng):
    r = np.sqrt(rng.uniform(0.7**2, 1.0**2, n))   # uniform in area
    theta = rng.uniform(0, 2 * np.pi, n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, (
        r"Annulus:  $X = r\cos\theta,\; Y = r\sin\theta$",
        r"$r$ uniform on $[R_1,R_2]$, $\theta$ uniform on $[0,2\pi)$",
        r"By symmetry $\langle XY\rangle = 0$, yet knowing $X$ "
        r"constrains $Y$ tightly (it must lie near a circle arc).",
    )


def gen_parabola(n, rng):
    x = rng.uniform(-2, 2, n)
    noise = rng.normal(0, 0.3, n)
    y = x**2 + noise
    return x, y, (
        r"Parabola:  $Y = X^2 + \varepsilon$",
        r"$X \sim \mathrm{Uniform}(-2,2)$,  $\varepsilon \sim \mathcal{N}(0,0.09)$",
        r"$\mathrm{Cov}(X,X^2) = E[X^3] - E[X]\,E[X^2] = 0$ by symmetry, "
        r"so $\rho = 0$ exactly — yet $Y$ is almost fully determined by $X$.",
    )


def gen_X_sinX(n, rng):
    x = rng.uniform(-np.pi, np.pi, n)
    noise = rng.normal(0, 0.15, n)
    y = np.sin(x) + noise
    return x, y, (
        r"Sine:  $Y = \sin(X) + \varepsilon$",
        r"$X \sim \mathrm{Uniform}(-\pi,\pi)$,  $\varepsilon \sim \mathcal{N}(0,0.02)$",
        r"Here $\rho \approx 0.47 \ne 0$! Nonlinear dependence can still "
        r"produce nonzero $\rho$. The message: $\rho = 0$ does not "
        r"mean independence, but $\rho \ne 0$ does not mean linear either.",
    )


def gen_sign_product(n, rng):
    x = rng.standard_normal(n)
    z = rng.standard_normal(n)
    y = np.sign(x) * z
    return x, y, (
        r"Sign product:  $Y = \mathrm{sign}(X)\cdot Z$",
        r"$X, Z \stackrel{\mathrm{iid}}{\sim} \mathcal{N}(0,1)$",
        r"$E[XY] = E[|X|]\,E[Z] = 0$, so $\rho = 0$. Yet the conditional "
        r"distribution of $Y$ depends on $X$: for $X>0$, $Y\sim\mathcal{N}(0,1)$; "
        r"for $X<0$, $Y\sim-\mathcal{N}(0,1)$.",
    )


def gen_cross(n, rng):
    half = n // 2
    x1 = rng.uniform(-2, 2, half)
    y1 = rng.normal(0, 0.15, half)
    x2 = rng.normal(0, 0.15, n - half)
    y2 = rng.uniform(-2, 2, n - half)
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    return x, y, (
        r"Cross mixture:  two perpendicular linear clusters",
        r"Horizontal arm: $Y|X \sim \mathcal{N}(0,0.02)$; "
        r"Vertical arm: $X|Y \sim \mathcal{N}(0,0.02)$",
        r"Each marginal is symmetric → $\rho = 0$. "
        r"Yet the 2D scatter reveals a cross that no 1D histogram shows.",
    )


DISTRIBUTIONS = {
    "Annulus  (ring of points)":     gen_ring,
    "Parabola  Y = X² + noise":      gen_parabola,
    "Sine  Y = sin(X) + noise":      gen_X_sinX,
    # "Sign product  Y = sign(X)·Z":   gen_sign_product,
    "Cross mixture  (two arms)":     gen_cross,
}


# ── main render function ─────────────────────────────────────────────────────

def render():

    st.title("Dependence and correlation")

    col_ctrl, col_plot = st.columns([1, 2])

    # ── LEFT COLUMN: all controls + explanation ──────────────────────────────
    with col_ctrl:
        dist_name = st.selectbox(
            "Distribution",
            list(DISTRIBUTIONS.keys()),
            index=0,
        )

        n_points = st.slider(
            "Sample size  n",
            min_value=200,
            max_value=5000,
            value=1000,
            step=100,
        )

        seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=9999,
            value=42,
            step=1,
        )

        # generate data here so we can show ρ in the left column too
        rng = np.random.default_rng(int(seed))
        gen_fn = DISTRIBUTIONS[dist_name]
        x, y, (subtitle, params_str, explanation) = gen_fn(n_points, rng)
        rho, pval = stats.pearsonr(x, y)

        st.markdown("---")
        st.markdown(f"**Formula:** {subtitle}")
        # st.markdown(f"**Parameters:** {params_str}")
        # st.markdown(f"**Pearson ρ = {rho:+.3f}**  (p = {pval:.1e})")
        # st.markdown("---")
        # st.markdown(f"*{explanation}*")
        # st.markdown("---")
        # st.caption(
        #     "Key takeaway: ρ measures only *linear* association. "
        #     "Two variables can be strongly dependent yet have ρ = 0. "
        #     "Always inspect the scatter plot."
        # )

    # ── RIGHT COLUMN: figure ─────────────────────────────────────────────────
    with col_plot:
        # compact figure — fits a projected screen without scrolling
        fig = plt.figure(figsize=(5.5, 4.5))
        gs = gridspec.GridSpec(
            2, 2,
            width_ratios=[4, 1],
            height_ratios=[1, 4],
            hspace=0.04,
            wspace=0.04,
        )

        ax_main  = fig.add_subplot(gs[1, 0])
        ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

        # scatter
        ax_main.scatter(x, y, s=3, alpha=0.30, color="steelblue", rasterized=True)
        ax_main.set_xlabel("X", fontsize=10)
        ax_main.set_ylabel("Y", fontsize=10)
        ax_main.tick_params(labelsize=8)

        # ρ annotation on plot
        rho_color = "crimson" if abs(rho) < 0.10 else "darkorange"
        ax_main.text(
            0.03, 0.97,
            rf"ρ = {rho:+.3f}",
            transform=ax_main.transAxes,
            va="top", ha="left",
            fontsize=9,
            color=rho_color,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=rho_color, alpha=0.9),
        )

        # marginal histograms
        bins = min(50, int(np.sqrt(n_points) * 1.5))
        ax_top.hist(x, bins=bins, color="steelblue", alpha=0.7, density=True)
        ax_right.hist(
            y, bins=bins, color="steelblue", alpha=0.7,
            density=True, orientation="horizontal",
        )

        # Gaussian overlays on marginals
        for ax_m, data, horiz in [(ax_top, x, True), (ax_right, y, False)]:
            mu, sig = data.mean(), data.std()
            t = np.linspace(data.min(), data.max(), 300)
            pdf = stats.norm.pdf(t, mu, sig)
            if horiz:
                ax_m.plot(t, pdf, "k--", lw=1.0)
            else:
                ax_m.plot(pdf, t, "k--", lw=1.0)

        # cosmetics
        plt.setp(ax_top.get_xticklabels(), visible=False)
        plt.setp(ax_right.get_yticklabels(), visible=False)
        ax_top.set_ylabel("p(x)", fontsize=8)
        ax_right.set_xlabel("p(y)", fontsize=8)
        ax_top.tick_params(labelsize=7)
        ax_right.tick_params(labelsize=7)

        fig.suptitle(subtitle, fontsize=10, y=1.01)
        fig.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.10)

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
