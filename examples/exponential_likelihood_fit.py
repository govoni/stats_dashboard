"""
Exponential likelihood fit — full chain
========================================
Demonstrates the complete workflow:
  1. Generate exponentially-distributed pseudo-data.
  2. Fit with iminuit (unbinned maximum-likelihood).
  3. Estimate uncertainty via the Hessian method.
  4. Estimate uncertainty via the graphical method (profile scan).
  5. Plot data histogram + fitted pdf (dual y-axes).
  6. Plot the log-likelihood profile with both uncertainty estimates.

Drop this file into examples/ and register it in lectures.py:
    import examples.exponential_likelihood_fit as exp_fit
    ...
    {"title": "Exponential likelihood fit", "render": exp_fit.render}
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from iminuit import Minuit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_data(lam: float, n: int, seed: int) -> np.ndarray:
    """Draw n samples from Exp(lam), i.e. mean = 1/lam."""
    rng = np.random.default_rng(seed)
    return rng.exponential(scale=1.0 / lam, size=n)


def _fit(data: np.ndarray):
    """
    Minimise the negative log-likelihood for Exp(lam).

    pdf: f(x; lam) = lam * exp(-lam * x),  x > 0
    NLL: -sum_i [ log(lam) - lam * x_i ]

    Returns the fitted Minuit object.
    """
    def nll(lam):
        if lam <= 0:
            return np.inf
        return -(np.log(lam) * len(data) - lam * data.sum())

    m = Minuit(nll, lam=1.0 / data.mean())
    m.limits["lam"] = (1e-6, None)
    m.errordef = 0.5          # likelihood convention: ±1σ at ΔNLL = 0.5
    m.migrad()
    m.hesse()
    return m


def _profile_scan(nll_func, lam_hat: float, n_points: int = 600):
    """Return (lam_scan, delta_nll_scan) arrays."""
    lam_scan = np.linspace(max(1e-6, lam_hat * 0.55), lam_hat * 1.45, n_points)
    nll_vals = np.array([nll_func(l) for l in lam_scan])
    return lam_scan, nll_vals - nll_vals.min()


def _graphical_crossings(lam_scan, dnll, lam_hat, target=0.5):
    """Find lambda values where delta-NLL = target on each side of the minimum."""
    idx_min = np.searchsorted(lam_scan, lam_hat)
    idx_min = np.clip(idx_min, 1, len(lam_scan) - 2)

    # Left side: scan is ascending in lam but dnll is descending → reverse
    left_lam = np.interp(
        target,
        dnll[:idx_min][::-1],
        lam_scan[:idx_min][::-1],
    )
    right_lam = np.interp(
        target,
        dnll[idx_min:],
        lam_scan[idx_min:],
    )
    return left_lam, right_lam


# ---------------------------------------------------------------------------
# Main render function (called by app.py)
# ---------------------------------------------------------------------------

def render():
    with st.container():
        st.title ("Maximum Likelihood fit")

    # ---- Controls ---------------------------------------------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        true_lam = st.slider(
            "True λ (rate)",
            min_value=0.5, max_value=5.0, value=2.0, step=0.1,
            help="Parameter of the exponential distribution used to generate data.",
        )
    with col2:
        n_events = st.slider(
            "Number of events",
            min_value=50, max_value=2000, value=500, step=50,
        )
    with col3:
        seed = st.slider("Random seed", min_value=0, max_value=99, value=42)

    # ---- Fit --------------------------------------------------------------
    data = _generate_data(true_lam, n_events, seed)
    m    = _fit(data)

    lam_hat   = m.values["lam"]
    err_hesse = m.errors["lam"]

    # Build the NLL function with the current data captured
    def nll_func(lam):
        if lam <= 0:
            return np.inf
        return -(np.log(lam) * len(data) - lam * data.sum())

    lam_scan, dnll = _profile_scan(nll_func, lam_hat)
    left_lam, right_lam = _graphical_crossings(lam_scan, dnll, lam_hat)

    err_graph_lo = lam_hat - left_lam
    err_graph_hi = right_lam - lam_hat

    # ---- Summary metrics --------------------------------------------------
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("True λ",          f"{true_lam:.3f}")
    mc2.metric("Fitted λ̂",        f"{lam_hat:.3f}")
    mc3.metric("σ (Hessian)",      f"{err_hesse:.4f}")
    mc4.metric("σ (graphical)",    f"+{err_graph_hi:.4f} / −{err_graph_lo:.4f}")

    # ---- Figures ----------------------------------------------------------
    fig, (ax_data, ax_nll) = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.patch.set_facecolor("white")

    # ── Left panel: histogram + pdf (dual y-axis) ──────────────────────────
    n_bins  = max(10, int(np.sqrt(n_events)))
    x_max   = np.percentile(data, 99)            # clip tail for readability
    bins    = np.linspace(0, x_max, n_bins + 1)
    bin_w   = bins[1] - bins[0]

    # Right axis: histogram (counts)
    ax_counts = ax_data.twinx()
    counts, _, patches = ax_counts.hist(
        data, bins=bins, color="steelblue", alpha=0.55, label="Data (counts)"
    )
    ax_counts.set_ylabel("Counts per bin", color="steelblue", fontsize=11)
    ax_counts.tick_params(axis="y", colors="steelblue")
    ax_counts.spines["right"].set_color("steelblue")
    ax_counts.set_ylim(bottom=0)

    # Left axis: pdf  f(x) = lam * exp(-lam * x)
    x_pdf = np.linspace(0, x_max, 500)
    y_pdf = lam_hat * np.exp(-lam_hat * x_pdf)
    ax_data.plot(x_pdf, y_pdf, color="crimson", lw=2.2,
                 label=rf"Fitted pdf  $\hat{{\lambda}}={lam_hat:.3f}$")
    ax_data.set_ylabel(r"pdf  $f(x;\hat{\lambda})$", color="crimson", fontsize=11)
    ax_data.tick_params(axis="y", colors="crimson")
    ax_data.spines["left"].set_color("crimson")
    ax_data.set_ylim(bottom=0)
    ax_data.set_xlabel(r"$x$", fontsize=12)
    ax_data.set_title(
        rf"Data ($n={n_events}$) and fitted exponential pdf", fontsize=12
    )

    # Combined legend
    lines_l, labels_l = ax_data.get_legend_handles_labels()
    lines_r, labels_r = ax_counts.get_legend_handles_labels()
    ax_data.legend(lines_l + lines_r, labels_l + labels_r,
                   loc="upper right", fontsize=9)

    ax_data.set_zorder(ax_counts.get_zorder() + 1)
    ax_data.patch.set_visible(False)

    # ── Right panel: likelihood profile ────────────────────────────────────
    DELTA_TARGET = 0.5

    ax_nll.axvline(true_lam,   color="crimson",  lw=1.5, ls="-",
                   label=r"true \lambda")

    ax_nll.plot(lam_scan, dnll, color="navy", lw=2, label=r"$\Delta\,\mathrm{NLL}(\lambda)$")
    ax_nll.axhline(DELTA_TARGET, color="grey", ls="--", lw=1.2,
                   label=rf"$\Delta\mathrm{{NLL}} = {DELTA_TARGET}$ (±1σ level)")

    # Graphical method: vertical bars + horizontal span
    ax_nll.axvline(lam_hat,   color="green",  lw=1.5, ls="-")
    ax_nll.axvline(left_lam,  color="darkorange", lw=1.5, ls="--",
                   label=rf"Graphical  $-\sigma={err_graph_lo:.4f}$")
    ax_nll.axvline(right_lam, color="darkorange", lw=1.5, ls="-.",
                   label=rf"Graphical  $+\sigma={err_graph_hi:.4f}$")
    ax_nll.fill_betweenx(
        [0, DELTA_TARGET], left_lam, right_lam,
        color="orange", alpha=0.15, label="Graphical 1σ band"
    )

    # Hessian method: error bar at minimum
    ax_nll.errorbar(
        lam_hat, 0,
        xerr=err_hesse,
        fmt="D", color="purple", ms=7, capsize=5, capthick=1.8, lw=1.8,
        zorder=5,
        label=rf"Hessian  $\sigma={err_hesse:.4f}$",
    )

    ax_nll.set_xlabel(r"$\lambda$", fontsize=12)
    ax_nll.set_ylabel(r"$\Delta\,\mathrm{NLL} = \mathrm{NLL}(\lambda) - \mathrm{NLL}_{\min}$",
                      fontsize=11)
    ax_nll.set_title("Log-likelihood profile", fontsize=12)
    ax_nll.set_ylim(-0.05, min(3.5, dnll.max() * 1.05))
    ax_nll.legend(fontsize=8.5, loc="upper right")
    ax_nll.grid(True, alpha=0.25)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ---- Expandable technical notes ---------------------------------------
    with st.expander("Technical notes"):
        st.markdown(r"""
**Model**  
$X \sim \mathrm{Exp}(\lambda)$ with pdf $f(x;\lambda)=\lambda\,e^{-\lambda x}$, $x>0$.  
The MLE has the closed-form solution $\hat\lambda = n/\sum x_i = 1/\bar x$.

**Negative log-likelihood (NLL)**  
$$
\mathrm{NLL}(\lambda) = -\sum_{i=1}^n \bigl[\ln\lambda - \lambda x_i\bigr]
= -n\ln\lambda + \lambda\sum_i x_i
$$

**Hessian method**  
The Fisher information for $\lambda$ is $I(\lambda)=n/\lambda^2$, giving  
$\sigma_{\hat\lambda}^{(H)} = \hat\lambda/\sqrt{n}$.  
`iminuit` evaluates this numerically via the Hessian of the NLL at the minimum.

**Graphical method**  
$\sigma$ is read from the profile $\Delta\mathrm{NLL}(\lambda)=\mathrm{NLL}(\lambda)-\mathrm{NLL}_{\min}$:
the ±1σ interval is bounded by where $\Delta\mathrm{NLL}=0.5$
(the factor ½ comes from the chi-squared relation $\Delta(-2\ln\mathcal{L})=1$).  
For asymmetric profiles the two crossings give unequal $\sigma_-$ and $\sigma_+$.

**Dual y-axis**  
The histogram shows raw counts (right axis, blue); the fitted pdf
$f(x;\hat\lambda)$ is plotted on the left axis (red) — the two are on different
scales but share the same $x$-axis, making the shape comparison straightforward.
        """)
