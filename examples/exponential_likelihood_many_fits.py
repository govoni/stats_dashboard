"""
Exponential likelihood fit — toy-ensemble explorer
====================================================
Full workflow:
  • Generate exponentially-distributed pseudo-data (one toy or 100 at a time).
  • Fit each toy with iminuit (unbinned maximum-likelihood).
  • Estimate the uncertainty via the Hessian method and the graphical method.
  • Top    — overlay of every NLL profile collected so far, with the latest
             one highlighted and both uncertainty estimates annotated on it.
  • Bottom — growing distribution of the MLE λ̂ across all toys,
             sharing the same x-axis (λ) as the profile panel above.

Integration into the dashboard
-------------------------------
Copy this file to  examples/exponential_likelihood_fit.py
In lectures.py add at the top:
    import examples.exponential_likelihood_fit as exp_fit
Then replace the relevant placeholder_for(...) entry with:
    {"title": "Exponential likelihood fit", "render": exp_fit.render}
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from iminuit import Minuit
from scipy.stats import norm as sp_norm

# ── session-state keys ────────────────────────────────────────────────────────
_KEY_RECORDS = "exp_lhfit_records"   # list[dict] — one entry per toy
_KEY_SEED    = "exp_lhfit_seed"      # int — global RNG counter (ensures unique toys)


# ── statistical helpers ───────────────────────────────────────────────────────

def _make_nll(data):
    """Return a NLL function for Exp(lam) closed over pre-computed sufficient stats."""
    n, s = len(data), float(data.sum())

    def nll(lam):                          # iminuit reads parameter names via inspection
        return np.inf if lam <= 0 else -(n * np.log(lam) - lam * s)

    return nll


def _fit(data):
    """Fit Exp(lam) by maximum likelihood; return the Minuit object."""
    m = Minuit(_make_nll(data), lam=1.0 / data.mean())
    m.limits["lam"] = (1e-6, None)
    m.errordef = 0.5          # likelihood convention: ±1σ ↔ ΔNLL = 0.5
    m.migrad()
    m.hesse()
    return m


def _profile_scan(lam_hat, n, s, n_pts=500):
    """Return (lam_array, delta_nll_array) for a dense scan around lam_hat."""
    lams    = np.linspace(max(1e-6, lam_hat * 0.55), lam_hat * 1.45, n_pts)
    nll_arr = np.where(lams > 0, -(n * np.log(lams) - lams * s), np.inf)
    return lams, nll_arr - nll_arr.min()


def _graphical_crossings(lams, dnll, lam_hat, level=0.5):
    """Interpolate λ values where ΔNLL = level on each side of the minimum."""
    idx   = int(np.clip(np.searchsorted(lams, lam_hat), 1, len(lams) - 2))
    left  = np.interp(level, dnll[:idx][::-1], lams[:idx][::-1])
    right = np.interp(level, dnll[idx:],        lams[idx:])
    return left, right


def _run_toys(true_lam, n_events, n_toys):
    """Generate, fit and store n_toys experiments into session state."""
    records = st.session_state[_KEY_RECORDS]
    for _ in range(n_toys):
        seed   = st.session_state[_KEY_SEED]
        st.session_state[_KEY_SEED] += 1

        data   = np.random.default_rng(seed).exponential(1.0 / true_lam, n_events)
        m      = _fit(data)
        lam_hat = m.values["lam"]
        err_H   = m.errors["lam"]

        n, s    = len(data), float(data.sum())
        lams, dnll = _profile_scan(lam_hat, n, s)
        left, right = _graphical_crossings(lams, dnll, lam_hat)

        records.append(dict(
            lam_hat  = lam_hat,
            err_H    = err_H,
            err_G_lo = lam_hat - left,
            err_G_hi = right   - lam_hat,
            lams     = lams,
            dnll     = dnll,
        ))


# ── figure builder ────────────────────────────────────────────────────────────

def _build_figure(records, true_lam, n_events):
    n_toys = len(records)
    arr    = np.array([r["lam_hat"] for r in records])
    last   = records[-1]

    # Shared x range: union of all profile domains, padded a little.
    lam_lo = min(r["lams"][ 0] for r in records)
    lam_hi = max(r["lams"][-1] for r in records)
    pad    = (lam_hi - lam_lo) * 0.04
    x_min  = max(0.0, lam_lo - pad)
    x_max  = lam_hi + pad

    # Two rows, one column; shared x-axis so λ ticks line up perfectly.
    fig, (ax_nll, ax_h) = plt.subplots(
        2, 1,
        figsize=(10, 5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1], "hspace": 0.08},
    )
    fig.patch.set_facecolor("white")

    # ── top: overlay of ALL NLL profiles, latest one annotated ───────────────
    for r in records[:-1]:
        ax_nll.plot(r["lams"], r["dnll"],
                    color="steelblue", lw=0.8, alpha=0.35)

    # Latest profile highlighted in crimson
    ax_nll.plot(last["lams"], last["dnll"],
                color="crimson", lw=2.0, label=f"Toy #{n_toys}", zorder=5)

    ax_nll.axhline(0.5, ls="--", color="grey", lw=1.3,
                   label=r"$\Delta\mathrm{NLL}=0.5\;(\pm1\sigma)$")
    ax_nll.axvline(true_lam, color="black", lw=1.3, ls=":",
                   label=f"True $\\lambda={true_lam}$")

    # Graphical uncertainty of the latest toy
    lh    = last["lam_hat"]
    left  = lh - last["err_G_lo"]
    right = lh + last["err_G_hi"]
    ax_nll.axvline(lh,    color="green",      lw=1.5, zorder=4)
    ax_nll.axvline(left,  color="darkorange", lw=1.5, ls="--",
                   label=f"Graph  $-\\sigma={last['err_G_lo']:.4f}$")
    ax_nll.axvline(right, color="darkorange", lw=1.5, ls="-.",
                   label=f"Graph  $+\\sigma={last['err_G_hi']:.4f}$")
    ax_nll.fill_betweenx([0, 0.5], left, right,
                         color="orange", alpha=0.15, label="Graphical 1σ band")

    # Hessian uncertainty of the latest toy
    ax_nll.errorbar(lh, 0, xerr=last["err_H"],
                    fmt="D", color="purple", ms=7,
                    capsize=5, capthick=1.8, lw=1.8, zorder=6,
                    label=f"Hessian  $\\sigma={last['err_H']:.4f}$")

    ax_nll.set_ylabel(r"$\Delta\mathrm{NLL}$", fontsize=11)
    ax_nll.set_ylim(-0.08, 3.5)
    ax_nll.set_xlim(x_min, x_max)
    ax_nll.set_title(
        f"NLL profiles — {n_toys} toy{'s' if n_toys > 1 else ''}  "
        f"(latest highlighted, uncertainty from toy #{n_toys})",
        fontsize=11,
    )
    ax_nll.legend(fontsize=8, loc="upper right")
    ax_nll.grid(alpha=0.25)
    # No x-label here: shared axis, label goes on the bottom panel only.
    ax_nll.tick_params(labelbottom=False)

    # ── bottom: estimator distribution (same x-axis = λ) ─────────────────────
    n_bins = 2 * max(6, int(np.sqrt(n_toys)) + 2)

    # Use the shared x range so histogram bins respect the panel boundaries.
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    counts, _, _ = ax_h.hist(
        arr, bins=bin_edges,
        color="steelblue", alpha=0.65, edgecolor="white",
        label=r"$\hat\lambda$ per toy",
    )

    ax_h.axvline(true_lam,   color="crimson", lw=2.2, ls="-",
                 label=f"True $\\lambda = {true_lam}$")
    ax_h.axvline(arr.mean(), color="green",   lw=2.2, ls="--",
                 label=f"Sample mean = {arr.mean():.4f}")

    # Expected Gaussian from asymptotic theory: σ = λ/√n
    expected_sigma = true_lam / np.sqrt(n_events)
    if n_toys >= 5:
        bin_w = bin_edges[1] - bin_edges[0]
        x_g   = np.linspace(x_min, x_max, 400)
        y_g   = sp_norm.pdf(x_g, true_lam, expected_sigma) * n_toys * bin_w
        ax_h.plot(x_g, y_g, color="darkorange", lw=2.0,
                  label=f"Expected Gaussian  $\\sigma={expected_sigma:.4f}$")

    bias      = arr.mean() - true_lam
    avg_err_H = float(np.mean([r["err_H"] for r in records]))
    ax_h.set_xlabel(r"$\lambda$", fontsize=12)
    ax_h.set_ylabel("Toys per bin", fontsize=11)
    # ax_h.set_title(
    #     f"Estimator distribution — "
    #     f"bias = {bias:+.4f},   "
    #     f"RMS = {arr.std():.4f},   "
    #     f"avg Hessian $\\sigma$ = {avg_err_H:.4f},   "
    #     f"expected $\\sigma$ = {expected_sigma:.4f}",
    #     fontsize=9.5,
    # )
    ax_h.legend(fontsize=9)
    ax_h.grid(alpha=0.20)

    return fig


# ── main render function (called by app.py) ───────────────────────────────────

def render():
    with st.container():
        st.title ("Maximum Likelihood Fit toys")

    # ── initialise session state ──────────────────────────────────────────────
    if _KEY_RECORDS not in st.session_state:
        st.session_state[_KEY_RECORDS] = []
    if _KEY_SEED not in st.session_state:
        st.session_state[_KEY_SEED]    = 0


    col_ctrl, col_plots = st.columns([1, 5])

    with col_ctrl:
        true_lam = st.slider(
            "True λ (rate)",
            min_value=0.5, max_value=5.0, value=2.0, step=0.1,
            help="Decay rate of the exponential distribution used to generate data.",
        )
        n_events = st.slider(
            "Events per toy",
            min_value=5, max_value=2000, value=300, step=10,
        )
        run_one = st.button("+ 1 toy", use_container_width=True)
        run_10 = st.button("+ 10 toys", use_container_width=True)
        run_100 = st.button("+ 100 toys", use_container_width=True)
        reset = st.button("Reset", use_container_width=True)
 
    # ── handle button actions ─────────────────────────────────────────────────
    if reset:
        st.session_state[_KEY_RECORDS] = []
        st.session_state[_KEY_SEED]    = 0

    if run_one:
        _run_toys(true_lam, n_events, 1)

    if run_10:
        _run_toys(true_lam, n_events, 10)

    if run_100:
        _run_toys(true_lam, n_events, 100)

    records = st.session_state[_KEY_RECORDS]

    # ── summary metrics ───────────────────────────────────────────────────────
    n_toys = len(records)
    if n_toys > 0:
        arr = np.array([r["lam_hat"] for r in records])
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Toys run",          f"{n_toys}")
        m2.metric("True λ",            f"{true_lam:.2f}")
        m3.metric("Mean λ̂",            f"{arr.mean():.4f}")
        m4.metric("Bias",              f"{arr.mean()-true_lam:+.4f}")
        m5.metric("RMS",               f"{arr.std():.4f}")

    with col_plots:
 
        # ── plot (only when at least one toy exists) ──────────────────────────────
        if n_toys == 0:
            st.info("waiting to start.")
        else:
            fig = _build_figure (records, true_lam, n_events)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # ── technical notes ───────────────────────────────────────────────────────
    with st.expander("Technical notes"):
        st.markdown(r"""
**Model**  
$X \sim \mathrm{Exp}(\lambda)$, pdf $f(x;\lambda)=\lambda e^{-\lambda x}$, $x>0$.  
MLE closed form: $\hat\lambda = n/\sum x_i = 1/\bar x$.

**Negative log-likelihood (NLL)**  
$$\mathrm{NLL}(\lambda)= -n\ln\lambda + \lambda\sum_i x_i$$

**Hessian uncertainty**  
`iminuit` evaluates the curvature at the minimum numerically.  
Asymptotically $\sigma_H = \hat\lambda/\sqrt{n}$.

**Graphical uncertainty**  
The ±1σ interval is where $\Delta\mathrm{NLL}=\mathrm{NLL}(\lambda)-\mathrm{NLL}_{\min}=0.5$.  
Asymmetric profiles give unequal $\sigma_-$ and $\sigma_+$.

**Toy ensemble**  
Each button press generates fresh data (using an incrementing seed so toys never repeat),
fits them independently, and appends the result.  Pressing **Reset** clears all accumulated
toys and restarts the seed counter.  Changing the sliders *does not* automatically reset —
press Reset first if you want a clean comparison.

**Expected Gaussian** (bottom panel)  
Drawn from asymptotic theory: $\hat\lambda\sim\mathcal{N}(\lambda,\,\lambda/\sqrt{n})$.
It appears once ≥ 5 toys are collected.
        """)
