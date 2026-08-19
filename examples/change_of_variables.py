"""
change_of_variables.py

Interactive lecture example: change of variables for continuous random
variables.

Given X with pdf f_X(x) and a monotonic transformation Y = u(X), the pdf of
Y is

    f_Y(y) = f_X(x) / |u'(x)|      evaluated at   x = u^{-1}(y)

which is equivalent to the statement that probability mass is conserved:

    f_X(x) |dx| = f_Y(y) |dy|

This module draws three linked panels:
  - bottom:  f_X(x) vs x
  - center:  the transformation curve y = u(x)
  - right:   f_Y(y) vs y

A slider selects a small interval [x0, x0+dx]. The corresponding interval
[y0, y1] = [u(x0), u(x0+dx)] is located through the curve, and the same
color is used to shade the area under f_X over [x0, x0+dx] and the area
under f_Y over [y0, y1] -- visually and numerically demonstrating that the
two probabilities are equal.

Usage from a multi-page Streamlit dashboard:

    from change_of_variables import render
    render()

The function is self-contained (own widgets, own state keys, own
matplotlib figure), so it can be dropped into a dashboard alongside other
per-topic example files without clashing with their widget keys.
"""

import numpy as np
import streamlit as st
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# numpy >= 2.0 renamed trapz -> trapezoid; keep this working on either.
_trapz = getattr(np, "trapezoid", None) or np.trapz

# All Streamlit widget keys in this module are prefixed with this string so
# the file can be safely combined with other example pages in one app.
_KEY = "covar_"


# --------------------------------------------------------------------------
# Distributions available for X
# --------------------------------------------------------------------------
def _distributions():
    return {
        "Normal": dict(
            build=lambda mu, sigma: stats.norm(loc=mu, scale=sigma),
            params=[("mu", "mean μ", -3.0, 3.0, 0.0, 0.1),
                    ("sigma", "std σ", 0.2, 3.0, 1.0, 0.1)],
        ),
        "Uniform": dict(
            build=lambda a, w: stats.uniform(loc=a, scale=w),
            params=[("a", "lower a", -3.0, 3.0, 0.0, 0.1),
                    ("w", "width w", 0.2, 6.0, 2.0, 0.1)],
        ),
        "Exponential": dict(
            build=lambda scale: stats.expon(scale=scale),
            params=[("scale", "scale 1/λ", 0.2, 4.0, 1.0, 0.1)],
        ),
        "Beta": dict(
            build=lambda a, b: stats.beta(a, b),
            params=[("a", "α", 0.5, 6.0, 2.0, 0.1),
                    ("b", "β", 0.5, 6.0, 2.0, 0.1)],
        ),
    }


# --------------------------------------------------------------------------
# Transformations y = u(x), with analytic derivative and a domain
# constraint (some transforms require x > 0 to stay monotonic).
# --------------------------------------------------------------------------
def _transforms():
    return {
        "Linear:  y = a\u00b7x + b": dict(
            u=lambda x, a, b: a * x + b,
            du=lambda x, a, b: np.full_like(x, a, dtype=float),
            params=[("a", "slope a", -3.0, 3.0, 2.0, 0.1),
                    ("b", "intercept b", -3.0, 3.0, 0.0, 0.1)],
            requires_positive=False,
            formula=r"y = a\,x + b \qquad \left|\frac{du}{dx}\right| = |a|",
        ),
        "Exponential:  y = exp(a\u00b7x)": dict(
            u=lambda x, a: np.exp(a * x),
            du=lambda x, a: a * np.exp(a * x),
            params=[("a", "rate a", -2.0, 2.0, 1.0, 0.1)],
            requires_positive=False,
            formula=r"y = e^{a x} \qquad \left|\frac{du}{dx}\right| = |a|\,e^{ax}",
        ),
        "Power:  y = x\u207f  (x > 0)": dict(
            u=lambda x, n: np.power(x, n),
            du=lambda x, n: n * np.power(x, n - 1.0),
            params=[("n", "exponent n", 0.3, 4.0, 2.0, 0.1)],
            requires_positive=True,
            formula=r"y = x^{n} \qquad \left|\frac{du}{dx}\right| = |n|\,x^{\,n-1}",
        ),
        "Logistic:  y = 1 / (1 + e\u207b\u02e3)": dict(
            u=lambda x: 1.0 / (1.0 + np.exp(-x)),
            du=lambda x: (1.0 / (1.0 + np.exp(-x))) * (1.0 - 1.0 / (1.0 + np.exp(-x))),
            params=[],
            requires_positive=False,
            formula=r"y = \sigma(x) \qquad \left|\frac{du}{dx}\right| = \sigma(x)\big(1-\sigma(x)\big)",
        ),
        "Probability integral transform:  y = F\u2093(x)": dict(
            # u and du are filled in at call time in render(), since this
            # transform is *the cdf of whichever X was picked* rather than a
            # fixed function of x. du/dx = f_X(x) is an exact identity
            # (derivative of the cdf is the pdf), so no derivative formula
            # is stored here.
            u=None,
            du=None,
            params=[],
            requires_positive=False,
            is_cdf=True,
            formula=r"y = F_X(x) \qquad \frac{du}{dx} = f_X(x) \ \Rightarrow\ f_Y(y) = 1,\ y\in(0,1)",
        ),
    }


def render():
    """Render the change-of-variables example. Call once per page load."""

    st.title("Change of variables for continuous pdfs")

    # st.markdown("### Change of variables for continuous pdfs")
    # st.markdown(
    #     "Pick a distribution for $X$ and a monotonic transform "
    #     "$Y = u(X)$. Move the interval on the $x$-axis and watch how it "
    #     "maps onto an interval on the $y$-axis: the shaded areas are "
    #     "equal, because probability cannot be created or destroyed by a "
    #     "change of variables."
    # )

    dists = _distributions()
    transforms = _transforms()
    dist_values = {}
    trans_values = {}

    col1, col2 = st.columns([1, 2])
    with col1:
        dist_name = st.selectbox("Distribution of X", list(dists.keys()), key=_KEY + "dist")
        transform_name = st.selectbox("Transformation", list(transforms.keys()), key=_KEY + "trans")

        dist_spec = dists[dist_name]
        trans_spec = transforms[transform_name]

        st.markdown(f"**{dist_name} parameters**")
        for key, label, lo, hi, default, step in dist_spec["params"]:
            dist_values[key] = st.slider(label, lo, hi, default, step, key=_KEY + "d_" + key)

        st.markdown(f"**{transform_name.split(':')[0]} parameters**")
        if trans_spec["params"]:
            for key, label, lo, hi, default, step in trans_spec["params"]:
                trans_values[key] = st.slider(label, lo, hi, default, step, key=_KEY + "t_" + key)
        else:
            st.caption("No free parameters for this transform.")
        # st.latex(trans_spec["formula"])


        rv = dist_spec["build"](**dist_values)

        is_cdf = bool(trans_spec.get("is_cdf"))
        if is_cdf:
            # y = F_X(x); dy/dx = f_X(x) exactly, by definition of the cdf.
            u_func = lambda x, **_: rv.cdf(x)
            du_func = lambda x, **_: rv.pdf(x)
            # st.info(
            #     "**Probability integral transform.** For *any* continuous X "
            #     "with cdf $F_X$, the variable $Y = F_X(X)$ is Uniform(0,1) — "
            #     "this is exactly what makes it useful for sampling and for "
            #     "goodness-of-fit tests. Try switching the distribution of X "
            #     "above: $f_Y(y)$ stays flat on $(0,1)$ every time."
            # )
        else:
            u_func = trans_spec["u"]
            du_func = trans_spec["du"]

        # --- domain for x --------------------------------------------------
        lo_q, hi_q = rv.ppf(0.001), rv.ppf(0.999)
        if not np.isfinite(lo_q):
            lo_q = rv.ppf(0.01)
        if not np.isfinite(hi_q):
            hi_q = rv.ppf(0.99)
        if trans_spec["requires_positive"]:
            lo_q = max(lo_q, 1e-3)
        if hi_q <= lo_q:
            hi_q = lo_q + 1.0

        x_grid = np.linspace(lo_q, hi_q, 800)
        fx = rv.pdf(x_grid)

        y_grid = u_func(x_grid, **trans_values)
        dudx = np.abs(du_func(x_grid, **trans_values))
        # y_grid = trans_spec["u"](x_grid, **trans_values)
        # dudx = np.abs(trans_spec["du"](x_grid, **trans_values))
        dudx = np.where(dudx < 1e-12, np.nan, dudx)
        fy = fx / dudx

        # --- interval selector on x ------------------------------------------
        span = hi_q - lo_q
        default_x0 = lo_q + 0.35 * span
        default_dx = 0.12 * span
        x0 = st.slider(
            "Interval start  x\u2080",
            float(lo_q), float(hi_q - 0.02 * span), float(default_x0),
            float(span / 200), key=_KEY + "x0",
        )
        dx = st.slider(
            "Interval width  dx",
            float(span / 100), float(span / 3), float(default_dx),
            float(span / 200), key=_KEY + "dx",
        )
        x1 = min(x0 + dx, hi_q)

        mask = (x_grid >= x0) & (x_grid <= x1)
        p_x = _trapz(fx[mask], x_grid[mask]) if mask.sum() > 1 else 0.0

        y_at_x0 = float(u_func(np.array([x0]), **trans_values)[0])
        y_at_x1 = float(u_func(np.array([x1]), **trans_values)[0])
        y_lo, y_hi = sorted([y_at_x0, y_at_x1])
        p_y = _trapz(fy[mask], y_grid[mask]) if mask.sum() > 1 else 0.0
        p_y = abs(p_y)  # trapz sign follows the direction y_grid is traversed

    # --- figure -------------------------------------------------------

    with col2:

        fig = plt.figure(figsize=(7.5, 6))
        gs = fig.add_gridspec(
            2, 2, width_ratios=[4, 1.3], height_ratios=[4, 1.3],
            wspace=0.05, hspace=0.05,
        )
        ax_main = fig.add_subplot(gs[0, 0])
        ax_y = fig.add_subplot(gs[0, 1], sharey=ax_main)
        ax_x = fig.add_subplot(gs[1, 0], sharex=ax_main)

        band_color = "#e0e0e0"
        fill_color = "#d9730d"   # same color on both sides -> equal probability
        curve_color = "#2c6e91"

        # main panel: the transformation curve
        ax_main.plot(x_grid, y_grid, color=curve_color, lw=2)
        ax_main.axvspan(x0, x1, color=band_color, alpha=0.6, zorder=0)
        ax_main.axhspan(y_lo, y_hi, color=band_color, alpha=0.6, zorder=0)

        ax_main.margins(x=0, y=0)  # Removes padding entirely

        ax_main.plot([x0, x0],[ax_main.get_ylim()[0], y_at_x0],
                     color="0.4", lw=0.8, ls="--")
        ax_main.plot([x1, x1],[ax_main.get_ylim()[0], y_at_x1],
                     color="0.4", lw=0.8, ls="--")
        ax_main.plot([x0, ax_main.get_xlim()[1]], [y_at_x0, y_at_x0],
                     color="0.4", lw=0.8, ls="--")
        ax_main.plot([x1, ax_main.get_xlim()[1]], [y_at_x1, y_at_x1],
                     color="0.4", lw=0.8, ls="--")

        ax_main.set_ylabel("y = u(x)")
        ax_main.tick_params(labelbottom=False)
        ax_main.set_title("y = u(x)", fontsize=10, loc="left", color=curve_color)

        # bottom panel: f_X(x), density increasing downward away from ax_main
        ax_x.plot(x_grid, fx, color=curve_color, lw=1.6)
        ax_x.fill_between(x_grid, fx, 0, where=mask, color=fill_color, alpha=0.6)
        ax_x.set_xlabel("x")
        ax_x.set_ylabel(r"$f_X(x)$", fontsize=9)
        ax_x.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3))

        # right panel: f_Y(y), density increasing rightward away from ax_main
        order = np.argsort(y_grid)
        ax_y.plot(fy[order], y_grid[order], color="#a13d1f", lw=1.6)
        y_mask = (y_grid[order] >= y_lo) & (y_grid[order] <= y_hi)
        ax_y.fill_betweenx(y_grid[order], fy[order], 0, where=y_mask, color=fill_color, alpha=0.6)
        ax_y.set_xlabel(r"$f_Y(y)$", fontsize=9)
        ax_y.tick_params(labelleft=False)
        ax_y.xaxis.set_major_locator(mticker.MaxNLocator(nbins=3))

        st.pyplot(fig, clear_figure=True)

    # c1, c2, c3 = st.columns(3)
    # c1.metric("P(x\u2080 < X < x\u2080+dx)", f"{p_x:.4f}")
    # c2.metric("P(y\u2080 < Y < y\u2081)", f"{p_y:.4f}")
    # c3.metric("difference", f"{abs(p_x - p_y):.2e}")

    # st.caption(
    #     "The two probabilities match up to the discretization error of the "
    #     "numerical integral: f_X(x)·|dx| = f_Y(y)·|dy| for corresponding "
    #     "intervals, which is exactly the Jacobian relation "
    #     "f_Y(y) = f_X(x) / |u'(x)|."
    # )


if __name__ == "__main__":
    render()
