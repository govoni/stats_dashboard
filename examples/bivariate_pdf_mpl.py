"""
bivariate_pdf.py  —  stats_dashboard example
=============================================
Independent case  : standard bivariate Gaussian (marginals are Gaussian)
Correlated case   : 3-component Gaussian mixture (marginals are NOT Gaussian,
                    conditionals can be bimodal — pedagogically richer)

Conditional slices follow the sliders AND, once you click on the 3-D surface,
they snap to the clicked (x, y) position.
"""

import numpy as np
from scipy.stats import multivariate_normal
import plotly.graph_objects as go
import streamlit as st


# ── Mixture definition ────────────────────────────────────────────────────────
# Three bivariate-Gaussian components whose marginals are clearly non-Gaussian:
#   • f(X) is bimodal  (peaks near x ≈ -1.3 and x ≈ +1.4)
#   • f(Y) is skewed / asymmetric  (single peak but heavy right tail)
#   • conditional f(y|X=0) is bimodal  (passes through the "saddle")
MIXTURE = [
    dict(w=0.50, mu=[-1.4,  1.0], cov=[[0.7, 0.5], [0.5, 0.6]]),
    dict(w=0.35, mu=[ 1.5, -1.2], cov=[[0.6,-0.4], [-0.4,0.8]]),
    dict(w=0.15, mu=[ 0.0,  0.0], cov=[[0.3, 0.0], [0.0, 0.3]]),
]

WALL = 3.5    # axis half-extent
N    = 120    # grid resolution


def _gauss2d(X, Y, mu, cov):
    """Bivariate Gaussian PDF evaluated on a meshgrid."""
    pos = np.dstack([X, Y])
    return multivariate_normal(mu, cov).pdf(pos)


def _build_joint(X, Y, rho):
    """Return joint PDF Z on the meshgrid for the chosen case."""
    if rho is None:
        # Independent: standard bivariate Gaussian, rho=0
        z_score = X**2 + Y**2
        return (1 / (2 * np.pi)) * np.exp(-0.5 * z_score)
    else:
        # Correlated: Gaussian mixture (rho slider is informational / unused
        # in the formula — the correlation is baked into the mixture)
        return sum(c['w'] * _gauss2d(X, Y, c['mu'], c['cov']) for c in MIXTURE)


def _conditional_y_given_x(x0, y, X, Y, rho):
    """
    Numerically exact conditional  f(y | X=x0):
    take the column of the joint at x≈x0 and normalise.
    Returns (z_slice, z_normalised).
    """
    x_arr = X[0, :]
    col   = int(np.abs(x_arr - x0).argmin())
    Z     = _build_joint(X, Y, rho)
    slice_raw = Z[:, col]                          # shape (N,)
    norm      = np.trapezoid(slice_raw, y)
    return slice_raw, slice_raw / max(norm, 1e-30)


def _conditional_x_given_y(y0, x, X, Y, rho):
    """
    Numerically exact conditional  f(x | Y=y0):
    take the row of the joint at y≈y0 and normalise.
    Returns (z_slice, z_normalised).
    """
    y_arr = Y[:, 0]
    row   = int(np.abs(y_arr - y0).argmin())
    Z     = _build_joint(X, Y, rho)
    slice_raw = Z[row, :]                          # shape (N,)
    norm      = np.trapezoid(slice_raw, x)
    return slice_raw, slice_raw / max(norm, 1e-30)


def _curtain(fixed_val, axis, coords, heights):
    """
    Return (px, py, pz) for a curtain of vertical drop-lines from
    `heights` down to 0, drawn as a single Scatter3d trace with NaN breaks.
    axis='x' → fixed x, axis='y' → fixed y.
    """
    n = len(coords)
    px = np.empty(n * 3); py = np.empty(n * 3); pz = np.empty(n * 3)
    if axis == 'x':
        px[:] = fixed_val
        py[0::3] = coords; py[1::3] = coords; py[2::3] = np.nan
    else:
        py[:] = fixed_val
        px[0::3] = coords; px[1::3] = coords; px[2::3] = np.nan
    pz[0::3] = 0; pz[1::3] = heights; pz[2::3] = np.nan
    return px, py, pz


# ─────────────────────────────────────────────────────────────────────────────
def render():
    st.header("2D Probability Density Functions")

    col1, col2 = st.columns([1,2])

    with col1:

    # ── Controls ──────────────────────────────────────────────────────────────
        case = st.radio(
            "Distribution",
            ["Independent Gaussian (ρ = 0)", "Gaussian Mixture (non-Gaussian marginals)"],
            help=(
                "The mixture has three correlated bivariate-Gaussian components. "
                "Its marginals are NOT Gaussian (f(X) is bimodal; f(Y) is skewed). "
                "The conditional f(y|X) can itself be bimodal."
            ),
        )
        is_mixture = "Mixture" in case
        rho = MIXTURE if is_mixture else None   # sentinel; None → Gaussian

        # st.markdown(
        #     "**Drag the sliders** to position the conditional slices, "
        #     "or **click anywhere on the 3-D surface** to snap them to that point."
        # )
        x0_slider = st.slider("Slice  X = x₀  (crimson)", -2.5, 2.5, 0.0, 0.05)
        y0_slider = st.slider("Slice  Y = y₀  (blue)",    -2.5, 2.5, 0.0, 0.05)

    with col2:

        # ── Grid ──────────────────────────────────────────────────────────────────
        x = np.linspace(-WALL, WALL, N)
        y = np.linspace(-WALL, WALL, N)
        X, Y = np.meshgrid(x, y)          # Z shape will be (N_y, N_x)

        # ── Joint PDF ─────────────────────────────────────────────────────────────
        Z     = _build_joint(X, Y, rho)
        Z_MAX = float(Z.max())

        # ── Read previous click from session state ─────────────────────────────────
        if "bpdf_click" not in st.session_state:
            st.session_state["bpdf_click"] = None

        click = st.session_state["bpdf_click"]
        if click is not None:
            x0 = float(np.clip(click["x"], -WALL, WALL))
            y0 = float(np.clip(click["y"], -WALL, WALL))
        else:
            x0, y0 = x0_slider, y0_slider

        # ── Conditional slices (unnormalised = value ON the surface) ──────────────
        z_x0_raw, z_x0_norm = _conditional_y_given_x(x0, y, X, Y, rho)
        z_y0_raw, z_y0_norm = _conditional_x_given_y(y0, x, X, Y, rho)

        # ── Marginals by numerical integration ────────────────────────────────────
        f_X_raw = np.trapezoid(Z, y, axis=0)   # integrate over y → shape (N_x,)
        f_Y_raw = np.trapezoid(Z, x, axis=1)   # integrate over x → shape (N_y,)

        # Scale marginals so their peak sits at Z_MAX (keeps z-axis uncluttered)
        marg_scale = Z_MAX / max(f_X_raw.max(), f_Y_raw.max())
        f_X = f_X_raw * marg_scale
        f_Y = f_Y_raw * marg_scale

        # Scale normalised conditionals to the same height as the raw slice for display
        # (so the shape is the true conditional PDF, but the height matches the surface)
        cond_x_scale = z_x0_raw.max() / max(z_x0_norm.max(), 1e-30)
        cond_y_scale = z_y0_raw.max() / max(z_y0_norm.max(), 1e-30)
        z_x0_disp = z_x0_norm * cond_x_scale
        z_y0_disp = z_y0_norm * cond_y_scale

        # ── Build figure ──────────────────────────────────────────────────────────
        fig = go.Figure()

        # 1. Joint PDF surface
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>f(x,y): %{z:.4f}<extra></extra>",
            colorscale="Viridis",
            opacity=0.55,
            showscale=False,
            name="Joint PDF f(x,y)",
            contours=dict(
                x=dict(show=False, highlight=False),
                y=dict(show=False, highlight=False),
                z=dict(show=False, highlight=False),
            ),
        ))

        # 2a. Conditional  f(y | X=x₀)  — crimson, ON the surface
        fig.add_trace(go.Scatter3d(
            x=np.full(N, x0),
            y=y,
            z=z_x0_raw,
            mode="lines",
            line=dict(color="crimson", width=10),
            name=f"f(y | X={x0:.2f})  [on surface]",
        ))

        # 2b. Conditional  f(x | Y=y₀)  — royalblue, ON the surface
        fig.add_trace(go.Scatter3d(
            x=x,
            y=np.full(N, y0),
            z=z_y0_raw,
            mode="lines",
            line=dict(color="royalblue", width=10),
            name=f"f(x | Y={y0:.2f})  [on surface]",
        ))

        # 2c. Normalised conditional  f(y|X=x₀)  on the X = -WALL back wall
        #     rescaled so its peak matches the surface value (shape = true cond. PDF)
        fig.add_trace(go.Scatter3d(
            x=np.full(N, -WALL),
            y=y,
            z=z_x0_disp,
            mode="lines",
            line=dict(color="crimson", width=5, dash="dot"),
            name=f"f(y | X={x0:.2f})  [normalised, back wall]",
        ))

        # 2d. Normalised conditional  f(x|Y=y₀)  on the Y = +WALL back wall
        fig.add_trace(go.Scatter3d(
            x=x,
            y=np.full(N, WALL),
            z=z_y0_disp,
            mode="lines",
            line=dict(color="royalblue", width=5, dash="dot"),
            name=f"f(x | Y={y0:.2f})  [normalised, back wall]",
        ))

        # 3a. Marginal  f(X)  on the Y = +WALL back wall  — lime green
        fig.add_trace(go.Scatter3d(
            x=x, y=np.full(N, WALL), z=f_X,
            mode="lines",
            line=dict(color="limegreen", width=7),
            name="Marginal f(X)  [back wall]",
        ))
        cx, cy, cz = _curtain(WALL, 'y', x, f_X)
        fig.add_trace(go.Scatter3d(
            x=cx, y=cy, z=cz,
            mode="lines", line=dict(color="limegreen", width=1),
            showlegend=False, hoverinfo="skip",
        ))

        # 3b. Marginal  f(Y)  on the X = -WALL back wall  — orange
        fig.add_trace(go.Scatter3d(
            x=np.full(N, -WALL), y=y, z=f_Y,
            mode="lines",
            line=dict(color="darkorange", width=7),
            name="Marginal f(Y)  [back wall]",
        ))
        cx2, cy2, cz2 = _curtain(-WALL, 'x', y, f_Y)
        fig.add_trace(go.Scatter3d(
            x=cx2, y=cy2, z=cz2,
            mode="lines", line=dict(color="darkorange", width=1),
            showlegend=False, hoverinfo="skip",
        ))

        # 4. Crosshair marker at (x0, y0, 0) so the slice position is obvious
        fig.add_trace(go.Scatter3d(
            x=[x0], y=[y0], z=[0],
            mode="markers",
            marker=dict(size=6, color="white", symbol="cross"),
            showlegend=False, hoverinfo="skip",
        ))

        # ── Layout ────────────────────────────────────────────────────────────────
        fig.update_layout(
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Probability Density",
                aspectratio=dict(x=1, y=1, z=0.65),
                camera=dict(eye=dict(x=1.7, y=-1.7, z=1.2)),
                xaxis=dict(range=[-WALL, WALL]),
                yaxis=dict(range=[-WALL, WALL]),
                zaxis=dict(range=[0, Z_MAX * 1.15]),
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            legend=dict(x=0.0, y=0.95, font=dict(size=11)),
            autosize=True,
        )

        # ── Render with click capture ─────────────────────────────────────────────
        event = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key="bpdf_chart",
        )

        # Parse click: Plotly returns selected points from ALL traces;
        # we only care about hits on trace 0 (the Surface).
        if event and hasattr(event, "selection") and event.selection:
            pts = event.selection.get("points", [])
            surface_hits = [p for p in pts if p.get("curve_number", -1) == 0]
            if surface_hits:
                p = surface_hits[0]
                st.session_state["bpdf_click"] = {"x": p["x"], "y": p["y"]}
                st.rerun()






        # # ── Annotation ────────────────────────────────────────────────────────────
        # if is_mixture:
        #     st.info(
        #         "**Mixture structure** — three bivariate-Gaussian components:  \n"
        #         "① w=0.50, μ=(−1.4, +1.0), positive correlation  \n"
        #         "② w=0.35, μ=(+1.5, −1.2), negative correlation  \n"
        #         "③ w=0.15, μ=(0, 0), spherical (bridge)  \n\n"
        #         "**f(X)** is bimodal (peaks near x≈−1.4 and x≈+1.5).  \n"
        #         "**f(Y)** is asymmetric / skewed.  \n"
        #         "**f(y|X≈0)** is bimodal — the slice crosses the saddle between the two main lobes."
        #     )

        # st.markdown(
        #     f"**Slice position:** x₀ = {x0:.2f},  y₀ = {y0:.2f}  "
        #     f"{'*(from click)*' if click else '*(from sliders)*'}  \n"
        #     "— **Crimson solid** = f(y|X=x₀) on the surface  "
        #     "· **Crimson dotted** = same, normalised, on the X=−3.5 wall  \n"
        #     "— **Blue solid** = f(x|Y=y₀) on the surface  "
        #     "· **Blue dotted** = same, normalised, on the Y=+3.5 wall  \n"
        #     "— **Green** = marginal f(X) on Y=+3.5 wall  "
        #     "· **Orange** = marginal f(Y) on X=−3.5 wall"
        # )

    if click:
        if st.button("Reset to sliders"):
            st.session_state["bpdf_click"] = None
            st.rerun()


if __name__ == "__main__":
    render()
