import numpy as np
import math
import plotly.graph_objects as go
import streamlit as st


def render():
    st.header("2D Probability Density Function: Independence vs. Dependence")

    case = st.radio(
        "Variable Relationship",
        ["Independent (ρ = 0)", "Correlated (ρ ≠ 0)"],
        help="Select whether X and Y are stochastically independent.",
    )

    rho = (
        0.0
        if "Independent" in case
        else st.slider(
            "Correlation Coefficient (ρ)", -0.9, 0.9, 0.65, step=0.05
        )
    )

    x0 = 0.
    y0 = 0.
    # --- Math & Meshgrid ---
    x = np.linspace(-3.5, 3.5, 100)
    y = np.linspace(-3.5, 3.5, 100)
    X, Y = np.meshgrid(x, y)

    mu_x, mu_y = 0.0, 0.0
    sigma_x, sigma_y = 1.0, 1.0

    # 2D Bivariate Normal PDF
    z_score = (
        ((X - mu_x) / sigma_x) ** 2
        + ((Y - mu_y) / sigma_y) ** 2
        - 2 * rho * ((X - mu_x) / sigma_x) * ((Y - mu_y) / sigma_y)
    )
    norm_factor = 1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho**2))
    Z = norm_factor * np.exp(-z_score / (2 * (1 - rho**2)))

    # Conditional slice at X = x0 across all Y
    z_x0 = norm_factor * np.exp(
        -(
            ((x0 - mu_x) / sigma_x) ** 2
            + ((y - mu_y) / sigma_y) ** 2
            - 2 * rho * ((x0 - mu_x) / sigma_x) * ((y - mu_y) / sigma_y)
        )
        / (2 * (1 - rho**2))
    )

    # Conditional slice at Y = y0 across all X
    z_y0 = norm_factor * np.exp(
        -(
            ((x - mu_x) / sigma_x) ** 2
            + ((y0 - mu_y) / sigma_y) ** 2
            - 2 * rho * ((x - mu_x) / sigma_x) * ((y0 - mu_y) / sigma_y)
        )
        / (2 * (1 - rho**2))
    )

    # Marginal Distributions (Projected on boundary walls)
    f_X = (1 / (np.sqrt(2 * np.pi) * sigma_x)) * np.exp(
        -0.5 * ((x - mu_x) / sigma_x) ** 2
    )
    f_Y = (1 / (np.sqrt(2 * np.pi) * sigma_y)) * np.exp(
        -0.5 * ((y - mu_y) / sigma_y) ** 2
    )

    # 1D Marginal Cumulative Distributions (CDF) for projection
    cdf_X = 0.5 * (1 + np.vectorize(math.erf)(x / np.sqrt(2)))
    cdf_Y = 0.5 * (1 + np.vectorize(math.erf)(y / np.sqrt(2)))

    # --- Plotly 3D Figure ---
    fig = go.Figure()

    # 1. Main 3D Surface
    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            hoverinfo="none",
            colorscale="Viridis",
            opacity=0.6,
            showscale=False,
            name="Joint PDF f(x,y)",
            contours={
                # 1. X Conditional Distribution (e.g., Red)
                "x": {
                    "show": False,  # Set to True if you want static grid lines
                    "highlight": True,  # Shows line on hover
                    "highlightcolor": "crimson",  # Color 1: Red
                    "highlightwidth": 16,
                },
                # 2. Y Conditional Distribution (e.g., Green)
                "y": {
                    "show": False,
                    "highlight": True,
                    "highlightcolor": "royalblue",  # Color 2: Green
                    "highlightwidth": 16,
                },
                # 3. Disable Horizontal (Z) Contours completely
                "z": {
                    "show": False,
                    "highlight": False,
                },
            },
        )
    )

    # 2. Conditional Slices (3D curves evaluated directly on the surface)
    # FIXME cambia il colore
    # fig.add_trace(
    #     go.Scatter3d(
    #         x=np.full_like(y, x0),
    #         y=y,
    #         z=z_x0,
    #         mode="lines",
    #         line=dict(color="crimson", width=7),
    #         name=f"Slice Y | X={x0:.1f}",
    #     )
    # )
    # fig.add_trace(
    #     go.Scatter3d(
    #         x=x,
    #         y=np.full_like(x, y0),
    #         z=z_y0,
    #         mode="lines",
    #         line=dict(color="royalblue", width=7),
    #         name=f"Slice X | Y={y0:.1f}",
    #     )
    # )

    # 3. Projected Marginal Densities (Back walls)
    # FIXME qui bisogna cambiare la scala di visualizzazione
    # fig.add_trace(
    #     go.Scatter3d(
    #         x=x,
    #         y=np.full_like(x, 3.5),
    #         z=f_X,
    #         mode="lines",
    #         line=dict(color="darkblue", width=4, dash="solid"),
    #         name="Marginal f(X)",
    #     )
    # )
    # fig.add_trace(
    #     go.Scatter3d(
    #         x=np.full_like(y, -3.5),
    #         y=y,
    #         z=f_Y,
    #         mode="lines",
    #         line=dict(color="darkred", width=4, dash="solid"),
    #         name="Marginal f(Y)",
    #     )
    # )

    # 4. Projected Cumulative Distributions (CDF wall curves scaled to z-max)
    # fig.add_trace(
    #     go.Scatter3d(
    #         x=x,
    #         y=np.full_like(x, 3.5),
    #         z=cdf_X * 0.15,
    #         mode="lines",
    #         line=dict(color="teal", width=3, dash="dot"),
    #         name="Marginal CDF F(X)",
    #     )
    # )
    # fig.add_trace(
    #     go.Scatter3d(
    #         x=np.full_like(y, -3.5),
    #         y=y,
    #         z=cdf_Y * 0.15,
    #         mode="lines",
    #         line=dict(color="orange", width=3, dash="dot"),
    #         name="Marginal CDF F(Y)",
    #     )
    # )

    # Layout styling
    fig.update_layout(
        scene=dict(
            xaxis_title="X Axis",
            yaxis_title="Y Axis",
            zaxis_title="Probability Density",
            aspectratio=dict(x=1, y=1, z=0.6),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.1)),
            xaxis=dict(range=[-3.5, 3.5]),
            yaxis=dict(range=[-3.5, 3.5]),
            zaxis=dict(range=[0, 0.22]),
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(x=0.0, y=0.9),
    )

    fig.update_layout(autosize=True)

    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    render()