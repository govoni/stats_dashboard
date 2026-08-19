import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def render():
    st.header("2D Gaussian Bird's-Eye View")

    # --- Controls ---
    col1, col2 = st.columns([1, 2])

    with col1:

        st.subheader("Covariance Parameters")
        sigma_x = st.slider("sigma_x", 0.2, 3.0, 1.0, 0.1)
        sigma_y = st.slider("sigma_y", 0.2, 3.0, 1.5, 0.1)
        rho = st.slider("rho", -0.99, 0.99, 0.60, 0.01)

        st.markdown(
            """
            - 🟦 **Blues Palette:** Density $f(x,y)$
            - **Solid Circle:** $1\\sigma$ Contour
            - **Dashed Circle:** $2\\sigma$ Contour
            - <span style="color:#DC2626; font-weight:bold; font-size:18px;">━</span> **Red Line:** Major Axis
            - <span style="color:#2563EB; font-weight:bold; font-size:18px;">━</span> **Blue Line:** Minor Axis
            """,
            unsafe_allow_html=True,
        )

    # --- Covariance & Eigensystem Math ---
    cov = np.array(
        [
            [sigma_x**2, rho * sigma_x * sigma_y],
            [rho * sigma_x * sigma_y, sigma_y**2],
        ]
    )
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]

    grid_bound = max(2.5 * sigma_x, 2.5 * sigma_y)
    x = np.linspace(-grid_bound, grid_bound, 200)
    y = np.linspace(-grid_bound, grid_bound, 200)
    X, Y = np.meshgrid(x, y)

    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)
    norm_factor = 1.0 / (2 * np.pi * np.sqrt(det_cov))

    mahalanobis_sq = (
        inv_cov[0, 0] * X**2
        + 2 * inv_cov[0, 1] * X * Y
        + inv_cov[1, 1] * Y**2
    )
    Z = norm_factor * np.exp(-0.5 * mahalanobis_sq)

    # 1-sigma / 2-sigma ellipses
    t = np.linspace(0, 2 * np.pi, 200)
    e1 = (
        vecs
        @ np.array([np.sqrt(vals[0]) * np.cos(t), np.sqrt(vals[1]) * np.sin(t)])
    )
    e2 = 2.0 * e1

    with col2:

        # --- Matplotlib Plot Generation ---
        fig, ax = plt.subplots(figsize=(5, 5), dpi=120)

        # Layer 1 (Bottom): Filled Contours (White -> Deep Blue)
        contour = ax.contourf(
            X, Y, Z, levels=15, cmap="Blues", origin="lower", zorder=1
        )

        # Layer 2: Grid Lines
        ax.grid(True, linestyle=":", color="#64748B", alpha=0.6, zorder=2)

        # Layer 3: Coordinate Zero-Axes
        ax.axhline(0, color="darkgray", linewidth=1.5, zorder=3)
        ax.axvline(0, color="darkgray", linewidth=1.5, zorder=3)

        # Layer 4: 1-sigma & 2-sigma Ellipses
        ax.plot(
            e1[0], e1[1], color="#0F172A", linewidth=2, label="1-σ Contour", zorder=4
        )
        ax.plot(
            e2[0],
            e2[1],
            color="#334155",
            linewidth=1.5,
            linestyle="--",
            label="2-σ Contour",
            zorder=4,
        )

        # Layer 5: Principal Axes
        len_maj, len_min = 2.0 * np.sqrt(vals[0]), 2.0 * np.sqrt(vals[1])
        ax.plot(
            [-len_maj * vecs[0, 0], len_maj * vecs[0, 0]],
            [-len_maj * vecs[1, 0], len_maj * vecs[1, 0]],
            color="#DC2626",
            linewidth=2,
            label="Major Axis",
            zorder=5,
        )
        ax.plot(
            [-len_min * vecs[0, 1], len_min * vecs[0, 1]],
            [-len_min * vecs[1, 1], len_min * vecs[1, 1]],
            color="#2563EB",
            linewidth=2,
            label="Minor Axis",
            zorder=5,
        )

        # # Layer 6 (Top): Peak Marker
        # ax.scatter(
        #     [0],
        #     [0],
        #     color="#F59E0B",
        #     marker="*",
        #     s=150,
        #     edgecolors="black",
        #     label="Peak",
        #     zorder=6,
        # )

        # Exact Geometric Aspect Ratio & Boundaries
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-grid_bound, grid_bound)
        ax.set_ylim(-grid_bound, grid_bound)
        ax.set_xlabel("X Values", fontsize=12)
        ax.set_ylabel("Y Values", fontsize=12)

        # Large Top Legend
        # ax.legend(
        #     loc="lower center",
        #     bbox_to_anchor=(0.5, 1.02),
        #     ncol=3,
        #     frameon=True,
        #     fontsize=11,
        # )

        plt.tight_layout()

        # --- Render cleanly in Streamlit ---
        st.pyplot(fig)


if __name__ == "__main__":
    render()