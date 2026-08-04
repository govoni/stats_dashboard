import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


def render () :
    """Renders an interactive demonstration of uniform random sampling

    and its resulting histogram/empirical density.
    """
    st.header("Uniform Distribution Sampling")

    # background-color: #fff9c4; /* yellow */
    st.markdown ("""
    <style>
    div[class*="st-key-prob_box"] {
        background-color: #ADD8E6; /* blue */
        padding: 36px;
        border-radius: 8px;
        text-align: center;
    }
    div[class*="st-key-prob_box"] p {
        font-size: 22px;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.container (key="prob_box"):
        # st.markdown ("**Probability definition:**")
        st.latex(
            r"""
            X \sim \mathcal{U}(a, b) \implies f(x) = 
            \begin{cases} 
              \frac{1}{b - a} & \text{for } a \le x \le b \\[8pt]
              0 & \text{otherwise} 
            \end{cases}
        """
        )


    # st.write(
    #     "Generate uniformly distributed random variables $X \\sim \\mathcal{U}(a, b)$ "
    #     "and observe how the empirical histogram converges to the true uniform probability density function."
    # )

    # --- Sidebar / Controls ---
    st.subheader("Parameters")

    col_ctrl1, col_ctrl2 = st.columns(2)

    with col_ctrl1:
        # Uniform bounds [a, b]
        bounds = st.slider(
            "Uniform Range [a, b]",
            min_value=-10.0,
            max_value=10.0,
            value=(-2.0, 5.0),
            step=0.5,
            help="Defines the lower (a) and upper (b) bounds of the uniform distribution.",
        )
        a, b = bounds

        # # Sample size
        # n_samples = st.slider(
        #     "Number of Tosses (N)",
        #     min_value=10,
        #     max_value=10000,
        #     value=500,
        #     step=10,
        #     help="Number of random draws generated.",
        # )

        # --- Logarithmic Slider for N ---
        # Exponents: 1.0 -> 10^1 = 10, 4.0 -> 10^4 = 10,000
        log_n = st.slider(
            "Number of Tosses (N)",
            min_value=1.0,
            max_value=4.0,
            value=1.0,  # 10^1 = 10 default
            step=0.01,  # Fine granular control along log scale
            # format="10^%.2f",  # Displays power exponent visually
            label_visibility="collapsed",
            # format_func = lambda x: f"{int(round(10**x)):,}",
            help="Logarithmic scale from 10 to 10,000 samples.",
        )

        # Compute actual N as integer
        n_samples = int(round(10**log_n))


    with col_ctrl2:
        # Histogram Bin Control
        n_bins = st.slider(
            "Number of Histogram Bins",
            min_value=5,
            max_value=100,
            value=25,
            step=1,
            help="Controls the resolution of the histogram bins.",
        )

        # Seed toggle for reproducible demos
        seed = st.number_input("Random Seed (0 for off)", min_value=0, value=0)

    # --- Data Generation ---
    if seed != 0:
        np.random.seed(seed)

    # Draw N samples from U(a, b)
    samples = np.random.uniform(low=a, high=b, size=n_samples)

    st.divider()  # Draws a subtle horizontal line across the layout

    # --- Visualization (Plotly) ---

    # Common X-axis range so both separate plots stay perfectly aligned visually
    marker_size = 20
    if n_samples > 10 : marker_size = 10

    x_range = [a - 1.5, b + 1.5]
    bin_size = (b - a) / n_bins if b > a else 0.1


   # ==========================================
    # DRAWING 1: 1D Scatter Points
    # ==========================================
    fig_scatter = go.Figure()

    # y_jitter = np.random.uniform(-0.1, 0.1, size=n_samples)
    y_jitter = np.zeros (n_samples)

    fig_scatter.add_trace(
        go.Scatter(
            x=samples,
            y=y_jitter,
            mode="markers",
            marker=dict(
                size=marker_size, opacity=0.6, color="#1f77b4"
            ),
            showlegend=False,
            hovertemplate="Value: %{x:.3f}<extra></extra>",
        )
    )

    fig_scatter.add_shape(
        type="line",
        x0=a - 1,
        x1=b + 1,
        y0=0,
        y1=0,
        line=dict(color="gray", width=1, dash="dash"),
    )

    fig_scatter.update_layout(
        title="1D Tossed Samples",
        height=180,
        margin=dict(l=20, r=20, t=35, b=20),
        xaxis_title="uniform variable",     # Horizontal axis title
        xaxis=dict(range=x_range),
        yaxis=dict(
            showticklabels=False, showgrid=False, zeroline=False, range=[-0.25, 0.25]
        ),
    )


    # ==========================================
    # DRAWING 2: Unnormalized Histogram (Raw Counts)
    # ==========================================
    fig_raw = go.Figure()

    fig_raw.add_trace(
        go.Histogram(
            x=samples,
            xbins=dict(start=a, end=b, size=bin_size),
            histnorm="",  # Default: Raw counts
            marker=dict(color="#B0C4DE", line=dict(color="#4682B4", width=1)),
            name="Raw Counts",
            opacity=0.75,
        )
    )

    fig_raw.update_layout(
        title="Raw Counts Histogram (Not Normalized)",
        height=350,
        margin=dict(l=20, r=20, t=35, b=20),
        xaxis=dict(title_text="Value (X)", range=x_range),
        yaxis=dict(title_text="Count"),
        showlegend=False,
    )

    # ==========================================
    # DRAWING 3: Normalized Density Histogram (density=True)
    # ==========================================
    fig_density = go.Figure()

    fig_density.add_trace(
        go.Histogram(
            x=samples,
            xbins=dict(start=a, end=b, size=bin_size),
            histnorm="density",  # Matplotlib density=True
            marker=dict(color="#ADD8E6", line=dict(color="#1f77b4", width=1)),
            name="Empirical Density",
            opacity=0.75,
        )
    )

    if b > a:
        pdf_height = 1.0 / (b - a)
        fig_density.add_trace(
            go.Scatter(
                x=[a - 1, a, a, b, b, b + 1],
                y=[0, 0, pdf_height, pdf_height, 0, 0],
                mode="lines",
                line=dict(color="#D32F2F", width=2.5, dash="dash"),
                name="Theoretical PDF",
            )
        )

    fig_density.update_layout(
        title="Density Histogram (Normalized, density=True)",
        height=350,
        margin=dict(l=20, r=20, t=35, b=20),
        xaxis=dict(title_text="Value (X)", range=x_range),
        yaxis=dict(title_text="Density"),
        showlegend=True,
    )

    # # ==========================================
    # # PLOT 1: 1D Sample Points on a Line
    # # ==========================================
    # fig_top = go.Figure()

    # # y_jitter = np.random.uniform(-0.1, 0.1, size=n_samples)
    # y_jitter = np.zeros (n_samples)

    # marker_size = 20
    # if n_samples > 10 : marker_size = 10

    # fig_top.add_trace(
    #     go.Scatter(
    #         x=samples,
    #         y=y_jitter,
    #         mode="markers",
    #         marker=dict(
    #             size=marker_size, opacity=0.7, color="#1f77b4"
    #         ),
    #         name="Samples",
    #         hovertemplate="Value: %{x:.3f}<extra></extra>",
    #     )
    # )

    # # Baseline y = 0
    # fig_top.add_shape(
    #     type="line",
    #     x0=a - 1,
    #     x1=b + 1,
    #     y0=0,
    #     y1=0,
    #     line=dict(color="gray", width=1, dash="dash"),
    # )

    # fig_top.update_layout(
    #     title="1D Sample Points on a Line",
    #     height=180,
    #     margin=dict(l=20, r=20, t=35, b=20),
    #     xaxis=dict(range=x_range),
    #     yaxis=dict(
    #         showticklabels=False, showgrid=False, zeroline=False, range=[-0.25, 0.25]
    #     ),
    #     showlegend=False,
    # )

    # # ==========================================
    # # PLOT 2: Sample Histogram & Theoretical PDF
    # # ==========================================
    # fig_bottom = go.Figure()

    # bin_size = (b - a) / n_bins if b > a else 0.1
    # fig_bottom.add_trace(
    #     go.Histogram(
    #         x=samples,
    #         xbins=dict(start=a, end=b, size=bin_size),
    #         histnorm="density",
    #         marker=dict(color="#ADD8E6", line=dict(color="#1f77b4", width=1)),
    #         name="Empirical Density",
    #         opacity=0.75,
    #     )
    # )


    # # ==========================================
    # # DRAWING 3: Normalized Density Histogram (density=True)
    # # ==========================================
    # fig_density = go.Figure()

    # fig_density.add_trace(
    #     go.Histogram(
    #         x=samples,
    #         xbins=dict(start=a, end=b, size=bin_size),
    #         histnorm="density",  # Matplotlib density=True
    #         marker=dict(color="#ADD8E6", line=dict(color="#1f77b4", width=1)),
    #         name="Empirical Density",
    #         opacity=0.75,
    #     )
    # )

    # # Theoretical Uniform PDF overlay
    # if b > a:
    #     pdf_height = 1.0 / (b - a)
    #     fig_bottom.add_trace(
    #         go.Scatter(
    #             x=[a - 1, a, a, b, b, b + 1],
    #             y=[0, 0, pdf_height, pdf_height, 0, 0],
    #             mode="lines",
    #             line=dict(color="#D32F2F", width=2.5, dash="dash"),
    #             name="Theoretical PDF",
    #         )
    #     )

    # fig_bottom.update_layout(
    #     title="Sample Histogram & PDF",
    #     height=380,
    #     margin=dict(l=20, r=20, t=35, b=20),
    #     xaxis=dict(title_text="Value (X)", range=x_range),
    #     yaxis=dict(title_text="Density"),
    #     showlegend=True,
    #     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    # )

    # ==========================================
    # DISPLAY PLOTS WITH INTERMEDIATE DIVIDER
    # ==========================================

    with st.container (key="prob_box_2"):
        st.markdown (f'**number of samples: {n_samples}**')

    st.plotly_chart (fig_scatter, use_container_width=True)

    st.divider()

    st.plotly_chart(fig_raw, use_container_width=True)

    st.divider()

    st.plotly_chart(fig_density, use_container_width=True)


# For standalone testing:
if __name__ == "__main__":
    st.set_page_config(page_title="Uniform Sampling Demo", layout="wide")
    render_uniform_distribution_example()