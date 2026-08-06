import time
import numpy as np
import plotly.graph_objects as go
import streamlit as st


def generate_pdf_samples(pdf_type: str, size: int) -> np.ndarray:
    """Generates random samples based on the selected Probability Density Function (PDF)."""
    if pdf_type == "Uniform":
        return np.random.uniform(low=-1.0, high=1.0, size=size)

    elif pdf_type == "Gaussian":
        return np.random.normal(loc=0.0, scale=1.0, size=size)

    elif pdf_type == "Cubic":
        # f(x) = 4 * x^3 on interval [0, 1]
        # Inverse CDF method: F(x) = x^4  =>  X = U^(1/4)
        u = np.random.uniform(low=0.0, high=1.0, size=size)
        return u ** (0.25)

    return np.array([])


def get_pdf_bounds(pdf_type: str):
    """Returns fixed x-axis limits for the single-sample plot based on the PDF domain."""
    if pdf_type == "Uniform":
        return [-1.2, 1.2]
    elif pdf_type == "Gaussian":
        return [-4.0, 4.0]
    elif pdf_type == "Cubic":
        return [-0.1, 1.1]
    return [-3.0, 3.0]


def render():
    st.title("Central Limit Theorem Explorer")
    # st.caption(
    #     "Demonstrating how the distribution of sample means $\\bar{X}$ approaches a Gaussian distribution regardless of the underlying population PDF."
    # )

    # -------------------------------------------------------------------------
    # 1. State Initialization
    # -------------------------------------------------------------------------
    if "sample_means" not in st.session_state:
        st.session_state.sample_means = []
    if "last_sample" not in st.session_state:
        st.session_state.last_sample = np.array([])
    if "current_pdf" not in st.session_state:
        st.session_state.current_pdf = "Uniform"

    # -------------------------------------------------------------------------
    # 2. Control Widgets
    # -------------------------------------------------------------------------
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 2])

    with ctrl_col1:
        pdf_choice = st.selectbox(
            "Parent Distribution PDF $f(x)$",
            options=["Uniform", "Gaussian", "Cubic"],
            index=0,
            key="pdf_selector",
        )
        # Reset state if distribution changes
        if pdf_choice != st.session_state.current_pdf:
            st.session_state.current_pdf = pdf_choice
            st.session_state.sample_means = []
            st.session_state.last_sample = np.array([])

    with ctrl_col2:
        sample_size = st.number_input(
            "Sample Size ($N$)",
            min_value=1,
            max_value=10000,
            value=30,
            step=5,
        )

    with ctrl_col3:
        delay_ms = st.slider(
            "Animation Delay (ms)",
            min_value=0,
            max_value=500,
            value=50,
            step=10,
        )

    # Action Buttons
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 4])
    gen_single = btn_col1.button("Generate 1")
    gen_100 = btn_col2.button("Generate 100")
    if btn_col3.button("Reset Data"):
        st.session_state.sample_means = []
        st.session_state.last_sample = np.array([])
        st.rerun()

    # -------------------------------------------------------------------------
    # 3. Layout Setup (Placeholders for zero-flicker re-rendering)
    # -------------------------------------------------------------------------
    # st.markdown("---")
    left_col, right_col = st.columns(2)

    with left_col:
        # st.subheader("Current Single Sample ($N$ draws)")
        sample_plot_place = st.empty()

    with right_col:
        # st.subheader("Distribution of Sample Means ($\ \\bar{X}\ $)")
        means_plot_place = st.empty()

    # -------------------------------------------------------------------------
    # 4. Plot Refresh Routine
    # -------------------------------------------------------------------------

    def update_plots():
        x_range = get_pdf_bounds(pdf_choice)

        # ---------------------------------------------------------------------
        # Font & Style Configuration
        # ---------------------------------------------------------------------
        title_style = dict(color="black", size=20, family="Arial")
        axis_title_style = dict(color="black", size=16, family="Arial")
        tick_style = dict(color="black", size=14, family="Arial")

        # --- Left Plot: Latest Single Sample ---
        fig_sample = go.Figure()
        if len(st.session_state.last_sample) > 0:
            fig_sample.add_trace(
                go.Histogram(
                    x=st.session_state.last_sample,
                    histnorm="probability density",
                    name="Sample",
                    marker_color="#3366CC",
                    opacity=0.75,
                    nbinsx=30,
                )
            )
        fig_sample.update_layout(
            title=dict(
                text=f"Current Sample", font=title_style
            ),
            xaxis=dict(
                title=dict(text="Variable x", font=axis_title_style),
                tickfont=tick_style,
                range=x_range,
                showgrid=True,
                gridcolor="#E5E5E5",
            ),
            yaxis=dict(
                title=dict(text="Probability Density", font=axis_title_style),
                tickfont=tick_style,
                autorange=True,
                showgrid=True,
                gridcolor="#E5E5E5",
            ),
            paper_bgcolor="white",  # Ensures high-contrast background
            plot_bgcolor="white",
            margin=dict(l=50, r=20, t=50, b=50),
            height=400,
            showlegend=False,
        )

        # --- Right Plot: Accumulated Means ---
        fig_means = go.Figure()
        if len(st.session_state.sample_means) > 0:
            fig_means.add_trace(
                go.Histogram(
                    x=st.session_state.sample_means,
                    histnorm="probability density",
                    name="Sample Means",
                    marker_color="#DC3912",
                    opacity=0.75,
                    nbinsx=35,
                )
            )
        fig_means.update_layout(
            title=dict(
                text=f"Distribution of Means",
                font=title_style,
            ),
            xaxis=dict(
                title=dict(text="Sample Mean (x̄)", font=axis_title_style),
                tickfont=tick_style,
                range=x_range,
                showgrid=True,
                gridcolor="#E5E5E5",
            ),
            yaxis=dict(
                title=dict(text="Density", font=axis_title_style),
                tickfont=tick_style,
                autorange=True,
                showgrid=True,
                gridcolor="#E5E5E5",
            ),
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=50, r=20, t=50, b=50),
            height=400,
            showlegend=False,
        )

        # Render inside empty slots
        sample_plot_place.plotly_chart(
            fig_sample, use_container_width=True, key=f"sample_{time.time()}"
        )
        means_plot_place.plotly_chart(
            fig_means, use_container_width=True, key=f"means_{time.time()}"
        )

    # -------------------------------------------------------------------------
    # 5. Event Loop & Generation Logic
    # -------------------------------------------------------------------------
    if gen_single:
        sample = generate_pdf_samples(pdf_choice, sample_size)
        st.session_state.last_sample = sample
        st.session_state.sample_means.append(float(np.mean(sample)))
        update_plots()

    elif gen_100:
        for _ in range(100):
            sample = generate_pdf_samples(pdf_choice, sample_size)
            st.session_state.last_sample = sample
            st.session_state.sample_means.append(float(np.mean(sample)))
            update_plots()
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)
    else:
        # Render initial or persisted state upon initial load/widget interaction
        update_plots()


# Standard entry point if running as a standalone script
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    plt.tight_layout()
    render()