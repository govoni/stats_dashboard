import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import streamlit as st


def sturges (N_events) :
    return int(np.ceil( 1 + np.log2 (N_events) ))


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def get_dist_info(dist_name):
    """Returns sampler, theoretical mean, theoretical sigma, and theoretical PDF function."""
    if dist_name == 'Uniform':
        mu, sigma = 0.5, 1.0 / np.sqrt(12.0)
        sampler = lambda size: np.random.uniform(0, 1, size)
        pdf = lambda x: np.where((x >= 0) & (x <= 1), 1.0, 0.0)
        x_lims = (-0.2, 1.2)

    elif dist_name == 'Gaussian':
        mu, sigma = 0.0, 1.0
        sampler = lambda size: np.random.normal(mu, sigma, size)
        pdf = lambda x: stats.norm.pdf(x, loc=mu, scale=sigma)
        x_lims = (-4.0, 4.0)

    elif dist_name == 'Triangular (Symmetric)':
        mu, sigma = 0.5, 1.0 / np.sqrt(24.0)
        sampler = lambda size: np.random.triangular(0, 0.5, 1, size)
        pdf = lambda x: stats.triang.pdf(x, c=0.5, loc=0, scale=1)
        x_lims = (-0.2, 1.2)

    elif dist_name == 'Triangular (Asymmetric)':
        c = 0.1  # Peak at 0.1
        mu = (1.0 + c) / 3.0
        sigma = np.sqrt((1.0 - c + c**2) / 18.0)
        sampler = lambda size: np.random.triangular(0, c, 1, size)
        pdf = lambda x: stats.triang.pdf(x, c=c, loc=0, scale=1)
        x_lims = (-0.2, 1.2)

    elif dist_name == 'Parabolic (U-shape)':
        mu, sigma = 0.5, np.sqrt(3.0 / 20.0)
        def sampler(size):
            u = np.random.uniform(0, 1, size)
            return 0.5 + np.cbrt((u - 0.5) / 4.0)
        pdf = lambda x: np.where((x >= 0) & (x <= 1), 6.0 * (x - 0.5)**2, 0.0)
        x_lims = (-0.2, 1.2)

    elif dist_name == 'Exponential':
        scale = 1.0  # rate lambda = 1
        mu, sigma = scale, scale
        sampler = lambda size: np.random.exponential(scale, size)
        pdf = lambda x: stats.expon.pdf(x, scale=scale)
        x_lims = (-0.2, 5.0)

    return sampler, mu, sigma, pdf, x_lims


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def render():
    st.header("Single Measurement vs. Mean of $N$ Measurements")
    # st.markdown(
    #     "Watch individual measurements scatter across the base distribution PDF while their "
    #     "**sample mean** $(\\bar{X})$ concentrates into a narrower Gaussian distribution with "
    #     "$\\sigma_{\\text{mean}} = \\sigma_{\\text{single}} / \\sqrt{N}$."
    # )


    # ------------------------------------------------------------------
    # 2. Controls & Session State Setup
    # ------------------------------------------------------------------


    c1, c2, c3, c4, c5 = st.columns (5)

    with c1 :
        dist_choice = st.selectbox(
            "Measurement Distribution:",
            ['Uniform', 'Gaussian', 'Triangular (Symmetric)', 'Triangular (Asymmetric)', 'Parabolic (U-shape)', 'Exponential']
        )

    if c2.button("Add 1 Point"):
        sampler, _, _, _, _ = get_dist_info(dist_choice)
        st.session_state.single_points.append(sampler(1)[0])

    if c3.button("Add 10 Points"):
        sampler, _, _, _, _ = get_dist_info(dist_choice)
        st.session_state.single_points.extend(sampler(10))

    if c4.button("Add 100 Points"):
        sampler, _, _, _, _ = get_dist_info(dist_choice)
        st.session_state.single_points.extend(sampler(100))

    if c5.button("Clear Points"):
        st.session_state.single_points = []

    # Initialize Session State for interactive point accumulation
    if 'single_points' not in st.session_state:
        st.session_state.single_points = []
    if 'current_dist' not in st.session_state or st.session_state.current_dist != dist_choice:
        st.session_state.single_points = []
        st.session_state.current_dist = dist_choice

    # ------------------------------------------------------------------
    # 3. Perform Calculations
    # ------------------------------------------------------------------
    sampler, mu_true, sigma_single_true, pdf_func, x_limits = get_dist_info (dist_choice)

    N_means = 1000 # FIXME da trasformare in un parametro configurabile
    sample_color      = 'red'
    dark_sample_color = 'darkred'
    mean_color        = 'darkblue'

    # for the trend plots
    n_range = np.arange(0, 6)
    n_points_list = []
    sample_mean_list, sample_sigma_list = [], []
    mean_mean_list, mean_sigma_list= [], []

    # Compute metrics over range for convergence curve
    # for each given number of samples, 
    # the metrics for the single measurement are calculated
    # on a sample of single measurements,
    # while the metrics for the mean are calculated
    # on a sample of means
    N_samples = 1
    for n in n_range:
        # generate a sample of means and calculate its stats
        means = [np.mean (sampler (N_samples)) for i in range (N_means)]
        mean_mean_list.append (np.mean (means))
        mean_sigma_list.append (np.std (means))

        n_points_list.append (N_samples)

        N_samples *= 3

    # ------------------------------------------------------------------
    # DISTRIBUTIONS AT FIXED NUMBER OF SAMPLES
    # ------------------------------------------------------------------

    fig, (ax_pdf, ax_sam, ax_ave) = plt.subplots(
        3, 1, 
        sharex=True, 
        figsize=(10, 3.5),
        gridspec_kw={
                'hspace': 0,               # Remove space between plots
                'height_ratios': [6, 1, 1] # 6:1:1 relative height
            }
        )
    ax_pdf2 = ax_pdf.twinx()  # Instantiate a second axes that shares the same x-axis 

    x_grid = np.linspace(x_limits[0], x_limits[1], 500)

    # 1. Base PDF (Single Measurements)
    ax_pdf.plot (x_grid, pdf_func(x_grid), color=sample_color, lw=1.5, label='Original PDF (Single Measurement)')

    ax_sam.axhline (y=0., color='gray', linestyle='--', linewidth=1)
    ax_sam.set_xlim (x_limits)
    ax_sam.get_yaxis ().set_visible (False)

    ax_ave.axhline (y=0., color='gray', linestyle='--', linewidth=1)
    ax_ave.set_xlim (x_limits)
    ax_ave.get_yaxis ().set_visible (False)

    N_events = 0

    # 4. Overlaid Interactive Step-by-Step Points
    if len (st.session_state.single_points) > 0:
        pts = np.array (st.session_state.single_points)
        y_jitter = np.zeros_like (pts)
        ax_sam.scatter (pts, y_jitter, color=sample_color, alpha=0.5, zorder=5, s=40, label='samples')
        ax_sam.scatter (pts[-1], y_jitter[-1], color=dark_sample_color, zorder=5, s=40)

        N_events = len (st.session_state.single_points)
        n_bins = 2 * sturges (N_events)
        bins = np.linspace (*x_limits, n_bins + 1)
        
        # ax_pdf.hist(
        #     st.session_state.single_points,
        #     bins      = bins,
        #     density   = True,
        #     histtype  = 'step',
        #     # color     = "#ADD8E6",
        #     lw        = 2.5,
        #     edgecolor = "red",
        #     alpha     = 0.8,
        #     label     = "Sample Histogram",
        # )

        means = [np.mean (sampler (N_events)) for i in range (N_means)]
        y_jitter = np.zeros_like (means)

        ax_ave.scatter (
          means, 
          y_jitter, 
          color  = mean_color, 
          marker = 'P',
          alpha  = 0.5,
          zorder = 5, 
          s      = 40, 
          label  = 'means'
        )
        ax_pdf2.set_ylabel ('Probability Density (means)', color = mean_color)
        ax_pdf2.tick_params (axis='y', labelcolor=mean_color)

        bins = np.linspace (*x_limits, 5 * n_bins + 1)

        ax_pdf2.hist(
            means,
            bins      = bins,
            density   = True,
            histtype  = 'step',
            # color     = "#ADD8E6",
            lw        = 2.5,
            edgecolor = mean_color,
            alpha     = 0.8,
            label     = "Mean Expected Histogram",
        )

    # ax_pdf.set_title(f"Spread of Single Measurements vs. Mean of $N={N_group}$ Measurements ({dist_choice})", fontweight='bold')
    ax_sam.set_xlabel ("Value (X)")
    ax_pdf.set_xlim (x_limits)
    ax_pdf.tick_params (axis='y', labelcolor = sample_color)
    ax_pdf.set_ylabel ("Probability Density (samples)", color = sample_color)
    ax_pdf.legend (loc='upper right', fontsize='small')
    # ax_pdf.grid(True, linestyle=':', alpha=0.6)

    # 3. Collect handles and labels from BOTH axes
    handles1, labels1 = ax_pdf.get_legend_handles_labels()
    handles4, labels4 = ax_pdf2.get_legend_handles_labels()
    handles2, labels2 = ax_sam.get_legend_handles_labels()
    handles3, labels3 = ax_ave.get_legend_handles_labels()

    # 4. Combine them and display single legend on top axis
    ax_pdf.legend (
       handles=handles1 + handles4 + handles2 + handles3, 
       labels=labels1 + labels4 + labels2 + labels3, 
       loc='upper right'
    )

    ax_pdf.set_title (
        f"Generation of {N_events} samples",
        fontsize=11,
        fontweight="bold",
    )

    plt.tight_layout()

    # ------------------------------------------------------------------
    # TRENDS VS NUMBER OF SAMPLES
    # ------------------------------------------------------------------

    fig2, ax_sig = plt.subplots (figsize=(10, 3.5))

    ax_sig.axhline (sigma_single_true, color=sample_color, linewidth=0.8, label = 'single measurement')
    ax_sig.plot (n_points_list, mean_sigma_list, color = mean_color, linewidth=2, label = 'mean')
    ax_sig.set_ylabel ("Sigma")
    ax_sig.legend ()

    ax_sig.set_xlabel ("sample size ($N$)")

    plt.tight_layout()

    # ------------------------------------------------------------------
    # PLOTTING
    # ------------------------------------------------------------------

    # Render Plot inside Streamlit
    st.pyplot(fig)
    st.divider()
    st.header("Sigma Trend vs. $N$")

    st.pyplot(fig2)


