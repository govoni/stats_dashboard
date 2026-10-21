"""
examples/radioactive_decay.py

Radioactive decay simulation.

Each "toy experiment" starts with N0 unstable atoms.  At every discrete
time step dt the surviving atoms each have an independent probability
p = 1 - exp(-dt / tau) of decaying in that step (exact binomial draw).
The population is tracked until the observation time T_obs is reached.

Controls (all in a row above the plot, no sidebar):
  - tau        : mean lifetime of one atom  [s]
  - N0         : initial number of atoms  (minimum 1)
  - dt         : time step size             [s]
  - T_obs      : total observation window   [s]
  - n_toys     : number of overlaid toy experiments (1 = single run)
  - show_theory: overlay the analytical N(t) = N0 * exp(-t/tau) curve

The multi-experiment overlay follows the same pattern as histogram_sampler.py:
each toy is a separate semi-transparent line; the mean over all toys is drawn
on top in a bold colour.  Randomness is re-drawn on every Streamlit rerun
(parameter change or manual page refresh) — no fixed seed.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


# ── helper ────────────────────────────────────────────────────────────────────

def run_decay(N0: int, tau: float, dt: float, T_obs: float) -> tuple:
    """
    Simulate one radioactive decay experiment.

    Returns
    -------
    times  : 1-D float array, shape (n_steps+1,)
    counts : 1-D float array, shape (n_steps+1,)  — surviving atoms at each step
    """
    p_decay = 1.0 - np.exp(-dt / tau)
    n_steps = int(np.ceil(T_obs / dt))
    times  = np.linspace(0.0, n_steps * dt, n_steps + 1)
    counts = np.empty(n_steps + 1, dtype=float)
    counts[0] = float(N0)
    N = N0
    for i in range(1, n_steps + 1):
        if N == 0:
            counts[i] = 0.0
        else:
            n_decayed = np.random.binomial(N, p_decay)
            N -= n_decayed
            counts[i] = float(N)
    return times, counts


# ── render ────────────────────────────────────────────────────────────────────

def render():
    with st.container():
        st.title ("Radioactive decay")
    # ── reset button (top-right, before widgets so it fires first) ────────────
    # A counter stored in session_state is used as the numpy seed, so clicking
    # Reset draws a new set of random numbers while leaving all parameters alone.
    if "decay_reset_counter" not in st.session_state:
        st.session_state["decay_reset_counter"] = 0

    # ── controls row ─────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6, c7 = st.columns([2, 2, 2, 2, 2, 2, 1])

    with c1:
        tau = st.number_input(
            "Mean lifetime τ (s)",
            min_value=0.1, max_value=1000.0, value=10.0, step=0.5,
        )
    with c2:
        N0 = st.number_input(
            "Initial atoms N₀",
            min_value=1, max_value=100_000, value=1, step=1,
        )
    with c3:
        dt = st.number_input(
            "Time step dt (s)",
            min_value=0.01, max_value=100.0, value=0.5, step=0.1,
        )
    with c4:
        T_obs = st.number_input(
            "Observation time T (s)",
            min_value=1.0, max_value=5000.0, value=50.0, step=5.0,
        )
    with c5:
        n_toys = st.number_input(
            "Experiments",
            min_value=1, max_value=200, value=1, step=1,
        )
    with c6:
        show_theory = st.checkbox("Show N₀ e^{-t/τ}", value=False)
    with c7:
        st.write("")   # vertical alignment spacer
        if st.button("Reset"):
            st.session_state["decay_reset_counter"] += 1

    # ── sanity guards ─────────────────────────────────────────────────────────
    if dt >= tau:
        st.warning(
            f"dt = {dt} s ≥ τ = {tau} s — time step too coarse.  Try dt ≪ τ."
        )

    n_steps = int(np.ceil(T_obs / dt))
    if n_steps > 50_000:
        st.error(
            f"T / dt = {n_steps} steps — too many.  Increase dt or reduce T."
        )
        return

    # ── simulation ────────────────────────────────────────────────────────────
    # The RNG is seeded from the reset counter so that:
    #   - clicking Reset draws a new set of random numbers,
    #   - changing a parameter widget reruns with the *same* random draws,
    #     letting students see the effect of the parameter cleanly.
    rng_seed = st.session_state["decay_reset_counter"]
    np.random.seed(rng_seed)

    all_counts = []
    for _ in range(n_toys):
        t_axis, c = run_decay(N0, tau, dt, T_obs)
        all_counts.append(c)

    # t_axis is identical for every toy
    mean_counts = np.mean(np.stack(all_counts, axis=0), axis=0)

    # ── figure ────────────────────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 3.5))

    if n_toys == 1:
        ax1.plot(t_axis, all_counts[0], color="steelblue", lw=1.5,
                 label="Simulation")
    else:
        for i, c in enumerate(all_counts):
            ax1.plot(t_axis, c, color="steelblue", alpha=0.25, lw=0.8,
                     label="Toy experiments" if i == 0 else None)
        ax1.plot(t_axis, mean_counts, color="tomato", lw=2.0,
                 label="Ensemble mean")

    if show_theory:
        t_th = np.linspace(0, T_obs, 800)
        ax1.plot(t_th, N0 * np.exp(-t_th / tau),
                 color="forestgreen", lw=1.8, ls="--",
                 label=r"$N_0\,e^{-t/\tau}$")

    # half-life marker
    t_half = tau * np.log(2)
    ax1.axvline(t_half, color="grey", lw=0.9, ls=":")
    ax1.text(t_half * 1.02, ax1.get_ylim()[1] * 0.95 if N0 > 1 else 0.95,
             rf"$t_{{1/2}}={t_half:.1f}$ s",
             color="grey", fontsize=8, va="top")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Surviving atoms")
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=8)
    fig1.tight_layout()

    st.pyplot(fig1)
    plt.close(fig1)

    # ── info caption ─────────────────────────────────────────────────────────
    p_step = 1.0 - np.exp(-dt / tau)
    st.caption(
        f"Per-step decay probability  p = 1 − e^(−dt/τ) = **{p_step:.4f}**"
        f"   |   Steps: **{n_steps}**"
        f"   |   Half-life: **{t_half:.2f} s**"
    )
