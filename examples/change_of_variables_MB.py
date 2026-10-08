import numpy as np
import streamlit as st
from scipy import stats
import matplotlib.pyplot as plt

KB = 1.380649e-23
H = 6.62607015e-34
M = 28.0134 * 1.66053906660e-27   # N2


def f_v(v, T):
    a = M / (2 * np.pi * KB * T)
    return 4 * np.pi * a**1.5 * v**2 * np.exp(-M * v**2 / (2 * KB * T))


def f_lam(lam, T):
    """f_λ(λ) = f_v(v(λ)) |dv/dλ|, with v = h/(mλ)."""
    return f_v(H / (M * lam), T) * H / (M * lam**2)


def f_nu(nu, T):
    """f_ν(ν) = f_λ(λ(ν)) |dλ/dν|, with λ = 1/ν and |dλ/dν| = 1/ν²."""
    return f_lam(1 / nu, T) / nu**2


def render():
    st.title ("La distribuzione di Maxwell-Boltzmann")

    T = 300 #K
    
    # Peak of f_λ (analytic) and grids in units of λ_M
    lam_M = H / np.sqrt(4 * M * KB * T)
    lam = np.linspace(0.2, 6, 3000) * lam_M
    nu = np.linspace(0.02, 4, 3000) / lam_M

    fl, fn = f_lam(lam, T), f_nu(nu, T)
    nu_peak = nu[np.argmax(fn)]            # numerical peak of f_ν
    nu_naive = 1 / lam_M                   # image of the peak of f_λ

    s = 1e12                               # λ in pm, ν in 1/pm
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    ax1.plot(lam * s, fl / s, "C0", lw=2)
    ax1.axvline(lam_M * s, color="k", ls="--", label=r"$\lambda_M$")
    ax1.set_xlabel("λ [pm]")
    ax1.set_ylabel(r"$f_\lambda$ [1/pm]")
    ax1.set_title("Distribution in wavelength")
    ax1.legend()

    ax2.plot(nu / s, fn * s, "C0", lw=2)
    ax2.axvline(nu_naive / s, color="k", ls="--", label=r"$1/\lambda_M$")
    ax2.axvline(nu_peak / s, color="C3", ls="-", label=r"actual peak of $f_\nu$")
    ax2.set_xlabel("ν = 1/λ [1/pm]")
    ax2.set_ylabel(r"$f_\nu$ [pm]")
    ax2.set_title("Distribution in ν = 1/λ")
    ax2.legend()

    fig.tight_layout()
    st.pyplot(fig)
