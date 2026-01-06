"""
threshold_utils_9V.py

Utilities to extract the breakdown "threshold" voltage V_th for the 9.1 V Zener
diode when the measured voltage is across (diode + 100 Ω series resistor).

We correct the voltage to the diode voltage:

    V_D = V_total - R_series * |I|

and then perform a local weighted linear fit I(V_D) around I_th.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

R_SERIES_DEFAULT = 100.0  # Ω, the series resistor you used


def _current_sd_from_ratio(df, I_abs, current_sd_col="current SD"):
    """
    Convert the 'current SD' column (relative uncertainty) into an
    absolute σ_I array. If the column is missing, returns None.
    """
    if current_sd_col not in df.columns:
        return None

    ratio = df[current_sd_col].to_numpy(dtype=float)
    I_sd = np.abs(I_abs) * ratio
    # Clean up any weird values
    I_sd[~np.isfinite(I_sd)] = np.nan
    return I_sd


def _weighted_linear_fit(x, y, y_sigma=None):
    """
    Weighted linear regression y = m x + c.

    Returns:
        m, c, sigma_m, sigma_c, cov_mc, chi2, dof
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if y_sigma is None or not np.any(np.isfinite(y_sigma)):
        w = np.ones_like(y)
        y_sigma = None
    else:
        y_sigma = np.asarray(y_sigma, dtype=float)
        w = 1.0 / (y_sigma ** 2)

    S = np.sum(w)
    Sx = np.sum(w * x)
    Sy = np.sum(w * y)
    Sxx = np.sum(w * x * x)
    Sxy = np.sum(w * x * y)

    Delta = S * Sxx - Sx ** 2
    if Delta == 0:
        raise RuntimeError("Degenerate points in weighted_linear_fit (Delta = 0).")

    m = (S * Sxy - Sx * Sy) / Delta
    c = (Sxx * Sy - Sx * Sxy) / Delta

    # Parameter variances and covariance (standard formulas for weighted LS)
    sigma_m2 = S / Delta
    sigma_c2 = Sxx / Delta
    cov_mc = -Sx / Delta

    # χ² (only meaningful if real σ provided)
    if y_sigma is None:
        chi2 = np.nan
    else:
        y_fit = m * x + c
        chi2 = np.sum(((y - y_fit) / y_sigma) ** 2)

    dof = len(x) - 2
    return m, c, np.sqrt(sigma_m2), np.sqrt(sigma_c2), cov_mc, chi2, dof


def find_threshold_at_fixed_current(
    filename,
    I_threshold=2e-3,
    R_series=R_SERIES_DEFAULT,
    voltage_col="voltage/V",
    current_col="current/A",
    current_sd_col="current SD",
    window_points=3,
    make_plot=True,
    show_errorbars=True,
    ax=None,
):
    """
    Find the breakdown "threshold" voltage V_th for a single IV file.

    Parameters
    ----------
    filename : str
        CSV file with columns including voltage_col and current_col.
        voltage_col is the TOTAL voltage across diode + R_series.
    I_threshold : float
        Threshold current (A) at which we define V_th.
    R_series : float
        Series resistor value (Ω).
    voltage_col, current_col, current_sd_col : str
        Column names in the CSV.
    window_points : int
        Half-width (in data points) of the local window around I_th
        used for the linear fit.
    make_plot : bool
        If True, plot the IV curve and the local linear fit.
    show_errorbars : bool
        If True and current SD available, plot error bars.
    ax : matplotlib.axes.Axes or None
        If provided, draw into this axis; otherwise create a new figure.

    Returns
    -------
    V_th : float
        Threshold diode voltage (V).
    I_th : float
        Threshold current (A), equals I_threshold.
    sigma_V_th : float or None
        Uncertainty on V_th from the linear fit (V).
    extra : dict
        Dictionary with extra details (fit params, chi2, etc.).
    """
    # ----- Load and clean data -----
    df = pd.read_csv(filename)
    df = df.dropna(subset=[voltage_col, current_col]).copy()

    V_total = df[voltage_col].to_numpy(dtype=float)
    I_raw = df[current_col].to_numpy(dtype=float)

    # Work with magnitude of current
    I = np.abs(I_raw)

    # Correct total voltage -> diode voltage
    V_diode = V_total - R_series * I

    # Sort by diode voltage (just for nice plotting / interpolation)
    order = np.argsort(V_diode)
    V = V_diode[order]
    I = I[order]
    df_sorted = df.iloc[order]

    # Current uncertainties (absolute)
    I_sd = _current_sd_from_ratio(df_sorted, I, current_sd_col=current_sd_col)

    # ----- Locate crossing of I_threshold -----
    idx_above = np.where(I >= I_threshold)[0]
    if len(idx_above) == 0:
        raise ValueError(f"No point reaches I >= {I_threshold:g} A in file {filename}")

    idx_cross = idx_above[0]

    # Choose a local window around the crossing
    half = window_points
    start = max(0, idx_cross - half)
    end = min(len(V), idx_cross + half + 1)
    if end - start < 2:  # guarantee at least 2 points
        start = max(0, idx_cross - 1)
        end = min(len(V), idx_cross + 2)

    V_seg = V[start:end]
    I_seg = I[start:end]
    I_sd_seg = I_sd[start:end] if I_sd is not None else None

    # ----- Weighted linear fit: I(V_D) ≈ m V_D + c -----
    m, c, sigma_m, sigma_c, cov_mc, chi2, dof = _weighted_linear_fit(
        V_seg, I_seg, y_sigma=I_sd_seg
    )

    # Solve for V_th s.t. I(V_th) = I_threshold
    V_th = (I_threshold - c) / m
    I_th = I_threshold

    # Propagate errors to V_th
    #   V_th = (I0 - c)/m
    dVth_dm = -(I_threshold - c) / (m ** 2)
    dVth_dc = -1.0 / m
    sigma_V_th2 = (
        dVth_dm ** 2 * sigma_m ** 2
        + dVth_dc ** 2 * sigma_c ** 2
        + 2.0 * dVth_dm * dVth_dc * cov_mc
    )
    sigma_V_th = float(np.sqrt(sigma_V_th2)) if np.isfinite(sigma_V_th2) else None

    # ----- Plot -----
    if make_plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))

        if show_errorbars and (I_sd is not None) and np.any(np.isfinite(I_sd)):
            ax.errorbar(V, I, yerr=I_sd, fmt=".", markersize=3, label="data")
        else:
            ax.plot(V, I, ".", label="data")

        # Linear fit segment
        V_fit = np.linspace(V_seg.min(), V_seg.max(), 100)
        I_fit = m * V_fit + c
        ax.plot(V_fit, I_fit, "--", label="local linear fit")

        # Threshold line + point
        ax.axhline(I_threshold, color="grey", linestyle=":", label=f"I_th = {I_threshold*1e3:.1f} mA")
        if sigma_V_th is not None:
            th_label = f"V_th = {V_th:.3f} ± {sigma_V_th*1e3:.2f} mV"
        else:
            th_label = f"V_th = {V_th:.3f} V"
        ax.plot(V_th, I_th, "o", color="red", label=th_label)

        ax.set_title(os.path.basename(filename), fontsize=9)
        ax.set_xlabel(r"Diode voltage $V_D$ (V)")
        ax.set_ylabel(r"$|I|$ (A)")
        ax.grid(True, linestyle=":")
        ax.legend(fontsize=7)

    extra = {
        "m": m,
        "c": c,
        "sigma_m": sigma_m,
        "sigma_c": sigma_c,
        "cov_mc": cov_mc,
        "chi2": chi2,
        "dof": dof,
        "idx_cross": idx_cross,
        "window": (start, end),
    }
    return V_th, I_th, sigma_V_th, extra


def plot_many_iv_fixed(
    filenames,
    I_threshold=2e-3,
    R_series=R_SERIES_DEFAULT,
    ncols=2,
    window_points=3,
    show_errorbars=True,
):
    """
    Make a grid of IV plots with V_th marked for each file.

    Returns a list of (filename, V_th, I_th, sigma_V_th).
    """
    n = len(filenames)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 2.4 * nrows),
        sharex=False,
        sharey=False,
    )
    axes = np.array(axes).reshape(-1)

    results = []

    for ax, fname in zip(axes, filenames):
        try:
            V_th, I_th, sigma_V_th, _ = find_threshold_at_fixed_current(
                fname,
                I_threshold=I_threshold,
                R_series=R_series,
                window_points=window_points,
                make_plot=True,
                show_errorbars=show_errorbars,
                ax=ax,
            )
            results.append((fname, V_th, I_th, sigma_V_th))
        except Exception as e:
            # If something goes wrong, annotate the subplot and continue
            ax.text(
                0.05,
                0.9,
                f"Error:\n{e}",
                transform=ax.transAxes,
                fontsize=7,
                va="top",
            )
            ax.set_title(os.path.basename(fname), fontsize=9)
            ax.set_xlabel(r"Diode voltage $V_D$ (V)")
            ax.set_ylabel(r"$|I|$ (A)")
            ax.grid(True, linestyle=":")
            results.append((fname, np.nan, np.nan, np.nan))

    # Hide any unused axes
    for j in range(len(filenames), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(
        "9.1 V Zener diode (diode voltage corrected for 100 Ω series resistor)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    return results
