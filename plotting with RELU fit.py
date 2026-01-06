

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math


# ---------- basic helpers ----------

def fit_line_weighted(x, y, sigma=None):
    """
    Weighted least-squares fit for y = m x + c.

    Parameters
    ----------
    x, y : 1D arrays
    sigma : 1D array or None
        Standard deviation of y at each point.  If None or unusable,
        falls back to unweighted fit.

    Returns
    -------
    m, c : float
        Slope and intercept of best-fit line.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Unweighted case
    if sigma is None:
        A = np.vstack([x, np.ones_like(x)]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return m, c

    sigma = np.asarray(sigma)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)

    # Not enough good points -> unweighted fit
    if mask.sum() < 2:
        A = np.vstack([x, np.ones_like(x)]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return m, c

    x = x[mask]
    y = y[mask]
    s = sigma[mask]
    w = 1.0 / (s ** 2)

    A = np.vstack([x, np.ones_like(x)]).T
    Aw = A * np.sqrt(w)[:, None]
    yw = y * np.sqrt(w)
    m, c = np.linalg.lstsq(Aw, yw, rcond=None)[0]
    return m, c


def weighted_mean(values, sigma=None):
    """
    Weighted mean of 'values' with weights 1/sigma^2.
    If sigma is None or unusable, return simple mean.
    """
    values = np.asarray(values)
    mask = np.isfinite(values)

    if sigma is None:
        return values[mask].mean()

    sigma = np.asarray(sigma)
    mask &= np.isfinite(sigma) & (sigma > 0)

    if mask.sum() == 0:
        return values[np.isfinite(values)].mean()

    v = values[mask]
    s = sigma[mask]
    w = 1.0 / (s ** 2)
    return np.sum(w * v) / np.sum(w)


# ---------- threshold finder using SD + ReLU ----------

def find_threshold_two_region_sd(
    filename,
    voltage_col="voltage/V",
    current_col="current/A",
    current_sd_col="current SD",
    baseline_points=20,
    high_points=20,
    make_plot=True,
    ax=None,
    show_errorbars=True,
):
    """
    Find breakdown 'threshold' voltage using a ReLU-style fit on linear I–V.

    Method:
    1. Sort data by V.
    2. Baseline region = first `baseline_points` at lowest V.
       -> baseline current I0 = weighted mean (using current SD if available).
    3. High-current region = last `high_points` at highest V.
       -> weighted straight-line fit I = m_high V + c_high.
    4. Threshold = intersection of baseline with that line:
           V_th = (I0 - c_high) / m_high.

    If no 'current SD' column is found, or SDs are NaN, the fit reverts
    to unweighted least squares.

    Returns
    -------
    V_th : float
        Threshold voltage.
    I_th : float
        Threshold current (≈ I0).
    (I0, m_high, c_high) : tuple
        Baseline current and high-region line parameters.
    """

    df = pd.read_csv(filename).dropna(subset=[voltage_col, current_col])

    V = df[voltage_col].values
    I = np.abs(df[current_col].values)
    I_sd = df[current_sd_col].values if current_sd_col in df.columns else None

    # sort by V
    order = np.argsort(V)
    V = V[order]
    I = I[order]
    if I_sd is not None:
        I_sd = I_sd[order]

    n = len(V)
    baseline_points = min(baseline_points, n // 2)
    high_points = min(high_points, n // 2)

    # ---- baseline (low-V) region ----
    V_low = V[:baseline_points]
    I_low = I[:baseline_points]
    I_sd_low = I_sd[:baseline_points] if I_sd is not None else None
    I0 = weighted_mean(I_low, I_sd_low)

    # ---- high-current (high-V) region ----
    V_high = V[-high_points:]
    I_high = I[-high_points:]
    I_sd_high = I_sd[-high_points:] if I_sd is not None else None
    m_high, c_high = fit_line_weighted(V_high, I_high, I_sd_high)

    # intersection
    V_th = (I0 - c_high) / m_high
    I_th = I0

    # ---- plotting ----
    if make_plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))

        if show_errorbars and (I_sd is not None) and np.any(np.isfinite(I_sd)):
            ax.errorbar(V, I, yerr=I_sd, fmt=".", markersize=3, label="data")
        else:
            ax.plot(V, I, ".", label="data")

        V_fit = np.linspace(V.min(), V.max(), 300)
        I_fit = np.where(V_fit <= V_th, I0, m_high * V_fit + c_high)
        ax.plot(V_fit, I_fit, "--", label="ReLU fit")
        ax.plot(V_th, I_th, "o", color="red", label=f"V_th ≈ {V_th:.3f} V")

        ax.set_title(os.path.basename(filename), fontsize=9)
        ax.set_xlabel("Voltage V (V)")
        ax.set_ylabel("|I| (A)")
        ax.grid(True, ls=":")
        ax.legend(fontsize=7)

    return V_th, I_th, (I0, m_high, c_high)


# ---------- helper to plot many files in subplots ----------

def plot_many_iv(files, ncols=3,
                 baseline_points=20, high_points=20,
                 sharey=False):
    """
    Make a grid of subplots with ReLU fits for an arbitrary list of files.

    Parameters
    ----------
    files : list of str
        CSV file paths.
    ncols : int
        Number of columns in the subplot grid.
    baseline_points, high_points : int
        Passed down to find_threshold_two_region_sd.
    sharey : bool
        If True, all subplots share the same y-axis.
    """
    n = len(files)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 3.5 * nrows),
        squeeze=False,
        sharey=sharey
    )
    axes = axes.ravel()

    results = []
    for i, fname in enumerate(files):
        ax = axes[i]
        V_th, I_th, params = find_threshold_two_region_sd(
            fname,
            baseline_points=baseline_points,
            high_points=high_points,
            make_plot=True,
            ax=ax,
        )
        results.append((os.path.basename(fname), V_th, I_th))

    # remove any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    return results


# ---------- Example usage ----------

if __name__ == "__main__":

    # Single file demo
    V_th, I_th, params = find_threshold_two_region_sd(
        "9.1 V zener diode at 160.7-155.5K.csv"
    )
    print(f"Single file: V_th = {V_th:.4f} V, I_th ≈ {I_th:.3e} A")

    # Multi-file subplot demo
    files = [
        "9.1 V zener diode at 124-125.4K.csv",
        "9.1 V zener diode at 160.7-155.5K.csv",
        "9.1 V zener diode at 217-212K.csv",
        "9.1 V zener diode at 218.5-214.3K.csv",
        "9.1 V zener diode at 244.5-246.5K.csv",
        "9.1 V zener diode at 244.7-246K.csv",
        "9.1 V zener diode at 273.7-273.8K.csv",
        "9.1 V zener diode at 293.4-293.9K.csv",
        "9.1 V zener diode at 306.7-305K.csv",
        "9.1 V zener diode at 309-307.8K.csv",
        # add more filenames here...
    ]
    results = plot_many_iv(files, ncols=2)
    for name, V_th, I_th in results:
        print(f"{name:40s}  V_th = {V_th:.4f} V,  I_th ≈ {I_th:.3e} A")

