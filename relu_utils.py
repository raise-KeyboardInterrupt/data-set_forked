# relu_utils.py
"""
Utilities for estimating Zener breakdown 'threshold' via a 2-region ReLU-style fit.

This is designed to be *importable*, similar to threshold_utils.py.

Key idea (piecewise model on |I| vs V):
- Low-voltage region: current is approximately constant (baseline I0).
- High-voltage region: current increases approximately linearly: I = m*V + c.
- Threshold V_th is the intersection of those two regimes:
      V_th = (I0 - c) / m

The CSV files used in this project typically store 'current SD' as a *relative*
standard deviation (dimensionless ratio) recorded once. We interpret it as:
      sigma_I = r * |I|
where r = median finite value in the 'current SD' column.
"""

from __future__ import annotations

import os
import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- helper: convert "current SD" ratio to absolute sigma_I ----------

def _current_sd_from_ratio(
    df: pd.DataFrame,
    I_abs: np.ndarray,
    current_sd_col: str = "current SD",
) -> Optional[np.ndarray]:
    """
    Interpret df[current_sd_col] as a *relative SD ratio* r (dimensionless),
    then return absolute SD array sigma_I = r * |I|.

    Returns None if the column is missing or has no finite values.
    """
    if current_sd_col not in df.columns:
        return None

    raw = np.asarray(df[current_sd_col].values, dtype=float)
    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        return None

    r = float(np.median(finite))
    return r * np.abs(I_abs)


# ---------- basic helpers ----------

def fit_line_weighted(
    x: np.ndarray,
    y: np.ndarray,
    sigma: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Weighted least-squares fit for y = m x + c.
    If sigma is None or unusable, falls back to an unweighted fit.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if sigma is None:
        A = np.vstack([x, np.ones_like(x)]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(m), float(c)

    sigma = np.asarray(sigma, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)

    if mask.sum() < 2:
        A = np.vstack([x, np.ones_like(x)]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(m), float(c)

    x_m = x[mask]
    y_m = y[mask]
    s_m = sigma[mask]
    w = 1.0 / (s_m ** 2)

    A = np.vstack([x_m, np.ones_like(x_m)]).T
    Aw = A * np.sqrt(w)[:, None]
    yw = y_m * np.sqrt(w)
    m, c = np.linalg.lstsq(Aw, yw, rcond=None)[0]
    return float(m), float(c)


def weighted_mean(values: np.ndarray, sigma: Optional[np.ndarray] = None) -> float:
    """
    Weighted mean with weights 1/sigma^2.
    If sigma is None or unusable, return the simple mean.
    """
    v = np.asarray(values, dtype=float)
    mask = np.isfinite(v)

    if sigma is None:
        return float(v[mask].mean())

    s = np.asarray(sigma, dtype=float)
    mask = mask & np.isfinite(s) & (s > 0)

    if mask.sum() == 0:
        return float(v[np.isfinite(v)].mean())

    v_m = v[mask]
    s_m = s[mask]
    w = 1.0 / (s_m ** 2)
    return float(np.sum(w * v_m) / np.sum(w))


def relu_piecewise(V: np.ndarray, V_th: float, I0: float, m: float, c: float) -> np.ndarray:
    """
    Piecewise ReLU-style model on current:
      I(V) = I0                 for V <= V_th
           = m*V + c            for V >  V_th
    """
    V = np.asarray(V, dtype=float)
    return np.where(V <= V_th, I0, m * V + c)


# ---------- threshold finder using baseline + high-V line (ReLU-style) ----------

def find_threshold_two_region_relu(
    filename: str,
    voltage_col: str = "voltage/V",
    current_col: str = "current/A",
    current_sd_col: str = "current SD",   # interpreted as relative SD ratio
    baseline_points: int = 20,
    high_points: int = 20,
    make_plot: bool = True,
    ax: Optional[plt.Axes] = None,
    show_errorbars: bool = True,
) -> Tuple[float, float, Tuple[float, float, float]]:
    """
    Find breakdown 'threshold' voltage using a 2-region ReLU-style fit on |I| vs V.

    Steps
    -----
    1) Sort data by V.
    2) Baseline region: first `baseline_points` points at lowest V.
       -> baseline current I0 = weighted mean (if SD available).
    3) High-V region: last `high_points` points at highest V.
       -> weighted line fit I = m*V + c.
    4) Threshold = intersection of baseline with that line:
           V_th = (I0 - c) / m

    Returns
    -------
    V_th : float
        Threshold voltage (intersection).
    I_th : float
        Threshold current (≈ I0).
    (I0, m, c) : tuple
        Baseline current and high-region line parameters.
    """
    df = pd.read_csv(filename).dropna(subset=[voltage_col, current_col])

    V = np.asarray(df[voltage_col].values, dtype=float)
    I = np.abs(np.asarray(df[current_col].values, dtype=float))

    I_sd = _current_sd_from_ratio(df, I, current_sd_col=current_sd_col)

    # sort by V
    order = np.argsort(V)
    V = V[order]
    I = I[order]
    if I_sd is not None:
        I_sd = I_sd[order]

    n = len(V)
    if n < 4:
        raise ValueError(f"{os.path.basename(filename)}: not enough points ({n}) for a 2-region fit.")

    baseline_points = int(min(max(2, baseline_points), n // 2))
    high_points = int(min(max(2, high_points), n // 2))

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

    if not np.isfinite(m_high) or abs(m_high) < 1e-30:
        raise ValueError(f"{os.path.basename(filename)}: high-region slope is ~0; cannot find intersection.")

    V_th = (I0 - c_high) / m_high
    I_th = I0

    # ---- plotting ----
    if make_plot:
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 5))

        if show_errorbars and (I_sd is not None) and np.any(np.isfinite(I_sd)) and np.any(I_sd > 0):
            ax.errorbar(V, I, yerr=I_sd, fmt=".", markersize=3, label="data")
        else:
            ax.plot(V, I, ".", label="data")

        V_fit = np.linspace(np.nanmin(V), np.nanmax(V), 300)
        I_fit = relu_piecewise(V_fit, V_th, I0, m_high, c_high)
        ax.plot(V_fit, I_fit, "--", label="ReLU fit")

        ax.plot(V_th, I_th, "o", color="red", label=f"V_th ≈ {V_th:.3f} V")

        ax.set_title(os.path.basename(filename), fontsize=9)
        ax.set_xlabel("Voltage V (V)")
        ax.set_ylabel("|I| (A)")
        ax.grid(True, ls=":")
        ax.legend(fontsize=7)

    return float(V_th), float(I_th), (float(I0), float(m_high), float(c_high))


# ---------- helper: plot many files in subplots ----------

def plot_many_iv_relu(
    files: Sequence[str],
    ncols: int = 3,
    baseline_points: int = 20,
    high_points: int = 20,
    sharey: bool = False,
) -> List[Tuple[str, float, float]]:
    """
    Make a grid of subplots with ReLU fits for a list of CSV files.

    Returns a list of (basename, V_th, I_th).
    """
    n = len(files)
    if n == 0:
        return []

    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 3.5 * nrows),
        squeeze=False,
        sharey=sharey,
    )
    axes = axes.ravel()

    results: List[Tuple[str, float, float]] = []

    for i, fname in enumerate(files):
        ax = axes[i]
        V_th, I_th, _ = find_threshold_two_region_relu(
            fname,
            baseline_points=baseline_points,
            high_points=high_points,
            make_plot=True,
            ax=ax,
        )
        results.append((os.path.basename(fname), V_th, I_th))

    # remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout(pad=1.6, h_pad=2.0, w_pad=1.2)
    plt.show()
    return results


# ---------- optional: uncertainty estimate for V_th (for V_th vs T plots) ----------

def _weighted_mean_and_var(
    values: np.ndarray,
    sigma: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Return (mean, var_of_mean) with optional weights 1/sigma^2."""
    v = np.asarray(values, dtype=float)
    mask = np.isfinite(v)

    # ----- unweighted -----
    if sigma is None:
        v_m = v[mask]
        if v_m.size == 0:
            return float("nan"), float("nan")
        mean = float(v_m.mean())
        dof = v_m.size - 1
        if dof <= 0:
            return mean, 0.0
        s2 = float(np.sum((v_m - mean) ** 2) / dof)  # sample variance
        var_mean = s2 / v_m.size
        return mean, float(var_mean)

    # ----- weighted -----
    s = np.asarray(sigma, dtype=float)
    mask = mask & np.isfinite(s) & (s > 0)
    if mask.sum() == 0:
        # fallback to unweighted
        return _weighted_mean_and_var(v, None)

    v_m = v[mask]
    s_m = s[mask]
    w = 1.0 / (s_m ** 2)

    mean = float(np.sum(w * v_m) / np.sum(w))
    dof = v_m.size - 1
    if dof <= 0:
        return mean, float(1.0 / np.sum(w))

    # inflate uncertainty if residuals are larger than expected
    chi2 = float(np.sum(w * (v_m - mean) ** 2))
    s2 = chi2 / dof
    var_mean = s2 / np.sum(w)
    return mean, float(var_mean)


def _line_fit_and_cov(
    x: np.ndarray,
    y: np.ndarray,
    sigma: Optional[np.ndarray] = None,
) -> Tuple[float, float, np.ndarray, float, int]:
    """Fit y=m x + c and return (m, c, cov(2x2), chi2, dof)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)

    # ----- unweighted -----
    if sigma is None:
        x_m = x[mask]
        y_m = y[mask]
        A = np.vstack([x_m, np.ones_like(x_m)]).T
        m, c = np.linalg.lstsq(A, y_m, rcond=None)[0]
        residuals = y_m - (m * x_m + c)
        dof = int(len(x_m) - 2)
        chi2 = float(np.sum(residuals ** 2))
        s2 = chi2 / dof if dof > 0 else 0.0
        cov = np.linalg.inv(A.T @ A) * s2 if len(x_m) >= 2 else np.full((2, 2), np.nan)
        return float(m), float(c), cov, chi2, dof

    # ----- weighted -----
    s = np.asarray(sigma, dtype=float)
    mask = mask & np.isfinite(s) & (s > 0)
    if mask.sum() < 2:
        return _line_fit_and_cov(x, y, None)

    x_m = x[mask]
    y_m = y[mask]
    s_m = s[mask]
    w = 1.0 / (s_m ** 2)

    A = np.vstack([x_m, np.ones_like(x_m)]).T
    Aw = A * np.sqrt(w)[:, None]
    yw = y_m * np.sqrt(w)
    m, c = np.linalg.lstsq(Aw, yw, rcond=None)[0]

    residuals = y_m - (m * x_m + c)
    chi2 = float(np.sum(w * residuals ** 2))
    dof = int(len(x_m) - 2)
    s2 = chi2 / dof if dof > 0 else 0.0
    cov = np.linalg.inv(Aw.T @ Aw) * s2
    return float(m), float(c), cov, chi2, dof


def find_threshold_two_region_relu_with_uncertainty(
    filename: str,
    voltage_col: str = "voltage/V",
    current_col: str = "current/A",
    current_sd_col: str = "current SD",
    baseline_points: int = 20,
    high_points: int = 20,
    make_plot: bool = True,
    ax: Optional[plt.Axes] = None,
    show_errorbars: bool = True,
) -> Tuple[float, float, Optional[float], Tuple[int, int]]:
    """
    Same ReLU-style threshold as `find_threshold_two_region_relu`, but also estimates
    a 1-sigma uncertainty on V_th by propagating the uncertainties of:
      - the baseline current mean I0 (from the low-V region)
      - the (m, c) line fit (from the high-V region)

    Returns
    -------
    V_th : float
    I_th : float
    V_th_sigma : float or None
    (baseline_points_used, high_points_used) : tuple
        Returned to mirror the 4-value return style of threshold_utils.
    """
    df = pd.read_csv(filename).dropna(subset=[voltage_col, current_col])

    V = np.asarray(df[voltage_col].values, dtype=float)
    I = np.abs(np.asarray(df[current_col].values, dtype=float))
    I_sd = _current_sd_from_ratio(df, I, current_sd_col=current_sd_col)

    order = np.argsort(V)
    V = V[order]
    I = I[order]
    if I_sd is not None:
        I_sd = I_sd[order]

    n = len(V)
    if n < 4:
        raise ValueError(f"{os.path.basename(filename)}: not enough points ({n}) for a 2-region fit.")

    bp = int(min(max(2, baseline_points), n // 2))
    hp = int(min(max(2, high_points), n // 2))

    V_low = V[:bp]
    I_low = I[:bp]
    I_sd_low = I_sd[:bp] if I_sd is not None else None

    V_high = V[-hp:]
    I_high = I[-hp:]
    I_sd_high = I_sd[-hp:] if I_sd is not None else None

    # Baseline current and its variance
    I0, var_I0 = _weighted_mean_and_var(I_low, I_sd_low)

    # High-V line fit and covariance
    m, c, cov_mc, _, _ = _line_fit_and_cov(V_high, I_high, I_sd_high)

    if not np.isfinite(m) or abs(m) < 1e-30:
        raise ValueError(f"{os.path.basename(filename)}: high-region slope is ~0; cannot find intersection.")

    V_th = (I0 - c) / m
    I_th = I0

    # Uncertainty propagation: V_th = (I0 - c)/m
    V_th_sigma: Optional[float] = None
    try:
        var_m = float(cov_mc[0, 0])
        var_c = float(cov_mc[1, 1])
        cov_m_c = float(cov_mc[0, 1])

        if np.isfinite(var_I0) and np.isfinite(var_m) and np.isfinite(var_c) and np.isfinite(cov_m_c):
            var_V = (var_I0 + (V_th ** 2) * var_m + var_c + 2.0 * V_th * cov_m_c) / (m ** 2)
            if np.isfinite(var_V) and var_V >= 0:
                V_th_sigma = float(np.sqrt(var_V))
    except Exception:
        V_th_sigma = None

    # Plot (same style as before, but add uncertainty if we have it)
    if make_plot:
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 5))

        if show_errorbars and (I_sd is not None) and np.any(np.isfinite(I_sd)) and np.any(I_sd > 0):
            ax.errorbar(V, I, yerr=I_sd, fmt=".", markersize=3, label="data")
        else:
            ax.plot(V, I, ".", label="data")

        V_fit = np.linspace(np.nanmin(V), np.nanmax(V), 300)
        I_fit = relu_piecewise(V_fit, V_th, I0, m, c)
        ax.plot(V_fit, I_fit, "--", label="ReLU fit")

        if V_th_sigma is not None:
            label = f"V_th ≈ {V_th:.3f} V ± {V_th_sigma*1e3:.2f} mV"
        else:
            label = f"V_th ≈ {V_th:.3f} V"
        ax.plot(V_th, I_th, "o", color="red", label=label)

        ax.set_title(os.path.basename(filename), fontsize=9)
        ax.set_xlabel("Voltage V (V)")
        ax.set_ylabel("|I| (A)")
        ax.grid(True, ls=":")
        ax.legend(fontsize=7)

    return float(V_th), float(I_th), V_th_sigma, (bp, hp)
