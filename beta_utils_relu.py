# beta_utils.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from relu_utils import find_threshold_two_region_relu, _current_sd_from_ratio


# ---------- basic I–V loader ----------

def load_iv_file(
    filename,
    voltage_col="voltage/V",
    current_col="current/A",
    current_sd_col="current SD",
    convert_to_abs=True,
    R_series=0.0,
):
    """
    Load an I–V CSV, sort by voltage, and return V, |I|, sigma_I.

    sigma_I is computed from the 'current SD' column, interpreted as a
    relative standard deviation (ratio), using the same routine as in
    relu_utils._current_sd_from_ratio.
    """
    df = pd.read_csv(filename).dropna(subset=[voltage_col, current_col])

    V = df[voltage_col].values
    I = np.abs(df[current_col].values)

    if current_sd_col in df.columns:
        I_sd = _current_sd_from_ratio(df, I, current_sd_col)
    else:
        I_sd = None

    order = np.argsort(V)
    V = V[order]
    I = I[order]
    if I_sd is not None:
        I_sd = I_sd[order]

    return V, I, I_sd


# ---------- generic weighted straight-line fit ----------

def weighted_linear_fit(x, y, sigma_y=None):
    """
    Weighted least-squares fit of y = m x + c.

    Returns
    -------
    m, c : float
        Best-fit slope and intercept.
    sigma_m, sigma_c : float
        1-sigma uncertainties on m and c.
    chi2, dof : float
        Chi-square and degrees of freedom.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if sigma_y is None:
        # unweighted fit
        A = np.vstack([x, np.ones_like(x)]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]

        residuals = y - (m * x + c)
        dof = len(x) - 2
        s2 = np.sum(residuals**2) / dof if dof > 0 else 0.0
        cov = np.linalg.inv(A.T @ A) * s2
        sigma_m = np.sqrt(cov[0, 0])
        sigma_c = np.sqrt(cov[1, 1])
        chi2 = np.sum((residuals**2) / s2) if s2 > 0 else 0.0
        return m, c, sigma_m, sigma_c, chi2, dof

    sigma_y = np.asarray(sigma_y, dtype=float)

    # mask bad / non-positive sigmas
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma_y) & (sigma_y > 0)
    if mask.sum() < 2:
        # fallback: unweighted
        return weighted_linear_fit(x, y, sigma_y=None)

    x = x[mask]
    y = y[mask]
    s = sigma_y[mask]

    w = 1.0 / (s**2)
    A = np.vstack([x, np.ones_like(x)]).T
    Aw = A * np.sqrt(w)[:, None]
    yw = y * np.sqrt(w)

    m, c = np.linalg.lstsq(Aw, yw, rcond=None)[0]

    residuals = y - (m * x + c)
    dof = len(x) - 2
    chi2 = np.sum(w * residuals**2)
    s2 = chi2 / dof if dof > 0 else 0.0
    cov = np.linalg.inv(Aw.T @ Aw) * s2
    sigma_m = np.sqrt(cov[0, 0])
    sigma_c = np.sqrt(cov[1, 1])

    return m, c, sigma_m, sigma_c, chi2, dof

def best_linear_window(x, y, sigma_y=None, npts=10):
    """
    Pick the *most linear* contiguous window (in x-sorted order) with at least
    npts points, by minimizing chi2/dof (weighted) or SSE/dof (unweighted).

    Returns
    -------
    idx_best : np.ndarray (int)
        Indices into the ORIGINAL input arrays (x, y, sigma_y) selecting the
        best window.
    m, c, sigma_m, sigma_c, chi2, dof : floats
        Fit results for the best window.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    idx0 = np.arange(len(x))

    # basic validity
    valid = np.isfinite(x) & np.isfinite(y)
    if sigma_y is not None:
        sigma_y = np.asarray(sigma_y, dtype=float)
        valid = valid & np.isfinite(sigma_y) & (sigma_y > 0)

    x = x[valid]
    y = y[valid]
    idx0 = idx0[valid]
    if sigma_y is not None:
        sigma_y = sigma_y[valid]

    # sort by x so "contiguous window" makes sense
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    idx0 = idx0[order]
    if sigma_y is not None:
        sigma_y = sigma_y[order]

    N = len(x)
    if N < npts:
        # not enough points → just fit everything we have
        m, c, sigma_m, sigma_c, chi2, dof = weighted_linear_fit(x, y, sigma_y=sigma_y)
        return idx0, m, c, sigma_m, sigma_c, chi2, dof

    best = None

    for start in range(0, N - npts + 1):
        for end in range(start + npts, N + 1):
            xs = x[start:end]
            ys = y[start:end]
            sigs = sigma_y[start:end] if sigma_y is not None else None

            m, c, sigma_m, sigma_c, chi2, dof = weighted_linear_fit(xs, ys, sigma_y=sigs)
            if dof <= 0:
                continue

            # For avalanche: expect m < 0 in ln|I| vs 1/V
            if m >= 0:
                continue

            # Score: weighted chi2/dof if possible; otherwise SSE/dof
            if sigs is None:
                resid = ys - (m * xs + c)
                score = np.sum(resid**2) / dof
            else:
                score = (chi2 / dof) if np.isfinite(chi2) else np.inf

            # Choose smallest score; if tied, prefer longer window
            nwin = end - start
            if (best is None) or (score < best["score"] - 1e-12) or (
                abs(score - best["score"]) < 1e-12 and nwin > best["nwin"]
            ):
                best = dict(
                    score=score, start=start, end=end, nwin=nwin,
                    m=m, c=c, sigma_m=sigma_m, sigma_c=sigma_c, chi2=chi2, dof=dof
                )

    if best is None:
        # fallback: fit everything
        m, c, sigma_m, sigma_c, chi2, dof = weighted_linear_fit(x, y, sigma_y=sigma_y)
        return idx0, m, c, sigma_m, sigma_c, chi2, dof

    idx_best = idx0[best["start"]:best["end"]]
    return idx_best, best["m"], best["c"], best["sigma_m"], best["sigma_c"], best["chi2"], best["dof"]

# ---------- choose breakdown region ----------

def select_breakdown_region(I, I_th, k_low=3.0, k_high=20.0, frac_max=0.8):
    """
    Select indices corresponding to the 'clean' breakdown region.

    We keep only points with currents between:

        I_min = k_low * I_th
        I_max = min(k_high * I_th, frac_max * I_max_all)

    This avoids:
        - very low currents (just before breakdown, strong curvature),
        - very high currents where series resistance makes the curve flatten.

    Returns a boolean mask over the array I.
    """
    I = np.asarray(I, dtype=float)
    I_min = k_low * I_th
    I_max_all = np.max(I)
    I_max = min(k_high * I_th, frac_max * I_max_all)

    if I_max <= I_min:
        # fallback: just keep points above I_min
        mask = I >= I_min
        return mask

    mask = (I >= I_min) & (I <= I_max)
    return mask


# ---------- avalanche fit (9.1 V diode) ----------

def fit_avalanche_beta(
    filename,
    I_threshold=2e-3,
    k_low=3.0,
    k_high=20.0,
    frac_max=0.8,
    npts=10,              # NEW: minimum points used in the best linear window
    auto_window=True,     # NEW: turn best-window selection on/off
    make_plot=True,
    ax=None,
):
    """
    For the 9.1 V avalanche diode.

    Steps:
    1. Load I–V data, find V_th at fixed I_threshold.
    2. Select points in the breakdown region via select_breakdown_region().
    3. Construct x = 1/V, y = ln|I|.
    4. Optionally pick the most linear contiguous window in (x,y).
    5. Fit y = m x + c  =>  beta_A = -m.

    Returns
    -------
    beta_A : float
    sigma_beta_A : float
    m, c, chi2, dof : floats
    """
    V, I, I_sd = load_iv_file(filename)

    # threshold at fixed current (also defines I_th)
    # Threshold from ReLU-style 2-region fit (V_th is the intersection voltage).
    # For breakdown-region selection we keep using I_threshold as the current scale by default,
    # so results are comparable to the fixed-current method. Set I_threshold=None to use the ReLU baseline.
    V_th, I_th_relu, _ = find_threshold_two_region_relu(
        filename,
        make_plot=False,
    )

    I_th = I_th_relu if (I_threshold is None) else I_threshold

    mask = select_breakdown_region(I, I_th, k_low=k_low, k_high=k_high, frac_max=frac_max)

    # ensure enough points; fallback if necessary
    if mask.sum() < 4:
        mask = I >= (k_low * I_th)

    V_fit = V[mask]
    I_fit = I[mask]

    # Build y uncertainties for ln(I): sigma_y = sigma_I / I
    if I_sd is not None:
        I_sd_fit = I_sd[mask]
        sigma_y = I_sd_fit / I_fit
    else:
        sigma_y = None

    # Transform for avalanche model
    x_all = 1.0 / V_fit
    y_all = np.log(I_fit)

    # --- NEW: pick best linear window ---
    if auto_window:
        idx_best, m, c, sigma_m, sigma_c, chi2, dof = best_linear_window(
            x_all, y_all, sigma_y=sigma_y, npts=npts
        )
        x = x_all[idx_best]
        y = y_all[idx_best]
        sigma_y_best = sigma_y[idx_best] if sigma_y is not None else None
    else:
        x = x_all
        y = y_all
        sigma_y_best = sigma_y
        m, c, sigma_m, sigma_c, chi2, dof = weighted_linear_fit(x, y, sigma_y=sigma_y_best)

    beta_A = -m
    sigma_beta_A = sigma_m

    # plotting
    if make_plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))

        # plot all candidate breakdown points faintly
        ax.errorbar(
            x_all,
            y_all,
            yerr=sigma_y,
            fmt="o",
            markersize=3,
            capsize=2,
            alpha=0.35,
            label="candidate region",
        )

        # highlight chosen best window
        ax.errorbar(
            x,
            y,
            yerr=sigma_y_best,
            fmt="o",
            markersize=3,
            capsize=2,
            label=f"best window (n={len(x)})",
        )

        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = m * x_line + c
        ax.plot(x_line, y_line, "--", label=f"fit: ln I = {m:.3e} x + {c:.3f}")

        ax.set_xlabel("1 / V (1/V)")
        ax.set_ylabel("ln |I|")
        ax.set_title(os.path.basename(filename))
        ax.grid(True, ls=":")
        ax.legend(fontsize=7)

    return beta_A, sigma_beta_A, m, c, chi2, dof



# ---------- Zener fit (2.7 V diode) ----------

def fit_zener_beta(
    filename,
    I_threshold=2e-3,
    k_low=3.0,
    k_high=20.0,
    frac_max=0.8,
    make_plot=True,
    ax=None,
):
    """
    For the 2.7 V Zener diode.

    Steps:
    1. Load I–V data, find V_th at fixed I_threshold.
    2. Select points in the breakdown region via select_breakdown_region().
    3. Construct x = 1/V, y = ln(|I| / V^2).
    4. Weighted linear fit y = m x + c  =>  beta_Z = -m.

    Returns
    -------
    beta_Z : float
        Effective Zener parameter (beta).
    sigma_beta_Z : float
        1-sigma uncertainty on beta_Z.
    m, c, chi2, dof : floats
        Full fit information.
    """
    V, I, I_sd = load_iv_file(filename)

    # Threshold from ReLU-style 2-region fit (V_th is the intersection voltage).
    # For breakdown-region selection we keep using I_threshold as the current scale by default,
    # so results are comparable to the fixed-current method. Set I_threshold=None to use the ReLU baseline.
    V_th, I_th_relu, _ = find_threshold_two_region_relu(
        filename,
        make_plot=False,
    )

    I_th = I_th_relu if (I_threshold is None) else I_threshold

    mask = select_breakdown_region(I, I_th, k_low=k_low, k_high=k_high, frac_max=frac_max)

    if mask.sum() < 4:
        mask = I >= (k_low * I_th)

    V_fit = V[mask]
    I_fit = I[mask]
    if I_sd is not None:
        I_sd_fit = I_sd[mask]
        sigma_y = I_sd_fit / I_fit
    else:
        sigma_y = None

    x = 1.0 / V_fit
    # ln(I / V^2) = ln I - 2 ln V
    y = np.log(I_fit) - 2.0 * np.log(V_fit)

    m, c, sigma_m, sigma_c, chi2, dof = weighted_linear_fit(x, y, sigma_y=sigma_y)

    beta_Z = -m
    sigma_beta_Z = sigma_m

    if make_plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))

        ax.errorbar(
            x,
            y,
            yerr=sigma_y,
            fmt="o",
            markersize=3,
            capsize=2,
            label="data",
        )

        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = m * x_line + c
        ax.plot(x_line, y_line, "--", label=f"fit: ln(I/V^2) = {m:.3e} x + {c:.3f}")

        ax.set_xlabel("1 / V (1/V)")
        ax.set_ylabel(r"ln (|I| / V$^2$)")
        ax.set_title(os.path.basename(filename))
        ax.grid(True, ls=":")
        ax.legend(fontsize=7)

    return beta_Z, sigma_beta_Z, m, c, chi2, dof
