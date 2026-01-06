# threshold_utils.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math


# ---------- helper for SD as ratio ----------

def _current_sd_from_ratio(df, I, current_sd_col="current SD"):
    """
    Interpret the 'current SD' column as a *relative* standard deviation
    (dimensionless ratio), common to all points.

    We:
    - look for finite values in that column,
    - take their median as the ratio r,
    - return absolute SD = r * |I| for every point.

    If the column is missing or has no finite entries, return None.
    """
    if current_sd_col not in df.columns:
        return None

    raw = df[current_sd_col].values
    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        return None

    r = np.median(finite)      # relative SD
    return r * np.abs(I)       # absolute SD for each point


# ---------- threshold at fixed current via interpolation ----------

def find_threshold_at_fixed_current(
    filename,
    I_threshold=2e-3,             # 2 mA
    voltage_col="voltage/V",
    current_col="current/A",
    current_sd_col="current SD",   # interpreted as relative SD
    make_plot=True,
    ax=None,
    show_errorbars=True,
):
    """
    Find breakdown voltage V_th by specifying a fixed threshold current I_threshold,
    using *linear interpolation between consecutive data points*.

    Also propagate the current uncertainties of the two points used in the
    interpolation to obtain an uncertainty sigma_Vth.

    Returns
    -------
    V_th : float
        Voltage at the fixed threshold current.
    I_th : float
        Equal to I_threshold.
    V_th_sigma : float or None
        1-sigma uncertainty on V_th (None if we cannot estimate it).
    (k1, k2) : tuple of ints
        Indices of the two points used for interpolation.
    """

    # ---- load & sort data ----
    df = pd.read_csv(filename).dropna(subset=[voltage_col, current_col])

    V = df[voltage_col].values
    I = np.abs(df[current_col].values)

    # SD is a relative ratio → convert to absolute SD for each point
    I_sd = _current_sd_from_ratio(df, I, current_sd_col=current_sd_col)

    order = np.argsort(V)
    V = V[order]
    I = I[order]
    if I_sd is not None:
        I_sd = I_sd[order]

    n = len(V)

    # ---- find first crossing of I_threshold ----
    idx_above = np.where(I >= I_threshold)[0]
    if len(idx_above) == 0:
        raise ValueError(
            f"{os.path.basename(filename)}: current never reaches "
            f"{I_threshold:.3e} A; choose a smaller I_threshold."
        )
    k = idx_above[0]

    # choose two points bracketing the threshold
    if k == 0:
        k1, k2 = 0, 1
    else:
        k1, k2 = k - 1, k

    V1, I1 = V[k1], I[k1]
    V2, I2 = V[k2], I[k2]

    if I2 == I1:
        raise ValueError(
            f"{os.path.basename(filename)}: two consecutive points at indices "
            f"{k1},{k2} have identical current; cannot interpolate."
        )

    # linear interpolation on the segment [k1, k2]
    V_th = V1 + (I_threshold - I1) * (V2 - V1) / (I2 - I1)
    I_th = I_threshold

    # ---- propagate uncertainty from I1, I2 to V_th ----
    V_th_sigma = None
    if I_sd is not None:
        sigma1 = I_sd[k1]
        sigma2 = I_sd[k2]

        denom_sq = (I1 - I2) ** 2
        dVdI1 = (I2 - I_th) * (V1 - V2) / denom_sq
        dVdI2 = -(I1 - I_th) * (V1 - V2) / denom_sq

        V_th_sigma = np.sqrt((dVdI1 * sigma1) ** 2 + (dVdI2 * sigma2) ** 2)

    # ---- plotting ----
    if make_plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))

        # data + errorbars if available
        if show_errorbars and (I_sd is not None):
            ax.errorbar(V, I, yerr=I_sd, fmt=".", markersize=3, label="data")
        else:
            ax.plot(V, I, ".", label="data")

        # short segment used for interpolation
        ax.plot([V1, V2], [I1, I2], "--", label="interpolation segment")

        # horizontal line at threshold current
        ax.axhline(I_threshold, color="grey", linestyle=":",
                   label=f"I_th = {I_threshold*1e3:.1f} mA")

        # label with or without uncertainty
        if V_th_sigma is not None:
            label_th = (f"V_th ≈ {V_th:.3f} V"
                        f" ± {V_th_sigma*1e3:.2f} mV")
        else:
            label_th = f"V_th ≈ {V_th:.3f} V"

        # marker at threshold voltage
        ax.plot(V_th, I_th, "o", color="red", label=label_th)

        ax.set_title(os.path.basename(filename), fontsize=9)
        ax.set_xlabel("Voltage V (V)")
        ax.set_ylabel("|I| (A)")
        ax.grid(True, ls=":")
        ax.legend(fontsize=7)

    return V_th, I_th, V_th_sigma, (k1, k2)


# ---------- multi-file subplot helper ----------

def plot_many_iv_fixed(
    files,
    I_threshold=2e-3,
    ncols=3,
    sharey=False,
):
    """
    Make a grid of subplots for an arbitrary list of I–V files,
    using the fixed-current threshold + interpolation method.
    """
    n = len(files)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 3.5 * nrows),
        squeeze=False,
        sharey=sharey,
    )
    axes = axes.ravel()

    results = []
    for i, fname in enumerate(files):
        ax = axes[i]
        V_th, I_th, V_th_sigma, _ = find_threshold_at_fixed_current(
            fname,
            I_threshold=I_threshold,
            make_plot=True,
            ax=ax,
        )
        results.append((os.path.basename(fname), V_th, V_th_sigma, I_th))

    # remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout(pad=1.6, h_pad=2.0, w_pad=1.2)
    plt.show()
    return results
