# plot_beta_9V.py

import numpy as np
import matplotlib.pyplot as plt

from beta_utils import fit_avalanche_beta
from temp_utils import parse_temperature_from_filename


FILES_9V = [
    "9.1 V zener diode at 124-125.4K.csv",
    "9.1 V zener diode at 160.7-155.5K.csv",
    #"9.1 V zener diode at 217-212K.csv",
    "9.1 V zener diode at 218.5-214.3K.csv",
    #"9.1 V zener diode at 244.5-246.5K.csv",
    #"9.1 V zener diode at 244.7-246K.csv",
    "9.1 V zener diode at 273.7-273.8K.csv",
    "9.1 V zener diode at 293.4-293.9K.csv",
    #"9.1 V zener diode at 306.7-305K.csv",
    "9.1 V zener diode at 309-307.8K.csv",
]

I_THRESHOLD = 2e-3  # 2 mA


def main():
    # ---- per-file ln|I| vs 1/V plots in a grid ----
    n = len(FILES_9V)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = axes.ravel()

    T_mean_list = []
    T_err_list = []
    beta_list = []
    beta_err_list = []
    chi2dof_list = []
    dof_list = []

    for i, fname in enumerate(FILES_9V):
        ax = axes[i]

        beta_A, sigma_beta_A, m, c, chi2, dof = fit_avalanche_beta(
            fname,
            I_threshold=I_THRESHOLD,
            make_plot=True,
            ax=ax,
        )

        T_mean, T_err = parse_temperature_from_filename(fname)
        T_mean_list.append(T_mean)
        T_err_list.append(T_err)
        beta_list.append(beta_A)
        beta_err_list.append(sigma_beta_A)

        chi2dof = chi2 / dof if dof > 0 else np.nan
        chi2dof_list.append(chi2dof)
        dof_list.append(dof)

        print(
            f"{fname:40s}  T = {T_mean:.1f}±{T_err:.1f} K   "
            f"beta_A = {beta_A:.3e} ± {sigma_beta_A:.1e}   "
            f"(chi2/dof = {chi2dof:.2f})"
        )

    # remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    # ---- summary: beta_A vs T with good/bad points and weighted mean ----
    T_mean = np.array(T_mean_list)
    T_err = np.array(T_err_list)
    beta = np.array(beta_list)
    beta_err = np.array(beta_err_list)
    chi2dof = np.array(chi2dof_list)
    dof_arr = np.array(dof_list)

    order = np.argsort(T_mean)
    T_mean = T_mean[order]
    T_err = T_err[order]
    beta = beta[order]
    beta_err = beta_err[order]
    chi2dof = chi2dof[order]
    dof_arr = dof_arr[order]

    # classify fits
    CHI2DOF_CUT = 10.0
    good = (dof_arr > 2) & (chi2dof < CHI2DOF_CUT)
    bad = ~good

    fig2, ax2 = plt.subplots(figsize=(6, 4))

    # good points (blue circles)
    if np.any(good):
        ax2.errorbar(
            T_mean[good],
            beta[good],
            xerr=T_err[good],
            yerr=beta_err[good],
            fmt="o",
            capsize=3,
            label="good fits",
        )

    # bad points (red crosses)
    if np.any(bad):
        ax2.errorbar(
            T_mean[bad],
            beta[bad],
            xerr=T_err[bad],
            yerr=beta_err[bad],
            fmt="x",
            color="red",
            capsize=3,
            label="poor fits",
        )

    # weighted mean beta_A over good points (no attempt at beta_A(T) slope)
    if np.any(good):
        w = 1.0 / beta_err[good] ** 2
        beta_mean = np.sum(w * beta[good]) / np.sum(w)
        sigma_beta_mean = np.sqrt(1.0 / np.sum(w))

        ax2.axhline(
            beta_mean,
            color="k",
            linestyle="--",
            label=r"weighted mean $\beta_A = %.2e\pm%.1e$"
            % (beta_mean, sigma_beta_mean),
        )

        print(
            f"Weighted mean beta_A (good fits only) = "
            f"{beta_mean:.3e} ± {sigma_beta_mean:.1e}"
        )

    ax2.set_xlabel("Temperature T (K)")
    ax2.set_ylabel(r"$\beta_A$ (1/V)")
    ax2.grid(True, ls=":")
    ax2.set_title("Avalanche parameter vs temperature (9.1 V diode)(fixed-I)")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
