# plot_beta_2V7.py

import numpy as np
import matplotlib.pyplot as plt

from beta_utils import fit_zener_beta, weighted_linear_fit
from temp_utils import parse_temperature_from_filename


FILES_2V7 = [
    "2.7 V zener diode at 125-124.9K.csv",
    "2.7 V zener diode at 155.5-153.6K.csv",
    "2.7 V zener diode at 183.7-182.3K.csv",
    "2.7 V zener diode at 211.9-212.3K.csv",
    "2.7 V zener diode at 240.7-241.3K.csv",
    "2.7 V zener diode at 272.1-271.7K.csv",
    "2.7 V zener diode at 301.7-301.2K.csv",
]

I_THRESHOLD = 2e-3  # 2 mA


def main():
    n = len(FILES_2V7)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = axes.ravel()

    T_mean_list = []
    T_err_list = []
    beta_list = []
    beta_err_list = []
    chi2dof_list = []

    for i, fname in enumerate(FILES_2V7):
        ax = axes[i]

        beta_Z, sigma_beta_Z, m, c, chi2, dof = fit_zener_beta(
            fname,
            I_threshold=I_THRESHOLD,
            make_plot=True,
            ax=ax,
        )

        T_mean, T_err = parse_temperature_from_filename(fname)
        T_mean_list.append(T_mean)
        T_err_list.append(T_err)
        beta_list.append(beta_Z)
        beta_err_list.append(sigma_beta_Z)
        chi2dof = chi2 / dof if dof > 0 else np.nan
        chi2dof_list.append(chi2dof)

        print(
            f"{fname:40s}  T = {T_mean:.1f}±{T_err:.1f} K   "
            f"beta_Z = {beta_Z:.3e} ± {sigma_beta_Z:.1e}   "
            f"(chi2/dof = {chi2dof:.2f})"
        )

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    # -------- summary: beta_Z vs T with linear fit --------
    T_mean = np.array(T_mean_list)
    T_err = np.array(T_err_list)
    beta = np.array(beta_list)
    beta_err = np.array(beta_err_list)
    chi2dof = np.array(chi2dof_list)

    order = np.argsort(T_mean)
    T_mean = T_mean[order]
    T_err = T_err[order]
    beta = beta[order]
    beta_err = beta_err[order]
    chi2dof = chi2dof[order]

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.errorbar(
        T_mean,
        beta,
        xerr=T_err,
        yerr=beta_err,
        fmt="o",
        capsize=3,
        label="data",
    )

    # weighted linear fit: beta_Z(T) = m_beta*T + c_beta
    m_beta, c_beta, sigma_m_beta, sigma_c_beta, chi2_lin, dof_lin = (
        weighted_linear_fit(T_mean, beta, sigma_y=beta_err)
    )
    T_fit = np.linspace(T_mean.min() - 5, T_mean.max() + 5, 200)
    ax2.plot(
        T_fit,
        m_beta * T_fit + c_beta,
        "--",
        label=(
            r"fit: $\beta_Z = (%.3e\pm%.1e)T + (%.2f\pm%.2f)$"
            % (m_beta, sigma_m_beta, c_beta, sigma_c_beta)
        ),
    )
    ax2.set_xlabel("Temperature T (K)")
    ax2.set_ylabel(r"$\beta_Z$ (1/V)")
    ax2.grid(True, ls=":")
    ax2.set_title("Zener parameter vs temperature (2.7 V diode)")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    print(
        f"d beta_Z / dT = {m_beta:.3e} ± {sigma_m_beta:.1e} 1/(V·K), "
        f"chi2/dof = {chi2_lin/dof_lin:.2f}"
    )

    # -------- convert beta_Z(T) -> effective band gap Eg(T) --------
    # Use Eg ∝ beta_Z^(2/3). Anchor Eg at ~300 K to Eg_ref ≈ 1.12 eV (Si).
    # Choose reference index closest to 300 K:
    idx_ref = np.argmin(np.abs(T_mean - 300.0))
    beta_ref = beta[idx_ref]
    Eg_ref = 1.12  # eV, approximate Si band gap near 300 K

    Eg = Eg_ref * (beta / beta_ref) ** (2.0 / 3.0)
    # error propagation: Eg ∝ beta^(2/3) -> sigma_Eg/Eg = (2/3)*(sigma_beta/beta)
    Eg_err = Eg * (2.0 / 3.0) * (beta_err / beta)

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.errorbar(
        T_mean,
        Eg,
        xerr=T_err,
        yerr=Eg_err,
        fmt="o",
        capsize=3,
        label="data (from Zener β)",
    )

    # optional linear fit Eg(T) over this T-range
    m_Eg, c_Eg, sigma_m_Eg, sigma_c_Eg, chi2_Eg, dof_Eg = (
        weighted_linear_fit(T_mean, Eg, sigma_y=Eg_err)
    )
    T_fit2 = np.linspace(T_mean.min() - 5, T_mean.max() + 5, 200)
    ax3.plot(
        T_fit2,
        m_Eg * T_fit2 + c_Eg,
        "--",
        label=(
            r"fit: $E_g = (%.3e\pm%.1e)T + (%.3f\pm%.3f)$ eV"
            % (m_Eg, sigma_m_Eg, c_Eg, sigma_c_Eg)
        ),
    )

    ax3.set_xlabel("Temperature T (K)")
    ax3.set_ylabel(r"Effective band gap $E_g$ (eV)")
    ax3.grid(True, ls=":")
    ax3.set_title("Effective band gap vs T (from 2.7 V Zener β)")
    ax3.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    print(
        f"d Eg / dT (from Zener) = {m_Eg:.3e} ± {sigma_m_Eg:.1e} eV/K, "
        f"chi2/dof = {chi2_Eg/dof_Eg:.2f}"
    )


if __name__ == "__main__":
    main()

