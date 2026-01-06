# plot_vth_vs_T_RELU_for_9_1.py

import numpy as np
import matplotlib.pyplot as plt

from relu_utils import find_threshold_two_region_relu_with_uncertainty
from temp_utils import parse_temperature_from_filename


# List of all data files to use
FILES = [
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
]

# ReLU-fit configuration
BASELINE_POINTS = 20  # low-V points used to estimate baseline current
HIGH_POINTS = 20      # high-V points used to fit the rising linear region


def main():
    T_mean_list = []
    T_err_list = []
    V_list = []
    V_err_list = []

    # --- compute V_th, sigma_V, T_mean, sigma_T for each file ---
    for fname in FILES:
        # temperature from filename
        T_mean, T_err = parse_temperature_from_filename(fname)

        # threshold voltage from I-V data (no I-V plot here)
        V_th, I_th, V_th_sigma, _ = find_threshold_two_region_relu_with_uncertainty(
            fname,
            baseline_points=BASELINE_POINTS,
            high_points=HIGH_POINTS,
            make_plot=False,
        )

        T_mean_list.append(T_mean)
        T_err_list.append(T_err)
        V_list.append(V_th)
        V_err_list.append(V_th_sigma if V_th_sigma is not None else 0.0)

        # print for sanity
        if V_th_sigma is not None:
            print(
                f"{fname:40s}  T = {T_mean:.1f}±{T_err:.1f} K   "
                f"V_th = {V_th:.4f} ± {V_th_sigma*1e3:.2f} mV"
            )
        else:
            print(
                f"{fname:40s}  T = {T_mean:.1f}±{T_err:.1f} K   "
                f"V_th = {V_th:.4f} V"
            )

    # --- convert to numpy arrays and sort by temperature ---
    T_mean = np.array(T_mean_list)
    T_err = np.array(T_err_list)
    V_th_arr = np.array(V_list)
    V_err = np.array(V_err_list)

    order = np.argsort(T_mean)
    T_mean = T_mean[order]
    T_err = T_err[order]
    V_th_arr = V_th_arr[order]
    V_err = V_err[order]

    # --- plot V_th vs T with error bars ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        T_mean,
        V_th_arr,
        xerr=T_err,
        yerr=V_err,
        fmt="o",
        capsize=3,
        label="data",
    )

    ax.set_title("Threshold voltage vs Temperature for 9.1 V diode (RELU)")
    ax.set_xlabel("Temperature T (K)")
    ax.set_ylabel(r"Threshold voltage $V_{\mathrm{th}}$ (V)")
    ax.grid(True, ls=":")

    # --- Optional: weighted linear fit V_th(T) ---
    sigma = V_err.copy()
    if np.any(sigma <= 0):
        nz = sigma[sigma > 0]
        if nz.size > 0:
            sigma[sigma <= 0] = nz.min()
        else:
            sigma[:] = 1.0

    w = 1.0 / sigma**2
    A = np.vstack([T_mean, np.ones_like(T_mean)]).T
    Aw = A * np.sqrt(w)[:, None]
    yw = V_th_arr * np.sqrt(w)

    m, c = np.linalg.lstsq(Aw, yw, rcond=None)[0]

    # parameter uncertainties
    ATA_inv = np.linalg.inv(Aw.T @ Aw)
    residuals = V_th_arr - (m * T_mean + c)
    dof = len(T_mean) - 2
    chi2 = np.sum(w * residuals**2)
    s2 = chi2 / dof if dof > 0 else 0.0
    cov = ATA_inv * s2
    sigma_m = np.sqrt(cov[0, 0])
    sigma_c = np.sqrt(cov[1, 1])

    print(f"chi^2 = {chi2:.2f}")
    print(f"dof   = {dof}")
    print(f"chi^2/dof = {chi2/dof:.2f}")

    # plot the fit
    T_fit = np.linspace(T_mean.min() - 5, T_mean.max() + 5, 200)
    ax.plot(
        T_fit,
        m * T_fit + c,
        "--",
        label=f"fit: V = ({m:.3e}±{sigma_m:.1e}) T + ({c:.3f}±{sigma_c:.3f})",
    )

    ax.legend()
    plt.tight_layout()
    plt.show()

    print()
    print(f"dV_th/dT = {m:.4e} ± {sigma_m:.4e} V/K")
    print(f"Intercept = {c:.4f} ± {sigma_c:.4f} V")


if __name__ == "__main__":
    main()
