# plotting_with_RELU_fit_for_9_1.py

import os
from relu_utils import find_threshold_two_region_relu, plot_many_iv_relu


if __name__ == "__main__":

    BASELINE_POINTS = 20   # low-V points used to estimate baseline current
    HIGH_POINTS = 20       # high-V points used to fit the rising linear region

    # ----- Single-file demo -----
    V_th, I_th, params = find_threshold_two_region_relu(
        "9.1 V zener diode at 160.7-155.5K.csv",
        baseline_points=BASELINE_POINTS,
        high_points=HIGH_POINTS,
    )

    print(
        f"Single file: V_th = {V_th:.4f} V, "
        f"I_th ≈ {I_th:.3e} A  (baseline_points={BASELINE_POINTS}, high_points={HIGH_POINTS})"
    )

    # ----- Multi-file subplot demo -----
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

    results = plot_many_iv_relu(
        files,
        ncols=2,
        baseline_points=BASELINE_POINTS,
        high_points=HIGH_POINTS,
    )

    for name, V_th, I_th in results:
        print(f"{name:40s}  V_th = {V_th:.4f} V,  I_th ≈ {I_th:.3e} A")
