# plotting_with_RELU_fit_for_2_7.py

import os
from relu_utils import find_threshold_two_region_relu, plot_many_iv_relu


if __name__ == "__main__":

    BASELINE_POINTS = 20   # low-V points used to estimate baseline current
    HIGH_POINTS = 20       # high-V points used to fit the rising linear region

    # ----- Single-file demo -----
    V_th, I_th, params = find_threshold_two_region_relu(
        "2.7 V zener diode at 125-124.9K.csv",
        baseline_points=BASELINE_POINTS,
        high_points=HIGH_POINTS,
    )

    print(
        f"Single file: V_th = {V_th:.4f} V, "
        f"I_th ≈ {I_th:.3e} A  (baseline_points={BASELINE_POINTS}, high_points={HIGH_POINTS})"
    )

    # ----- Multi-file subplot demo -----
    files = [
        "2.7 V zener diode at 125-124.9K.csv",
        "2.7 V zener diode at 155.5-153.6K.csv",
        "2.7 V zener diode at 183.7-182.3K.csv",
        "2.7 V zener diode at 211.9-212.3K.csv",
        "2.7 V zener diode at 240.7-241.3K.csv",
        "2.7 V zener diode at 247.7-247.9K.csv",
        #"2.7 V zener diode at 260.3-261.8K.csv",
        #"2.7 V zener diode at 270.1-272.4K.csv",
        #"2.7 V zener diode at 272.1-271.7K.csv",
        #"2.7 V zener diode at 280.2-282.8K.csv",
        #"2.7 V zener diode at 290.2-292.6K.csv",
        #"2.7 V zener diode at 300-302.3K.csv",
        #"2.7 V zener diode at 301.7-301.2K.csv",
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
