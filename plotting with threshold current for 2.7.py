import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math


# plot_diode_iv.py

from threshold_utils import find_threshold_at_fixed_current, plot_many_iv_fixed


if __name__ == "__main__":

    I_THRESHOLD = 2e-3  # 2 mA, can change here

    # ----- Single-file demo -----
    V_th, I_th, V_th_sigma, params = find_threshold_at_fixed_current(
        "2.7 V zener diode at 125-124.9K.csv",
        I_threshold=I_THRESHOLD,
    )

    if V_th_sigma is not None:
        print(
            f"Single file: V_th = {V_th:.4f} ± {V_th_sigma*1e3:.1f} mV "
            f"at I_th = {I_th*1e3:.1f} mA"
        )
    else:
        print(
            f"Single file: V_th = {V_th:.4f} V at I_th = {I_th*1e3:.1f} mA"
        )


    # ----- Multi-file subplot demo -----
    files = [
        "2.7 V zener diode at 125-124.9K.csv",
        "2.7 V zener diode at 155.5-153.6K.csv",
        "2.7 V zener diode at 183.7-182.3K.csv",
        "2.7 V zener diode at 211.9-212.3K.csv",
        "2.7 V zener diode at 240.7-241.3K.csv",
        "2.7 V zener diode at 247.7-247.9K.csv",
        "2.7 V zener diode at 260.3-261.8K.csv",
        "2.7 V zener diode at 270.1-272.4K.csv",
        "2.7 V zener diode at 272.1-271.7K.csv",
        "2.7 V zener diode at 280.2-282.8K.csv",
        "2.7 V zener diode at 290.2-292.6K.csv",
        "2.7 V zener diode at 300-302.3K.csv",
        "2.7 V zener diode at 301.7-301.2K.csv"

        # add more filenames here...
    ]

    results = plot_many_iv_fixed(
        files,
        I_threshold=I_THRESHOLD,
        ncols=2,
    )

    for name, V_th, V_th_sigma, I_th in results:
        if V_th_sigma is not None:
            print(
                f"{name:40s}  V_th = {V_th:.4f} ± {V_th_sigma*1e3:.2f} mV"
                f"  at  I_th = {I_th*1e3:.1f} mA"
            )
        else:
            print(
                f"{name:40s}  V_th = {V_th:.4f} V"
                f"  at  I_th = {I_th*1e3:.2f} mA"
            )
