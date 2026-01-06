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
        "9.1 V zener diode at 160.7-155.5K.csv",
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


