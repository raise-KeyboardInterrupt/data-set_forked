# temp_utils.py

import os
import re


def parse_temperature_from_filename(filename):
    """
    Extract mean temperature and uncertainty from filenames like
    '9.1 V zener diode at 124-125.4K.csv' or '... at 293.8K.csv'.

    Returns
    -------
    T_mean : float
        Mean temperature in K.
    T_err : float
        Half-range, used as 1-sigma uncertainty in K.
    """
    base = os.path.basename(filename)

    # Case 1: range, e.g. 'at 124-125.4K'
    m_range = re.search(r'at\s*([0-9.]+)-([0-9.]+)K', base)
    if m_range:
        t1 = float(m_range.group(1))
        t2 = float(m_range.group(2))
        t_low, t_high = sorted([t1, t2])
        T_mean = 0.5 * (t_low + t_high)
        T_err = 0.5 * (t_high - t_low)
        return T_mean, T_err

    # Case 2: single temperature, e.g. 'at 300K'
    m_single = re.search(r'at\s*([0-9.]+)K', base)
    if m_single:
        T = float(m_single.group(1))
        return T, 0.0

    raise ValueError(f"Could not parse temperature from filename: {base}")
