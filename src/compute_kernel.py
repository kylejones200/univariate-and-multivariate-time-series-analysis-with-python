"""Sample autocorrelation function up to max_lag."""

from __future__ import annotations

import numpy as np


def autocorrelation(series: np.ndarray, max_lag: int) -> np.ndarray:
    s = np.asarray(series, dtype=float)
    n = len(s)
    if n == 0:
        return np.empty(0, dtype=float)
    mean = float(s.sum()) / n
    var = float(((s - mean) ** 2).sum()) / n
    out = []
    for lag in range(max_lag + 1):
        if lag >= n:
            out.append(0.0)
            continue
        cov = 0.0
        for i in range(n - lag):
            cov += (s[i] - mean) * (s[i + lag] - mean)
        cov /= n
        out.append(cov / var if var > 1e-18 else 0.0)
    return np.asarray(out, dtype=float)
