#!/usr/bin/env python3
"""Python vs Rust kernel benchmark."""

from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from compute_kernel import autocorrelation  # noqa: E402

def main() -> None:
    s = np.ascontiguousarray(np.sin(np.arange(5000) * 0.01) + 10.0)
    max_lag = 40
    t0 = time.perf_counter()
    for _ in range(200):
        autocorrelation(s, max_lag)
    py_s = time.perf_counter() - t0
    try:
        import univariate_and_multivariate_time_series_analysis_with_python_rs as rs
    except ImportError:
        print("Build: maturin develop --release -m rust/py/Cargo.toml")
        print(f"Python {py_s:.3f}s")
        return
    rs_s = rs.bench_kernel_py(s, max_lag, 2000)
    print(f"Python {py_s:.3f}s Rust {rs_s:.3f}s speedup {py_s / max(rs_s, 1e-9):.1f}x")
    np.testing.assert_allclose(
        autocorrelation(s, max_lag),
        np.asarray(rs.autocorrelation_py(s, max_lag)),
        rtol=1e-10,
    )
    print("Correctness: OK")

if __name__ == "__main__":
    main()
