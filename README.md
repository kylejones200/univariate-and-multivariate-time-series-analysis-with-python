# Univariate and Multivariate Time Series Analysis with Python

Published: 2024-12-19
Medium: [https://medium.com/@kyle-t-jones/univariate-and-multivariate-time-series-analysis-with-python-b22c6ec8f133](https://medium.com/@kyle-t-jones/univariate-and-multivariate-time-series-analysis-with-python-b22c6ec8f133)

## Business context

Traditional statistical approaches for time series are univariate, meaning they focus on a single sequence of values.

<figcaption>Photo by <a class="markup--anchor markup--figure-anchor" rel="photo-creator noopener" target="_blank">Christoph</a> on <a class="markup--anchor markup--figure-anchor"

However, in the real world, time series data often consists of multiple variables that interact with one another. This interaction introduces an opportunity to move beyond univariate analysis and leverage multivariate time series, where relationships between features play a central role.



## Rust performance port

Side-by-side **Python vs Rust** implementation of the numeric hot loop — autocorrelation function. Reference PyO3 benchmark: **see `benchmark_rust.py`** on a release build (local machine; run `benchmark_rust.py` to reproduce).

| Path | Role |
|------|------|
| `src/compute_kernel.py` | Python/numpy reference kernel |
| `rust/core/` | Pure Rust library |
| `rust/py/` | PyO3 bindings |
| `rust/bench/` | Standalone CLI benchmark |
| `benchmark_rust.py` | Python vs Rust timing + correctness check |

```bash
# Rust-only CLI benchmark
cd rust && cargo run --release -p univariate_and_multivariate_time_series_analysis_with_python_bench

# Python vs Rust (PyO3)
pip install maturin numpy
maturin develop --release -m rust/py/Cargo.toml
python benchmark_rust.py
```

Python ML training, solvers, and orchestration stay in Python; Rust targets the numeric hot loops. Stochastic generators validate output shapes; deterministic kernels match at tight floating-point tolerance.


## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).