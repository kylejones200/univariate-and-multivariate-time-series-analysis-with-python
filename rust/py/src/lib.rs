use univariate_and_multivariate_time_series_analysis_with_python_core::autocorrelation;
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray};
use pyo3::prelude::*;

#[pyfunction]
fn autocorrelation_py<'py>(py: Python<'py>, series: PyReadonlyArray1<f64>, max_lag: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(autocorrelation(series.as_slice()?, max_lag).into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (series, max_lag, iterations=500))]
fn bench_kernel_py(series: PyReadonlyArray1<f64>, max_lag: usize, iterations: usize) -> PyResult<f64> {
    let series_buf = series.as_slice()?.to_vec();
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = autocorrelation(&series_buf, max_lag);
    }
    Ok(start.elapsed().as_secs_f64())
}

#[pymodule]
fn univariate_and_multivariate_time_series_analysis_with_python_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(autocorrelation_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_kernel_py, m)?)?;
    Ok(())
}
