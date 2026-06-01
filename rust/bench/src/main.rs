use univariate_and_multivariate_time_series_analysis_with_python_core::autocorrelation;

fn main() {
    let s: Vec<f64> = (0..5000).map(|i| (i as f64 * 0.01).sin() + 10.0).collect();
    for _ in 0..2000 {
        let _ = autocorrelation(&s, 40);
    }
}
