//! Sample autocorrelation function up to max_lag.

pub fn autocorrelation(series: &[f64], max_lag: usize) -> Vec<f64> {
    let n = series.len();
    if n == 0 {
        return vec![];
    }
    let mean = series.iter().sum::<f64>() / n as f64;
    let var = series.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let mut out = Vec::with_capacity(max_lag + 1);
    for lag in 0..=max_lag {
        if lag >= n {
            out.push(0.0);
            continue;
        }
        let mut cov = 0.0;
        for i in 0..(n - lag) {
            cov += (series[i] - mean) * (series[i + lag] - mean);
        }
        cov /= n as f64;
        out.push(if var > 1e-18 { cov / var } else { 0.0 });
    }
    out
}
