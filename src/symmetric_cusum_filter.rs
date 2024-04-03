#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
struct CusumKwargs {
    threshold: f64,
}

fn calculate_cusum_filter(diff_series: &ChunkedArray<Float64Type>, threshold: f64) -> Vec<i8> {
    let mut out: Vec<i8> = Vec::with_capacity(diff_series.len());
    let mut s_pos = 0.0;
    let mut s_neg = 0.0;
    for val in diff_series.iter() {
        match val {
            Some(v) => {
                s_pos = (s_pos + v).max(0.0);
                s_neg = (s_neg + v).min(0.0);
                if s_neg < -threshold {
                    s_neg = 0.0;
                    out.push(-1);
                } else if s_pos > threshold {
                    s_pos = 0.0;
                    out.push(1);
                } else {
                    out.push(0);
                }
            }
            None => out.push(0),
        }
    }
    out
}

#[polars_expr(output_type=Int8)]
pub fn symmetric_cusum_filter(inputs: &[Series], kwargs: CusumKwargs) -> PolarsResult<Series> {
    let diff_series = inputs[0].f64()?;
    let out = calculate_cusum_filter(diff_series, kwargs.threshold);
    Ok(Series::from_vec("cusum_filter", out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_cusum_filter() {
        let diff_series = Float64Chunked::from_slice("diff_series", &[1.0, 2.0, -3.0, -4.0, 5.0]);
        let threshold = 2.0;
        let expected = vec![0, 1, -1, -1, 1];

        let result = calculate_cusum_filter(&diff_series, threshold);
        assert_eq!(result, expected);
    }
}
