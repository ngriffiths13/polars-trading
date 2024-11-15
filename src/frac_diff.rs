use polars::prelude::*;
use polars_arrow::bitmap::MutableBitmap;
use pyo3_polars::derive::polars_expr;

use serde::Deserialize;

pub fn get_weights_ffd(d: f64, threshold: f64) -> Vec<f64> {
    let mut w = vec![1.];
    let mut k = 1.0;
    loop {
        let w_: f64 = -w.last().unwrap() / k * (d - k + 1.0);
        if w_.abs() < threshold {
            break;
        }
        w.push(w_);
        k += 1.0;
    }
    w.reverse();
    w
}

fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(a, b)| a * b).sum()
}

#[derive(Deserialize)]
struct FracDiffKwargs {
    d: f64,
    threshold: f64,
}

#[polars_expr(output_type=Float64)]
fn frac_diff(inputs: &[Series], kwargs: FracDiffKwargs) -> PolarsResult<Series> {
    let prices = inputs[0].f64().unwrap().to_vec_null_aware();
    let prices = if prices.is_left() {
        prices.left().unwrap()
    } else {
        return Err(PolarsError::InvalidOperation("Null price found".into()));
    };
    let weights = get_weights_ffd(kwargs.d, kwargs.threshold);
    let n_weights = weights.len();
    let mut outputs: Vec<f64> = Vec::with_capacity(prices.len());
    let mut validity_mask = MutableBitmap::with_capacity(prices.len());
    validity_mask.extend_constant(prices.len(), true);
    for i in 0..prices.len() {
        if i < (n_weights - 1) {
            outputs.push(0.0);
            validity_mask.set(i, false);
        } else {
            let window = &prices[i + 1 - n_weights..i + 1];
            let output = dot_product(window, &weights);
            outputs.push(output);
        }
    }
    Ok(
        Float64Chunked::from_vec_validity("frac_diff".into(), outputs, validity_mask.into())
            .into_series(),
    )
}
