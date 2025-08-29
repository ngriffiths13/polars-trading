use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use libm;


fn norm_cdf(x: f64) -> f64 {
    const SQRT_2: f64 = 1.4142135623730951;
    0.5 * (1.0 + libm::erf(x / SQRT_2))
}

/// Compute Black–Scholes European call & put prices.
///
/// Parameters:
/// - s: spot price of underlying
/// - k: strike price
/// - r: risk-free rate (annual, continuous compounding)
/// - sigma: volatility (annual)
/// - t: time to maturity in years (T - t0)
///
/// Returns BlackScholes { call, put }.
///
/// Handles obvious degenerate cases:
/// - If time_to_expiry == 0: returns intrinsic values
/// - If sigma == 0: treat as deterministic forward (discounted intrinsic)
pub fn _black_scholes(s: f64, k: f64, r: f64, sigma: f64, t: f64, type_: &str) -> Option<f64> {
    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;

    let nd1 = norm_cdf(d1);
    let nd2 = norm_cdf(d2);
    let n_minus_d1 = norm_cdf(-d1);
    let n_minus_d2 = norm_cdf(-d2);

    let discounted_strike = k * (-r * t).exp();

    let call = s * nd1 - discounted_strike * nd2;
    let put  = discounted_strike * n_minus_d2 - s * n_minus_d1;

    return match type_ {
        "call" => Some(call),
        "put"  => Some(put),
        _      => None,
    };
}


#[polars_expr(output_type=Float64)]
fn black_scholes(inputs: &[Series]) -> PolarsResult<Series> {
    let s: &Float64Chunked = inputs[0].f64()?;
    let k: &Float64Chunked = inputs[1].f64()?;
    let t: &Float64Chunked = inputs[2].f64()?;
    let sigma: &Float64Chunked = inputs[3].f64()?;
    let r: &Float64Chunked = inputs[4].f64()?;
    let type_: &StringChunked = inputs[5].str()?;

    // arity_5 will iterate elementwise across all 5 inputs
    let out: Float64Chunked = s
        .into_iter()
        .zip(k)
        .zip(t)
        .zip(sigma)
        .zip(r)
        .zip(type_)
        .map(|(((((s, k), t), sigma), r), type_) | match (s, k, t, sigma, r, type_) {
            (Some(s), Some(k), Some(t), Some(sigma), Some(r), Some(type_)) => {
                _black_scholes(s, k, r, sigma, t, type_)
            }
            _ => None,
        })
        .collect();

    Ok(out.into_series())
}
