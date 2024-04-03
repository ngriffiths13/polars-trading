#![allow(clippy::unused_unit)]
use polars::error::ErrString;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
struct TripleBarrierLabelKwargs {
    stop_loss: Option<f64>,
    profit_taker: Option<f64>,
    use_vertical_barrier_sign: bool,
    min_return: f64,
}

struct HorizontalBarrier {
    lower: Option<f64>,
    upper: Option<f64>,
}

struct Label {
    event: Option<i8>,
    ret: f64,
    n_bars: i64,
}

fn get_event(
    path_prices: &[f64],
    stop_loss: Option<f64>,
    profit_taker: Option<f64>,
    use_vertical_barrier_sign: bool,
    min_return: f64,
) -> Label {
    for (i, price) in path_prices.iter().enumerate() {
        match (stop_loss, profit_taker) {
            (Some(sl), Some(pt)) => {
                if *price <= -sl && *price <= -min_return {
                    return Label {
                        event: Some(-1),
                        ret: *price,
                        n_bars: i as i64,
                    };
                } else if *price >= pt && *price >= min_return {
                    return Label {
                        event: Some(1),
                        ret: *price,
                        n_bars: i as i64,
                    };
                }
            }
            (None, Some(pt)) => {
                if *price >= pt && *price >= min_return {
                    return Label {
                        event: Some(1),
                        ret: *price,
                        n_bars: i as i64,
                    };
                }
            }
            (Some(sl), None) => {
                if *price <= -sl && *price <= -min_return {
                    return Label {
                        event: Some(-1),
                        ret: *price,
                        n_bars: i as i64,
                    };
                }
            }
            _ => {}
        }
    }
    if use_vertical_barrier_sign {
        if *path_prices.last().unwrap_or(&0.0) < -min_return {
            Label {
                event: Some(-1),
                ret: *path_prices.last().unwrap_or(&0.0),
                n_bars: path_prices.len() as i64,
            }
        } else if *path_prices.last().unwrap_or(&0.0) > min_return {
            Label {
                event: Some(1),
                ret: *path_prices.last().unwrap_or(&0.0),
                n_bars: path_prices.len() as i64,
            }
        } else {
            Label {
                event: None,
                ret: *path_prices.last().unwrap_or(&0.0),
                n_bars: path_prices.len() as i64,
            }
        }
    } else {
        Label {
            event: Some(0),
            ret: *path_prices.last().unwrap_or(&0.0),
            n_bars: path_prices.len() as i64,
        }
    }
}

fn get_horizontal_barriers(
    horizontal_widths: &[Option<f64>],
    stop_loss: Option<f64>,
    profit_taker: Option<f64>,
) -> Vec<HorizontalBarrier> {
    let mut horizontal_barriers = Vec::new();
    for width in horizontal_widths {
        let (lower, upper) = match width {
            Some(w) => match (stop_loss, profit_taker) {
                (Some(sl), Some(pt)) => (Some(sl * w), Some(pt * w)),
                (None, Some(pt)) => (None, Some(pt * w)),
                (Some(sl), None) => (Some(sl * w), None),
                _ => (None, None),
            },
            None => (None, None),
        };
        horizontal_barriers.push(HorizontalBarrier { lower, upper });
    }
    horizontal_barriers
}

fn get_path_prices(prices: &[f64]) -> Vec<f64> {
    let first_price = prices[0];
    let mut path_prices = Vec::new();
    for price in prices {
        path_prices.push(price / first_price - 1.0);
    }
    path_prices
}

fn tbl_struct_type(_input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "triple_barrier_label",
        DataType::Struct(vec![
            Field::new("label", DataType::Int8),
            Field::new("ret", DataType::Float64),
            Field::new("n_bars", DataType::Int64),
        ]),
    ))
}

#[polars_expr(output_type_func=tbl_struct_type)]
pub fn triple_barrier_label(
    inputs: &[Series],
    kwargs: TripleBarrierLabelKwargs,
) -> PolarsResult<Series> {
    let prices = inputs[0].f64()?.to_vec();
    let horizontal_widths = inputs[1].f64()?;
    let vertical_barriers = inputs[2].i64()?;
    let seed_indicator = inputs[3].bool()?;
    let stop_loss = kwargs.stop_loss;
    let profit_taker = kwargs.profit_taker;

    if prices.iter().any(|&x| x.is_none()) {
        return Err(PolarsError::ComputeError(ErrString::from(
            "Missing prices in the input".to_string(),
        )));
    }
    let prices: Vec<f64> = prices.iter().map(|&x| x.unwrap()).collect();
    let horizontal_barriers =
        get_horizontal_barriers(&horizontal_widths.to_vec(), stop_loss, profit_taker);

    let mut event_builder: PrimitiveChunkedBuilder<Int8Type> =
        PrimitiveChunkedBuilder::new("triple_barrier_label_event", prices.len());
    let mut ret_builder: PrimitiveChunkedBuilder<Float64Type> =
        PrimitiveChunkedBuilder::new("triple_barrier_label_ret", prices.len());
    let mut n_bar_builder: PrimitiveChunkedBuilder<Int64Type> =
        PrimitiveChunkedBuilder::new("triple_barrier_label_n_bars", prices.len());
    for i in 0..prices.len() {
        if !seed_indicator.get(i).unwrap_or(false) {
            event_builder.append_null();
            ret_builder.append_null();
            n_bar_builder.append_null();
        } else {
            let path_prices = get_path_prices(
                &prices[i..vertical_barriers.get(i).unwrap_or(prices.len() as i64) as usize],
            );
            let label = get_event(
                &path_prices,
                horizontal_barriers[i].lower,
                horizontal_barriers[i].upper,
                kwargs.use_vertical_barrier_sign,
                kwargs.min_return,
            );
            // TODO: Add n_bars to the output
            match label {
                Label {
                    event: Some(e),
                    ret: _,
                    n_bars: _,
                } => {
                    event_builder.append_value(e);
                    ret_builder.append_value(label.ret);
                    n_bar_builder.append_value(label.n_bars);
                }

                Label {
                    event: None,
                    ret: _,
                    n_bars: _,
                } => {
                    event_builder.append_null();
                    ret_builder.append_null();
                    n_bar_builder.append_null();
                }
            }
        }
    }
    let s = df!(
        "triple_barrier_label_event" => event_builder.finish(),
        "triple_barrier_label_ret" => ret_builder.finish(),
        "triple_barrier_label_n_bars" => n_bar_builder.finish()
    )?
    .lazy()
    .select([as_struct(vec![
        col("triple_barrier_label_event"),
        col("triple_barrier_label_ret"),
        col("triple_barrier_label_n_bars")
    ])
    .alias("triple_barrier_label")])
    .collect()?
    .column("triple_barrier_label")?
    .clone();
    Ok(s)
}
