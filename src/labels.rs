#![allow(clippy::unused_unit)]
/// TODOS:
/// - [ X ] Add bitmask
/// - [ X ] Handle 0 size price paths
/// - [ ] Calculate barrier touch from index
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

/// Returns the start and end indices of a slice range within a vector of i64 values.
///
/// # Arguments
///
/// * `data` - A vector of i64 values to search within.
/// * `start` - The value to search for as the start of the range.
/// * `end` - The value to search for as the end of the range.
///
/// # Returns
///
/// * `Ok((usize, usize))` - A tuple containing the start and end indices if both are found.
/// * `Err(String)` - An error message if either the start or end value (or both) are not found in the data.
///
/// # Examples
///
/// ```
/// let data = vec![1, 2, 3, 4, 5];
/// assert_eq!(get_slice_range(data, 2, 4), Ok((1, 3)));
/// ```
fn get_slice_range(data: &Vec<i64>, start: i64, end: i64) -> Result<(usize, usize), String> {
    let start_idx = data.iter().position(|&r| r == start);
    let end_idx = data.iter().position(|&r| r == end);
    match (start_idx, end_idx) {
        (Some(start_idx), Some(end_idx)) => Ok((start_idx, end_idx + 1)),
        (Some(_), None) => Err(format!("End index {} not found in index", end).into()),
        (None, Some(_)) => Err(format!("Start index {} not found in index", start).into()),
        (None, None) => Err(format!(
            "Both start index {} and end index {} not found in index",
            start, end
        )
        .into()),
    }
}

/// Calculate the returns of a given price path
///
/// I do this slightly differently than Lopez de Prado. In AFML pg. 46, he calculates
/// the returns by setting the first price to the price before the price path. This
/// seems a little off to me, since it means the first price in your price path does
/// not have a 0 return. This means when you use this label to train a model, you have
/// to be careful to not use the data from the date of the label. I prefer to set the
/// returns so the first return in the price path is 0. This way, you can use all the
/// data up to the close price of the date of the label.
///
/// # Arguments
///
/// * `prices` - A vector of prices to calculate the returns of.
///
/// # Returns
///
/// * `Vec<f64>` - A vector of returns for the given price path.
///
/// # Examples
///
/// ```
/// let prices = vec![1.0, 2.0, 3.0];
/// assert_eq!(calculate_price_path_return(prices), vec![Some(0.0), Some(1.0), Some(0.5)]);
/// ```
fn calculate_price_path_return(prices: Vec<f64>) -> Vec<f64> {
    let first_price = prices[0];
    prices.iter().map(|x| x / first_price - 1.0).collect()
}

#[derive(Debug)]
struct TripleBarrierLabel {
    ret: f64,
    label: i64,
    barrier_touch: i64,
}

/// Calculate the label for a given price path
fn get_label(
    returns: &[f64],
    profit_taking: Option<f64>,
    stop_loss: Option<f64>,
    zero_vertical_barrier: bool,
) -> TripleBarrierLabel {
    let pt_touch_idx = match profit_taking {
        Some(pt) => returns.iter().position(|&r| r >= pt),
        None => None,
    };
    let sl_touch_idx = match stop_loss {
        Some(sl) => returns.iter().position(|&r| r <= sl),
        None => None,
    };
    match (pt_touch_idx, sl_touch_idx) {
        (Some(pt_touch_idx), Some(sl_touch_idx)) => {
            if pt_touch_idx < sl_touch_idx {
                TripleBarrierLabel {
                    ret: returns[pt_touch_idx],
                    label: 1,
                    barrier_touch: pt_touch_idx as i64,
                }
            } else {
                TripleBarrierLabel {
                    ret: returns[sl_touch_idx],
                    label: -1,
                    barrier_touch: sl_touch_idx as i64,
                }
            }
        },
        (Some(pt_touch_idx), None) => TripleBarrierLabel {
            ret: returns[pt_touch_idx],
            label: 1,
            barrier_touch: pt_touch_idx as i64,
        },
        (None, Some(sl_touch_idx)) => TripleBarrierLabel {
            ret: returns[sl_touch_idx],
            label: -1,
            barrier_touch: sl_touch_idx as i64,
        },
        (None, None) => {
            if zero_vertical_barrier {
                TripleBarrierLabel {
                    ret: returns[returns.len() - 1],
                    label: 0,
                    barrier_touch: (returns.len() - 1) as i64,
                }
            } else {
                TripleBarrierLabel {
                    ret: returns[returns.len() - 1],
                    label: returns[returns.len() - 1].signum() as i64,
                    barrier_touch: (returns.len() - 1) as i64,
                }
            }
        },
    }
}

struct TripleBarrierLabels {
    rets: Vec<f64>,
    labels: Vec<i64>,
    barrier_touches: Vec<i64>,
}

impl TripleBarrierLabels {
    fn new() -> Self {
        TripleBarrierLabels {
            rets: Vec::new(),
            labels: Vec::new(),
            barrier_touches: Vec::new(),
        }
    }
    fn new_with_capacity(capacity: usize) -> Self {
        TripleBarrierLabels {
            rets: Vec::with_capacity(capacity),
            labels: Vec::with_capacity(capacity),
            barrier_touches: Vec::with_capacity(capacity),
        }
    }
}

fn calculate_labels(
    index: Vec<i64>,
    prices: Vec<f64>,
    profit_taking: Vec<Option<f64>>,
    stop_loss: Vec<Option<f64>>,
    vertical_barriers: Vec<Option<i64>>,
    validity_mask: Vec<bool>,
    zero_vertical_barrier: bool,
) -> TripleBarrierLabels {
    let mut labels = TripleBarrierLabels::new_with_capacity(prices.len());

    for i in 0..index.len() {
        if !validity_mask[i] {
            labels.rets.push(0.0);
            labels.labels.push(0);
            labels.barrier_touches.push(0);
            continue;
        }
        let mut barrier_touch_start_idx = 0 as usize;
        let price_path = match vertical_barriers[i] {
            Some(vb) => {
                let (start_idx, end_idx) = get_slice_range(&index, index[i], vb).unwrap();
                barrier_touch_start_idx = start_idx;
                println!("{:?}", { start_idx });
                println!("{:?}", { end_idx });
                println!("{:?}", { prices[start_idx..end_idx].to_vec() });
                calculate_price_path_return(prices[start_idx..end_idx].into())
            },
            None => {
                barrier_touch_start_idx = i;
                calculate_price_path_return(prices[i..].into())
            },
        };
        let label = get_label(
            &price_path,
            profit_taking[i],
            stop_loss[i],
            zero_vertical_barrier,
        );
        println!("{:?}", label);
        labels.rets.push(label.ret);
        labels.labels.push(label.label);
        labels
            .barrier_touches
            .push(label.barrier_touch + barrier_touch_start_idx as i64);
    }
    labels
}

fn triple_barrier_struct(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "triple_barrier_label".into(),
        DataType::Struct(vec![
            Field::new("price_path_return", DataType::Float64),
            Field::new("price_path_label", DataType::Float64),
            Field::new("barrier_touch", DataType::Int64),
        ]),
    ))
}

#[polars_expr(output_type_func=triple_barrier_struct)]
fn triple_barrier_label(inputs: &[Series]) -> PolarsResult<Series> {
    // There should be no nulls in index
    let index = &inputs[0];
    let index = if index.null_count() == 0 {
        index.i64()?.to_vec_null_aware().left().unwrap()
    } else {
        return Err(PolarsError::InvalidOperation(
            "Index should not contain null values".into(),
        ));
    };
    // There should be no null prices
    let prices = &inputs[1];
    let prices = if prices.null_count() == 0 {
        prices.f64()?.to_vec_null_aware().left().unwrap()
    } else {
        return Err(PolarsError::InvalidOperation(
            "Prices should not contain null values".into(),
        ));
    };
    // Null price taking means we don't implement
    let price_taking = inputs[2].f64()?.to_vec();
    // Null stop loss means we don't implement
    let stop_loss = inputs[3].f64()?.to_vec();
    // Null vertical barrier means we don't implement
    let vertical_barrier = inputs[4].i64()?.to_vec();
    let validity_mask = inputs[5].bool()?.into_no_null_iter().collect();
    let labels = calculate_labels(
        index,
        prices,
        price_taking,
        stop_loss,
        vertical_barrier,
        validity_mask,
        false,
    );

    // TODO
    let ret_series = Float64Chunked::from_vec("ret", labels.rets);
    let label_series = Int64Chunked::from_vec("label", labels.labels);
    let barrier_touch_series = Int64Chunked::from_vec("barrier_touch", labels.barrier_touches);
    let fields = vec![
        ret_series.into_series(),
        label_series.into_series(),
        barrier_touch_series.into_series(),
    ];
    let struct_series = StructChunked::from_series("row_groups", &fields).unwrap();
    Ok(struct_series.into_series())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests for get_slice_range function
    #[test]
    fn test_get_slice_range_normal() {
        let data = vec![1, 2, 3, 4, 5];
        assert_eq!(get_slice_range(&data, 2, 4), Ok((1, 4)));
    }

    #[test]
    fn test_get_slice_range_same_start_end() {
        let data = vec![1, 2, 3, 4, 5];
        assert_eq!(get_slice_range(&data, 3, 3), Ok((2, 3)));
    }

    #[test]
    fn test_get_slice_range_full_range() {
        let data = vec![1, 2, 3, 4, 5];
        assert_eq!(get_slice_range(&data, 1, 5), Ok((0, 5)));
        assert_eq!(data[0..5], [1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_get_slice_range_start_not_found() {
        let data = vec![1, 2, 3, 4, 5];
        assert_eq!(
            get_slice_range(&data, 0, 4),
            Err("Start index 0 not found in index".to_string())
        );
    }

    #[test]
    fn test_get_slice_range_end_not_found() {
        let data = vec![1, 2, 3, 4, 5];
        assert_eq!(
            get_slice_range(&data, 2, 6),
            Err("End index 6 not found in index".to_string())
        );
    }

    #[test]
    fn test_get_slice_range_both_not_found() {
        let data = vec![1, 2, 3, 4, 5];
        assert_eq!(
            get_slice_range(&data, 0, 6),
            Err("Both start index 0 and end index 6 not found in index".to_string())
        );
    }

    #[test]
    fn test_get_slice_range_empty_vector() {
        let data: Vec<i64> = vec![];
        assert_eq!(
            get_slice_range(&data, 1, 2),
            Err("Both start index 1 and end index 2 not found in index".to_string())
        );
    }

    // Tests for calculate_price_path_return function
    #[test]
    fn test_calculate_price_path_return_normal() {
        let prices = vec![1.0, 2.0, 3.0];
        assert_eq!(calculate_price_path_return(prices), vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_calculate_price_path_return_single_price() {
        let prices = vec![1.0];
        assert_eq!(calculate_price_path_return(prices), vec![0.0]);
    }

    #[test]
    fn test_calculate_price_path_return_decreasing_prices() {
        use approx::assert_relative_eq;
        let prices = vec![3.0, 2.0, 1.0];
        let result = calculate_price_path_return(prices);
        let expected = vec![0.0, -1.0 / 3.0, -2.0 / 3.0];
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(r, e, max_relative = 1e-5);
        }
    }

    // Tests for get_label function
    #[test]
    fn test_get_label_profit_taking() {
        let returns = vec![0.0, 0.1, 0.2, 0.3];
        let label = get_label(&returns, Some(0.25), Some(-0.1), false);
        assert_eq!(label.label, 1);
        assert_eq!(label.barrier_touch, 3);
        assert_eq!(label.ret, 0.3);
    }

    #[test]
    fn test_get_label_stop_loss() {
        let returns = vec![0.0, -0.05, -0.1, -0.15];
        let label = get_label(&returns, Some(0.2), Some(-0.1), false);
        assert_eq!(label.label, -1);
        assert_eq!(label.barrier_touch, 2);
        assert_eq!(label.ret, -0.1);
    }

    #[test]
    fn test_get_label_no_barrier_touch_zero_vertical() {
        let returns = vec![0.0, 0.05, 0.08, 0.09];
        let label = get_label(&returns, Some(0.1), Some(-0.1), true);
        assert_eq!(label.label, 0);
        assert_eq!(label.barrier_touch, 3);
        assert_eq!(label.ret, 0.09);
    }

    #[test]
    fn test_get_label_no_barrier_touch_non_zero_vertical() {
        let returns = vec![0.0, 0.05, 0.08, 0.09];
        let label = get_label(&returns, Some(0.1), Some(-0.1), false);
        assert_eq!(label.label, 1);
        assert_eq!(label.barrier_touch, 3);
        assert_eq!(label.ret, 0.09);
    }

    #[test]
    fn test_get_label_only_profit_taking() {
        let returns = vec![0.0, 0.1, 0.2, 0.3];
        let label = get_label(&returns, Some(0.25), None, false);
        assert_eq!(label.label, 1);
        assert_eq!(label.barrier_touch, 3);
        assert_eq!(label.ret, 0.3);
    }

    #[test]
    fn test_get_label_only_stop_loss() {
        let returns = vec![0.0, -0.05, -0.1, -0.15];
        let label = get_label(&returns, None, Some(-0.1), false);
        assert_eq!(label.label, -1);
        assert_eq!(label.barrier_touch, 2);
        assert_eq!(label.ret, -0.1);
    }

    #[test]
    fn test_get_label_no_barriers() {
        let returns = vec![0.0, 0.05, -0.05, 0.1];
        let label = get_label(&returns, None, None, false);
        assert_eq!(label.label, 1);
        assert_eq!(label.barrier_touch, 3);
        assert_eq!(label.ret, 0.1);
    }

    #[test]
    fn test_get_label_touches_pt_then_sl() {
        let returns = vec![0.0, 0.1, -0.1, -0.15];
        let label = get_label(&returns, Some(0.1), Some(-0.1), false);
        assert_eq!(label.label, 1);
        assert_eq!(label.barrier_touch, 1);
        assert_eq!(label.ret, 0.1);
    }

    #[test]
    fn test_get_label_touches_sl_then_pt() {
        let returns = vec![0.0, -0.1, 0.1, -0.15];
        let label = get_label(&returns, Some(0.1), Some(-0.1), false);
        assert_eq!(label.label, -1);
        assert_eq!(label.barrier_touch, 1);
        assert_eq!(label.ret, -0.1);
    }

    #[test]
    fn test_calculate_labels_basic() {
        let index = vec![1, 2, 3, 4, 5];
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let profit_taking = vec![Some(0.02); 5];
        let stop_loss = vec![Some(-0.01); 5];
        let vertical_barriers = vec![Some(5), Some(5), Some(5), None, None];
        let validity_mask = vec![true; 5];
        let zero_vertical_barrier = false;

        let result = calculate_labels(
            index,
            prices,
            profit_taking,
            stop_loss,
            vertical_barriers,
            validity_mask,
            zero_vertical_barrier,
        );

        assert_eq!(
            result.rets,
            vec![
                102.0 / 100.0 - 1.0,
                104.0 / 101.0 - 1.0,
                104.0 / 102.0 - 1.0,
                104.0 / 103.0 - 1.0,
                0.0
            ]
        );
        assert_eq!(result.labels, vec![1, 1, 1, 1, 1]);
        assert_eq!(result.barrier_touches, vec![2, 4, 4, 4, 4]);
    }

    #[test]
    fn test_calculate_labels_with_zero_vertical_barrier() {
        let index = vec![1, 2, 3, 4, 5];
        let prices = vec![100.0, 99.0, 98.0, 97.0, 96.0];
        let profit_taking = vec![Some(0.02); 5];
        let stop_loss = vec![Some(-0.01); 5];
        let vertical_barriers = vec![Some(5); 5];
        let validity_mask = vec![true; 5];
        let zero_vertical_barrier = true;

        let result = calculate_labels(
            index,
            prices,
            profit_taking,
            stop_loss,
            vertical_barriers,
            validity_mask,
            zero_vertical_barrier,
        );

        assert_eq!(
            result.rets,
            vec![
                99.0 / 100.0 - 1.0,
                98.0 / 99.0 - 1.0,
                97.0 / 98.0 - 1.0,
                96.0 / 97.0 - 1.0,
                0.0
            ]
        );
        assert_eq!(result.labels, vec![-1, -1, -1, -1, 0]);
        assert_eq!(result.barrier_touches, vec![1, 2, 3, 4, 4]);
    }

    #[test]
    fn test_calculate_labels_with_mixed_barriers() {
        let index = vec![1, 2, 3, 4, 5];
        let prices = vec![100.0, 102.0, 99.0, 103.0, 101.0];
        let profit_taking = vec![Some(0.03), Some(0.02), None, Some(0.01), Some(0.02)];
        let stop_loss = vec![Some(-0.02), None, Some(-0.01), Some(-0.02), Some(-0.01)];
        let vertical_barriers = vec![Some(3), Some(4), None, Some(5), None];
        let validity_mask = vec![true, true, false, true, true];
        let zero_vertical_barrier = false;

        let result = calculate_labels(
            index,
            prices,
            profit_taking,
            stop_loss,
            vertical_barriers,
            validity_mask,
            zero_vertical_barrier,
        );

        assert_eq!(result.rets.len(), 5);
        assert_eq!(result.labels.len(), 5);
        assert_eq!(result.barrier_touches.len(), 5);
        assert_eq!(result.labels[2], 0); // Invalid due to validity_mask
    }

    #[test]
    fn test_calculate_labels_no_barriers_hit() {
        let index = vec![1, 2, 3, 4, 5];
        let prices = vec![100.0, 101.0, 100.5, 101.5, 102.0];
        let profit_taking = vec![Some(0.05); 5];
        let stop_loss = vec![Some(-0.05); 5];
        let vertical_barriers = vec![None; 5];
        let validity_mask = vec![true; 5];
        let zero_vertical_barrier = false;

        let result = calculate_labels(
            index,
            prices,
            profit_taking,
            stop_loss,
            vertical_barriers,
            validity_mask,
            zero_vertical_barrier,
        );

        assert!(result.rets.iter().all(|&r| r >= 0.0));
        assert!(result.labels.iter().all(|&l| l == 1));
        assert_eq!(result.barrier_touches, vec![4, 4, 4, 4, 4]);
    }

    #[test]
    fn test_calculate_labels_empty_input() {
        let result = calculate_labels(vec![], vec![], vec![], vec![], vec![], vec![], false);

        assert!(result.rets.is_empty());
        assert!(result.labels.is_empty());
        assert!(result.barrier_touches.is_empty());
    }

    #[test]
    fn test_calculate_labels_all_invalid() {
        let index = vec![1, 2, 3];
        let prices = vec![100.0, 101.0, 102.0];
        let profit_taking = vec![Some(0.02); 3];
        let stop_loss = vec![Some(-0.01); 3];
        let vertical_barriers = vec![Some(3); 3];
        let validity_mask = vec![false; 3];
        let zero_vertical_barrier = false;

        let result = calculate_labels(
            index,
            prices,
            profit_taking,
            stop_loss,
            vertical_barriers,
            validity_mask,
            zero_vertical_barrier,
        );

        assert_eq!(result.rets, vec![0.0, 0.0, 0.0]);
        assert_eq!(result.labels, vec![0, 0, 0]);
        assert_eq!(result.barrier_touches, vec![0, 0, 0]);
    }
}
