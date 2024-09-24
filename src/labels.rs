// #![allow(clippy::unused_unit)]
// use std::cmp::PartialOrd;

// use polars::prelude::*;
// use pyo3_polars::derive::polars_expr;
// use serde::Deserialize;

// fn apply_profit_taking_stop_loss<T>(
//     index: &ChunkedArray<T>,
//     prices: &Float64Chunked,
//     profit_taking: &Float64Chunked,
//     stop_loss: &Float64Chunked,
// ) -> (Option<T>, Option<T>)
// where
//     T: PartialOrd + Clone,
// {
//     let returns: Vec<f64> = prices
//         .iter()
//         .map(|x| x.unwrap() / prices.get(0).unwrap() - 1.0)
//         .collect();
//     // Get the minimum index where profit take is greater than returns
//     let profit_taking_index = returns
//         .iter()
//         .zip(profit_taking.iter())
//         .position(|(&ret, &pt)| ret >= pt);
//     let stop_loss_index = returns
//         .iter()
//         .zip(stop_loss.iter())
//         .position(|(&ret, &sl)| ret <= sl);

//     match (profit_taking_index, stop_loss_index) {
//         (Some(pt), Some(sl)) => {
//             return (
//                 Some(index.get(pt).unwrap().clone()),
//                 Some(index.get(sl).unwrap().clone()),
//             )
//         },
//         (Some(pt), None) => return (Some(index.get(pt).unwrap().clone()), None),
//         (None, Some(sl)) => return (None, Some(index.get(sl).unwrap().clone())),
//         (None, None) => return (None, None),
//     }
// }

// fn barrier_touch_struct(input_fields: &[Field]) -> PolarsResult<Field> {
//     let dtype = input_fields[0].data_type();
//     Ok(Field::new(
//         input_fields[0].name(),
//         DataType::Struct(vec![
//             Field::new("barrier_touch_start", dtype.clone()),
//             Field::new("barrier_touch_profit_take", dtype.clone()),
//             Field::new("barrier_touch_stop_loss", dtype.clone()),
//             Field::new("barrier_touch_vertical_barrier", dtype.clone()),
//         ]),
//     ))
// }

// #[polars_expr(output_type_func=barrier_touch_struct)]
// fn get_barrier_touches(inputs: &[Series]) -> PolarsResult<Series> {
//     let targets = inputs[0].datetime()?; // Not sure what to do with this type yet.
//     let prices = inputs[1].f64()?;
//     let profit_taking = inputs[2].f64()?;
//     let stop_loss = inputs[3].f64()?;
//     let (pt, sl) = apply_profit_taking_stop_loss(targets, prices, profit_taking, stop_loss);
// }
