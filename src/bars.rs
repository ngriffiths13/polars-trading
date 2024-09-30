#![allow(clippy::unused_unit)]
use std::cmp::PartialOrd;

use num::traits::{Signed, Zero};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

fn create_row_groups<T>(ca: &ChunkedArray<T>, bar_size: T::Native) -> PolarsResult<StructChunked>
where
    T: PolarsNumericType,
    T::Native: Signed + Zero + PartialOrd,
    ChunkedArray<T>: IntoSeries,
{
    let mut row_group_ids: Vec<i32> = Vec::with_capacity(ca.len());
    let mut row_group_amounts: Vec<T::Native> = Vec::with_capacity(ca.len());
    let mut current_sum = T::Native::zero();
    let mut group_id = 0;

    for val in ca.into_no_null_iter() {
        if current_sum + val >= bar_size {
            let remaining_amount = bar_size - current_sum;
            row_group_ids.push(group_id);
            row_group_amounts.push(remaining_amount);
            group_id += 1;
            current_sum = val - remaining_amount;
        } else {
            row_group_ids.push(group_id);
            row_group_amounts.push(val);
            current_sum += val;
        }
    }

    // Create ChunkedArrays
    let id_ca = Int32Chunked::new("bar_group__id", &row_group_ids);
    let amount_ca = ChunkedArray::<T>::from_slice("bar_group__amount", &row_group_amounts);

    // Create a StructChunked
    let fields = vec![id_ca.into_series(), amount_ca.into_series()];

    StructChunked::from_series("row_groups", &fields)
}

#[derive(Deserialize)]
struct BarGroupKwargs {
    bar_size: f64,
}

fn bar_group_struct(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        input_fields[0].name(),
        DataType::Struct(vec![
            Field::new("bar_group__id", DataType::Int32),
            Field::new("bar_group__amount", input_fields[0].data_type().clone()),
        ]),
    ))
}

#[polars_expr(output_type_func=bar_group_struct)]
fn bar_groups(inputs: &[Series], kwargs: BarGroupKwargs) -> PolarsResult<Series> {
    let groups = match inputs[0].dtype() {
        DataType::Float64 => create_row_groups(inputs[0].f64().unwrap(), kwargs.bar_size)?,
        DataType::Float32 => create_row_groups(inputs[0].f32().unwrap(), kwargs.bar_size as f32)?,
        DataType::Int64 => create_row_groups(inputs[0].i64().unwrap(), kwargs.bar_size as i64)?,
        DataType::Int32 => create_row_groups(inputs[0].i32().unwrap(), kwargs.bar_size as i32)?,
        _ => return Err(PolarsError::ComputeError("Unsupported type".into())),
    };
    Ok(groups.into_series())
}
