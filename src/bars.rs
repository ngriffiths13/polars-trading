#![allow(clippy::unused_unit)]
use std::cmp::PartialOrd;
use std::ops::{Add, Sub};

use num::traits::{Signed, Zero};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

struct RowGroup<T> {
    id: i32,
    amount: T,
}

fn create_row_groups<T, I>(ca: &ChunkedArray<T>, bar_size: T::Native) -> StructChunked
where
    T: PolarsNumericType,
    T::Native: Signed + Zero + PartialOrd,
{
    let mut row_groups: Vec<RowGroup<T>> = Vec::with_capacity(ca.len());
    let mut current_sum = T::zero();
    let zero_value = T::zero();
    let mut group_id = 0;
    // iterate over ca and add a group number and the value to the row groups as a RowGroup struct
    // if the sum of the values in the group is greater than or equal to the bar_size, only keep the
    // amount of the value up to the bar size. Reset the sum and start a new group. If a value is missing,
    // include it in the current group with a zero value. Collect the vector of structs into a StructChunked.
    todo!()
}

#[derive(Deserialize)]
struct VolumeBars {
    bar_size: f64,
}

#[polars_expr(output_type=Float64)]
fn volume_bars(inputs: &[Series], kwargs: VolumeBars) -> PolarsResult<Series> {
    let groups = match inputs[0].dtype() {
        DataType::Float64 => create_row_groups(inputs[0].f64().unwrap(), kwargs.bar_size as f64),
        // DataType::Float32 => inputs[0].f32()?.cast(&DataType::Float64),
        // DataType::Int64 => inputs[0].i64()?,
        // DataType::Int32 => inputs[0].i32()?,
        _ => return Err(PolarsError::ComputeError("Unsupported type".into())),
    };
}
