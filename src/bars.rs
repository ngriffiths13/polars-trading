#![allow(clippy::unused_unit)]
use std::cmp::PartialOrd;

use num::traits::{Signed, Zero};
use polars::lazy::prelude::*;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

fn compute_bar_groups<T>(
    values: impl Iterator<Item = T>,
    bar_size: T,
    allow_splits: bool,
) -> (Vec<i32>, Vec<i32>, Vec<T>)
where
    T: Signed
        + Zero
        + PartialOrd
        + Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::AddAssign,
{
    let mut transaction_ids: Vec<i32> = Vec::new();
    let mut group_ids: Vec<i32> = Vec::new();
    let mut amounts: Vec<T> = Vec::new();
    let mut current_sum = T::zero();
    let mut group_id = 0;
    let mut transaction_id = 0;

    for val in values {
        if allow_splits {
            // Allow splitting a single value across multiple bars
            let mut remaining_val = val;

            while remaining_val > T::zero() {
                if current_sum + remaining_val >= bar_size {
                    let amount_to_add = bar_size - current_sum;
                    transaction_ids.push(transaction_id);
                    group_ids.push(group_id);
                    amounts.push(amount_to_add);
                    group_id += 1;
                    current_sum = T::zero();
                    remaining_val = remaining_val - amount_to_add;
                } else {
                    transaction_ids.push(transaction_id);
                    group_ids.push(group_id);
                    amounts.push(remaining_val);
                    current_sum += remaining_val;
                    remaining_val = T::zero();
                }
            }
        } else {
            // Don't allow splitting - entire value goes to one bar, allow overflow
            transaction_ids.push(transaction_id);
            group_ids.push(group_id);
            amounts.push(val);
            current_sum += val;

            // If we've met or exceeded the bar size, start a new bar for the next value
            if current_sum >= bar_size {
                group_id += 1;
                current_sum = T::zero();
            }
        }

        transaction_id += 1;
    }

    (transaction_ids, group_ids, amounts)
}

fn create_row_groups<T>(
    ca: &ChunkedArray<T>,
    bar_size: T::Native,
    allow_splits: bool,
) -> PolarsResult<Series>
where
    T: PolarsNumericType,
    T::Native: Signed + Zero + PartialOrd,
    ChunkedArray<T>: IntoSeries,
{
    let (transaction_ids, group_ids, amounts) =
        compute_bar_groups(ca.into_no_null_iter(), bar_size, allow_splits);

    let transaction_id_ca = Int32Chunked::new("transaction_id".into(), &transaction_ids);
    let id_ca = Int32Chunked::new("bar_group__id".into(), &group_ids);
    let amount_ca = ChunkedArray::<T>::from_slice("bar_group__amount".into(), &amounts);

    let fields = vec![id_ca.into_series(), amount_ca.into_series()];
    let struct_series =
        StructChunked::from_series("row_groups".into(), fields[0].len(), fields.iter())?
            .into_series();

    let df = DataFrame::new(vec![
        transaction_id_ca.into_series().into(),
        struct_series.into(),
    ])?;

    let result = df
        .lazy()
        .group_by([col("transaction_id")])
        .agg([col("row_groups")])
        .sort(["transaction_id"], Default::default())
        .collect()?;

    Ok(result
        .column("row_groups")?
        .as_materialized_series()
        .clone())
}

#[derive(Deserialize)]
struct BarGroupKwargs {
    bar_size: f64,
    #[serde(default = "default_allow_splits")]
    allow_splits: bool,
}

fn default_allow_splits() -> bool {
    true
}

fn bar_group_struct(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        input_fields[0].name().clone(),
        DataType::List(Box::new(DataType::Struct(vec![
            Field::new("bar_group__id".into(), DataType::Int32),
            Field::new("bar_group__amount".into(), input_fields[0].dtype().clone()),
        ]))),
    ))
}

#[polars_expr(output_type_func=bar_group_struct)]
fn bar_groups(inputs: &[Series], kwargs: BarGroupKwargs) -> PolarsResult<Series> {
    match inputs[0].dtype() {
        DataType::Float64 => create_row_groups(
            inputs[0].f64().unwrap(),
            kwargs.bar_size,
            kwargs.allow_splits,
        ),
        DataType::Float32 => create_row_groups(
            inputs[0].f32().unwrap(),
            kwargs.bar_size as f32,
            kwargs.allow_splits,
        ),
        DataType::Int64 => create_row_groups(
            inputs[0].i64().unwrap(),
            kwargs.bar_size as i64,
            kwargs.allow_splits,
        ),
        DataType::Int32 => create_row_groups(
            inputs[0].i32().unwrap(),
            kwargs.bar_size as i32,
            kwargs.allow_splits,
        ),
        _ => Err(PolarsError::ComputeError("Unsupported type".into())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_bar_groups_simple() {
        let values = vec![1, 2, 3, 4, 5];
        let bar_size = 4;

        let expected_transaction_ids = vec![
            0, // transaction 0 (value 1)
            1, // transaction 1 (value 2)
            2, 2, // transaction 2 (value 3, split into 2 parts)
            3, 3, // transaction 3 (value 4, split into 2 parts)
            4, 4, // transaction 4 (value 5, split into 2 parts)
        ];
        let expected_group_ids = vec![
            0, // value 1 goes to group 0
            0, // value 2 goes to group 0
            0, 1, // value 3 splits: group 0, then group 1
            1, 2, // value 4 splits: group 1, then group 2
            2, 3, // value 5 splits: group 2, then group 3
        ];
        let expected_amounts = vec![
            1, // value 1: amount 1 to group 0
            2, // value 2: amount 2 to group 0
            1, 2, // value 3: amount 1 to group 0, amount 2 to group 1
            2, 2, // value 4: amount 2 to group 1, amount 2 to group 2
            2, 3, // value 5: amount 2 to group 2, amount 3 to group 3
        ];

        let (result_transaction_ids, result_group_ids, result_amounts) =
            compute_bar_groups(values.into_iter(), bar_size, true);

        assert_eq!(result_transaction_ids, expected_transaction_ids);
        assert_eq!(result_group_ids, expected_group_ids);
        assert_eq!(result_amounts, expected_amounts);
    }

    #[test]
    fn test_compute_bar_groups_no_splits() {
        let values = vec![1, 2, 3, 4, 5];
        let bar_size = 4;

        // When allow_splits is false, bars can overflow
        // Bar 0: 1 + 2 = 3 (< 4, continue)
        // Bar 0: 3 + 3 = 6 (>= 4, overflow allowed, then start new bar)
        // Bar 1: 4 = 4 (>= 4, then start new bar)
        // Bar 2: 5 = 5 (>= 4, overflow allowed)
        let expected_transaction_ids = vec![
            0, // transaction 0 (value 1)
            1, // transaction 1 (value 2)
            2, // transaction 2 (value 3)
            3, // transaction 3 (value 4)
            4, // transaction 4 (value 5)
        ];
        let expected_group_ids = vec![
            0, // value 1 goes to group 0
            0, // value 2 goes to group 0 (sum=3)
            0, // value 3 goes to group 0 (sum=6, overflow)
            1, // value 4 goes to group 1 (new bar)
            2, // value 5 goes to group 2 (new bar)
        ];
        let expected_amounts = vec![
            1, // value 1: full amount to group 0
            2, // value 2: full amount to group 0
            3, // value 3: full amount to group 0 (causes overflow)
            4, // value 4: full amount to group 1
            5, // value 5: full amount to group 2
        ];

        let (result_transaction_ids, result_group_ids, result_amounts) =
            compute_bar_groups(values.into_iter(), bar_size, false);

        assert_eq!(result_transaction_ids, expected_transaction_ids);
        assert_eq!(result_group_ids, expected_group_ids);
        assert_eq!(result_amounts, expected_amounts);
    }

    #[test]
    fn test_compute_bar_groups_overflow_example() {
        let values = vec![2, 2, 5, 1, 3];
        let bar_size = 4;

        // When allow_splits is false, bars can overflow
        // Bar 0: 2 + 2 = 4 (>= 4, then start new bar)
        // Bar 1: 5 = 5 (>= 4, overflow, then start new bar)
        // Bar 2: 1 (< 4, continue)
        // Bar 2: 1 + 3 = 4 (>= 4, then would start new bar if more values)
        let expected_transaction_ids = vec![
            0, // transaction 0 (value 2)
            1, // transaction 1 (value 2)
            2, // transaction 2 (value 5)
            3, // transaction 3 (value 1)
            4, // transaction 4 (value 3)
        ];
        let expected_group_ids = vec![
            0, // value 2 goes to group 0
            0, // value 2 goes to group 0 (sum=4, meets bar_size exactly)
            1, // value 5 goes to group 1 (new bar, overflows)
            2, // value 1 goes to group 2 (new bar)
            2, // value 3 goes to group 2 (sum=4)
        ];
        let expected_amounts = vec![
            2, // value 2: full amount to group 0
            2, // value 2: full amount to group 0
            5, // value 5: full amount to group 1 (overflow)
            1, // value 1: full amount to group 2
            3, // value 3: full amount to group 2
        ];

        let (result_transaction_ids, result_group_ids, result_amounts) =
            compute_bar_groups(values.into_iter(), bar_size, false);

        assert_eq!(result_transaction_ids, expected_transaction_ids);
        assert_eq!(result_group_ids, expected_group_ids);
        assert_eq!(result_amounts, expected_amounts);
    }

    #[test]
    fn test_compare_split_vs_overflow() {
        // Test with the same data to show the difference between split and overflow modes
        let values = vec![3, 3, 3, 3];
        let bar_size = 4;

        // With splits enabled
        let (split_transaction_ids, split_group_ids, split_amounts) =
            compute_bar_groups(values.clone().into_iter(), bar_size, true);

        // Expected with splits: values get split to fit exactly into bars
        // Transaction 0: value 3, goes to bar 0
        // Transaction 1: value 3, 1 to bar 0 (fills it), 2 to bar 1
        // Transaction 2: value 3, 2 to bar 1 (fills it), 1 to bar 2
        // Transaction 3: value 3, goes to bar 2
        assert_eq!(split_transaction_ids, vec![0, 1, 1, 2, 2, 3]);
        assert_eq!(split_group_ids, vec![0, 0, 1, 1, 2, 2]);
        assert_eq!(split_amounts, vec![3, 1, 2, 2, 1, 3]);

        // Without splits (overflow allowed)
        let (overflow_transaction_ids, overflow_group_ids, overflow_amounts) =
            compute_bar_groups(values.into_iter(), bar_size, false);

        // Expected with overflow: entire values go to bars, allowing overflow
        // Transaction 0: value 3 goes to bar 0 (sum=3)
        // Transaction 1: value 3 goes to bar 0 (sum=6, overflow, then new bar)
        // Transaction 2: value 3 goes to bar 1 (sum=3)
        // Transaction 3: value 3 goes to bar 1 (sum=6, overflow)
        assert_eq!(overflow_transaction_ids, vec![0, 1, 2, 3]);
        assert_eq!(overflow_group_ids, vec![0, 0, 1, 1]);
        assert_eq!(overflow_amounts, vec![3, 3, 3, 3]);
    }

    #[test]
    fn test_create_row_groups() {
        let values = vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)];
        let ca = Float64Chunked::new("test".into(), values);
        let bar_size = 4.0;

        let result = create_row_groups(&ca, bar_size, true).unwrap();

        assert_eq!(
            result.dtype(),
            &DataType::List(Box::new(DataType::Struct(vec![
                Field::new("bar_group__id".into(), DataType::Int32),
                Field::new("bar_group__amount".into(), DataType::Float64),
            ])))
        );

        assert_eq!(result.len(), 5);

        let list_ca = result.list().unwrap();

        let first_transaction = list_ca.get_as_series(0).unwrap();
        assert_eq!(first_transaction.len(), 1);

        let second_transaction = list_ca.get_as_series(1).unwrap();
        assert_eq!(second_transaction.len(), 1);

        let third_transaction = list_ca.get_as_series(2).unwrap();
        assert_eq!(third_transaction.len(), 2);

        let fourth_transaction = list_ca.get_as_series(3).unwrap();
        assert_eq!(fourth_transaction.len(), 2);

        let fifth_transaction = list_ca.get_as_series(4).unwrap();
        assert_eq!(fifth_transaction.len(), 2);
    }
}
