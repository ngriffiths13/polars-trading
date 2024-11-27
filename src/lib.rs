use pyo3::prelude::*;
use pyo3::types::{PyModule, PyModuleMethods};
use pyo3::{pymodule, Bound, PyResult};
use pyo3_polars::PolarsAllocator;

mod bars;
mod frac_diff;
mod labels;

#[pyfunction]
fn get_weights_ffd_py(d: f64, threshold: f64) -> Vec<f64> {
    frac_diff::get_weights_ffd(d, threshold)
}

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(get_weights_ffd_py, m)?)?;
    Ok(())
}

// #[global_allocator]
// static ALLOC: PolarsAllocator = PolarsAllocator::new();
