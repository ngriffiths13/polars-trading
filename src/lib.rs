mod bars;
mod labels;
mod nbbo;
mod symmetric_cusum_filter;
// use pyo3::prelude::*;

#[cfg(target_os = "linux")]
use jemallocator::Jemalloc;

#[global_allocator]
#[cfg(target_os = "linux")]
static ALLOC: Jemalloc = Jemalloc;

// Prints a message.
// #[pyfunction]
// fn hello() -> PyResult<String> {
//     Ok("Hello from polars-finance!".into())
// }

// /// A Python module implemented in Rust.
// #[pymodule]
// fn _lowlevel(_py: Python, m: &PyModule) -> PyResult<()> {
//     m.add_function(wrap_pyfunction!(hello, m)?)?;
//     Ok(())
// }
