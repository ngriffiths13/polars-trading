mod bars;
mod frac_diff;
mod labels;
mod black_scholes;

use pyo3::prelude::*;

#[pymodule]
fn _internal(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

// #[global_allocator]
// static ALLOC: PolarsAllocator = PolarsAllocator::new();
