mod bars;
mod frac_diff;
mod labels;

use pyo3::prelude::*;

#[pymodule]
fn _internal(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(get_weights_ffd_py, m)?)?;
    Ok(())
}

// #[global_allocator]
// static ALLOC: PolarsAllocator = PolarsAllocator::new();
