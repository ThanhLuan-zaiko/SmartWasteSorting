mod config;
mod domain;
mod error;
mod inference;
mod python_api;
mod telemetry;
mod training_cache;

pub use config::RuntimeConfig;
pub use domain::{ClassificationResult, Prediction};
pub use error::AgentError;
pub use inference::WasteClassifier;
pub use telemetry::init_tracing;

use pyo3::{prelude::*, types::PyModule, Bound};

#[pymodule]
fn _rust(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    python_api::register(module)
}
