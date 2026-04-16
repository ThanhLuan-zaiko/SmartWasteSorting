use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyModule, wrap_pyfunction, Bound};

use crate::{RuntimeConfig, WasteClassifier};

#[pyclass]
pub struct RustBackend {
    classifier: WasteClassifier,
}

#[pymethods]
impl RustBackend {
    #[new]
    #[pyo3(signature = (model_path=None, labels_path=None, device=None, score_threshold=None))]
    fn new(
        model_path: Option<String>,
        labels_path: Option<String>,
        device: Option<String>,
        score_threshold: Option<f32>,
    ) -> Self {
        let mut config = RuntimeConfig::default();
        if let Some(value) = model_path {
            config.model_path = value;
        }
        if let Some(value) = labels_path {
            config.labels_path = value;
        }
        if let Some(value) = device {
            config.device = value;
        }
        if let Some(value) = score_threshold {
            config.score_threshold = value;
        }

        Self {
            classifier: WasteClassifier::new(config),
        }
    }

    fn classify_stub(&self, sample_id: String) -> PyResult<String> {
        let sample_ids = [sample_id];
        let result = self
            .classifier
            .classify_batch(&sample_ids)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
            .into_iter()
            .next()
            .ok_or_else(|| PyRuntimeError::new_err("classifier returned no result"))?;

        serde_json::to_string(&result).map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }
}

#[pyfunction]
fn ping() -> &'static str {
    "localagent-rs-ready"
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<RustBackend>()?;
    module.add_function(wrap_pyfunction!(ping, module)?)?;
    Ok(())
}
