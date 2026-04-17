use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyModule, wrap_pyfunction, Bound};

use crate::{
    build_classification_report, compute_class_weight_map,
    training_cache::{build_training_cache, CacheFormat},
    RuntimeConfig, WasteClassifier,
};

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

#[pyfunction]
fn compute_class_weight_map_json(
    train_labels: Vec<String>,
    class_names: Vec<String>,
) -> PyResult<String> {
    let weights = compute_class_weight_map(&train_labels, &class_names);
    serde_json::to_string(&weights).map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

#[pyfunction]
fn build_classification_report_json(
    predictions: Vec<usize>,
    targets: Vec<usize>,
    class_names: Vec<String>,
) -> PyResult<String> {
    let report = build_classification_report(&predictions, &targets, &class_names)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    serde_json::to_string(&report).map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

#[pyfunction]
#[pyo3(signature = (entries, cache_dir, image_size, cache_format="png", failure_report_path=None, force=false, show_progress=true))]
fn prepare_image_cache(
    py: Python<'_>,
    entries: Vec<(String, String)>,
    cache_dir: String,
    image_size: u32,
    cache_format: &str,
    failure_report_path: Option<String>,
    force: bool,
    show_progress: bool,
) -> PyResult<String> {
    py.allow_threads(move || {
        let cache_format = CacheFormat::parse(cache_format)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let summary = build_training_cache(
            &entries,
            std::path::Path::new(&cache_dir),
            failure_report_path.as_deref().map(std::path::Path::new),
            image_size,
            cache_format,
            force,
            show_progress,
        )
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;

        serde_json::to_string(&summary).map_err(|error| PyRuntimeError::new_err(error.to_string()))
    })
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<RustBackend>()?;
    module.add_function(wrap_pyfunction!(ping, module)?)?;
    module.add_function(wrap_pyfunction!(compute_class_weight_map_json, module)?)?;
    module.add_function(wrap_pyfunction!(build_classification_report_json, module)?)?;
    module.add_function(wrap_pyfunction!(prepare_image_cache, module)?)?;
    Ok(())
}
