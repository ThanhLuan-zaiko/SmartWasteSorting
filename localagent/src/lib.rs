mod artifacts;
mod cluster_review;
mod config;
mod domain;
mod error;
mod inference;
mod jobs;
mod python_api;
mod telemetry;
mod training_cache;
mod training_metrics;
mod workflow;

pub use artifacts::{to_api_envelope, ArtifactKind, ArtifactStore};
pub use cluster_review::{
    load_review_state_from_paths, ClusterReviewError, ClusterReviewSaveRequest, ClusterReviewState,
    ClusterReviewStore,
};
pub use config::RuntimeConfig;
pub use domain::{ClassificationResult, Prediction};
pub use error::AgentError;
pub use inference::WasteClassifier;
pub use jobs::{
    BenchmarkJobRequest, JobLogStream, JobLogsSnapshot, JobManager, JobRecord, JobStatus,
    JobStreamEvent, PipelineCommand, PipelineJobRequest, TrainingCommand, TrainingJobRequest,
};
pub use telemetry::init_tracing;
pub use training_metrics::{build_classification_report, compute_class_weight_map};
pub use workflow::{WorkflowCommandState, WorkflowState, WorkflowStatus, WorkflowStepState};

use pyo3::{prelude::*, types::PyModule, Bound};

#[pymodule]
fn _rust(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    python_api::register(module)
}
