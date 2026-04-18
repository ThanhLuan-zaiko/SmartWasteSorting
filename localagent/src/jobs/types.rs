use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobRecord {
    pub schema_version: u8,
    pub job_id: String,
    pub job_type: String,
    pub command: String,
    pub experiment_name: Option<String>,
    pub status: JobStatus,
    pub progress_hint: Option<String>,
    pub created_at: String,
    pub started_at: Option<String>,
    pub finished_at: Option<String>,
    pub exit_code: Option<i32>,
    pub stdout_log_path: String,
    pub stderr_log_path: String,
    pub artifacts: BTreeMap<String, String>,
    pub error: Option<String>,
    pub cancel_requested: bool,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JobLogStream {
    Stdout,
    Stderr,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobLogsSnapshot {
    pub job_id: String,
    pub status: JobStatus,
    pub stdout: Vec<String>,
    pub stderr: Vec<String>,
    pub stdout_log_path: String,
    pub stderr_log_path: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum JobStreamEvent {
    Snapshot {
        jobs: Vec<JobRecord>,
        active_logs: Option<JobLogsSnapshot>,
    },
    JobUpdated {
        job: JobRecord,
    },
    LogLine {
        job_id: String,
        stream: JobLogStream,
        line: String,
    },
    ResyncRequired {
        reason: String,
    },
}

impl JobStreamEvent {
    pub fn matches_job(&self, job_id: Option<&str>) -> bool {
        match self {
            Self::LogLine {
                job_id: event_job_id,
                ..
            } => job_id == Some(event_job_id.as_str()),
            _ => true,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PipelineCommand {
    Scan,
    Split,
    Report,
    RunAll,
    Embed,
    Cluster,
    ExportClusterReview,
    PromoteClusterLabels,
    ExportLabelingTemplate,
    ImportLabels,
    ValidateLabels,
}

impl PipelineCommand {
    pub(super) fn as_cli(self) -> &'static str {
        match self {
            Self::Scan => "scan",
            Self::Split => "split",
            Self::Report => "report",
            Self::RunAll => "run-all",
            Self::Embed => "embed",
            Self::Cluster => "cluster",
            Self::ExportClusterReview => "export-cluster-review",
            Self::PromoteClusterLabels => "promote-cluster-labels",
            Self::ExportLabelingTemplate => "export-labeling-template",
            Self::ImportLabels => "import-labels",
            Self::ValidateLabels => "validate-labels",
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct PipelineJobRequest {
    pub command: PipelineCommand,
    pub raw_dir: Option<String>,
    pub manifest_dir: Option<String>,
    pub report_dir: Option<String>,
    pub min_width: Option<i64>,
    pub min_height: Option<i64>,
    pub train_ratio: Option<f64>,
    pub val_ratio: Option<f64>,
    pub test_ratio: Option<f64>,
    pub seed: Option<i64>,
    pub num_clusters: Option<i64>,
    pub infer_filename_labels: Option<bool>,
    pub labels_file: Option<String>,
    pub review_file: Option<String>,
    pub output: Option<String>,
    pub no_progress: Option<bool>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TrainingCommand {
    Summary,
    ExportSpec,
    ExportLabels,
    WarmCache,
    PseudoLabel,
    Fit,
    Evaluate,
    ExportOnnx,
    Report,
}

impl TrainingCommand {
    pub(super) fn as_cli(self) -> &'static str {
        match self {
            Self::Summary => "summary",
            Self::ExportSpec => "export-spec",
            Self::ExportLabels => "export-labels",
            Self::WarmCache => "warm-cache",
            Self::PseudoLabel => "pseudo-label",
            Self::Fit => "fit",
            Self::Evaluate => "evaluate",
            Self::ExportOnnx => "export-onnx",
            Self::Report => "report",
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct TrainingJobRequest {
    pub command: Option<TrainingCommand>,
    pub manifest: Option<String>,
    pub training_preset: Option<String>,
    pub experiment_name: Option<String>,
    pub training_backend: Option<String>,
    pub model_name: Option<String>,
    pub pretrained_backbone: Option<bool>,
    pub train_backbone: Option<bool>,
    pub image_size: Option<i64>,
    pub batch_size: Option<i64>,
    pub epochs: Option<i64>,
    pub num_workers: Option<i64>,
    pub device: Option<String>,
    pub cache_dir: Option<String>,
    pub resume_from: Option<String>,
    pub checkpoint: Option<String>,
    pub onnx_output: Option<String>,
    pub spec_output: Option<String>,
    pub cache_format: Option<String>,
    pub use_rust_cache: Option<bool>,
    pub force_cache: Option<bool>,
    pub class_bias: Option<String>,
    pub early_stopping_patience: Option<i64>,
    pub early_stopping_min_delta: Option<f64>,
    pub enable_early_stopping: Option<bool>,
    pub onnx_opset: Option<i64>,
    pub export_batch_size: Option<i64>,
    pub verify_onnx: Option<bool>,
    pub pseudo_label_threshold: Option<f64>,
    pub pseudo_label_margin: Option<f64>,
    pub no_progress: Option<bool>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct BenchmarkJobRequest {
    #[serde(flatten)]
    pub training: TrainingJobRequest,
    pub compare_to: Option<String>,
    pub compare_experiment: Option<String>,
}
