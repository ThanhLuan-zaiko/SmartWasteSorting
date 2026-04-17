use std::{
    collections::{BTreeMap, HashMap},
    path::{Path, PathBuf},
    process::Stdio,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use time::{format_description::well_known::Rfc3339, OffsetDateTime};
use tokio::{
    fs::OpenOptions,
    io::{AsyncBufReadExt, AsyncRead, AsyncWriteExt, BufReader},
    process::Child,
    process::Command,
    sync::{broadcast, Mutex as AsyncMutex},
};
use tracing::warn;

use crate::{ArtifactKind, ArtifactStore, RuntimeConfig};

const DEFAULT_LOG_TAIL_LINES: usize = 200;
const JOB_EVENT_BUFFER: usize = 2_048;

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
    ExportLabelingTemplate,
    ImportLabels,
    ValidateLabels,
}

impl PipelineCommand {
    fn as_cli(self) -> &'static str {
        match self {
            Self::Scan => "scan",
            Self::Split => "split",
            Self::Report => "report",
            Self::RunAll => "run-all",
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
    pub infer_filename_labels: Option<bool>,
    pub labels_file: Option<String>,
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
    Fit,
    Evaluate,
    ExportOnnx,
    Report,
}

impl TrainingCommand {
    fn as_cli(self) -> &'static str {
        match self {
            Self::Summary => "summary",
            Self::ExportSpec => "export-spec",
            Self::ExportLabels => "export-labels",
            Self::WarmCache => "warm-cache",
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
    pub no_progress: Option<bool>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct BenchmarkJobRequest {
    #[serde(flatten)]
    pub training: TrainingJobRequest,
    pub compare_to: Option<String>,
    pub compare_experiment: Option<String>,
}

#[derive(Clone)]
pub struct JobManager {
    config: RuntimeConfig,
    jobs: Arc<Mutex<BTreeMap<String, JobRecord>>>,
    children: Arc<Mutex<HashMap<String, Arc<AsyncMutex<Child>>>>>,
    events: broadcast::Sender<JobStreamEvent>,
    sequence: Arc<AtomicU64>,
}

impl JobManager {
    pub fn new(config: RuntimeConfig) -> Self {
        let (events, _) = broadcast::channel(JOB_EVENT_BUFFER);
        let manager = Self {
            config,
            jobs: Arc::new(Mutex::new(BTreeMap::new())),
            children: Arc::new(Mutex::new(HashMap::new())),
            events,
            sequence: Arc::new(AtomicU64::new(0)),
        };
        if let Err(error) = manager.load_existing_jobs() {
            warn!("unable to load existing jobs: {error}");
        }
        manager
    }

    pub fn list_jobs(&self) -> Vec<JobRecord> {
        let mut jobs = self
            .jobs
            .lock()
            .expect("job registry mutex poisoned")
            .values()
            .cloned()
            .collect::<Vec<_>>();
        jobs.sort_by(|left, right| right.created_at.cmp(&left.created_at));
        jobs
    }

    pub fn get_job(&self, job_id: &str) -> Option<JobRecord> {
        self.jobs
            .lock()
            .expect("job registry mutex poisoned")
            .get(job_id)
            .cloned()
    }

    pub fn jobs_for_experiment(&self, experiment_name: &str) -> Vec<JobRecord> {
        let mut jobs = self
            .jobs
            .lock()
            .expect("job registry mutex poisoned")
            .values()
            .filter(|job| job.experiment_name.as_deref() == Some(experiment_name))
            .cloned()
            .collect::<Vec<_>>();
        jobs.sort_by(|left, right| right.created_at.cmp(&left.created_at));
        jobs
    }

    pub fn subscribe_events(&self) -> broadcast::Receiver<JobStreamEvent> {
        self.events.subscribe()
    }

    pub fn stream_snapshot(
        &self,
        active_job_id: Option<&str>,
        tail_lines: Option<usize>,
    ) -> Result<JobStreamEvent> {
        let active_logs = match active_job_id {
            Some(job_id) => Some(self.job_logs_snapshot(job_id, tail_lines)?),
            None => None,
        };
        Ok(JobStreamEvent::Snapshot {
            jobs: self.list_jobs(),
            active_logs,
        })
    }

    pub async fn spawn_pipeline_job(&self, request: PipelineJobRequest) -> Result<JobRecord> {
        let (_command_name, args) = self.pipeline_args(&request)?;

        self.spawn_job(
            "dataset_pipeline",
            None,
            args,
            "Runs dataset pipeline steps through the Python CLI.".to_string(),
        )
        .await
    }

    pub async fn spawn_training_job(&self, request: TrainingJobRequest) -> Result<JobRecord> {
        let command_name = request
            .command
            .context("command is required for training jobs")?
            .as_cli()
            .to_string();
        let experiment_name = request.experiment_name.clone();
        let args = self.training_args(&request, &command_name, None)?;
        self.spawn_job(
            "training_pipeline",
            experiment_name,
            args,
            "Runs training pipeline steps through the Python CLI.".to_string(),
        )
        .await
    }

    pub async fn spawn_benchmark_job(&self, request: BenchmarkJobRequest) -> Result<JobRecord> {
        let experiment_name = request.training.experiment_name.clone();
        let compare_to = request.compare_to.clone().or_else(|| {
            request.compare_experiment.as_ref().map(|experiment| {
                PathBuf::from(&self.config.artifact_dir)
                    .join("reports")
                    .join(format!("{experiment}_benchmark.json"))
                    .display()
                    .to_string()
            })
        });
        let args = self.training_args(&request.training, "benchmark", compare_to.as_deref())?;
        self.spawn_job(
            "benchmark_pipeline",
            experiment_name,
            args,
            "Runs a benchmark workflow using persisted artifact reports.".to_string(),
        )
        .await
    }

    pub async fn cancel_job(&self, job_id: &str) -> Result<JobRecord> {
        let updated_record = {
            let mut jobs = self.jobs.lock().expect("job registry mutex poisoned");
            let job = jobs
                .get_mut(job_id)
                .with_context(|| format!("unknown job_id: {job_id}"))?;
            if matches!(
                job.status,
                JobStatus::Completed | JobStatus::Failed | JobStatus::Cancelled
            ) {
                return Ok(job.clone());
            }
            job.cancel_requested = true;
            job.progress_hint = Some("Cancellation requested".to_string());
            self.persist_job(job)?;
            job.clone()
        };
        self.emit_event(JobStreamEvent::JobUpdated {
            job: updated_record,
        });

        let handle = self
            .children
            .lock()
            .expect("job child registry mutex poisoned")
            .get(job_id)
            .cloned()
            .with_context(|| format!("job is not currently running: {job_id}"))?;
        let mut child = handle.lock().await;
        child
            .kill()
            .await
            .context("failed to cancel child process")?;

        self.get_job(job_id)
            .with_context(|| format!("job disappeared after cancellation: {job_id}"))
    }

    pub fn job_logs(&self, job_id: &str, tail_lines: Option<usize>) -> Result<Value> {
        let snapshot = self.job_logs_snapshot(job_id, tail_lines)?;
        Ok(json!(snapshot))
    }

    pub fn job_logs_snapshot(
        &self,
        job_id: &str,
        tail_lines: Option<usize>,
    ) -> Result<JobLogsSnapshot> {
        let job = self
            .get_job(job_id)
            .with_context(|| format!("unknown job_id: {job_id}"))?;
        let max_lines = tail_lines.unwrap_or(DEFAULT_LOG_TAIL_LINES);
        Ok(JobLogsSnapshot {
            job_id: job.job_id,
            status: job.status,
            stdout: read_tail_lines(Path::new(&job.stdout_log_path), max_lines)?,
            stderr: read_tail_lines(Path::new(&job.stderr_log_path), max_lines)?,
            stdout_log_path: job.stdout_log_path,
            stderr_log_path: job.stderr_log_path,
        })
    }

    fn pipeline_args(&self, request: &PipelineJobRequest) -> Result<(String, Vec<String>)> {
        let command_name = request.command.as_cli().to_string();
        let mut args = vec![
            "run".to_string(),
            "python".to_string(),
            "-m".to_string(),
            "localagent.data.pipeline".to_string(),
            command_name.clone(),
        ];

        push_optional_path(&mut args, "--raw-dir", request.raw_dir.as_deref());
        push_optional_path(&mut args, "--manifest-dir", request.manifest_dir.as_deref());
        push_optional_path(&mut args, "--report-dir", request.report_dir.as_deref());
        push_optional_i64(&mut args, "--min-width", request.min_width);
        push_optional_i64(&mut args, "--min-height", request.min_height);
        push_optional_f64(&mut args, "--train-ratio", request.train_ratio);
        push_optional_f64(&mut args, "--val-ratio", request.val_ratio);
        push_optional_f64(&mut args, "--test-ratio", request.test_ratio);
        push_optional_i64(&mut args, "--seed", request.seed);
        if matches!(request.infer_filename_labels, Some(false)) {
            args.push("--no-filename-labels".to_string());
        }
        if matches!(request.command, PipelineCommand::ImportLabels) {
            let labels_file = request
                .labels_file
                .as_deref()
                .context("labels_file is required for import-labels")?;
            push_optional_path(&mut args, "--labels-file", Some(labels_file));
        }
        if matches!(request.command, PipelineCommand::ExportLabelingTemplate) {
            push_optional_path(&mut args, "--output", request.output.as_deref());
        }
        if request.no_progress.unwrap_or(true) {
            args.push("--no-progress".to_string());
        }

        Ok((command_name, args))
    }

    fn training_args(
        &self,
        request: &TrainingJobRequest,
        command_name: &str,
        compare_to: Option<&str>,
    ) -> Result<Vec<String>> {
        let mut args = vec![
            "run".to_string(),
            "python".to_string(),
            "-m".to_string(),
            "localagent.training.train".to_string(),
            command_name.to_string(),
        ];

        push_optional_path(&mut args, "--manifest", request.manifest.as_deref());
        push_optional(
            &mut args,
            "--training-preset",
            request.training_preset.as_deref(),
        );
        push_optional(
            &mut args,
            "--experiment-name",
            request.experiment_name.as_deref(),
        );
        push_optional(
            &mut args,
            "--training-backend",
            request.training_backend.as_deref(),
        );
        push_optional(&mut args, "--model-name", request.model_name.as_deref());
        if matches!(request.pretrained_backbone, Some(false)) {
            args.push("--no-pretrained".to_string());
        }
        if matches!(request.train_backbone, Some(true)) {
            args.push("--train-backbone".to_string());
        }
        push_optional_i64(&mut args, "--image-size", request.image_size);
        push_optional_i64(&mut args, "--batch-size", request.batch_size);
        push_optional_i64(&mut args, "--epochs", request.epochs);
        push_optional_i64(&mut args, "--num-workers", request.num_workers);
        push_optional(&mut args, "--device", request.device.as_deref());
        push_optional_path(&mut args, "--cache-dir", request.cache_dir.as_deref());
        push_optional_path(&mut args, "--resume-from", request.resume_from.as_deref());
        push_optional_path(&mut args, "--checkpoint", request.checkpoint.as_deref());
        push_optional_path(&mut args, "--onnx-output", request.onnx_output.as_deref());
        push_optional_path(&mut args, "--spec-output", request.spec_output.as_deref());
        push_optional(&mut args, "--cache-format", request.cache_format.as_deref());
        if matches!(request.use_rust_cache, Some(false)) {
            args.push("--no-rust-cache".to_string());
        }
        if matches!(request.force_cache, Some(true)) {
            args.push("--force-cache".to_string());
        }
        push_optional(&mut args, "--class-bias", request.class_bias.as_deref());
        push_optional_i64(
            &mut args,
            "--early-stopping-patience",
            request.early_stopping_patience,
        );
        push_optional_f64(
            &mut args,
            "--early-stopping-min-delta",
            request.early_stopping_min_delta,
        );
        if matches!(request.enable_early_stopping, Some(false)) {
            args.push("--disable-early-stopping".to_string());
        }
        push_optional_i64(&mut args, "--onnx-opset", request.onnx_opset);
        push_optional_i64(&mut args, "--export-batch-size", request.export_batch_size);
        if matches!(request.verify_onnx, Some(false)) {
            args.push("--skip-onnx-verify".to_string());
        }
        if request.no_progress.unwrap_or(true) {
            args.push("--no-progress".to_string());
        }
        if command_name == "benchmark" {
            push_optional_path(&mut args, "--compare-to", compare_to);
        } else if compare_to.is_some() {
            anyhow::bail!("compare_to is only valid for benchmark jobs");
        }
        Ok(args)
    }

    async fn spawn_job(
        &self,
        job_type: &str,
        experiment_name: Option<String>,
        args: Vec<String>,
        progress_hint: String,
    ) -> Result<JobRecord> {
        let job_id = self.next_job_id();
        let stdout_log_path = self.logs_dir().join(format!("{job_id}.stdout.log"));
        let stderr_log_path = self.logs_dir().join(format!("{job_id}.stderr.log"));
        std::fs::File::create(&stdout_log_path).with_context(|| {
            format!("failed to create stdout log: {}", stdout_log_path.display())
        })?;
        std::fs::File::create(&stderr_log_path).with_context(|| {
            format!("failed to create stderr log: {}", stderr_log_path.display())
        })?;

        let started_at = timestamp_now();
        let mut child = Command::new("uv");
        child
            .args(&args)
            .current_dir(self.project_root())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = child
            .spawn()
            .with_context(|| format!("failed to spawn uv command for job {job_id}"))?;
        let stdout = child
            .stdout
            .take()
            .context("failed to capture stdout for spawned job")?;
        let stderr = child
            .stderr
            .take()
            .context("failed to capture stderr for spawned job")?;
        let child = Arc::new(AsyncMutex::new(child));

        let record = JobRecord {
            schema_version: 1,
            job_id: job_id.clone(),
            job_type: job_type.to_string(),
            command: format!("uv {}", args.join(" ")),
            experiment_name,
            status: JobStatus::Running,
            progress_hint: Some(progress_hint),
            created_at: started_at.clone(),
            started_at: Some(started_at),
            finished_at: None,
            exit_code: None,
            stdout_log_path: stdout_log_path.display().to_string(),
            stderr_log_path: stderr_log_path.display().to_string(),
            artifacts: BTreeMap::new(),
            error: None,
            cancel_requested: false,
        };

        {
            let mut jobs = self.jobs.lock().expect("job registry mutex poisoned");
            jobs.insert(job_id.clone(), record.clone());
            self.persist_job(
                jobs.get(&job_id)
                    .expect("job must exist immediately after insertion"),
            )?;
        }
        self.children
            .lock()
            .expect("job child registry mutex poisoned")
            .insert(job_id.clone(), child.clone());
        self.emit_event(JobStreamEvent::JobUpdated {
            job: record.clone(),
        });

        let stdout_manager = self.clone();
        let stdout_job_id = job_id.clone();
        let stdout_log_path_clone = stdout_log_path.clone();
        let stdout_task = tokio::spawn(async move {
            if let Err(error) = stream_child_output(
                stdout_manager,
                stdout_job_id,
                stdout,
                stdout_log_path_clone,
                JobLogStream::Stdout,
            )
            .await
            {
                warn!("failed to stream stdout: {error}");
            }
        });

        let stderr_manager = self.clone();
        let stderr_job_id = job_id.clone();
        let stderr_log_path_clone = stderr_log_path.clone();
        let stderr_task = tokio::spawn(async move {
            if let Err(error) = stream_child_output(
                stderr_manager,
                stderr_job_id,
                stderr,
                stderr_log_path_clone,
                JobLogStream::Stderr,
            )
            .await
            {
                warn!("failed to stream stderr: {error}");
            }
        });

        let manager = self.clone();
        tokio::spawn(async move {
            let wait_result = {
                let mut child = child.lock().await;
                child.wait().await
            };
            let _ = stdout_task.await;
            let _ = stderr_task.await;
            if let Err(error) = manager.finish_job(&job_id, wait_result) {
                warn!("failed to finish job {job_id}: {error}");
            }
        });

        Ok(record)
    }

    fn finish_job(
        &self,
        job_id: &str,
        wait_result: std::io::Result<std::process::ExitStatus>,
    ) -> Result<()> {
        let exit_status = wait_result.context("failed while waiting for child process")?;
        let completed_record = {
            let mut jobs = self.jobs.lock().expect("job registry mutex poisoned");
            let record = jobs
                .get_mut(job_id)
                .with_context(|| format!("missing job record during completion: {job_id}"))?;
            record.finished_at = Some(timestamp_now());
            record.exit_code = exit_status.code();
            record.progress_hint = None;
            record.artifacts = self.discover_artifacts(record.experiment_name.as_deref());

            record.status = if record.cancel_requested {
                JobStatus::Cancelled
            } else if exit_status.success() {
                JobStatus::Completed
            } else {
                JobStatus::Failed
            };
            if !exit_status.success() && !record.cancel_requested {
                record.error = Some(format!(
                    "Process exited with status {:?}",
                    exit_status.code()
                ));
            }
            self.persist_job(record)?;
            record.clone()
        };

        self.children
            .lock()
            .expect("job child registry mutex poisoned")
            .remove(job_id);
        self.emit_event(JobStreamEvent::JobUpdated {
            job: completed_record,
        });

        let artifact_store = ArtifactStore::new(self.config.clone());
        let _ = artifact_store.sync_run_index();
        Ok(())
    }

    fn discover_artifacts(&self, experiment_name: Option<&str>) -> BTreeMap<String, String> {
        let artifact_store = ArtifactStore::new(self.config.clone());
        let experiment = experiment_name.unwrap_or(artifact_store.default_experiment());
        let mut artifacts = BTreeMap::new();

        for (name, kind) in [
            ("dataset_summary", ArtifactKind::DatasetSummary),
            ("training", ArtifactKind::Training),
            ("evaluation", ArtifactKind::Evaluation),
            ("export", ArtifactKind::Export),
            ("benchmark", ArtifactKind::Benchmark),
            ("experiment_spec", ArtifactKind::ExperimentSpec),
            ("bundle", ArtifactKind::Bundle),
            ("model_manifest", ArtifactKind::ModelManifest),
        ] {
            let path = artifact_store.artifact_path(kind, Some(experiment));
            if path.is_file() {
                artifacts.insert(name.to_string(), path.display().to_string());
            }
        }
        artifacts
    }

    fn load_existing_jobs(&self) -> Result<()> {
        let mut latest_sequence = 0_u64;
        let mut jobs = self.jobs.lock().expect("job registry mutex poisoned");
        jobs.clear();
        for entry in std::fs::read_dir(self.jobs_dir()).context("failed to read jobs dir")? {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() || path.extension().and_then(|value| value.to_str()) != Some("json")
            {
                continue;
            }
            let payload = std::fs::read_to_string(&path)
                .with_context(|| format!("failed to read job manifest: {}", path.display()))?;
            let job = serde_json::from_str::<JobRecord>(&payload)
                .with_context(|| format!("failed to parse job manifest: {}", path.display()))?;
            latest_sequence = latest_sequence.max(sequence_from_job_id(&job.job_id));
            jobs.insert(job.job_id.clone(), job);
        }
        self.sequence.store(latest_sequence, Ordering::SeqCst);
        Ok(())
    }

    fn persist_job(&self, job: &JobRecord) -> Result<()> {
        let path = self.jobs_dir().join(format!("{}.json", job.job_id));
        let payload = serde_json::to_string_pretty(job)?;
        std::fs::write(&path, payload)
            .with_context(|| format!("failed to persist job manifest: {}", path.display()))
    }

    fn emit_event(&self, event: JobStreamEvent) {
        let _ = self.events.send(event);
    }

    fn next_job_id(&self) -> String {
        let sequence = self.sequence.fetch_add(1, Ordering::SeqCst) + 1;
        let millis = OffsetDateTime::now_utc().unix_timestamp_nanos() / 1_000_000;
        format!("job-{millis}-{sequence}")
    }

    fn project_root(&self) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    fn jobs_dir(&self) -> PathBuf {
        let path = PathBuf::from(&self.config.artifact_dir).join("jobs");
        let _ = std::fs::create_dir_all(&path);
        path
    }

    fn logs_dir(&self) -> PathBuf {
        let path = self.jobs_dir().join("logs");
        let _ = std::fs::create_dir_all(&path);
        path
    }
}

fn push_optional(args: &mut Vec<String>, flag: &str, value: Option<&str>) {
    if let Some(value) = value.filter(|value| !value.trim().is_empty()) {
        args.push(flag.to_string());
        args.push(value.to_string());
    }
}

fn push_optional_path(args: &mut Vec<String>, flag: &str, value: Option<&str>) {
    push_optional(args, flag, value);
}

fn push_optional_i64(args: &mut Vec<String>, flag: &str, value: Option<i64>) {
    if let Some(value) = value {
        args.push(flag.to_string());
        args.push(value.to_string());
    }
}

fn push_optional_f64(args: &mut Vec<String>, flag: &str, value: Option<f64>) {
    if let Some(value) = value {
        args.push(flag.to_string());
        args.push(value.to_string());
    }
}

async fn stream_child_output<R>(
    manager: JobManager,
    job_id: String,
    reader: R,
    log_path: PathBuf,
    stream: JobLogStream,
) -> Result<()>
where
    R: AsyncRead + Unpin + Send + 'static,
{
    let mut lines = BufReader::new(reader).lines();
    let mut log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .await
        .with_context(|| format!("failed to open log file for append: {}", log_path.display()))?;

    while let Some(line) = lines.next_line().await? {
        log_file.write_all(line.as_bytes()).await?;
        log_file.write_all(b"\n").await?;
        manager.emit_event(JobStreamEvent::LogLine {
            job_id: job_id.clone(),
            stream,
            line,
        });
    }

    log_file.flush().await?;
    Ok(())
}

fn read_tail_lines(path: &Path, max_lines: usize) -> Result<Vec<String>> {
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read log file: {}", path.display()))?;
    let mut lines = content.lines().map(str::to_string).collect::<Vec<_>>();
    if lines.len() > max_lines {
        lines = lines.split_off(lines.len() - max_lines);
    }
    Ok(lines)
}

fn timestamp_now() -> String {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| OffsetDateTime::now_utc().unix_timestamp().to_string())
}

fn sequence_from_job_id(job_id: &str) -> u64 {
    job_id
        .rsplit('-')
        .next()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::{
        read_tail_lines, BenchmarkJobRequest, JobManager, PipelineCommand, PipelineJobRequest,
        TrainingCommand, TrainingJobRequest,
    };
    use crate::RuntimeConfig;

    #[tokio::test]
    async fn builds_pipeline_command_with_expected_args() {
        let manager = JobManager::new(RuntimeConfig::default());
        let request = PipelineJobRequest {
            command: PipelineCommand::ImportLabels,
            raw_dir: None,
            manifest_dir: None,
            report_dir: None,
            min_width: None,
            min_height: None,
            train_ratio: None,
            val_ratio: None,
            test_ratio: None,
            seed: None,
            infer_filename_labels: None,
            labels_file: Some("artifacts/manifests/labels.csv".to_string()),
            output: None,
            no_progress: Some(true),
        };

        let (command_name, args) = manager
            .pipeline_args(&request)
            .expect("failed to build pipeline args");
        assert_eq!(command_name, "import-labels");
        assert!(args
            .join(" ")
            .contains("localagent.data.pipeline import-labels"));
        assert!(args
            .join(" ")
            .contains("--labels-file artifacts/manifests/labels.csv"));
    }

    #[test]
    fn reads_log_tail() {
        let temp_path =
            std::env::temp_dir().join(format!("localagent_logs_test_{}.log", std::process::id()));
        std::fs::write(&temp_path, "a\nb\nc\nd\n").expect("failed to write temp log");
        let lines = read_tail_lines(&temp_path, 2).expect("failed to read tail");
        assert_eq!(lines, vec!["c".to_string(), "d".to_string()]);
        let _ = std::fs::remove_file(temp_path);
    }

    #[test]
    fn benchmark_job_request_serializes_expected_compare_flag() {
        let request = BenchmarkJobRequest {
            training: TrainingJobRequest {
                command: Some(TrainingCommand::Summary),
                manifest: None,
                training_preset: Some("cpu_fast".to_string()),
                experiment_name: Some("demo".to_string()),
                training_backend: Some("pytorch".to_string()),
                model_name: None,
                pretrained_backbone: None,
                train_backbone: None,
                image_size: None,
                batch_size: None,
                epochs: None,
                num_workers: None,
                device: None,
                cache_dir: None,
                resume_from: None,
                checkpoint: None,
                onnx_output: None,
                spec_output: None,
                cache_format: None,
                use_rust_cache: None,
                force_cache: None,
                class_bias: None,
                early_stopping_patience: None,
                early_stopping_min_delta: None,
                enable_early_stopping: None,
                onnx_opset: None,
                export_batch_size: None,
                verify_onnx: None,
                no_progress: None,
            },
            compare_to: Some("artifacts/reports/left.json".to_string()),
            compare_experiment: None,
        };

        assert_eq!(
            request.compare_to.as_deref(),
            Some("artifacts/reports/left.json")
        );
    }
}
