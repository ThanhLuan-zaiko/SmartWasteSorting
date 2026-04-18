use std::{
    path::{Path, PathBuf},
    process::Stdio,
};

use anyhow::{Context, Result};
use serde_json::{json, Value};
use time::OffsetDateTime;
use tokio::{process::Command, sync::broadcast};
use tracing::warn;

use super::{
    io::{read_tail_lines, sequence_from_job_id, stream_child_output, timestamp_now},
    JobLogsSnapshot, JobManager, JobRecord, JobStatus, JobStreamEvent, JOB_EVENT_BUFFER,
};
use crate::{ArtifactKind, ArtifactStore, RuntimeConfig};

const DEFAULT_LOG_TAIL_LINES: usize = 200;

impl JobManager {
    pub fn new(config: RuntimeConfig) -> Self {
        let (events, _) = broadcast::channel(JOB_EVENT_BUFFER);
        let manager = Self {
            config,
            jobs: std::sync::Arc::new(std::sync::Mutex::new(std::collections::BTreeMap::new())),
            children: std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            events,
            sequence: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
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

    pub(super) async fn spawn_job(
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
        let child = std::sync::Arc::new(tokio::sync::Mutex::new(child));

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
            artifacts: std::collections::BTreeMap::new(),
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
                super::JobLogStream::Stdout,
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
                super::JobLogStream::Stderr,
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

    pub(super) fn finish_job(
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

    pub(super) fn discover_artifacts(
        &self,
        experiment_name: Option<&str>,
    ) -> std::collections::BTreeMap<String, String> {
        let artifact_store = ArtifactStore::new(self.config.clone());
        let experiment = experiment_name.unwrap_or(artifact_store.default_experiment());
        let mut artifacts = std::collections::BTreeMap::new();

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

    pub(super) fn load_existing_jobs(&self) -> Result<()> {
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
        self.sequence
            .store(latest_sequence, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    pub(super) fn persist_job(&self, job: &JobRecord) -> Result<()> {
        let path = self.jobs_dir().join(format!("{}.json", job.job_id));
        let payload = serde_json::to_string_pretty(job)?;
        std::fs::write(&path, payload)
            .with_context(|| format!("failed to persist job manifest: {}", path.display()))
    }

    pub(super) fn emit_event(&self, event: JobStreamEvent) {
        let _ = self.events.send(event);
    }

    pub(super) fn next_job_id(&self) -> String {
        let sequence = self
            .sequence
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;
        let millis = OffsetDateTime::now_utc().unix_timestamp_nanos() / 1_000_000;
        format!("job-{millis}-{sequence}")
    }

    pub(super) fn project_root(&self) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    pub(super) fn jobs_dir(&self) -> PathBuf {
        let path = PathBuf::from(&self.config.artifact_dir).join("jobs");
        let _ = std::fs::create_dir_all(&path);
        path
    }

    pub(super) fn logs_dir(&self) -> PathBuf {
        let path = self.jobs_dir().join("logs");
        let _ = std::fs::create_dir_all(&path);
        path
    }
}
