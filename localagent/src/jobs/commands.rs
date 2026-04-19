use std::path::PathBuf;

use anyhow::{Context, Result};
use serde_json::Value;

use super::{
    BenchmarkJobRequest, JobManager, PipelineCommand, PipelineJobRequest, TrainingJobRequest,
};
use crate::load_review_state_from_paths;

const STEP1_REQUIRED_MESSAGE: &str =
    "Complete Step 1 first. Run `run-all` so the dataset manifest and summary exist.";
const STEP2_REQUIRED_MESSAGE: &str = "Complete Step 2 first. Review clusters, save the review, and run `promote-cluster-labels` before using Step 3.";

impl JobManager {
    pub async fn spawn_pipeline_job(
        &self,
        request: PipelineJobRequest,
    ) -> Result<super::JobRecord> {
        let command_name = request.command.as_cli().to_string();
        self.validate_pipeline_request(&request, &command_name)?;
        let (_command_name, args) = self.pipeline_args(&request)?;

        self.spawn_job(
            "dataset_pipeline",
            None,
            args,
            "Runs dataset pipeline steps through the Python CLI.".to_string(),
        )
        .await
    }

    pub async fn spawn_training_job(
        &self,
        request: TrainingJobRequest,
    ) -> Result<super::JobRecord> {
        let command_name = request
            .command
            .context("command is required for training jobs")?
            .as_cli()
            .to_string();
        let experiment_name = request.experiment_name.clone();
        self.validate_training_request(&request, &command_name, None)?;
        let args = self.training_args(&request, &command_name, None)?;
        self.spawn_job(
            "training_pipeline",
            experiment_name,
            args,
            "Runs training pipeline steps through the Python CLI.".to_string(),
        )
        .await
    }

    pub async fn spawn_benchmark_job(
        &self,
        request: BenchmarkJobRequest,
    ) -> Result<super::JobRecord> {
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
        self.validate_training_request(&request.training, "benchmark", compare_to.as_deref())?;
        let args = self.training_args(&request.training, "benchmark", compare_to.as_deref())?;
        self.spawn_job(
            "benchmark_pipeline",
            experiment_name,
            args,
            "Runs a benchmark workflow using persisted artifact reports.".to_string(),
        )
        .await
    }

    pub(super) fn pipeline_args(
        &self,
        request: &PipelineJobRequest,
    ) -> Result<(String, Vec<String>)> {
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
        push_optional_i64(&mut args, "--num-clusters", request.num_clusters);
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
        if matches!(
            request.command,
            PipelineCommand::ExportLabelingTemplate | PipelineCommand::ExportClusterReview
        ) {
            push_optional_path(&mut args, "--output", request.output.as_deref());
        }
        if matches!(request.command, PipelineCommand::PromoteClusterLabels) {
            let review_file = request
                .review_file
                .as_deref()
                .context("review_file is required for promote-cluster-labels")?;
            push_optional_path(&mut args, "--review-file", Some(review_file));
        }
        if request.no_progress.unwrap_or(true) {
            args.push("--no-progress".to_string());
        }

        Ok((command_name, args))
    }

    pub(super) fn validate_pipeline_request(
        &self,
        request: &PipelineJobRequest,
        command_name: &str,
    ) -> Result<()> {
        if !matches!(
            request.command,
            PipelineCommand::Embed
                | PipelineCommand::Cluster
                | PipelineCommand::ExportClusterReview
                | PipelineCommand::PromoteClusterLabels
        ) {
            return Ok(());
        }

        let manifest_path = self.dataset_manifest_path_for_pipeline_request(request);
        let summary_path = self.dataset_summary_path_for_pipeline_request(request);
        if !manifest_path.is_file() || !summary_path.is_file() {
            anyhow::bail!(
                "Discovery command `{command_name}` requires Step 1 to be complete. {STEP1_REQUIRED_MESSAGE} Expected manifest at `{}` and summary at `{}`.",
                manifest_path.display(),
                summary_path.display()
            );
        }

        let summary = self
            .read_pipeline_dataset_summary(request)?
            .with_context(|| format!("dataset summary missing at {}", summary_path.display()))?;

        if matches!(request.command, PipelineCommand::Cluster) {
            let embeddings_path = self.embeddings_path_for_pipeline_request(request);
            let embedding_exists = summary
                .get("embedding_artifact_exists")
                .and_then(Value::as_bool)
                .unwrap_or_else(|| embeddings_path.is_file());
            if !embedding_exists {
                anyhow::bail!(
                    "Discovery command `{command_name}` requires `embed` first so image similarity vectors exist. Expected embeddings at `{}`.",
                    embeddings_path.display()
                );
            }
        }

        if matches!(
            request.command,
            PipelineCommand::ExportClusterReview | PipelineCommand::PromoteClusterLabels
        ) {
            let cluster_summary_path = self.cluster_summary_path_for_pipeline_request(request);
            let cluster_ready = summary
                .get("cluster_summary_exists")
                .and_then(Value::as_bool)
                .unwrap_or_else(|| cluster_summary_path.is_file())
                || summary
                    .get("clustered_files")
                    .and_then(Value::as_u64)
                    .unwrap_or(0)
                    > 0;
            if !cluster_ready {
                anyhow::bail!(
                    "Discovery command `{command_name}` requires `cluster` first so cluster assignments exist in the manifest."
                );
            }
        }

        if matches!(request.command, PipelineCommand::PromoteClusterLabels) {
            let review_path = self.review_path_for_pipeline_request(request);
            let manifest_csv_path = self.dataset_manifest_csv_path_for_pipeline_request(request);
            let state = load_review_state_from_paths(&manifest_csv_path, &review_path)?;
            if state.reviewed_count == 0 {
                anyhow::bail!(
                    "Discovery command `promote-cluster-labels` requires at least one saved cluster decision in `{}`. Review clusters, save the review, then retry.",
                    review_path.display()
                );
            }
        }

        Ok(())
    }

    pub(super) fn training_args(
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
        if command_name == "pseudo-label" {
            push_optional_f64(
                &mut args,
                "--pseudo-label-threshold",
                request.pseudo_label_threshold,
            );
            push_optional_f64(
                &mut args,
                "--pseudo-label-margin",
                request.pseudo_label_margin,
            );
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

    pub(super) fn validate_training_request(
        &self,
        request: &TrainingJobRequest,
        command_name: &str,
        compare_to: Option<&str>,
    ) -> Result<()> {
        self.validate_training_step_unlocked(request)?;

        let training_backend = request
            .training_backend
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or("pytorch");

        match training_backend {
            "pytorch" | "rust_tch" => {}
            _ => {
                anyhow::bail!(
                    "Unsupported training backend `{training_backend}`. Supported backends are `pytorch` and `rust_tch`."
                );
            }
        }

        if matches!(
            command_name,
            "fit" | "evaluate" | "export-onnx" | "pseudo-label"
        ) && training_backend != "pytorch"
        {
            anyhow::bail!(
                "Training command `{command_name}` requires backend `pytorch`. `rust_tch` is preview-only for `summary`, `export-spec`, and `benchmark` in this build."
            );
        }

        if matches!(
            command_name,
            "warm-cache" | "fit" | "evaluate" | "benchmark" | "export-labels" | "pseudo-label"
        ) {
            let manifest_path = self.manifest_path_for_request(request);
            if !manifest_path.is_file() {
                anyhow::bail!(
                    "Training command `{command_name}` requires a dataset manifest at `{}`. Run the dataset pipeline `run-all` first or pass `--manifest`.",
                    manifest_path.display()
                );
            }
        }
        if matches!(command_name, "fit" | "benchmark") {
            self.validate_dataset_training_readiness()?;
        }

        if matches!(command_name, "evaluate" | "export-onnx" | "pseudo-label") {
            let checkpoint_path = self.checkpoint_path_for_request(request);
            if !checkpoint_path.is_file() {
                anyhow::bail!(
                    "Training command `{command_name}` requires a checkpoint at `{}`. Run `fit` first or pass `--checkpoint`.",
                    checkpoint_path.display()
                );
            }
        }

        if command_name == "benchmark" {
            if let Some(compare_to_path) = compare_to {
                let resolved_compare_path = self.resolve_project_path(compare_to_path);
                if !resolved_compare_path.is_file() {
                    anyhow::bail!(
                        "Benchmark comparison report not found at `{}`. Choose an experiment with an existing benchmark report or omit comparison.",
                        resolved_compare_path.display()
                    );
                }
            }
        } else if compare_to.is_some() {
            anyhow::bail!("compare_to is only valid for benchmark jobs");
        }

        Ok(())
    }

    pub(super) fn validate_training_step_unlocked(
        &self,
        request: &TrainingJobRequest,
    ) -> Result<()> {
        let summary_path = self.dataset_summary_path_for_training_request(request);
        if !summary_path.is_file() {
            anyhow::bail!("{STEP2_REQUIRED_MESSAGE}");
        }

        let payload = std::fs::read_to_string(&summary_path).with_context(|| {
            format!("failed to read dataset summary: {}", summary_path.display())
        })?;
        let summary: Value = serde_json::from_str(&payload).with_context(|| {
            format!(
                "failed to parse dataset summary JSON: {}",
                summary_path.display()
            )
        })?;
        let accepted_cluster_review_labels = summary
            .get("accepted_label_source_counts")
            .and_then(|value| value.get("cluster_review"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        if accepted_cluster_review_labels == 0 {
            anyhow::bail!("{STEP2_REQUIRED_MESSAGE}");
        }
        Ok(())
    }

    pub(super) fn validate_dataset_training_readiness(&self) -> Result<()> {
        let summary_path = self.dataset_summary_path();
        if !summary_path.is_file() {
            return Ok(());
        }

        let payload = std::fs::read_to_string(&summary_path).with_context(|| {
            format!("failed to read dataset summary: {}", summary_path.display())
        })?;
        let summary: Value = serde_json::from_str(&payload).with_context(|| {
            format!(
                "failed to parse dataset summary JSON: {}",
                summary_path.display()
            )
        })?;

        if summary.get("training_ready_files").and_then(Value::as_u64) == Some(0) {
            anyhow::bail!(
                "Dataset summary reports zero training-ready files. Re-run the dataset pipeline or curate labels before training."
            );
        }

        let label_counts = summary
            .get("trainable_label_counts")
            .and_then(Value::as_object)
            .or_else(|| summary.get("label_counts").and_then(Value::as_object));
        if let Some(label_counts) = label_counts {
            let labels = label_counts
                .keys()
                .filter(|label| label.as_str() != "unknown")
                .cloned()
                .collect::<Vec<_>>();
            if labels.len() < 2 {
                anyhow::bail!(
                    "Training requires at least 2 labels in the dataset summary. Found {} label(s): {}. The current dataset filenames appear to map everything into one class.",
                    labels.len(),
                    if labels.is_empty() {
                        "<none>".to_string()
                    } else {
                        labels.join(", ")
                    }
                );
            }
        }

        Ok(())
    }

    pub(super) fn resolve_project_path(&self, raw_path: &str) -> PathBuf {
        let path = PathBuf::from(raw_path);
        if path.is_absolute() {
            path
        } else {
            self.project_root().join(path)
        }
    }

    pub(super) fn artifact_root(&self) -> PathBuf {
        self.resolve_project_path(&self.config.artifact_dir)
    }

    pub(super) fn artifact_reports_dir(&self) -> PathBuf {
        self.artifact_root().join("reports")
    }

    pub(super) fn dataset_summary_path(&self) -> PathBuf {
        self.artifact_reports_dir().join("summary.json")
    }

    pub(super) fn pipeline_manifest_dir(&self, request: &PipelineJobRequest) -> PathBuf {
        request
            .manifest_dir
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .map(|value| self.resolve_project_path(value))
            .unwrap_or_else(|| self.artifact_root().join("manifests"))
    }

    pub(super) fn pipeline_report_dir(&self, request: &PipelineJobRequest) -> PathBuf {
        request
            .report_dir
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .map(|value| self.resolve_project_path(value))
            .unwrap_or_else(|| self.artifact_root().join("reports"))
    }

    pub(super) fn dataset_manifest_path_for_pipeline_request(
        &self,
        request: &PipelineJobRequest,
    ) -> PathBuf {
        self.pipeline_manifest_dir(request)
            .join("dataset_manifest.parquet")
    }

    pub(super) fn dataset_manifest_csv_path_for_pipeline_request(
        &self,
        request: &PipelineJobRequest,
    ) -> PathBuf {
        self.pipeline_manifest_dir(request)
            .join("dataset_manifest.csv")
    }

    pub(super) fn embeddings_path_for_pipeline_request(
        &self,
        request: &PipelineJobRequest,
    ) -> PathBuf {
        self.pipeline_manifest_dir(request)
            .join("dataset_embeddings.npz")
    }

    pub(super) fn cluster_summary_path_for_pipeline_request(
        &self,
        request: &PipelineJobRequest,
    ) -> PathBuf {
        self.pipeline_report_dir(request)
            .join("cluster_summary.json")
    }

    pub(super) fn dataset_summary_path_for_pipeline_request(
        &self,
        request: &PipelineJobRequest,
    ) -> PathBuf {
        self.pipeline_report_dir(request).join("summary.json")
    }

    pub(super) fn review_path_for_pipeline_request(&self, request: &PipelineJobRequest) -> PathBuf {
        request
            .review_file
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .map(|value| self.resolve_project_path(value))
            .unwrap_or_else(|| {
                self.pipeline_manifest_dir(request)
                    .join("cluster_review.csv")
            })
    }

    pub(super) fn read_pipeline_dataset_summary(
        &self,
        request: &PipelineJobRequest,
    ) -> Result<Option<Value>> {
        let summary_path = self.dataset_summary_path_for_pipeline_request(request);
        if !summary_path.is_file() {
            return Ok(None);
        }
        let payload = std::fs::read_to_string(&summary_path).with_context(|| {
            format!("failed to read dataset summary: {}", summary_path.display())
        })?;
        let summary: Value = serde_json::from_str(&payload).with_context(|| {
            format!(
                "failed to parse dataset summary JSON: {}",
                summary_path.display()
            )
        })?;
        Ok(Some(summary))
    }

    pub(super) fn dataset_summary_path_for_training_request(
        &self,
        request: &TrainingJobRequest,
    ) -> PathBuf {
        if let Some(manifest) = request
            .manifest
            .as_deref()
            .filter(|value| !value.trim().is_empty())
        {
            let manifest_path = self.resolve_project_path(manifest);
            if let Some(artifact_root) = manifest_path.parent().and_then(|path| path.parent()) {
                return artifact_root.join("reports").join("summary.json");
            }
        }
        self.dataset_summary_path()
    }

    pub(super) fn manifest_path_for_request(&self, request: &TrainingJobRequest) -> PathBuf {
        if let Some(manifest) = request
            .manifest
            .as_deref()
            .filter(|value| !value.trim().is_empty())
        {
            return self.resolve_project_path(manifest);
        }
        self.artifact_root()
            .join("manifests")
            .join("dataset_manifest.parquet")
    }

    pub(super) fn checkpoint_path_for_request(&self, request: &TrainingJobRequest) -> PathBuf {
        if let Some(checkpoint) = request
            .checkpoint
            .as_deref()
            .filter(|value| !value.trim().is_empty())
        {
            return self.resolve_project_path(checkpoint);
        }

        let experiment_name = request
            .experiment_name
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or(&self.config.experiment_name);
        self.artifact_root()
            .join("checkpoints")
            .join(format!("{experiment_name}.pt"))
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
