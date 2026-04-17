use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use serde_json::{json, Map, Value};
use time::{format_description::well_known::Rfc3339, OffsetDateTime};

use crate::RuntimeConfig;

#[derive(Clone, Copy, Debug)]
pub enum ArtifactKind {
    DatasetSummary,
    Training,
    Evaluation,
    Export,
    Benchmark,
    ExperimentSpec,
    Bundle,
    ModelManifest,
}

impl ArtifactKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DatasetSummary => "dataset",
            Self::Training => "training",
            Self::Evaluation => "evaluation",
            Self::Export => "export",
            Self::Benchmark => "benchmark",
            Self::ExperimentSpec => "experiment-spec",
            Self::Bundle => "bundle",
            Self::ModelManifest => "model-manifest",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ArtifactStore {
    config: RuntimeConfig,
}

impl ArtifactStore {
    pub fn new(config: RuntimeConfig) -> Self {
        Self { config }
    }

    pub fn default_experiment(&self) -> &str {
        &self.config.experiment_name
    }

    pub fn read_artifact(
        &self,
        kind: ArtifactKind,
        experiment: Option<&str>,
    ) -> Result<Option<Value>> {
        let path = self.artifact_path(kind, experiment);
        self.read_json_file(&path)
    }

    pub fn overview(&self, experiment: Option<&str>) -> Result<Value> {
        let experiment_name = self.experiment_name(experiment);
        if let Some(bundle) = self.read_artifact(ArtifactKind::Bundle, Some(&experiment_name))? {
            return Ok(bundle);
        }

        Ok(json!({
            "schema_version": 1,
            "experiment_name": experiment_name,
            "dataset_summary": self.read_artifact(ArtifactKind::DatasetSummary, None)?,
            "training": self.read_artifact(ArtifactKind::Training, Some(&experiment_name))?,
            "evaluation": self.read_artifact(ArtifactKind::Evaluation, Some(&experiment_name))?,
            "export": self.read_artifact(ArtifactKind::Export, Some(&experiment_name))?,
            "benchmark": self.read_artifact(ArtifactKind::Benchmark, Some(&experiment_name))?,
            "experiment_spec": self.read_artifact(ArtifactKind::ExperimentSpec, Some(&experiment_name))?,
            "model_manifest": self.read_artifact(ArtifactKind::ModelManifest, None)?,
        }))
    }

    pub fn training_overview(&self, experiment: Option<&str>) -> Result<Option<Value>> {
        let experiment_name = self.experiment_name(experiment);
        let payload = match self.read_artifact(ArtifactKind::Training, Some(&experiment_name))? {
            Some(value) => value,
            None => return Ok(None),
        };

        let history = payload
            .get("history")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let chart = json!({
            "epochs": self.history_series_int(&history, "epoch"),
            "train_loss": self.history_series_f64(&history, "train_loss"),
            "val_loss": self.history_series_f64(&history, "val_loss"),
            "train_accuracy": self.history_series_f64(&history, "train_accuracy"),
            "val_accuracy": self.history_series_f64(&history, "val_accuracy"),
        });

        Ok(Some(json!({
            "schema_version": 1,
            "experiment_name": experiment_name,
            "training_backend": payload.get("training_backend").cloned().unwrap_or(Value::Null),
            "best_epoch": payload.get("best_epoch").cloned().unwrap_or(Value::Null),
            "best_loss": payload.get("best_loss").cloned().unwrap_or(Value::Null),
            "epochs_completed": payload.get("epochs_completed").cloned().unwrap_or(Value::Null),
            "stopped_early": payload.get("stopped_early").cloned().unwrap_or(Value::Bool(false)),
            "stop_reason": payload.get("stop_reason").cloned().unwrap_or(Value::Null),
            "class_bias_strategy": payload.get("class_bias_strategy").cloned().unwrap_or(Value::Null),
            "class_weight_map": payload.get("class_weight_map").cloned().unwrap_or(Value::Null),
            "device": payload.get("device").cloned().unwrap_or(Value::Null),
            "chart": chart,
            "history": history,
        })))
    }

    pub fn benchmark_overview(&self, experiment: Option<&str>) -> Result<Value> {
        let experiment_name = self.experiment_name(experiment);
        Ok(json!({
            "schema_version": 1,
            "experiment_name": experiment_name,
            "benchmark": self.read_artifact(ArtifactKind::Benchmark, Some(&experiment_name))?,
            "experiment_spec": self.read_artifact(ArtifactKind::ExperimentSpec, Some(&experiment_name))?,
            "evaluation": self.read_artifact(ArtifactKind::Evaluation, Some(&experiment_name))?,
            "export": self.read_artifact(ArtifactKind::Export, Some(&experiment_name))?,
            "model_manifest": self.read_artifact(ArtifactKind::ModelManifest, None)?,
        }))
    }

    pub fn dashboard_summary(&self, experiment: Option<&str>) -> Result<Value> {
        let experiment_name = self.experiment_name(experiment);
        let dataset_summary = self.read_artifact(ArtifactKind::DatasetSummary, None)?;
        let training = self.training_overview(Some(&experiment_name))?;
        let benchmark = self.read_artifact(ArtifactKind::Benchmark, Some(&experiment_name))?;
        let experiment_spec =
            self.read_artifact(ArtifactKind::ExperimentSpec, Some(&experiment_name))?;
        let evaluation = self.read_artifact(ArtifactKind::Evaluation, Some(&experiment_name))?;
        let export = self.read_artifact(ArtifactKind::Export, Some(&experiment_name))?;
        let model_manifest = self.read_artifact(ArtifactKind::ModelManifest, None)?;
        let benchmark_metrics = benchmark
            .as_ref()
            .and_then(|value| value.get("metrics"))
            .cloned()
            .unwrap_or(Value::Null);
        let benchmark_capability = benchmark
            .as_ref()
            .and_then(|value| value.get("backend_capability"))
            .cloned()
            .or_else(|| {
                experiment_spec
                    .as_ref()
                    .and_then(|value| value.get("backend_capability"))
                    .cloned()
            })
            .unwrap_or(Value::Null);
        let training_backend = benchmark
            .as_ref()
            .and_then(|value| value.get("training_backend"))
            .cloned()
            .or_else(|| {
                training
                    .as_ref()
                    .and_then(|value| value.get("training_backend"))
                    .cloned()
            })
            .or_else(|| {
                experiment_spec
                    .as_ref()
                    .and_then(|value| value.get("training_backend"))
                    .cloned()
            })
            .unwrap_or(Value::Null);

        let cards = json!({
            "best_epoch": training
                .as_ref()
                .and_then(|value| value.get("best_epoch"))
                .cloned()
                .unwrap_or(Value::Null),
            "best_loss": training
                .as_ref()
                .and_then(|value| value.get("best_loss"))
                .cloned()
                .unwrap_or(Value::Null),
            "epochs_completed": training
                .as_ref()
                .and_then(|value| value.get("epochs_completed"))
                .cloned()
                .unwrap_or(Value::Null),
            "accuracy": evaluation
                .as_ref()
                .and_then(|value| value.get("accuracy"))
                .cloned()
                .or_else(|| benchmark_metrics.get("accuracy").cloned())
                .unwrap_or(Value::Null),
            "macro_f1": evaluation
                .as_ref()
                .and_then(|value| value.get("macro_f1"))
                .cloned()
                .or_else(|| benchmark_metrics.get("macro_f1").cloned())
                .unwrap_or(Value::Null),
            "weighted_f1": evaluation
                .as_ref()
                .and_then(|value| value.get("weighted_f1"))
                .cloned()
                .or_else(|| benchmark_metrics.get("weighted_f1").cloned())
                .unwrap_or(Value::Null),
            "training_backend": training_backend,
            "benchmark_status": benchmark
                .as_ref()
                .and_then(|value| value.get("status"))
                .cloned()
                .unwrap_or(Value::Null),
            "benchmark_duration_seconds": benchmark_metrics
                .get("total_duration_seconds")
                .cloned()
                .unwrap_or(Value::Null),
            "peak_stage_rss_mb": benchmark_metrics
                .get("peak_stage_rss_mb")
                .cloned()
                .unwrap_or(Value::Null),
            "backend_supported": benchmark_capability
                .get("supported")
                .cloned()
                .unwrap_or(Value::Null),
            "onnx_verified": export
                .as_ref()
                .and_then(|value| value.get("verification"))
                .and_then(|value| value.get("verified"))
                .cloned()
                .unwrap_or(Value::Null),
        });

        let charts = training
            .as_ref()
            .and_then(|value| value.get("chart"))
            .cloned()
            .unwrap_or_else(|| json!({}));
        let confusion_matrix = evaluation
            .as_ref()
            .and_then(|value| value.get("confusion_matrix"))
            .cloned()
            .unwrap_or(Value::Null);
        let per_class = evaluation
            .as_ref()
            .and_then(|value| value.get("per_class"))
            .cloned()
            .unwrap_or(Value::Null);

        Ok(json!({
            "schema_version": 1,
            "experiment_name": experiment_name,
            "status": {
                "dataset_ready": dataset_summary.is_some(),
                "training_ready": training.is_some(),
                "benchmark_ready": benchmark.is_some(),
                "experiment_spec_ready": experiment_spec.is_some(),
                "evaluation_ready": evaluation.is_some(),
                "export_ready": export.is_some(),
                "model_manifest_ready": model_manifest.is_some(),
            },
            "cards": cards,
            "charts": charts,
            "dataset_summary": dataset_summary,
            "benchmark": {
                "report": benchmark,
                "evaluation": evaluation,
                "confusion_matrix": confusion_matrix,
                "per_class": per_class,
            },
            "export": export,
            "experiment_spec": experiment_spec,
            "model_manifest": model_manifest,
        }))
    }

    pub fn run_index(&self) -> Result<Value> {
        self.sync_run_index()
    }

    pub fn sync_run_index(&self) -> Result<Value> {
        let mut runs = self
            .detect_experiments()?
            .into_iter()
            .map(|experiment_name| self.build_run_entry(&experiment_name))
            .collect::<Result<Vec<_>>>()?;
        runs.sort_by(|left, right| {
            let left_time = left
                .get("latest_generated_at")
                .and_then(Value::as_str)
                .unwrap_or("");
            let right_time = right
                .get("latest_generated_at")
                .and_then(Value::as_str)
                .unwrap_or("");
            right_time.cmp(left_time)
        });

        let payload = json!({
            "schema_version": 1,
            "generated_at": timestamp_now(),
            "runs": runs,
        });
        let path = self.run_index_path();
        std::fs::create_dir_all(self.artifact_reports_dir())
            .context("failed to create artifact reports dir for run index")?;
        std::fs::write(&path, serde_json::to_string_pretty(&payload)?)
            .with_context(|| format!("failed to write run index: {}", path.display()))?;
        Ok(payload)
    }

    pub fn run_detail(&self, experiment: Option<&str>) -> Result<Value> {
        let experiment_name = self.experiment_name(experiment);
        Ok(json!({
            "schema_version": 1,
            "experiment_name": experiment_name,
            "dashboard_summary": self.dashboard_summary(Some(&experiment_name))?,
            "overview": self.overview(Some(&experiment_name))?,
            "training": self.read_artifact(ArtifactKind::Training, Some(&experiment_name))?,
            "evaluation": self.read_artifact(ArtifactKind::Evaluation, Some(&experiment_name))?,
            "export": self.read_artifact(ArtifactKind::Export, Some(&experiment_name))?,
            "benchmark": self.read_artifact(ArtifactKind::Benchmark, Some(&experiment_name))?,
            "experiment_spec": self.read_artifact(ArtifactKind::ExperimentSpec, Some(&experiment_name))?,
            "model_manifest": self.read_artifact(ArtifactKind::ModelManifest, None)?,
        }))
    }

    pub fn compare_runs(&self, left: &str, right: &str) -> Result<Value> {
        let left_experiment = self.experiment_name(Some(left));
        let right_experiment = self.experiment_name(Some(right));
        let left_payload = self
            .read_artifact(ArtifactKind::Benchmark, Some(&left_experiment))?
            .with_context(|| format!("missing benchmark artifact for {}", left_experiment))?;
        let right_payload = self
            .read_artifact(ArtifactKind::Benchmark, Some(&right_experiment))?
            .with_context(|| format!("missing benchmark artifact for {}", right_experiment))?;

        Ok(json!({
            "schema_version": 1,
            "left_experiment_name": left_experiment,
            "right_experiment_name": right_experiment,
            "left_backend": left_payload.get("training_backend").cloned().unwrap_or(Value::Null),
            "right_backend": right_payload.get("training_backend").cloned().unwrap_or(Value::Null),
            "duration_delta_seconds": self.metric_delta(&left_payload, &right_payload, "total_duration_seconds"),
            "fit_stage_delta_seconds": self.stage_duration_delta(&left_payload, &right_payload, "fit"),
            "accuracy_delta": self.metric_delta(&left_payload, &right_payload, "accuracy"),
            "macro_f1_delta": self.metric_delta(&left_payload, &right_payload, "macro_f1"),
            "weighted_f1_delta": self.metric_delta(&left_payload, &right_payload, "weighted_f1"),
        }))
    }

    pub fn artifact_path(&self, kind: ArtifactKind, experiment: Option<&str>) -> PathBuf {
        match kind {
            ArtifactKind::DatasetSummary => {
                self.artifact_dir().join("reports").join("summary.json")
            }
            ArtifactKind::Training => self.artifact_reports_dir().join(format!(
                "{}_training.json",
                self.experiment_name(experiment)
            )),
            ArtifactKind::Evaluation => self.artifact_reports_dir().join(format!(
                "{}_evaluation.json",
                self.experiment_name(experiment)
            )),
            ArtifactKind::Export => self
                .artifact_reports_dir()
                .join(format!("{}_export.json", self.experiment_name(experiment))),
            ArtifactKind::Benchmark => self.artifact_reports_dir().join(format!(
                "{}_benchmark.json",
                self.experiment_name(experiment)
            )),
            ArtifactKind::ExperimentSpec => self.artifact_reports_dir().join(format!(
                "{}_experiment_spec.json",
                self.experiment_name(experiment)
            )),
            ArtifactKind::Bundle => self
                .artifact_reports_dir()
                .join(format!("{}_report.json", self.experiment_name(experiment))),
            ArtifactKind::ModelManifest => PathBuf::from(&self.config.model_manifest_path),
        }
    }

    fn artifact_dir(&self) -> PathBuf {
        PathBuf::from(&self.config.artifact_dir)
    }

    fn artifact_reports_dir(&self) -> PathBuf {
        self.artifact_dir().join("reports")
    }

    fn run_index_path(&self) -> PathBuf {
        self.artifact_reports_dir().join("run_index.json")
    }

    fn experiment_name(&self, experiment: Option<&str>) -> String {
        experiment
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or(&self.config.experiment_name)
            .to_string()
    }

    fn read_json_file(&self, path: &Path) -> Result<Option<Value>> {
        if !path.is_file() {
            return Ok(None);
        }

        let payload = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read artifact file: {}", path.display()))?;
        let value = serde_json::from_str::<Value>(&payload)
            .with_context(|| format!("failed to parse artifact json: {}", path.display()))?;
        Ok(Some(value))
    }

    fn detect_experiments(&self) -> Result<Vec<String>> {
        let mut experiments = BTreeSet::new();
        if !self.artifact_reports_dir().is_dir() {
            return Ok(Vec::new());
        }
        for entry in std::fs::read_dir(self.artifact_reports_dir())
            .context("failed to read artifact reports dir")?
        {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|value| value.to_str()) else {
                continue;
            };
            if let Some(experiment_name) = detect_experiment_name(stem) {
                experiments.insert(experiment_name.to_string());
            }
        }
        Ok(experiments.into_iter().collect())
    }

    fn build_run_entry(&self, experiment_name: &str) -> Result<Value> {
        let training = self.read_artifact(ArtifactKind::Training, Some(experiment_name))?;
        let evaluation = self.read_artifact(ArtifactKind::Evaluation, Some(experiment_name))?;
        let export = self.read_artifact(ArtifactKind::Export, Some(experiment_name))?;
        let benchmark = self.read_artifact(ArtifactKind::Benchmark, Some(experiment_name))?;
        let experiment_spec =
            self.read_artifact(ArtifactKind::ExperimentSpec, Some(experiment_name))?;
        let dashboard = self.dashboard_summary(Some(experiment_name))?;

        let latest_generated_at = [
            training.as_ref(),
            evaluation.as_ref(),
            export.as_ref(),
            benchmark.as_ref(),
            experiment_spec.as_ref(),
        ]
        .into_iter()
        .flatten()
        .filter_map(|payload| payload.get("generated_at").and_then(Value::as_str))
        .max()
        .unwrap_or("")
        .to_string();

        let training_backend = benchmark
            .as_ref()
            .and_then(|value| value.get("training_backend"))
            .cloned()
            .or_else(|| {
                experiment_spec
                    .as_ref()
                    .and_then(|value| value.get("training_backend"))
                    .cloned()
            })
            .or_else(|| {
                training
                    .as_ref()
                    .and_then(|value| value.get("training_backend"))
                    .cloned()
            })
            .unwrap_or(Value::Null);

        Ok(json!({
            "schema_version": 1,
            "experiment_name": experiment_name,
            "latest_generated_at": latest_generated_at,
            "training_backend": training_backend,
            "available": {
                "training": training.is_some(),
                "evaluation": evaluation.is_some(),
                "export": export.is_some(),
                "benchmark": benchmark.is_some(),
                "experiment_spec": experiment_spec.is_some(),
            },
            "cards": dashboard.get("cards").cloned().unwrap_or(Value::Null),
            "artifact_paths": {
                "training": self.path_if_exists(ArtifactKind::Training, Some(experiment_name)),
                "evaluation": self.path_if_exists(ArtifactKind::Evaluation, Some(experiment_name)),
                "export": self.path_if_exists(ArtifactKind::Export, Some(experiment_name)),
                "benchmark": self.path_if_exists(ArtifactKind::Benchmark, Some(experiment_name)),
                "experiment_spec": self.path_if_exists(ArtifactKind::ExperimentSpec, Some(experiment_name)),
                "bundle": self.path_if_exists(ArtifactKind::Bundle, Some(experiment_name)),
            }
        }))
    }

    fn path_if_exists(&self, kind: ArtifactKind, experiment: Option<&str>) -> Value {
        let path = self.artifact_path(kind, experiment);
        if path.is_file() {
            Value::String(path.display().to_string())
        } else {
            Value::Null
        }
    }

    fn metric_delta(&self, left_payload: &Value, right_payload: &Value, metric_key: &str) -> Value {
        delta_value(
            left_payload
                .get("metrics")
                .and_then(|value| value.get(metric_key))
                .and_then(Value::as_f64),
            right_payload
                .get("metrics")
                .and_then(|value| value.get(metric_key))
                .and_then(Value::as_f64),
        )
    }

    fn stage_duration_delta(
        &self,
        left_payload: &Value,
        right_payload: &Value,
        stage_name: &str,
    ) -> Value {
        delta_value(
            left_payload
                .get("stages")
                .and_then(|value| value.get(stage_name))
                .and_then(|value| value.get("duration_seconds"))
                .and_then(Value::as_f64),
            right_payload
                .get("stages")
                .and_then(|value| value.get(stage_name))
                .and_then(|value| value.get("duration_seconds"))
                .and_then(Value::as_f64),
        )
    }

    fn history_series_int(&self, history: &[Value], key: &str) -> Value {
        Value::Array(
            history
                .iter()
                .map(|entry| entry.get(key).cloned().unwrap_or(Value::Null))
                .collect(),
        )
    }

    fn history_series_f64(&self, history: &[Value], key: &str) -> Value {
        Value::Array(
            history
                .iter()
                .map(|entry| {
                    entry
                        .get(key)
                        .and_then(Value::as_f64)
                        .map(Value::from)
                        .unwrap_or(Value::Null)
                })
                .collect(),
        )
    }
}

pub fn to_api_envelope(kind: &str, experiment: &str, payload: Value) -> Value {
    let mut map = Map::new();
    map.insert("kind".to_string(), Value::String(kind.to_string()));
    map.insert(
        "experiment_name".to_string(),
        Value::String(experiment.to_string()),
    );
    map.insert("payload".to_string(), payload);
    Value::Object(map)
}

fn detect_experiment_name(stem: &str) -> Option<&str> {
    for suffix in [
        "_training",
        "_evaluation",
        "_export",
        "_benchmark",
        "_experiment_spec",
        "_report",
    ] {
        if let Some(experiment_name) = stem.strip_suffix(suffix) {
            return Some(experiment_name);
        }
    }
    None
}

fn delta_value(left: Option<f64>, right: Option<f64>) -> Value {
    match (left, right) {
        (Some(left), Some(right)) => Value::from(right - left),
        _ => Value::Null,
    }
}

fn timestamp_now() -> String {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| OffsetDateTime::now_utc().unix_timestamp().to_string())
}

#[cfg(test)]
mod tests {
    use super::{ArtifactKind, ArtifactStore};
    use crate::RuntimeConfig;

    #[test]
    fn reads_training_artifact_for_selected_experiment() {
        let test_root =
            std::env::temp_dir().join(format!("localagent_artifacts_test_{}", std::process::id()));
        let report_dir = test_root.join("artifacts").join("reports");
        let model_dir = test_root.join("models");
        std::fs::create_dir_all(&report_dir).expect("failed to create report dir");
        std::fs::create_dir_all(&model_dir).expect("failed to create model dir");

        let training_path = report_dir.join("demo_training.json");
        let manifest_path = model_dir.join("model_manifest.json");
        std::fs::write(
            &training_path,
            r#"{"best_loss":0.25,"history":[{"epoch":1}]}"#,
        )
        .expect("failed to write training report");
        std::fs::write(&manifest_path, r#"{"labels":["glass","plastic"]}"#)
            .expect("failed to write model manifest");

        let store = ArtifactStore::new(RuntimeConfig {
            artifact_dir: test_root.join("artifacts").display().to_string(),
            model_manifest_path: manifest_path.display().to_string(),
            experiment_name: "demo".to_string(),
            ..RuntimeConfig::default()
        });

        let training = store
            .read_artifact(ArtifactKind::Training, None)
            .expect("failed to read training artifact")
            .expect("missing training artifact");
        let overview = store
            .training_overview(None)
            .expect("failed to build training overview")
            .expect("missing training overview");

        assert_eq!(training["best_loss"], 0.25);
        assert_eq!(overview["chart"]["epochs"][0], 1);
        assert_eq!(overview["best_loss"], 0.25);

        let _ = std::fs::remove_dir_all(&test_root);
    }

    #[test]
    fn builds_dashboard_summary_payload() {
        let test_root =
            std::env::temp_dir().join(format!("localagent_dashboard_test_{}", std::process::id()));
        let report_dir = test_root.join("artifacts").join("reports");
        let model_dir = test_root.join("models");
        std::fs::create_dir_all(&report_dir).expect("failed to create report dir");
        std::fs::create_dir_all(&model_dir).expect("failed to create model dir");

        std::fs::write(
            report_dir.join("summary.json"),
            r#"{"training_ready_files":10}"#,
        )
        .expect("failed to write dataset summary");
        std::fs::write(
            report_dir.join("demo_training.json"),
            r#"{"best_loss":0.2,"best_epoch":3,"epochs_completed":3,"history":[{"epoch":1,"train_loss":1.0,"train_accuracy":0.5}]}"#,
        )
        .expect("failed to write training report");
        std::fs::write(
            report_dir.join("demo_evaluation.json"),
            r#"{"accuracy":0.8,"macro_f1":0.75,"weighted_f1":0.78,"confusion_matrix":[[4,1],[0,5]],"per_class":{"glass":{"support":5}}}"#,
        )
        .expect("failed to write evaluation report");
        std::fs::write(
            report_dir.join("demo_export.json"),
            r#"{"verification":{"verified":true},"onnx_path":"models/waste_classifier.onnx"}"#,
        )
        .expect("failed to write export report");
        std::fs::write(
            report_dir.join("demo_benchmark.json"),
            r#"{"status":"completed","training_backend":"pytorch","backend_capability":{"supported":true},"metrics":{"total_duration_seconds":12.5,"peak_stage_rss_mb":256.0,"accuracy":0.8,"macro_f1":0.75,"weighted_f1":0.78}}"#,
        )
        .expect("failed to write benchmark report");
        std::fs::write(
            report_dir.join("demo_experiment_spec.json"),
            r#"{"training_backend":"pytorch","model_name":"resnet18","backend_capability":{"supported":true}}"#,
        )
        .expect("failed to write experiment spec");
        std::fs::write(
            model_dir.join("model_manifest.json"),
            r#"{"labels":["glass","plastic"]}"#,
        )
        .expect("failed to write model manifest");

        let store = ArtifactStore::new(RuntimeConfig {
            artifact_dir: test_root.join("artifacts").display().to_string(),
            model_manifest_path: model_dir.join("model_manifest.json").display().to_string(),
            experiment_name: "demo".to_string(),
            ..RuntimeConfig::default()
        });

        let dashboard = store
            .dashboard_summary(None)
            .expect("failed to build dashboard summary");

        assert_eq!(dashboard["cards"]["best_epoch"], 3);
        assert_eq!(dashboard["cards"]["accuracy"], 0.8);
        assert_eq!(dashboard["cards"]["onnx_verified"], true);
        assert_eq!(dashboard["cards"]["training_backend"], "pytorch");
        assert_eq!(dashboard["cards"]["benchmark_status"], "completed");
        assert_eq!(dashboard["status"]["training_ready"], true);
        assert_eq!(dashboard["status"]["benchmark_ready"], true);
        assert_eq!(dashboard["status"]["experiment_spec_ready"], true);
        assert_eq!(dashboard["benchmark"]["evaluation"]["macro_f1"], 0.75);
        assert_eq!(
            dashboard["benchmark"]["report"]["metrics"]["peak_stage_rss_mb"],
            256.0
        );

        let _ = std::fs::remove_dir_all(&test_root);
    }

    #[test]
    fn builds_benchmark_overview_payload() {
        let test_root =
            std::env::temp_dir().join(format!("localagent_benchmark_test_{}", std::process::id()));
        let report_dir = test_root.join("artifacts").join("reports");
        let model_dir = test_root.join("models");
        std::fs::create_dir_all(&report_dir).expect("failed to create report dir");
        std::fs::create_dir_all(&model_dir).expect("failed to create model dir");

        std::fs::write(
            report_dir.join("demo_benchmark.json"),
            r#"{"status":"unsupported","training_backend":"rust_tch","backend_capability":{"supported":false}}"#,
        )
        .expect("failed to write benchmark report");
        std::fs::write(
            report_dir.join("demo_experiment_spec.json"),
            r#"{"training_backend":"rust_tch","model_name":"resnet18"}"#,
        )
        .expect("failed to write experiment spec");
        std::fs::write(
            model_dir.join("model_manifest.json"),
            r#"{"labels":["glass","plastic"]}"#,
        )
        .expect("failed to write model manifest");

        let store = ArtifactStore::new(RuntimeConfig {
            artifact_dir: test_root.join("artifacts").display().to_string(),
            model_manifest_path: model_dir.join("model_manifest.json").display().to_string(),
            experiment_name: "demo".to_string(),
            ..RuntimeConfig::default()
        });

        let overview = store
            .benchmark_overview(None)
            .expect("failed to build benchmark overview");

        assert_eq!(overview["benchmark"]["status"], "unsupported");
        assert_eq!(overview["experiment_spec"]["training_backend"], "rust_tch");

        let _ = std::fs::remove_dir_all(&test_root);
    }

    #[test]
    fn writes_run_index_from_report_files() {
        let test_root =
            std::env::temp_dir().join(format!("localagent_run_index_test_{}", std::process::id()));
        let report_dir = test_root.join("artifacts").join("reports");
        let model_dir = test_root.join("models");
        std::fs::create_dir_all(&report_dir).expect("failed to create report dir");
        std::fs::create_dir_all(&model_dir).expect("failed to create model dir");

        std::fs::write(
            report_dir.join("alpha_training.json"),
            r#"{"generated_at":"2026-04-17T10:00:00Z","best_epoch":3,"best_loss":0.2,"training_backend":"pytorch","history":[{"epoch":1}]}"#,
        )
        .expect("failed to write training report");
        std::fs::write(
            report_dir.join("alpha_benchmark.json"),
            r#"{"generated_at":"2026-04-17T10:05:00Z","training_backend":"pytorch","status":"completed","metrics":{"accuracy":0.8,"macro_f1":0.75,"weighted_f1":0.78}}"#,
        )
        .expect("failed to write benchmark report");
        std::fs::write(
            model_dir.join("model_manifest.json"),
            r#"{"labels":["glass","plastic"]}"#,
        )
        .expect("failed to write model manifest");

        let store = ArtifactStore::new(RuntimeConfig {
            artifact_dir: test_root.join("artifacts").display().to_string(),
            model_manifest_path: model_dir.join("model_manifest.json").display().to_string(),
            experiment_name: "alpha".to_string(),
            ..RuntimeConfig::default()
        });

        let payload = store.sync_run_index().expect("failed to write run index");
        let run_index_path = test_root
            .join("artifacts")
            .join("reports")
            .join("run_index.json");

        assert!(run_index_path.is_file());
        assert_eq!(payload["runs"][0]["experiment_name"], "alpha");
        assert_eq!(payload["runs"][0]["cards"]["benchmark_status"], "completed");

        let _ = std::fs::remove_dir_all(&test_root);
    }

    #[test]
    fn compares_two_benchmark_runs() {
        let test_root =
            std::env::temp_dir().join(format!("localagent_compare_test_{}", std::process::id()));
        let report_dir = test_root.join("artifacts").join("reports");
        let model_dir = test_root.join("models");
        std::fs::create_dir_all(&report_dir).expect("failed to create report dir");
        std::fs::create_dir_all(&model_dir).expect("failed to create model dir");

        std::fs::write(
            report_dir.join("baseline_benchmark.json"),
            r#"{"training_backend":"pytorch","stages":{"fit":{"duration_seconds":12.0}},"metrics":{"total_duration_seconds":20.0,"accuracy":0.78,"macro_f1":0.74,"weighted_f1":0.76}}"#,
        )
        .expect("failed to write left benchmark");
        std::fs::write(
            report_dir.join("candidate_benchmark.json"),
            r#"{"training_backend":"rust_tch","stages":{"fit":{"duration_seconds":10.0}},"metrics":{"total_duration_seconds":17.5,"accuracy":0.8,"macro_f1":0.75,"weighted_f1":0.78}}"#,
        )
        .expect("failed to write right benchmark");
        std::fs::write(
            model_dir.join("model_manifest.json"),
            r#"{"labels":["glass","plastic"]}"#,
        )
        .expect("failed to write model manifest");

        let store = ArtifactStore::new(RuntimeConfig {
            artifact_dir: test_root.join("artifacts").display().to_string(),
            model_manifest_path: model_dir.join("model_manifest.json").display().to_string(),
            experiment_name: "baseline".to_string(),
            ..RuntimeConfig::default()
        });

        let comparison = store
            .compare_runs("baseline", "candidate")
            .expect("failed to compare runs");

        assert_eq!(comparison["duration_delta_seconds"], -2.5);
        assert_eq!(comparison["fit_stage_delta_seconds"], -2.0);
        assert!(comparison["accuracy_delta"]
            .as_f64()
            .map(|value| (value - 0.02).abs() < 1e-9)
            .unwrap_or(false));

        let _ = std::fs::remove_dir_all(&test_root);
    }
}
