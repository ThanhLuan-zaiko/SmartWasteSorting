use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::{json, Map, Value};

use crate::RuntimeConfig;

#[derive(Clone, Copy, Debug)]
pub enum ArtifactKind {
    DatasetSummary,
    Training,
    Evaluation,
    Export,
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
            "evaluation": self.read_artifact(ArtifactKind::Evaluation, Some(&experiment_name))?,
            "export": self.read_artifact(ArtifactKind::Export, Some(&experiment_name))?,
            "model_manifest": self.read_artifact(ArtifactKind::ModelManifest, None)?,
        }))
    }

    pub fn dashboard_summary(&self, experiment: Option<&str>) -> Result<Value> {
        let experiment_name = self.experiment_name(experiment);
        let dataset_summary = self.read_artifact(ArtifactKind::DatasetSummary, None)?;
        let training = self.training_overview(Some(&experiment_name))?;
        let evaluation = self.read_artifact(ArtifactKind::Evaluation, Some(&experiment_name))?;
        let export = self.read_artifact(ArtifactKind::Export, Some(&experiment_name))?;
        let model_manifest = self.read_artifact(ArtifactKind::ModelManifest, None)?;

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
                .unwrap_or(Value::Null),
            "macro_f1": evaluation
                .as_ref()
                .and_then(|value| value.get("macro_f1"))
                .cloned()
                .unwrap_or(Value::Null),
            "weighted_f1": evaluation
                .as_ref()
                .and_then(|value| value.get("weighted_f1"))
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
                "evaluation_ready": evaluation.is_some(),
                "export_ready": export.is_some(),
                "model_manifest_ready": model_manifest.is_some(),
            },
            "cards": cards,
            "charts": charts,
            "dataset_summary": dataset_summary,
            "benchmark": {
                "evaluation": evaluation,
                "confusion_matrix": confusion_matrix,
                "per_class": per_class,
            },
            "export": export,
            "model_manifest": model_manifest,
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
        assert_eq!(dashboard["status"]["training_ready"], true);
        assert_eq!(dashboard["benchmark"]["evaluation"]["macro_f1"], 0.75);

        let _ = std::fs::remove_dir_all(&test_root);
    }
}
