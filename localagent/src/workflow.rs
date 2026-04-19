use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use anyhow::Result;
use serde::Serialize;
use serde_json::Value;

use crate::{
    load_review_state_from_paths, ArtifactKind, ArtifactStore, ClusterReviewError, RuntimeConfig,
};

const STEP1_LOCK_REASON: &str =
    "Complete Step 1 first. Run `run-all` so the dataset manifest and summary exist.";
const STEP3_LOCK_REASON: &str = "Complete Step 2 first. Review clusters, save the review, and run `promote-cluster-labels` before using Step 3.";
const EMBED_LOCK_REASON: &str =
    "Complete Step 1 first. Run `run-all` so the dataset manifest and summary exist.";
const CLUSTER_LOCK_REASON: &str = "Run `embed` first so image similarity vectors exist.";
const CLUSTER_REVIEW_LOCK_REASON: &str =
    "Run `cluster` first so cluster assignments exist in the manifest.";
const PROMOTE_LOCK_REASON: &str =
    "Review at least one cluster, save the review, then run `promote-cluster-labels`.";

const TRAINING_COMMANDS: [&str; 8] = [
    "summary",
    "export-spec",
    "warm-cache",
    "pseudo-label",
    "fit",
    "evaluate",
    "export-onnx",
    "report",
];
const BENCHMARK_COMMAND: &str = "benchmark";

#[derive(Clone, Debug, Serialize)]
pub struct WorkflowStepState {
    pub title: String,
    pub completed: bool,
    pub enabled: bool,
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct WorkflowCommandState {
    pub allowed: bool,
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct WorkflowStatus {
    pub dataset_manifest_exists: bool,
    pub dataset_summary_exists: bool,
    pub embedding_artifact_exists: bool,
    pub cluster_summary_exists: bool,
    pub cluster_ready: bool,
    pub clustered_files: usize,
    pub cluster_outlier_files: usize,
    pub reviewed_cluster_count: usize,
    pub cluster_review_cluster_count: usize,
    pub cluster_review_stale_reset_count: usize,
    pub cluster_review_error: Option<String>,
    pub cluster_review_file: String,
    pub accepted_cluster_review_labels: usize,
    pub training_ready_files: usize,
    pub trainable_labels: Vec<String>,
    pub effective_training_mode: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct WorkflowState {
    pub schema_version: u8,
    pub steps: BTreeMap<String, WorkflowStepState>,
    pub commands: BTreeMap<String, WorkflowCommandState>,
    pub status: WorkflowStatus,
    pub dataset_summary: Option<Value>,
}

impl WorkflowState {
    pub fn from_config(config: RuntimeConfig, review_file: Option<&str>) -> Result<Self> {
        let artifact_store = ArtifactStore::new(config.clone());
        let dataset_summary = artifact_store.read_artifact(ArtifactKind::DatasetSummary, None)?;
        let artifact_root = artifact_root(&config);
        let manifest_path = artifact_root
            .join("manifests")
            .join("dataset_manifest.parquet");
        let manifest_csv_path = artifact_root.join("manifests").join("dataset_manifest.csv");
        let embeddings_path = artifact_root
            .join("manifests")
            .join("dataset_embeddings.npz");
        let cluster_summary_path = artifact_root.join("reports").join("cluster_summary.json");
        let review_path = resolve_review_path(&config, review_file);

        let dataset_manifest_exists = manifest_path.is_file();
        let dataset_summary_exists = dataset_summary.is_some();
        let embedding_artifact_exists = dataset_summary
            .as_ref()
            .and_then(|value| value.get("embedding_artifact_exists"))
            .and_then(Value::as_bool)
            .unwrap_or_else(|| embeddings_path.is_file());
        let cluster_summary_exists = dataset_summary
            .as_ref()
            .and_then(|value| value.get("cluster_summary_exists"))
            .and_then(Value::as_bool)
            .unwrap_or_else(|| cluster_summary_path.is_file());
        let clustered_files = dataset_summary
            .as_ref()
            .and_then(|value| value.get("clustered_files"))
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        let cluster_outlier_files = dataset_summary
            .as_ref()
            .and_then(|value| value.get("cluster_outlier_files"))
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        let accepted_cluster_review_labels = dataset_summary
            .as_ref()
            .and_then(|value| value.get("accepted_label_source_counts"))
            .and_then(|value| value.get("cluster_review"))
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        let training_ready_files = dataset_summary
            .as_ref()
            .and_then(|value| value.get("training_ready_files"))
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        let effective_training_mode = dataset_summary
            .as_ref()
            .and_then(|value| value.get("effective_training_mode"))
            .and_then(Value::as_str)
            .map(str::to_string);
        let trainable_labels = collect_label_keys(
            dataset_summary
                .as_ref()
                .and_then(|value| value.get("trainable_label_counts")),
        );
        let cluster_ready = cluster_summary_exists || clustered_files > 0;
        let step1_completed = dataset_manifest_exists && dataset_summary_exists;
        let step2_completed = accepted_cluster_review_labels > 0;

        let (reviewed_cluster_count, cluster_review_cluster_count, stale_reset_count, review_error) =
            load_review_progress(
                &manifest_csv_path,
                &review_path,
                step1_completed,
                cluster_ready,
            );

        let mut steps = BTreeMap::new();
        steps.insert(
            "dataset".to_string(),
            WorkflowStepState {
                title: "Step 1: Dataset pipeline".to_string(),
                completed: step1_completed,
                enabled: true,
                reason: None,
            },
        );
        steps.insert(
            "discovery".to_string(),
            WorkflowStepState {
                title: "Step 2: Discovery workflow".to_string(),
                completed: step2_completed,
                enabled: step1_completed,
                reason: (!step1_completed).then(|| STEP1_LOCK_REASON.to_string()),
            },
        );
        steps.insert(
            "training".to_string(),
            WorkflowStepState {
                title: "Step 3: Training studio".to_string(),
                completed: false,
                enabled: step2_completed,
                reason: (!step2_completed).then(|| STEP3_LOCK_REASON.to_string()),
            },
        );

        let mut commands = BTreeMap::new();
        commands.insert(
            "embed".to_string(),
            command_state(
                step1_completed,
                (!step1_completed).then(|| EMBED_LOCK_REASON.to_string()),
            ),
        );
        commands.insert(
            "cluster".to_string(),
            command_state(
                step1_completed && embedding_artifact_exists,
                if !step1_completed {
                    Some(EMBED_LOCK_REASON.to_string())
                } else if !embedding_artifact_exists {
                    Some(CLUSTER_LOCK_REASON.to_string())
                } else {
                    None
                },
            ),
        );
        commands.insert(
            "export-cluster-review".to_string(),
            command_state(
                step1_completed && cluster_ready,
                if !step1_completed {
                    Some(EMBED_LOCK_REASON.to_string())
                } else if !cluster_ready {
                    Some(CLUSTER_REVIEW_LOCK_REASON.to_string())
                } else {
                    None
                },
            ),
        );
        commands.insert(
            "promote-cluster-labels".to_string(),
            command_state(
                step1_completed
                    && cluster_ready
                    && reviewed_cluster_count > 0
                    && review_error.is_none(),
                if !step1_completed {
                    Some(EMBED_LOCK_REASON.to_string())
                } else if !cluster_ready {
                    Some(CLUSTER_REVIEW_LOCK_REASON.to_string())
                } else if let Some(message) = review_error.clone() {
                    Some(message)
                } else if reviewed_cluster_count == 0 {
                    Some(PROMOTE_LOCK_REASON.to_string())
                } else {
                    None
                },
            ),
        );
        for command in TRAINING_COMMANDS {
            commands.insert(
                command.to_string(),
                command_state(
                    step2_completed,
                    (!step2_completed).then(|| STEP3_LOCK_REASON.to_string()),
                ),
            );
        }
        commands.insert(
            BENCHMARK_COMMAND.to_string(),
            command_state(
                step2_completed,
                (!step2_completed).then(|| STEP3_LOCK_REASON.to_string()),
            ),
        );

        Ok(Self {
            schema_version: 1,
            steps,
            commands,
            status: WorkflowStatus {
                dataset_manifest_exists,
                dataset_summary_exists,
                embedding_artifact_exists,
                cluster_summary_exists,
                cluster_ready,
                clustered_files,
                cluster_outlier_files,
                reviewed_cluster_count,
                cluster_review_cluster_count,
                cluster_review_stale_reset_count: stale_reset_count,
                cluster_review_error: review_error,
                cluster_review_file: review_path.display().to_string(),
                accepted_cluster_review_labels,
                training_ready_files,
                trainable_labels,
                effective_training_mode,
            },
            dataset_summary,
        })
    }
}

fn command_state(allowed: bool, reason: Option<String>) -> WorkflowCommandState {
    WorkflowCommandState { allowed, reason }
}

fn collect_label_keys(value: Option<&Value>) -> Vec<String> {
    let Some(object) = value.and_then(Value::as_object) else {
        return Vec::new();
    };
    let mut labels = object
        .keys()
        .filter(|label| label.as_str() != "unknown")
        .cloned()
        .collect::<Vec<_>>();
    labels.sort();
    labels
}

fn load_review_progress(
    manifest_csv_path: &Path,
    review_path: &Path,
    step1_completed: bool,
    cluster_ready: bool,
) -> (usize, usize, usize, Option<String>) {
    if !step1_completed || !cluster_ready {
        return (0, 0, 0, None);
    }

    match load_review_state_from_paths(manifest_csv_path, review_path) {
        Ok(state) => (
            state.reviewed_count,
            state.cluster_count,
            state.stale_reset_count,
            None,
        ),
        Err(error) => match error {
            ClusterReviewError::InvalidRequest(message)
                if message.contains("Run `run-all` first")
                    || message.contains("Run `cluster` first")
                    || message.contains("No cluster assignments are available") =>
            {
                (0, 0, 0, None)
            }
            other => (0, 0, 0, Some(other.to_string())),
        },
    }
}

fn artifact_root(config: &RuntimeConfig) -> PathBuf {
    let path = PathBuf::from(&config.artifact_dir);
    if path.is_absolute() {
        path
    } else {
        project_root().join(path)
    }
}

fn resolve_review_path(config: &RuntimeConfig, review_file: Option<&str>) -> PathBuf {
    let Some(trimmed) = review_file.map(str::trim).filter(|value| !value.is_empty()) else {
        return artifact_root(config)
            .join("manifests")
            .join("cluster_review.csv");
    };
    let path = PathBuf::from(trimmed);
    if path.is_absolute() {
        path
    } else {
        project_root().join(path)
    }
}

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::WorkflowState;
    use crate::RuntimeConfig;

    fn temp_root(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "localagent_workflow_state_{name}_{}",
            std::process::id()
        ))
    }

    #[test]
    fn step_one_can_complete_without_any_training_run() {
        let root = temp_root("step1_only");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("artifacts").join("manifests"))
            .expect("failed to create manifests dir");
        fs::create_dir_all(root.join("artifacts").join("reports"))
            .expect("failed to create reports dir");
        fs::write(
            root.join("artifacts")
                .join("manifests")
                .join("dataset_manifest.parquet"),
            b"placeholder",
        )
        .expect("failed to write manifest");
        fs::write(
            root.join("artifacts").join("manifests").join("dataset_manifest.csv"),
            "sample_id,relative_path,label,label_source,annotation_status,review_status,cluster_id,cluster_distance,is_cluster_outlier\nsample-1,a.jpg,unknown,unknown,unlabeled,pending_review,,,,false\n",
        )
        .expect("failed to write manifest csv");
        fs::write(
            root.join("artifacts").join("reports").join("summary.json"),
            r#"{"embedding_artifact_exists":false,"cluster_summary_exists":false,"clustered_files":0,"accepted_label_source_counts":{},"training_ready_files":0,"trainable_label_counts":{},"effective_training_mode":"weak_inferred"}"#,
        )
        .expect("failed to write summary");

        let state = WorkflowState::from_config(
            RuntimeConfig {
                artifact_dir: root.join("artifacts").display().to_string(),
                ..RuntimeConfig::default()
            },
            None,
        )
        .expect("failed to build workflow state");

        assert!(state.steps["dataset"].completed);
        assert!(state.steps["discovery"].enabled);
        assert!(!state.steps["training"].enabled);
        assert!(state.commands["embed"].allowed);
        assert!(!state.commands["cluster"].allowed);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn step_three_unlocks_after_cluster_review_labels_are_accepted() {
        let root = temp_root("step3_unlock");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("artifacts").join("manifests"))
            .expect("failed to create manifests dir");
        fs::create_dir_all(root.join("artifacts").join("reports"))
            .expect("failed to create reports dir");
        fs::write(
            root.join("artifacts")
                .join("manifests")
                .join("dataset_manifest.parquet"),
            b"placeholder",
        )
        .expect("failed to write manifest");
        fs::write(
            root.join("artifacts").join("manifests").join("dataset_manifest.csv"),
            [
                "sample_id,relative_path,label,label_source,annotation_status,review_status,cluster_id,cluster_distance,is_cluster_outlier",
                "sample-a,a.jpg,glass,cluster_review,labeled,cluster_accepted,1,0.10,false",
                "sample-b,b.jpg,glass,cluster_review,labeled,cluster_accepted,1,0.15,false",
            ]
            .join("\n"),
        )
        .expect("failed to write manifest csv");
        fs::write(
            root.join("artifacts").join("manifests").join("cluster_review.csv"),
            "cluster_id,cluster_size,outlier_count,representative_sample_ids,representative_paths,current_majority_label,label,status,notes\n1,2,0,sample-a|sample-b,a.jpg|b.jpg,glass,glass,labeled,\n",
        )
        .expect("failed to write review file");
        fs::write(
            root.join("artifacts").join("reports").join("summary.json"),
            r#"{"embedding_artifact_exists":true,"cluster_summary_exists":true,"clustered_files":2,"accepted_label_source_counts":{"cluster_review":2},"training_ready_files":2,"trainable_label_counts":{"glass":2},"effective_training_mode":"accepted_labels_only"}"#,
        )
        .expect("failed to write summary");

        let state = WorkflowState::from_config(
            RuntimeConfig {
                artifact_dir: root.join("artifacts").display().to_string(),
                ..RuntimeConfig::default()
            },
            None,
        )
        .expect("failed to build workflow state");

        assert!(state.steps["discovery"].completed);
        assert!(state.steps["training"].enabled);
        assert!(state.commands["summary"].allowed);
        assert_eq!(state.status.accepted_cluster_review_labels, 2);
        assert_eq!(state.status.reviewed_cluster_count, 1);

        let _ = fs::remove_dir_all(&root);
    }
}
