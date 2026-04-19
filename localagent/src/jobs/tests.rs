use super::{
    io::read_tail_lines, BenchmarkJobRequest, JobManager, PipelineCommand, PipelineJobRequest,
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
        num_clusters: None,
        infer_filename_labels: None,
        labels_file: Some("artifacts/manifests/labels.csv".to_string()),
        review_file: None,
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
            pseudo_label_threshold: None,
            pseudo_label_margin: None,
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

#[test]
fn rejects_rust_tch_for_fit_jobs() {
    let temp_root = std::env::temp_dir().join(format!(
        "localagent_jobs_rust_tch_validate_{}",
        std::process::id()
    ));
    let artifact_dir = temp_root.join("artifacts");
    let manifest_dir = artifact_dir.join("manifests");
    let reports_dir = artifact_dir.join("reports");
    std::fs::create_dir_all(&manifest_dir).expect("failed to create manifest dir");
    std::fs::create_dir_all(&reports_dir).expect("failed to create reports dir");
    std::fs::write(
        manifest_dir.join("dataset_manifest.parquet"),
        b"placeholder",
    )
    .expect("failed to create manifest placeholder");
    std::fs::write(
        reports_dir.join("summary.json"),
        r#"{"training_ready_files":4,"trainable_label_counts":{"glass":2,"plastic":2},"accepted_label_source_counts":{"cluster_review":4}}"#,
    )
    .expect("failed to write dataset summary");

    let manager = JobManager::new(RuntimeConfig {
        artifact_dir: artifact_dir.display().to_string(),
        ..RuntimeConfig::default()
    });
    let request = TrainingJobRequest {
        command: Some(TrainingCommand::Fit),
        manifest: Some(
            manifest_dir
                .join("dataset_manifest.parquet")
                .display()
                .to_string(),
        ),
        training_preset: None,
        experiment_name: Some("demo".to_string()),
        training_backend: Some("rust_tch".to_string()),
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
        pseudo_label_threshold: None,
        pseudo_label_margin: None,
        no_progress: None,
    };

    let error = manager
        .validate_training_request(&request, "fit", None)
        .expect_err("fit should reject rust_tch");
    assert!(error.to_string().contains("requires backend `pytorch`"));
}

#[test]
fn warm_cache_requires_existing_manifest() {
    let temp_root = std::env::temp_dir().join(format!(
        "localagent_missing_manifest_{}_warm_cache",
        std::process::id()
    ));
    let missing_manifest = temp_root
        .join("artifacts")
        .join("manifests")
        .join("dataset_manifest.parquet");
    let reports_dir = temp_root.join("artifacts").join("reports");
    if let Some(parent) = missing_manifest.parent() {
        std::fs::create_dir_all(parent).expect("failed to create manifest dir");
    }
    std::fs::create_dir_all(&reports_dir).expect("failed to create reports dir");
    std::fs::write(
        reports_dir.join("summary.json"),
        r#"{"training_ready_files":4,"trainable_label_counts":{"glass":2,"plastic":2},"accepted_label_source_counts":{"cluster_review":4}}"#,
    )
    .expect("failed to write dataset summary");

    let manager = JobManager::new(RuntimeConfig::default());
    let _ = std::fs::remove_file(&missing_manifest);
    let request = TrainingJobRequest {
        command: Some(TrainingCommand::WarmCache),
        manifest: Some(missing_manifest.display().to_string()),
        training_preset: None,
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
        pseudo_label_threshold: None,
        pseudo_label_margin: None,
        no_progress: None,
    };

    let error = manager
        .validate_training_request(&request, "warm-cache", None)
        .expect_err("warm-cache should require a manifest");
    assert!(error
        .to_string()
        .contains("Run the dataset pipeline `run-all` first"));
}

#[test]
fn discovery_commands_require_completed_step_one() {
    let temp_root = std::env::temp_dir().join(format!(
        "localagent_jobs_discovery_step1_{}",
        std::process::id()
    ));
    let artifact_dir = temp_root.join("artifacts");
    std::fs::create_dir_all(artifact_dir.join("manifests"))
        .expect("failed to create manifests dir");
    std::fs::create_dir_all(artifact_dir.join("reports")).expect("failed to create reports dir");

    let manager = JobManager::new(RuntimeConfig {
        artifact_dir: artifact_dir.display().to_string(),
        ..RuntimeConfig::default()
    });
    let request = PipelineJobRequest {
        command: PipelineCommand::Embed,
        raw_dir: None,
        manifest_dir: None,
        report_dir: None,
        min_width: None,
        min_height: None,
        train_ratio: None,
        val_ratio: None,
        test_ratio: None,
        seed: None,
        num_clusters: None,
        infer_filename_labels: None,
        labels_file: None,
        review_file: None,
        output: None,
        no_progress: Some(true),
    };

    let error = manager
        .validate_pipeline_request(&request, "embed")
        .expect_err("embed should require step 1");
    assert!(error.to_string().contains("Step 1"));
    assert!(error.to_string().contains("run-all"));
}

#[test]
fn promote_cluster_labels_requires_saved_review_decisions() {
    let temp_root = std::env::temp_dir().join(format!(
        "localagent_jobs_promote_review_{}",
        std::process::id()
    ));
    let artifact_dir = temp_root.join("artifacts");
    let manifest_dir = artifact_dir.join("manifests");
    let reports_dir = artifact_dir.join("reports");
    std::fs::create_dir_all(&manifest_dir).expect("failed to create manifest dir");
    std::fs::create_dir_all(&reports_dir).expect("failed to create reports dir");
    std::fs::write(
        manifest_dir.join("dataset_manifest.parquet"),
        b"placeholder",
    )
    .expect("failed to write manifest placeholder");
    std::fs::write(
        manifest_dir.join("dataset_manifest.csv"),
        [
            "sample_id,relative_path,label,label_source,annotation_status,review_status,cluster_id,cluster_distance,is_cluster_outlier",
            "sample-a,a.jpg,unknown,unknown,unlabeled,pending_review,1,0.10,false",
            "sample-b,b.jpg,unknown,unknown,unlabeled,pending_review,1,0.15,false",
        ]
        .join("\n"),
    )
    .expect("failed to write manifest csv");
    std::fs::write(
        reports_dir.join("summary.json"),
        r#"{"embedding_artifact_exists":true,"cluster_summary_exists":true,"clustered_files":2,"accepted_label_source_counts":{},"training_ready_files":0,"trainable_label_counts":{}}"#,
    )
    .expect("failed to write summary");

    let manager = JobManager::new(RuntimeConfig {
        artifact_dir: artifact_dir.display().to_string(),
        ..RuntimeConfig::default()
    });
    let request = PipelineJobRequest {
        command: PipelineCommand::PromoteClusterLabels,
        raw_dir: None,
        manifest_dir: None,
        report_dir: None,
        min_width: None,
        min_height: None,
        train_ratio: None,
        val_ratio: None,
        test_ratio: None,
        seed: None,
        num_clusters: None,
        infer_filename_labels: None,
        labels_file: None,
        review_file: None,
        output: None,
        no_progress: Some(true),
    };

    let error = manager
        .validate_pipeline_request(&request, "promote-cluster-labels")
        .expect_err("promote should require saved review decisions");
    assert!(error.to_string().contains("saved cluster decision"));
}

#[test]
fn evaluate_requires_existing_checkpoint() {
    let manager = JobManager::new(RuntimeConfig::default());
    let temp_root =
        std::env::temp_dir().join(format!("localagent_jobs_validate_{}", std::process::id()));
    let manifest_path = temp_root
        .join("artifacts")
        .join("manifests")
        .join("dataset_manifest.parquet");
    let reports_dir = temp_root.join("artifacts").join("reports");
    let checkpoint_path = temp_root.join("missing_checkpoint.pt");
    if let Some(parent) = manifest_path.parent() {
        std::fs::create_dir_all(parent).expect("failed to create temp manifest dir");
    }
    std::fs::create_dir_all(&reports_dir).expect("failed to create reports dir");
    std::fs::write(&manifest_path, b"placeholder").expect("failed to create temp manifest");
    std::fs::write(
        reports_dir.join("summary.json"),
        r#"{"training_ready_files":4,"trainable_label_counts":{"glass":2,"plastic":2},"accepted_label_source_counts":{"cluster_review":4}}"#,
    )
    .expect("failed to write dataset summary");
    let _ = std::fs::remove_file(&checkpoint_path);

    let request = TrainingJobRequest {
        command: Some(TrainingCommand::Evaluate),
        manifest: Some(manifest_path.display().to_string()),
        training_preset: None,
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
        checkpoint: Some(checkpoint_path.display().to_string()),
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
        pseudo_label_threshold: None,
        pseudo_label_margin: None,
        no_progress: None,
    };

    let error = manager
        .validate_training_request(&request, "evaluate", None)
        .expect_err("evaluate should require a checkpoint");
    assert!(error
        .to_string()
        .contains("Run `fit` first or pass `--checkpoint`"));

    let _ = std::fs::remove_file(&manifest_path);
}

#[test]
fn fit_requires_at_least_two_labels_in_dataset_summary() {
    let temp_root = std::env::temp_dir().join(format!(
        "localagent_jobs_summary_validate_{}",
        std::process::id()
    ));
    let artifact_dir = temp_root.join("artifacts");
    let manifest_dir = artifact_dir.join("manifests");
    let reports_dir = artifact_dir.join("reports");
    std::fs::create_dir_all(&manifest_dir).expect("failed to create manifest dir");
    std::fs::create_dir_all(&reports_dir).expect("failed to create reports dir");
    std::fs::write(
        manifest_dir.join("dataset_manifest.parquet"),
        b"placeholder",
    )
    .expect("failed to create manifest placeholder");
    std::fs::write(
        reports_dir.join("summary.json"),
        r#"{"training_ready_files":4,"trainable_label_counts":{"r":4},"accepted_label_source_counts":{"cluster_review":4}}"#,
    )
    .expect("failed to write dataset summary");

    let manager = JobManager::new(RuntimeConfig {
        artifact_dir: artifact_dir.display().to_string(),
        ..RuntimeConfig::default()
    });
    let request = TrainingJobRequest {
        command: Some(TrainingCommand::Fit),
        manifest: None,
        training_preset: None,
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
        pseudo_label_threshold: None,
        pseudo_label_margin: None,
        no_progress: None,
    };

    let error = manager
        .validate_training_request(&request, "fit", None)
        .expect_err("fit should require at least two labels");
    assert!(error
        .to_string()
        .contains("Training requires at least 2 labels in the dataset summary"));
}

#[test]
fn training_commands_require_completed_step_two() {
    let temp_root = std::env::temp_dir().join(format!(
        "localagent_jobs_training_step2_{}",
        std::process::id()
    ));
    let artifact_dir = temp_root.join("artifacts");
    let manifest_dir = artifact_dir.join("manifests");
    let reports_dir = artifact_dir.join("reports");
    std::fs::create_dir_all(&manifest_dir).expect("failed to create manifest dir");
    std::fs::create_dir_all(&reports_dir).expect("failed to create reports dir");
    std::fs::write(
        manifest_dir.join("dataset_manifest.parquet"),
        b"placeholder",
    )
    .expect("failed to create manifest placeholder");
    std::fs::write(
        reports_dir.join("summary.json"),
        r#"{"training_ready_files":4,"trainable_label_counts":{"glass":2,"plastic":2},"accepted_label_source_counts":{}}"#,
    )
    .expect("failed to write dataset summary");

    let manager = JobManager::new(RuntimeConfig {
        artifact_dir: artifact_dir.display().to_string(),
        ..RuntimeConfig::default()
    });
    let request = TrainingJobRequest {
        command: Some(TrainingCommand::Summary),
        manifest: Some(
            manifest_dir
                .join("dataset_manifest.parquet")
                .display()
                .to_string(),
        ),
        training_preset: None,
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
        pseudo_label_threshold: None,
        pseudo_label_margin: None,
        no_progress: None,
    };

    let error = manager
        .validate_training_request(&request, "summary", None)
        .expect_err("summary should require completed step 2");
    assert!(error.to_string().contains("Step 2"));
    assert!(error.to_string().contains("promote-cluster-labels"));
}
