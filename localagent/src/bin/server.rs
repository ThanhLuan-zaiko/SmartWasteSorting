use actix_web::{get, post, web, App, HttpRequest, HttpResponse, HttpServer, Responder};
use actix_ws::{Message, MessageStream, Session};
use futures_util::StreamExt;
use localagent_rs::{
    init_tracing, to_api_envelope, ArtifactKind, ArtifactStore, BenchmarkJobRequest, JobManager,
    JobStreamEvent, PipelineJobRequest, RuntimeConfig, TrainingJobRequest, WasteClassifier,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::{Component, Path, PathBuf};
use tokio::{fs, sync::broadcast, time::Duration};

#[derive(Debug, Deserialize)]
struct ClassificationRequest {
    sample_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ArtifactQuery {
    experiment: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DatasetImageQuery {
    relative_path: String,
}

#[derive(Debug, Deserialize)]
struct JobLogsQuery {
    tail_lines: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct JobStreamQuery {
    job_id: Option<String>,
    tail_lines: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct CompareQuery {
    #[serde(rename = "with")]
    with_experiment: String,
}

#[derive(Debug, Deserialize)]
struct ExperimentPath {
    experiment_name: String,
}

#[derive(Debug, Deserialize)]
struct JobPath {
    job_id: String,
}

#[get("/health")]
async fn health(store: web::Data<ArtifactStore>, jobs: web::Data<JobManager>) -> impl Responder {
    let experiment = store.default_experiment().to_string();
    HttpResponse::Ok().json(json!({
        "status": "ok",
        "service": "localagent-server",
        "default_experiment": experiment,
        "jobs_tracked": jobs.list_jobs().len(),
        "artifacts": {
            "training": store
                .artifact_path(ArtifactKind::Training, Some(store.default_experiment()))
                .is_file(),
            "evaluation": store
                .artifact_path(ArtifactKind::Evaluation, Some(store.default_experiment()))
                .is_file(),
            "export": store
                .artifact_path(ArtifactKind::Export, Some(store.default_experiment()))
                .is_file(),
            "benchmark": store
                .artifact_path(ArtifactKind::Benchmark, Some(store.default_experiment()))
                .is_file(),
            "experiment_spec": store
                .artifact_path(ArtifactKind::ExperimentSpec, Some(store.default_experiment()))
                .is_file(),
            "run_index": store.run_index().is_ok(),
            "model_manifest": store
                .artifact_path(ArtifactKind::ModelManifest, None)
                .is_file(),
        }
    }))
}

#[post("/classify")]
async fn classify(
    classifier: web::Data<WasteClassifier>,
    request: web::Json<ClassificationRequest>,
) -> impl Responder {
    match classifier.classify_batch(&request.sample_ids) {
        Ok(results) => HttpResponse::Ok().json(results),
        Err(error) => HttpResponse::InternalServerError().json(json!({
            "error": error.to_string(),
        })),
    }
}

#[get("/jobs")]
async fn list_jobs(manager: web::Data<JobManager>) -> impl Responder {
    HttpResponse::Ok().json(json!({
        "schema_version": 1,
        "jobs": manager.list_jobs(),
    }))
}

#[get("/jobs/{job_id}")]
async fn get_job(manager: web::Data<JobManager>, path: web::Path<JobPath>) -> impl Responder {
    match manager.get_job(&path.job_id) {
        Some(job) => HttpResponse::Ok().json(job),
        None => HttpResponse::NotFound().json(json!({
            "error": "job not found",
            "job_id": path.job_id,
        })),
    }
}

#[post("/jobs/pipeline")]
async fn create_pipeline_job(
    manager: web::Data<JobManager>,
    request: web::Json<PipelineJobRequest>,
) -> impl Responder {
    match manager.spawn_pipeline_job(request.into_inner()).await {
        Ok(job) => HttpResponse::Accepted().json(job),
        Err(error) => HttpResponse::BadRequest().json(json!({
            "error": error.to_string(),
        })),
    }
}

#[post("/jobs/training")]
async fn create_training_job(
    manager: web::Data<JobManager>,
    request: web::Json<TrainingJobRequest>,
) -> impl Responder {
    match manager.spawn_training_job(request.into_inner()).await {
        Ok(job) => HttpResponse::Accepted().json(job),
        Err(error) => HttpResponse::BadRequest().json(json!({
            "error": error.to_string(),
        })),
    }
}

#[post("/jobs/benchmark")]
async fn create_benchmark_job(
    manager: web::Data<JobManager>,
    request: web::Json<BenchmarkJobRequest>,
) -> impl Responder {
    match manager.spawn_benchmark_job(request.into_inner()).await {
        Ok(job) => HttpResponse::Accepted().json(job),
        Err(error) => HttpResponse::BadRequest().json(json!({
            "error": error.to_string(),
        })),
    }
}

#[post("/jobs/{job_id}/cancel")]
async fn cancel_job(manager: web::Data<JobManager>, path: web::Path<JobPath>) -> impl Responder {
    match manager.cancel_job(&path.job_id).await {
        Ok(job) => HttpResponse::Ok().json(job),
        Err(error) => HttpResponse::BadRequest().json(json!({
            "error": error.to_string(),
            "job_id": path.job_id,
        })),
    }
}

#[get("/jobs/{job_id}/logs")]
async fn job_logs(
    manager: web::Data<JobManager>,
    path: web::Path<JobPath>,
    query: web::Query<JobLogsQuery>,
) -> impl Responder {
    match manager.job_logs(&path.job_id, query.tail_lines) {
        Ok(payload) => HttpResponse::Ok().json(payload),
        Err(error) => HttpResponse::BadRequest().json(json!({
            "error": error.to_string(),
            "job_id": path.job_id,
        })),
    }
}

#[get("/ws/jobs")]
async fn jobs_ws(
    request: HttpRequest,
    payload: web::Payload,
    manager: web::Data<JobManager>,
    query: web::Query<JobStreamQuery>,
) -> actix_web::Result<HttpResponse> {
    let (response, session, msg_stream) = actix_ws::handle(&request, payload)?;
    let manager = manager.get_ref().clone();
    let query = query.into_inner();

    actix_web::rt::spawn(async move {
        stream_job_events(session, msg_stream, manager, query).await;
    });

    Ok(response)
}

#[get("/runs")]
async fn list_runs(store: web::Data<ArtifactStore>) -> impl Responder {
    match store.run_index() {
        Ok(payload) => HttpResponse::Ok().json(payload),
        Err(error) => HttpResponse::InternalServerError().json(json!({
            "error": error.to_string(),
        })),
    }
}

#[get("/runs/{experiment_name}")]
async fn run_detail(
    store: web::Data<ArtifactStore>,
    jobs: web::Data<JobManager>,
    path: web::Path<ExperimentPath>,
) -> impl Responder {
    match store.run_detail(Some(&path.experiment_name)) {
        Ok(mut payload) => {
            if let Some(object) = payload.as_object_mut() {
                object.insert(
                    "job_history".to_string(),
                    json!(jobs.jobs_for_experiment(&path.experiment_name)),
                );
            }
            HttpResponse::Ok().json(payload)
        }
        Err(error) => HttpResponse::InternalServerError().json(json!({
            "error": error.to_string(),
            "experiment_name": path.experiment_name,
        })),
    }
}

#[get("/runs/{experiment_name}/compare")]
async fn compare_runs(
    store: web::Data<ArtifactStore>,
    path: web::Path<ExperimentPath>,
    query: web::Query<CompareQuery>,
) -> impl Responder {
    match store.compare_runs(&path.experiment_name, &query.with_experiment) {
        Ok(payload) => HttpResponse::Ok().json(payload),
        Err(error) => HttpResponse::BadRequest().json(json!({
            "error": error.to_string(),
            "left_experiment_name": path.experiment_name,
            "right_experiment_name": query.with_experiment,
        })),
    }
}

#[get("/presets/training")]
async fn training_presets() -> impl Responder {
    HttpResponse::Ok().json(json!({
        "schema_version": 1,
        "presets": {
            "cpu_fast": {
                "model_name": "mobilenet_v3_small",
                "image_size": 160,
                "batch_size": 32,
                "cache_format": "raw",
                "class_bias": "loss",
            },
            "cpu_balanced": {
                "model_name": "resnet18",
                "image_size": 224,
                "batch_size": 16,
                "cache_format": "raw",
                "class_bias": "loss",
            },
            "cpu_stronger": {
                "model_name": "efficientnet_b0",
                "image_size": 224,
                "batch_size": 8,
                "cache_format": "raw",
                "class_bias": "loss",
            }
        }
    }))
}

#[get("/presets/pipeline")]
async fn pipeline_presets() -> impl Responder {
    HttpResponse::Ok().json(json!({
        "schema_version": 1,
        "dataset_commands": [
            "scan",
            "split",
            "report",
            "run-all",
            "embed",
            "cluster",
            "export-cluster-review",
            "promote-cluster-labels",
            "export-labeling-template",
            "import-labels",
            "validate-labels"
        ],
        "training_commands": [
            "summary",
            "export-spec",
            "export-labels",
            "warm-cache",
            "pseudo-label",
            "fit",
            "evaluate",
            "export-onnx",
            "report",
            "benchmark"
        ]
    }))
}

#[get("/artifacts/overview")]
async fn artifacts_overview(
    store: web::Data<ArtifactStore>,
    query: web::Query<ArtifactQuery>,
) -> impl Responder {
    match store.overview(query.experiment.as_deref()) {
        Ok(payload) => HttpResponse::Ok().json(payload),
        Err(error) => HttpResponse::InternalServerError().json(json!({
            "error": error.to_string(),
        })),
    }
}

#[get("/artifacts/training-overview")]
async fn artifacts_training_overview(
    store: web::Data<ArtifactStore>,
    query: web::Query<ArtifactQuery>,
) -> impl Responder {
    let experiment = query
        .experiment
        .as_deref()
        .unwrap_or(store.default_experiment());
    match store.training_overview(query.experiment.as_deref()) {
        Ok(Some(payload)) => {
            HttpResponse::Ok().json(to_api_envelope("training-overview", experiment, payload))
        }
        Ok(None) => HttpResponse::NotFound().json(json!({
            "error": "training overview artifact not found",
            "experiment_name": experiment,
            "path": store
                .artifact_path(ArtifactKind::Training, Some(experiment))
                .display()
                .to_string(),
        })),
        Err(error) => HttpResponse::InternalServerError().json(json!({
            "error": error.to_string(),
        })),
    }
}

#[get("/artifacts/benchmarks")]
async fn artifacts_benchmarks(
    store: web::Data<ArtifactStore>,
    query: web::Query<ArtifactQuery>,
) -> impl Responder {
    match store.benchmark_overview(query.experiment.as_deref()) {
        Ok(payload) => HttpResponse::Ok().json(payload),
        Err(error) => HttpResponse::InternalServerError().json(json!({
            "error": error.to_string(),
        })),
    }
}

#[get("/artifacts/benchmark")]
async fn artifacts_benchmark(
    store: web::Data<ArtifactStore>,
    query: web::Query<ArtifactQuery>,
) -> impl Responder {
    artifact_response(&store, ArtifactKind::Benchmark, query.experiment.as_deref())
}

#[get("/artifacts/experiment-spec")]
async fn artifacts_experiment_spec(
    store: web::Data<ArtifactStore>,
    query: web::Query<ArtifactQuery>,
) -> impl Responder {
    artifact_response(
        &store,
        ArtifactKind::ExperimentSpec,
        query.experiment.as_deref(),
    )
}

#[get("/dashboard/summary")]
async fn dashboard_summary(
    store: web::Data<ArtifactStore>,
    query: web::Query<ArtifactQuery>,
) -> impl Responder {
    let experiment = query
        .experiment
        .as_deref()
        .unwrap_or(store.default_experiment());
    match store.dashboard_summary(query.experiment.as_deref()) {
        Ok(payload) => {
            HttpResponse::Ok().json(to_api_envelope("dashboard-summary", experiment, payload))
        }
        Err(error) => HttpResponse::InternalServerError().json(json!({
            "error": error.to_string(),
        })),
    }
}

#[get("/dataset/image")]
async fn dataset_image(
    store: web::Data<ArtifactStore>,
    query: web::Query<DatasetImageQuery>,
) -> impl Responder {
    let sanitized_relative = match sanitize_relative_dataset_path(&query.relative_path) {
        Some(path) => path,
        None => {
            return HttpResponse::BadRequest().json(json!({
                "error": "relative_path must be a non-empty dataset-relative path",
                "relative_path": query.relative_path,
            }));
        }
    };

    let dataset_root = match dataset_root_from_store(&store) {
        Ok(path) => path,
        Err(error) => {
            return HttpResponse::NotFound().json(json!({
                "error": error,
            }));
        }
    };

    let canonical_root = match std::fs::canonicalize(&dataset_root) {
        Ok(path) => path,
        Err(error) => {
            return HttpResponse::InternalServerError().json(json!({
                "error": format!(
                    "failed to resolve dataset root {}: {}",
                    dataset_root.display(),
                    error
                ),
            }));
        }
    };

    let candidate = canonical_root.join(&sanitized_relative);
    let canonical_candidate = match std::fs::canonicalize(&candidate) {
        Ok(path) => path,
        Err(error) => {
            return HttpResponse::NotFound().json(json!({
                "error": format!(
                    "dataset image not found for {}: {}",
                    sanitized_relative.display(),
                    error
                ),
                "relative_path": query.relative_path,
            }));
        }
    };

    if !canonical_candidate.starts_with(&canonical_root) {
        return HttpResponse::BadRequest().json(json!({
            "error": "relative_path resolved outside the dataset root",
            "relative_path": query.relative_path,
        }));
    }

    match fs::read(&canonical_candidate).await {
        Ok(bytes) => HttpResponse::Ok()
            .insert_header(("Cache-Control", "no-store"))
            .content_type(content_type_for_path(&canonical_candidate))
            .body(bytes),
        Err(error) => HttpResponse::InternalServerError().json(json!({
            "error": format!(
                "failed to read dataset image {}: {}",
                canonical_candidate.display(),
                error
            ),
            "relative_path": query.relative_path,
        })),
    }
}

#[get("/artifacts/dataset")]
async fn artifacts_dataset(
    store: web::Data<ArtifactStore>,
    query: web::Query<ArtifactQuery>,
) -> impl Responder {
    artifact_response(
        &store,
        ArtifactKind::DatasetSummary,
        query.experiment.as_deref(),
    )
}

#[get("/artifacts/training")]
async fn artifacts_training(
    store: web::Data<ArtifactStore>,
    query: web::Query<ArtifactQuery>,
) -> impl Responder {
    artifact_response(&store, ArtifactKind::Training, query.experiment.as_deref())
}

#[get("/artifacts/evaluation")]
async fn artifacts_evaluation(
    store: web::Data<ArtifactStore>,
    query: web::Query<ArtifactQuery>,
) -> impl Responder {
    artifact_response(
        &store,
        ArtifactKind::Evaluation,
        query.experiment.as_deref(),
    )
}

#[get("/artifacts/export")]
async fn artifacts_export(
    store: web::Data<ArtifactStore>,
    query: web::Query<ArtifactQuery>,
) -> impl Responder {
    artifact_response(&store, ArtifactKind::Export, query.experiment.as_deref())
}

#[get("/artifacts/model-manifest")]
async fn artifacts_model_manifest(
    store: web::Data<ArtifactStore>,
    query: web::Query<ArtifactQuery>,
) -> impl Responder {
    artifact_response(
        &store,
        ArtifactKind::ModelManifest,
        query.experiment.as_deref(),
    )
}

fn artifact_response(
    store: &ArtifactStore,
    kind: ArtifactKind,
    experiment: Option<&str>,
) -> HttpResponse {
    let experiment_name = experiment.unwrap_or(store.default_experiment());
    match store.read_artifact(kind, experiment) {
        Ok(Some(payload)) => {
            HttpResponse::Ok().json(to_api_envelope(kind.as_str(), experiment_name, payload))
        }
        Ok(None) => HttpResponse::NotFound().json(json!({
            "error": format!("{} artifact not found", kind.as_str()),
            "kind": kind.as_str(),
            "experiment_name": experiment_name,
            "path": store
                .artifact_path(kind, experiment)
                .display()
                .to_string(),
        })),
        Err(error) => HttpResponse::InternalServerError().json(json!({
            "error": error.to_string(),
        })),
    }
}

fn dataset_root_from_store(store: &ArtifactStore) -> Result<PathBuf, String> {
    let summary_path = store.artifact_path(ArtifactKind::DatasetSummary, None);
    let payload = std::fs::read_to_string(&summary_path).map_err(|error| {
        format!(
            "dataset summary not found at {}. Run the dataset pipeline first. {}",
            summary_path.display(),
            error
        )
    })?;
    let summary: Value = serde_json::from_str(&payload).map_err(|error| {
        format!(
            "failed to parse dataset summary at {}: {}",
            summary_path.display(),
            error
        )
    })?;
    let dataset_root = summary
        .get("dataset_root")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            format!(
                "dataset summary at {} does not contain a usable dataset_root",
                summary_path.display()
            )
        })?;
    Ok(PathBuf::from(dataset_root))
}

fn sanitize_relative_dataset_path(raw: &str) -> Option<PathBuf> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let path = Path::new(trimmed);
    if path.is_absolute() {
        return None;
    }

    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(segment) => normalized.push(segment),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => return None,
        }
    }

    if normalized.as_os_str().is_empty() {
        return None;
    }

    Some(normalized)
}

fn content_type_for_path(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase())
        .as_deref()
    {
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("png") => "image/png",
        Some("webp") => "image/webp",
        Some("bmp") => "image/bmp",
        Some("gif") => "image/gif",
        _ => "application/octet-stream",
    }
}

async fn stream_job_events(
    mut session: Session,
    mut msg_stream: MessageStream,
    manager: JobManager,
    query: JobStreamQuery,
) {
    let mut receiver = manager.subscribe_events();
    let active_job_id = query.job_id.clone();
    let tail_lines = query.tail_lines;

    match manager.stream_snapshot(active_job_id.as_deref(), tail_lines) {
        Ok(snapshot) => {
            if send_stream_event(&mut session, &snapshot).await.is_err() {
                return;
            }
        }
        Err(error) => {
            let _ = send_stream_event(
                &mut session,
                &JobStreamEvent::ResyncRequired {
                    reason: error.to_string(),
                },
            )
            .await;
        }
    }

    let mut heartbeat = tokio::time::interval(Duration::from_secs(20));

    loop {
        tokio::select! {
            maybe_message = msg_stream.next() => match maybe_message {
                Some(Ok(Message::Ping(bytes))) => {
                    if session.pong(&bytes).await.is_err() {
                        break;
                    }
                }
                Some(Ok(Message::Close(reason))) => {
                    let _ = session.close(reason).await;
                    break;
                }
                Some(Ok(Message::Pong(_)))
                | Some(Ok(Message::Text(_)))
                | Some(Ok(Message::Binary(_)))
                | Some(Ok(Message::Continuation(_)))
                | Some(Ok(Message::Nop)) => {}
                Some(Err(_)) | None => break,
            },
            received = receiver.recv() => match received {
                Ok(event) => {
                    if event.matches_job(active_job_id.as_deref())
                        && send_stream_event(&mut session, &event).await.is_err()
                    {
                        break;
                    }
                }
                Err(broadcast::error::RecvError::Lagged(skipped)) => {
                    let reason = format!("Lagged behind the job stream by {skipped} messages.");
                    if send_stream_event(
                        &mut session,
                        &JobStreamEvent::ResyncRequired { reason },
                    )
                    .await
                    .is_err()
                    {
                        break;
                    }
                    match manager.stream_snapshot(active_job_id.as_deref(), tail_lines) {
                        Ok(snapshot) => {
                            if send_stream_event(&mut session, &snapshot).await.is_err() {
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }
                Err(broadcast::error::RecvError::Closed) => break,
            },
            _ = heartbeat.tick() => {
                if session.ping(b"localagent").await.is_err() {
                    break;
                }
            }
        }
    }
}

async fn send_stream_event(session: &mut Session, event: &JobStreamEvent) -> Result<(), ()> {
    let payload = serde_json::to_string(event).map_err(|_| ())?;
    session.text(payload).await.map_err(|_| ())
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let _ = init_tracing();

    let config = RuntimeConfig::default();
    let bind_address = config.server_addr();
    let classifier = web::Data::new(WasteClassifier::new(config.clone()));
    let artifact_store = web::Data::new(ArtifactStore::new(config.clone()));
    let job_manager = web::Data::new(JobManager::new(config));

    HttpServer::new(move || {
        App::new()
            .app_data(classifier.clone())
            .app_data(artifact_store.clone())
            .app_data(job_manager.clone())
            .service(health)
            .service(classify)
            .service(list_jobs)
            .service(get_job)
            .service(create_pipeline_job)
            .service(create_training_job)
            .service(create_benchmark_job)
            .service(cancel_job)
            .service(job_logs)
            .service(jobs_ws)
            .service(list_runs)
            .service(run_detail)
            .service(compare_runs)
            .service(training_presets)
            .service(pipeline_presets)
            .service(artifacts_overview)
            .service(artifacts_training_overview)
            .service(artifacts_benchmarks)
            .service(artifacts_benchmark)
            .service(artifacts_experiment_spec)
            .service(dashboard_summary)
            .service(dataset_image)
            .service(artifacts_dataset)
            .service(artifacts_training)
            .service(artifacts_evaluation)
            .service(artifacts_export)
            .service(artifacts_model_manifest)
    })
    .bind(bind_address)?
    .run()
    .await
}
