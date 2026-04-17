use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use localagent_rs::{
    init_tracing, to_api_envelope, ArtifactKind, ArtifactStore, RuntimeConfig, WasteClassifier,
};
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Deserialize)]
struct ClassificationRequest {
    sample_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ArtifactQuery {
    experiment: Option<String>,
}

#[get("/health")]
async fn health(store: web::Data<ArtifactStore>) -> impl Responder {
    let experiment = store.default_experiment().to_string();
    HttpResponse::Ok().json(json!({
        "status": "ok",
        "service": "localagent-server",
        "default_experiment": experiment,
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

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let _ = init_tracing();

    let config = RuntimeConfig::default();
    let bind_address = config.server_addr();
    let classifier = web::Data::new(WasteClassifier::new(config.clone()));
    let artifact_store = web::Data::new(ArtifactStore::new(config));

    HttpServer::new(move || {
        App::new()
            .app_data(classifier.clone())
            .app_data(artifact_store.clone())
            .service(health)
            .service(classify)
            .service(artifacts_overview)
            .service(artifacts_training_overview)
            .service(artifacts_benchmarks)
            .service(dashboard_summary)
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
