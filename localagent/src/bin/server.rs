use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use localagent_rs::{init_tracing, RuntimeConfig, WasteClassifier};
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Deserialize)]
struct ClassificationRequest {
    sample_ids: Vec<String>,
}

#[get("/health")]
async fn health() -> impl Responder {
    HttpResponse::Ok().json(json!({
        "status": "ok",
        "service": "localagent-server",
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

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let _ = init_tracing();

    let config = RuntimeConfig::default();
    let bind_address = config.server_addr();
    let classifier = web::Data::new(WasteClassifier::new(config));

    HttpServer::new(move || {
        App::new()
            .app_data(classifier.clone())
            .service(health)
            .service(classify)
    })
    .bind(bind_address)?
    .run()
    .await
}
