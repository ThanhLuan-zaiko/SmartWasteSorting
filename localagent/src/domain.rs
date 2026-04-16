use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Prediction {
    pub label: String,
    pub score: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub sample_id: String,
    pub predictions: Vec<Prediction>,
    pub backend: String,
}
