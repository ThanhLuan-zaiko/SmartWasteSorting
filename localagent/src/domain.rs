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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImageClassificationResult {
    pub file_name: String,
    pub predictions: Vec<Prediction>,
    pub backend: String,
    pub model_name: String,
    pub image_size: usize,
}
