use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeConfig {
    pub model_path: String,
    pub labels_path: String,
    pub device: String,
    pub score_threshold: f32,
    pub server_host: String,
    pub server_port: u16,
}

impl RuntimeConfig {
    pub fn server_addr(&self) -> String {
        format!("{}:{}", self.server_host, self.server_port)
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            model_path: "models/waste_classifier.onnx".to_string(),
            labels_path: "models/labels.json".to_string(),
            device: "cpu".to_string(),
            score_threshold: 0.45,
            server_host: "127.0.0.1".to_string(),
            server_port: 8080,
        }
    }
}
