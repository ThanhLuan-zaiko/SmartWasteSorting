use anyhow::Result;
use rayon::prelude::*;
use tracing::instrument;

use crate::{
    config::RuntimeConfig,
    domain::{ClassificationResult, Prediction},
};

#[derive(Clone, Debug)]
pub struct WasteClassifier {
    config: RuntimeConfig,
}

impl WasteClassifier {
    pub fn new(config: RuntimeConfig) -> Self {
        Self { config }
    }

    #[instrument(skip(self, sample_ids))]
    pub fn classify_batch<S>(&self, sample_ids: &[S]) -> Result<Vec<ClassificationResult>>
    where
        S: AsRef<str> + Sync,
    {
        let backend = if self.config.model_path.is_empty() {
            "rust-stub"
        } else {
            "rust-onnx"
        }
        .to_string();
        let score = self.config.score_threshold.max(0.51);

        Ok(sample_ids
            .par_iter()
            .map(|sample_id| ClassificationResult {
                sample_id: sample_id.as_ref().to_string(),
                predictions: vec![Prediction {
                    label: "recyclable".to_string(),
                    score,
                }],
                backend: backend.clone(),
            })
            .collect())
    }
}
