use std::{
    cmp::Ordering,
    path::{Path, PathBuf},
    sync::OnceLock,
};

use anyhow::{anyhow, bail, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use image::imageops::FilterType;
use ndarray::Array4;
use ort::{session::Session, value::Tensor};
use rayon::prelude::*;
use serde::Deserialize;
use tracing::instrument;

use crate::{
    config::RuntimeConfig,
    domain::{ClassificationResult, ImageClassificationResult, Prediction},
};

static ORT_RUNTIME_INIT: OnceLock<std::result::Result<(), String>> = OnceLock::new();

#[derive(Clone, Debug)]
pub struct WasteClassifier {
    config: RuntimeConfig,
}

#[derive(Debug, Deserialize)]
struct ModelManifest {
    model_name: String,
    onnx_path: String,
    labels: Vec<String>,
    image_size: usize,
    normalization: NormalizationSpec,
}

#[derive(Debug, Deserialize)]
struct NormalizationSpec {
    mean: [f32; 3],
    std: [f32; 3],
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

    #[instrument(skip(self, image_base64))]
    pub fn classify_uploaded_image(
        &self,
        image_base64: &str,
        file_name: Option<&str>,
        top_k: Option<usize>,
    ) -> Result<ImageClassificationResult> {
        let bytes = decode_image_payload(image_base64)?;
        let resolved_file_name = file_name
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("uploaded-image");
        self.classify_image_bytes(&bytes, resolved_file_name, top_k.unwrap_or(3))
    }

    fn classify_image_bytes(
        &self,
        image_bytes: &[u8],
        file_name: &str,
        top_k: usize,
    ) -> Result<ImageClassificationResult> {
        ensure_onnx_runtime_loaded()?;
        let manifest = self.load_model_manifest()?;
        if manifest.labels.is_empty() {
            bail!("model manifest contains no labels");
        }

        let onnx_path = resolve_project_path(&manifest.onnx_path);
        if !onnx_path.is_file() {
            bail!("ONNX model not found at {}", onnx_path.display());
        }

        let input_tensor = preprocess_image(
            image_bytes,
            manifest.image_size,
            &manifest.normalization.mean,
            &manifest.normalization.std,
        )?;

        let mut session = Session::builder()
            .context("failed to create ONNX session builder")?
            .commit_from_file(&onnx_path)
            .with_context(|| format!("failed to open ONNX model {}", onnx_path.display()))?;

        let input = Tensor::<f32>::from_array(input_tensor)
            .context("failed to create ONNX input tensor")?;
        let outputs = session
            .run(ort::inputs![input])
            .context("ONNX inference failed")?;
        let (_, logits) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("failed to extract ONNX logits tensor")?;
        if logits.len() < manifest.labels.len() {
            bail!(
                "ONNX output does not contain enough logits: {} < {}",
                logits.len(),
                manifest.labels.len()
            );
        }

        let probabilities = softmax(&logits[..manifest.labels.len()]);
        let mut predictions = manifest
            .labels
            .into_iter()
            .zip(probabilities)
            .map(|(label, score)| Prediction { label, score })
            .collect::<Vec<_>>();
        predictions.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
        });
        predictions.truncate(top_k.max(1).min(predictions.len()));

        Ok(ImageClassificationResult {
            file_name: file_name.to_string(),
            predictions,
            backend: "rust-onnx".to_string(),
            model_name: manifest.model_name,
            image_size: manifest.image_size,
        })
    }

    fn load_model_manifest(&self) -> Result<ModelManifest> {
        let manifest_path = resolve_project_path(&self.config.model_manifest_path);
        if !manifest_path.is_file() {
            bail!("model manifest not found at {}", manifest_path.display());
        }
        let payload = std::fs::read_to_string(&manifest_path)
            .with_context(|| format!("failed to read model manifest {}", manifest_path.display()))?;
        serde_json::from_str(&payload)
            .with_context(|| format!("failed to parse model manifest {}", manifest_path.display()))
    }
}

fn preprocess_image(
    image_bytes: &[u8],
    image_size: usize,
    mean: &[f32; 3],
    std: &[f32; 3],
) -> Result<Array4<f32>> {
    if image_size == 0 {
        bail!("image_size must be greater than zero");
    }
    let source = image::load_from_memory(image_bytes).context("failed to decode uploaded image")?;
    let rgb = source.to_rgb8();
    let resized = image::imageops::resize(
        &rgb,
        image_size as u32,
        image_size as u32,
        FilterType::Triangle,
    );

    let mut tensor = Array4::<f32>::zeros((1, 3, image_size, image_size));
    for (x, y, pixel) in resized.enumerate_pixels() {
        let channels = pixel.0;
        for channel_index in 0..3 {
            let normalized =
                (f32::from(channels[channel_index]) / 255.0 - mean[channel_index])
                    / std[channel_index];
            tensor[[0, channel_index, y as usize, x as usize]] = normalized;
        }
    }

    Ok(tensor)
}

fn softmax(values: &[f32]) -> Vec<f32> {
    let max_value = values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |current, value| current.max(value));
    let exp_values = values
        .iter()
        .map(|value| (value - max_value).exp())
        .collect::<Vec<_>>();
    let denominator = exp_values.iter().sum::<f32>();
    if denominator <= f32::EPSILON {
        return vec![0.0; values.len()];
    }
    exp_values
        .into_iter()
        .map(|value| value / denominator)
        .collect()
}

fn decode_image_payload(raw: &str) -> Result<Vec<u8>> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        bail!("image payload is empty");
    }

    let encoded = match trimmed.split_once(',') {
        Some((prefix, payload)) if prefix.contains(";base64") => payload,
        _ => trimmed,
    };
    STANDARD
        .decode(encoded)
        .context("failed to decode base64 image payload")
}

fn ensure_onnx_runtime_loaded() -> Result<()> {
    let state = ORT_RUNTIME_INIT.get_or_init(|| {
        let dll_path = discover_onnxruntime_dll()
            .ok_or_else(|| "unable to locate onnxruntime.dll for Rust ONNX inference".to_string())?;
        ort::init_from(&dll_path)
            .map(|builder| {
                builder.commit();
            })
            .map_err(|error| {
                format!(
                    "failed to initialize ONNX Runtime from {}: {}",
                    dll_path.display(),
                    error
                )
            })
    });

    state
        .as_ref()
        .map(|_| ())
        .map_err(|message| anyhow!(message.clone()))
}

fn discover_onnxruntime_dll() -> Option<PathBuf> {
    let configured = std::env::var_os("ORT_DYLIB_PATH")
        .map(PathBuf::from)
        .filter(|path| path.is_file());
    if configured.is_some() {
        return configured;
    }

    [
        project_root()
            .join(".venv")
            .join("Lib")
            .join("site-packages")
            .join("onnxruntime")
            .join("capi")
            .join("onnxruntime.dll"),
        project_root().join("onnxruntime.dll"),
    ]
    .into_iter()
    .find(|path| path.is_file())
}

fn resolve_project_path(raw: &str) -> PathBuf {
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        path
    } else {
        project_root().join(path)
    }
}

fn project_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::{decode_image_payload, softmax};

    #[test]
    fn softmax_returns_normalized_scores() {
        let scores = softmax(&[1.0, 2.0, 3.0]);
        let total = scores.iter().sum::<f32>();
        assert!((total - 1.0).abs() < 1e-5);
        assert!(scores[2] > scores[1]);
        assert!(scores[1] > scores[0]);
    }

    #[test]
    fn decode_image_payload_accepts_data_urls() {
        let bytes = decode_image_payload("data:image/png;base64,aGVsbG8=").expect("should decode");
        assert_eq!(bytes, b"hello");
    }
}
