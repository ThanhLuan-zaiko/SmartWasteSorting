use thiserror::Error;

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("model artifact is not configured")]
    MissingModelArtifact,
    #[error("onnx runtime is not initialized yet")]
    RuntimeUnavailable,
    #[error("serialization failed: {0}")]
    Serialization(String),
}
