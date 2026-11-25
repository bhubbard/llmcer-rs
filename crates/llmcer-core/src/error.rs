//! Error types for `llmcer-core`.

use thiserror::Error;

/// Errors returned by LLMCER blocking, clustering, or guardrail checks.
#[derive(Debug, Error)]
pub enum LlmcerError {
    /// Invalid configuration or mismatched embedding dimensions.
    #[error("configuration or dimension error: {0}")]
    Dimension(String),

    /// LLM clustering failed Misclustering Detection Guardrail (MDG) checks.
    #[error("misclustering guardrail rejected partition: {0}")]
    Guardrail(String),
}
