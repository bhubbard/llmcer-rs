//! Pure-Rust port of ZJU-DAILY's LLMCER (LLM-assisted Entity Resolution).
//!
//! Implements the paper's 4-stage block-partitioned pipeline:
//! 1. **NRS** ([`record_set`]): Neighboring Record Set creation (`<= S_s`).
//! 2. **In-Context Clustering + MDG** ([`pipeline`]): LLM clustering with
//!    Misclustering Detection Guardrail verification.
//! 3. **CMR** ([`mod@cluster_merge`]): Anti-transitive Hierarchical Cluster Merge.
//! 4. **Pairwise Metrics** ([`metrics`]): Precision, Recall, and F1 evaluation.

pub mod cluster_merge;
pub mod error;
pub mod metrics;
pub mod pipeline;
pub mod record_set;
pub mod similarity;

pub use cluster_merge::{cluster_merge, representative_of};
pub use error::LlmcerError;
pub use metrics::{pairwise_f1, PairwiseMetrics};
pub use pipeline::{
    process_block, run_blocks, LlmClusterer, MockLlmClusterer, PipelineConfig, Record,
};
pub use record_set::create_record_sets;
pub use similarity::{cosine_similarity, pairwise_similarity_matrix};
