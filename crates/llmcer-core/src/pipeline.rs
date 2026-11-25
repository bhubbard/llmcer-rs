//! End-to-end parallel LLMCER block pipeline with Misclustering Detection Guardrail (MDG).

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::cluster_merge::cluster_merge;
use crate::error::LlmcerError;
use crate::record_set::create_record_sets;

/// An entity record entering the LLMCER pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Record {
    /// Zero-based record index.
    pub id: usize,
    /// Serialized entity text (name, phone, address, etc.).
    pub text: String,
}

/// Pipeline hyper-parameters matching LLMCER (`S_s`, `S_d`).
#[derive(Debug, Clone, Copy)]
pub struct PipelineConfig {
    /// Maximum records per in-context LLM batch (`S_s`).
    pub set_size: usize,
    /// Target diversity (`S_d`).
    pub diversity: usize,
    /// Similarity threshold used by the mock/guardrail baseline.
    pub guardrail_min_sim: f32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            set_size: 4,
            diversity: 2,
            guardrail_min_sim: 0.25,
        }
    }
}

/// Interface for in-context entity clustering with Misclustering Detection Guardrail.
pub trait LlmClusterer: Send + Sync {
    /// Clusters `record_ids` into disjoint entity groups.
    ///
    /// # Errors
    /// Returns [`LlmcerError`] if clustering or guardrail verification fails.
    fn cluster_records(
        &self,
        record_ids: &[usize],
        sim_matrix: &[Vec<f32>],
    ) -> Result<Vec<Vec<usize>>, LlmcerError>;

    /// Runs [`Self::cluster_records`] and validates the Misclustering Detection
    /// Guardrail (exact partition coverage, no dropped/duplicated IDs).
    ///
    /// # Errors
    /// Returns [`LlmcerError::Guardrail`] if the partition is invalid.
    fn cluster_with_guardrail(
        &self,
        record_ids: &[usize],
        sim_matrix: &[Vec<f32>],
    ) -> Result<Vec<Vec<usize>>, LlmcerError> {
        let clusters = self.cluster_records(record_ids, sim_matrix)?;
        let expected: HashSet<usize> = record_ids.iter().copied().collect();
        let mut seen = HashSet::new();
        for cl in &clusters {
            for &id in cl {
                if !expected.contains(&id) || !seen.insert(id) {
                    return Err(LlmcerError::Guardrail(format!(
                        "invalid or duplicate record id {id} in cluster output"
                    )));
                }
            }
        }
        if seen.len() != expected.len() {
            return Err(LlmcerError::Guardrail(
                "cluster output dropped input records".into(),
            ));
        }
        Ok(clusters)
    }
}

/// Deterministic similarity-threshold [`LlmClusterer`] for offline testing and
/// fast Fellegi-Sunter-style pre-clustering.
#[derive(Debug, Clone, Copy)]
pub struct MockLlmClusterer {
    /// Minimum cosine similarity to merge two records.
    pub merge_threshold: f32,
}

impl LlmClusterer for MockLlmClusterer {
    fn cluster_records(
        &self,
        record_ids: &[usize],
        sim_matrix: &[Vec<f32>],
    ) -> Result<Vec<Vec<usize>>, LlmcerError> {
        let mut clusters: Vec<Vec<usize>> = Vec::new();
        for &id in record_ids {
            let mut placed = false;
            for cl in &mut clusters {
                if cl
                    .iter()
                    .any(|&m| sim_matrix[id][m] >= self.merge_threshold)
                {
                    cl.push(id);
                    placed = true;
                    break;
                }
            }
            if !placed {
                clusters.push(vec![id]);
            }
        }
        Ok(clusters)
    }
}

/// Resolves a single block end-to-end (`NRS -> in-context clustering + MDG -> CMR`).
///
/// # Errors
/// Returns [`LlmcerError`] if clustering or merging fails.
pub fn process_block(
    block: &[usize],
    vectors: &[Vec<f32>],
    sim_matrix: &[Vec<f32>],
    clusterer: &dyn LlmClusterer,
    config: PipelineConfig,
) -> Result<Vec<Vec<usize>>, LlmcerError> {
    if block.is_empty() {
        return Ok(Vec::new());
    }
    if block.len() == 1 {
        return Ok(vec![block.to_vec()]);
    }

    let record_sets = create_record_sets(block, sim_matrix, config.set_size, config.diversity);
    let mut clustered_sets = Vec::with_capacity(record_sets.len());
    for rs in record_sets {
        if rs.len() == 1 {
            clustered_sets.push(vec![rs]);
        } else {
            clustered_sets.push(clusterer.cluster_with_guardrail(&rs, sim_matrix)?);
        }
    }

    cluster_merge(
        clustered_sets,
        vectors,
        sim_matrix,
        clusterer,
        config.set_size,
    )
}

/// Resolves all blocks in parallel using scoped OS threads (`std::thread::scope`)
/// and concatenates the final entity partitions.
///
/// # Errors
/// Returns [`LlmcerError`] if any block fails.
pub fn run_blocks(
    vectors: &[Vec<f32>],
    sim_matrix: &[Vec<f32>],
    blocks: &[Vec<usize>],
    clusterer: &dyn LlmClusterer,
    config: PipelineConfig,
) -> Result<Vec<Vec<usize>>, LlmcerError> {
    std::thread::scope(|scope| {
        let handles: Vec<_> = blocks
            .iter()
            .map(|b| scope.spawn(move || process_block(b, vectors, sim_matrix, clusterer, config)))
            .collect();

        let mut all_clusters = Vec::new();
        for h in handles {
            let block_clusters = h
                .join()
                .map_err(|_| LlmcerError::Guardrail("worker thread panicked".into()))??;
            all_clusters.extend(block_clusters);
        }
        Ok(all_clusters)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::pairwise_f1;
    use crate::similarity::pairwise_similarity_matrix;

    #[test]
    fn parallel_blocks_resolve_entities_across_nrs_and_cmr() {
        // Records 0, 1, 2 are entity A; records 3, 4 are entity B; record 5 is singleton C
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.98, 0.05, 0.0],
            vec![0.96, 0.08, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.04, 0.99, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let sim = pairwise_similarity_matrix(&vectors);
        let blocks = vec![vec![0, 1, 2, 3, 4], vec![5]];
        let clusterer = MockLlmClusterer {
            merge_threshold: 0.85,
        };
        let pred = run_blocks(
            &vectors,
            &sim,
            &blocks,
            &clusterer,
            PipelineConfig {
                set_size: 2,
                diversity: 2,
                guardrail_min_sim: 0.25,
            },
        )
        .expect("run_blocks succeeds");

        let truth = vec![vec![0, 1, 2], vec![3, 4], vec![5]];
        let metrics = pairwise_f1(&pred, &truth);
        assert!(
            (metrics.f1 - 1.0).abs() < 1e-6,
            "expected F1=1.0, got {:?}",
            metrics
        );
    }
}
