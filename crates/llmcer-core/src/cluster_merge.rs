//! Anti-transitive Hierarchical Cluster Merge (CMR, Algorithm 3 in LLMCER).

use std::collections::{HashMap, HashSet};

use crate::error::LlmcerError;
use crate::pipeline::LlmClusterer;

/// Picks the representative record index whose embedding is closest to the
/// cluster's centroid.
#[must_use]
pub fn representative_of(cluster: &[usize], vectors: &[Vec<f32>]) -> usize {
    if cluster.len() <= 1 {
        return cluster[0];
    }
    let dim = vectors[cluster[0]].len();
    let mut centroid = vec![0.0f32; dim];
    for &idx in cluster {
        for (c, &v) in centroid.iter_mut().zip(vectors[idx].iter()) {
            *c += v;
        }
    }
    let inv = 1.0 / (cluster.len() as f32);
    for c in &mut centroid {
        *c *= inv;
    }

    let mut best_id = cluster[0];
    let mut best_dist = f32::INFINITY;
    for &idx in cluster {
        let dist: f32 = vectors[idx]
            .iter()
            .zip(centroid.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        if dist < best_dist {
            best_dist = dist;
            best_id = idx;
        }
    }
    best_id
}

/// Hierarchically merges clusters across record sets within a single block
/// while enforcing anti-transitivity (never packing two clusters from the same
/// source set into the same merge batch).
///
/// # Errors
/// Returns [`LlmcerError`] if the underlying [`LlmClusterer`] fails.
pub fn cluster_merge(
    clustered_sets: Vec<Vec<Vec<usize>>>,
    vectors: &[Vec<f32>],
    sim_matrix: &[Vec<f32>],
    clusterer: &dyn LlmClusterer,
    set_size: usize,
) -> Result<Vec<Vec<usize>>, LlmcerError> {
    if clustered_sets.len() <= 1 {
        return Ok(clustered_sets.into_iter().flatten().collect());
    }

    let mut current_sets = clustered_sets;
    let cap = set_size.max(2);

    loop {
        if current_sets.len() <= 1 {
            break;
        }

        // Map each (set_idx, cluster_idx) -> representative record id
        let mut items: Vec<(usize, usize, usize)> = Vec::new();
        for (si, set_clusters) in current_sets.iter().enumerate() {
            for (ci, cl) in set_clusters.iter().enumerate() {
                if !cl.is_empty() {
                    items.push((si, ci, representative_of(cl, vectors)));
                }
            }
        }

        // Pack batches with at most one cluster per source set (anti-transitivity)
        let mut batches: Vec<Vec<(usize, usize, usize)>> = Vec::new();
        for item in items {
            if let Some(batch) = batches
                .iter_mut()
                .find(|b| b.len() < cap && !b.iter().any(|existing| existing.0 == item.0))
            {
                batch.push(item);
            } else {
                batches.push(vec![item]);
            }
        }

        let mut merged_any = false;
        let mut next_sets: Vec<Vec<Vec<usize>>> = Vec::new();

        for batch in batches {
            if batch.len() <= 1 {
                let (si, ci, _) = batch[0];
                next_sets.push(vec![current_sets[si][ci].clone()]);
                continue;
            }
            let rep_ids: Vec<usize> = batch.iter().map(|x| x.2).collect();
            let rep_to_cluster: HashMap<usize, Vec<usize>> = batch
                .iter()
                .map(|&(si, ci, rep)| (rep, current_sets[si][ci].clone()))
                .collect();

            let grouped_reps = clusterer.cluster_with_guardrail(&rep_ids, sim_matrix)?;
            let mut merged_set = Vec::new();
            for rep_group in grouped_reps {
                if rep_group.len() > 1 {
                    merged_any = true;
                }
                let mut combined = HashSet::new();
                for rep in rep_group {
                    if let Some(members) = rep_to_cluster.get(&rep) {
                        combined.extend(members.iter().copied());
                    }
                }
                let mut sorted: Vec<usize> = combined.into_iter().collect();
                sorted.sort_unstable();
                merged_set.push(sorted);
            }
            next_sets.push(merged_set);
        }

        current_sets = next_sets;
        if !merged_any {
            break;
        }
    }

    Ok(current_sets.into_iter().flatten().collect())
}
