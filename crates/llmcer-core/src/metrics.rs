//! Pairwise entity resolution evaluation metrics (Precision, Recall, F1).

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

/// Pairwise Precision, Recall, and F1 score.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PairwiseMetrics {
    /// Pairwise precision.
    pub precision: f64,
    /// Pairwise recall.
    pub recall: f64,
    /// Harmonic mean (F1).
    pub f1: f64,
}

fn pairs_from_clusters(clusters: &[Vec<usize>]) -> HashSet<(usize, usize)> {
    let mut pairs = HashSet::new();
    for cl in clusters {
        for i in 0..cl.len() {
            for j in (i + 1)..cl.len() {
                let a = cl[i].min(cl[j]);
                let b = cl[i].max(cl[j]);
                pairs.insert((a, b));
            }
        }
    }
    pairs
}

/// Computes pairwise Precision, Recall, and F1 between predicted and ground-truth clusters.
#[must_use]
pub fn pairwise_f1(predicted: &[Vec<usize>], truth: &[Vec<usize>]) -> PairwiseMetrics {
    let pred_pairs = pairs_from_clusters(predicted);
    let true_pairs = pairs_from_clusters(truth);
    let tp = pred_pairs.intersection(&true_pairs).count() as f64;
    let precision = if pred_pairs.is_empty() {
        1.0
    } else {
        tp / (pred_pairs.len() as f64)
    };
    let recall = if true_pairs.is_empty() {
        1.0
    } else {
        tp / (true_pairs.len() as f64)
    };
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    PairwiseMetrics {
        precision,
        recall,
        f1,
    }
}
