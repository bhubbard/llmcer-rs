//! Embedding cosine similarity and pairwise similarity matrix utilities.

/// Computes cosine similarity between two equal-length `f32` vectors.
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// Computes a dense `N x N` cosine similarity matrix from `vectors`.
#[must_use]
pub fn pairwise_similarity_matrix(vectors: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = vectors.len();
    let mut mat = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        mat[i][i] = 1.0;
        for j in (i + 1)..n {
            let s = cosine_similarity(&vectors[i], &vectors[j]);
            mat[i][j] = s;
            mat[j][i] = s;
        }
    }
    mat
}
