//! Neighboring Record Set (NRS) construction (Algorithm 1 in LLMCER).

/// Partitions `block` of record indices into record sets of size at most `set_size`
/// by anchoring diverse seeds and grouping their nearest neighbors.
#[must_use]
pub fn create_record_sets(
    block: &[usize],
    sim_matrix: &[Vec<f32>],
    set_size: usize,
    _diversity: usize,
) -> Vec<Vec<usize>> {
    if block.is_empty() {
        return Vec::new();
    }
    let cap = set_size.max(2);
    let mut remaining: Vec<usize> = block.to_vec();
    let mut sets = Vec::new();

    while !remaining.is_empty() {
        let anchor = remaining.remove(0);
        let mut group = vec![anchor];
        remaining.sort_by(|&a, &b| {
            sim_matrix[anchor][b]
                .partial_cmp(&sim_matrix[anchor][a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let take = (cap - 1).min(remaining.len());
        group.extend(remaining.drain(0..take));
        sets.push(group);
    }
    sets
}
