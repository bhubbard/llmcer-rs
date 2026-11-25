//! Release benchmark for `llmcer-core` similarity matrix, NRS, CMR, and parallel blocks.

use std::hint::black_box;
use std::time::Instant;

use llmcer_core::{pairwise_similarity_matrix, run_blocks, MockLlmClusterer, PipelineConfig};

fn main() {
    let n = 64;
    let dim = 32;
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let cluster = i / 4;
            (0..dim)
                .map(|d| {
                    if d == cluster % dim {
                        1.0
                    } else {
                        (i as f32 * 0.01).sin() * 0.05
                    }
                })
                .collect()
        })
        .collect();
    let blocks: Vec<Vec<usize>> = (0..n)
        .collect::<Vec<_>>()
        .chunks(16)
        .map(|c| c.to_vec())
        .collect();
    let clusterer = MockLlmClusterer {
        merge_threshold: 0.85,
    };

    let mut sim_times = Vec::with_capacity(15);
    let mut pipe_times = Vec::with_capacity(15);
    let sim = pairwise_similarity_matrix(&vectors);

    for i in 0..18 {
        let t0 = Instant::now();
        let s = black_box(pairwise_similarity_matrix(&vectors));
        let dt_sim = t0.elapsed().as_secs_f64() * 1000.0;
        black_box(s.len());

        let t1 = Instant::now();
        let res = black_box(run_blocks(
            &vectors,
            &sim,
            &blocks,
            &clusterer,
            PipelineConfig::default(),
        ));
        let dt_pipe = t1.elapsed().as_secs_f64() * 1000.0;
        let _ = black_box(res);

        if i >= 3 {
            sim_times.push(dt_sim);
            pipe_times.push(dt_pipe);
        }
    }
    sim_times.sort_by(f64::total_cmp);
    pipe_times.sort_by(f64::total_cmp);
    println!(
        "pairwise_similarity_matrix(64x32): median={:.4} ms | run_blocks(64 records, 4 blocks): median={:.4} ms",
        sim_times[sim_times.len() / 2],
        pipe_times[pipe_times.len() / 2]
    );
}
