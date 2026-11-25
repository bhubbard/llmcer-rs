//! CLI for LLMCER entity resolution and pairwise F1 evaluation.

use anyhow::Result;
use clap::{Parser, Subcommand};
use llmcer_core::{
    pairwise_f1, pairwise_similarity_matrix, run_blocks, MockLlmClusterer, PipelineConfig,
};

#[derive(Parser)]
#[command(name = "llmcer", about = "Pure-Rust LLMCER entity resolution CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a parallel NRS + MDG + CMR entity resolution demo.
    Demo,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Demo => {
            let vectors = vec![
                vec![1.0, 0.0, 0.0],
                vec![0.97, 0.05, 0.0],
                vec![0.95, 0.08, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.04, 0.99, 0.0],
            ];
            let sim = pairwise_similarity_matrix(&vectors);
            let blocks = vec![vec![0, 1, 2, 3, 4]];
            let clusterer = MockLlmClusterer {
                merge_threshold: 0.85,
            };
            let pred = run_blocks(
                &vectors,
                &sim,
                &blocks,
                &clusterer,
                PipelineConfig::default(),
            )?;
            let metrics = pairwise_f1(&pred, &[vec![0, 1, 2], vec![3, 4]]);
            println!(
                "Clusters: {} | Metrics: {}",
                serde_json::to_string(&pred)?,
                serde_json::to_string(&metrics)?
            );
        }
    }
    Ok(())
}
