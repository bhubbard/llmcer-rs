# llmcer-rs

Pure-Rust port of [ZJU-DAILY/LLMCER](https://github.com/ZJU-DAILY/LLMCER) — modular, parallelized LLM entity resolution featuring Neighboring Record Sets (NRS), Misclustering Detection Guardrails (MDG), and anti-transitive Hierarchical Cluster Merge (CMR).

Zero Python runtime required.

## Workspace Crates

| Crate | Purpose |
| :--- | :--- |
| `llmcer-core` | Cosine similarity matrix, NRS record set partitioning, `LlmClusterer` + MDG guardrail, anti-transitive `cluster_merge`, parallel `run_blocks`, and `pairwise_f1` |
| `llmcer-cli` | `llmcer` CLI (`demo`) |

## Quickstart

```bash
cargo test --workspace
cargo run -p llmcer-cli -- demo
```
