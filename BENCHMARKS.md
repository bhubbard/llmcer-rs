# Benchmarks & Coverage

Measured on **Apple M3 Pro** (`arm64`, 11 cores, 18 GiB RAM, macOS) with `rustc 1.100.0` (`--release`) vs. `Python 3.15.2` (same host) and upstream `ZJU-DAILY/LLMCER`.

## 1. Code Coverage (`cargo llvm-cov --workspace --summary-only`)

| File | Region Coverage | Function Coverage | Line Coverage |
| :--- | ---: | ---: | ---: |
| `llmcer-core/src/similarity.rs` | 96.97% | 100.00% | 95.83% |
| `llmcer-core/src/record_set.rs` | 95.65% | 100.00% | 96.15% |
| `llmcer-core/src/pipeline.rs` | 91.72% | 81.82% | 87.69% |
| `llmcer-core/src/metrics.rs` | 90.00% | 100.00% | 89.66% |
| `llmcer-core/src/cluster_merge.rs` | 89.47% | 100.00% | 92.05% |
| **Workspace Total** *(incl. CLI)* | **78.68%** | **88.00%** | **83.02%** |

## 2. Footprint, Startup & Microbenchmarks

| Metric / Operation | `llmcer-rs` (Rust) | Upstream `ZJU-DAILY/LLMCER` (Python) | Comparison |
| :--- | ---: | ---: | :--- |
| **Standalone Binary Size** | **1.04 MiB** | Python + `scikit-learn` + `numpy` + `pandas` | Single native binary |
| **Cold Startup (5-run median)** | **4.72 ms** | **23.61 ms** (`python3 -c ""` bare stdlib) | **5.00× faster** cold start |
| `pairwise_similarity_matrix(64x32)` | **0.0990 ms** | **7.0802 ms** (Python 3.15 loop) | **71.5× faster** |
| `run_blocks(64 records, 4 blocks)` | **3.8130 ms** | Python `ThreadPoolExecutor` (GIL-bound) | True OS-thread parallelism |

Reproduce locally:
```bash
cargo run --release -p llmcer-core --example bench
cargo llvm-cov --workspace --summary-only
```
