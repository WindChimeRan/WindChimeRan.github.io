---
layout: post
title: "The Same Stack on DGX Spark"
description: "NVIDIA's DGX Spark is another unified-memory target for the same vLLM stack. Same SiliconBench agent split, same client, two machines: where the Mac leads, where it does not, and what the shape of the gap says."
tags: [LLM, inference, Apple Silicon, DGX Spark, benchmarks, vllm-metal]
categories: [research]
giscus_comments: false
related_posts: false
toc:
  beginning: true
---

*Draft — work in progress. Moved verbatim out of the vllm-metal v0.4.0 announcement post; the prose below has not been rewritten for a standalone piece yet.*

NVIDIA's DGX Spark is another unified-memory target for the same vLLM stack. It exposes a shared CPU/GPU memory pool and runs the same V1 scheduler, chunked prefill, and paged KV management. The execution layer changes from vllm-metal, MLX, and Metal to upstream vLLM and CUDA.

The two machines balance memory and compute differently:

| | Apple M5 Pro | DGX Spark (GB10) |
|---|---|---|
| Unified memory | 64 GB LPDDR5X-9600 | 128 GB LPDDR5X-8533 |
| Memory bandwidth | 307 GB/s | 273 GB/s |
| Nominal scalar shader width (rough) | ≈2,560 lanes (20 cores × ≈128) | 6,144 CUDA cores (48 SMs × 128) |
| Serving stack | vllm-metal (MLX + Metal) | upstream vLLM (CUDA) |

*Nominal scalar width gives a rough architectural comparison. It does not measure equivalent FLOPS. [Apple publishes](https://www.apple.com/macbook-pro/specs/) the 20-core GPU but not the M5 Pro's lane count or absolute GPU throughput; the Apple value estimates 128 lanes per core from prior Apple GPU designs. [NVIDIA publishes](https://docs.nvidia.com/dgx/dgx-spark/dgx-spark.pdf) 6,144 CUDA cores. Clock rates, instruction issue, matrix accelerators, and kernel efficiency differ.*

Neural Accelerators and Tensor Cores are omitted because Apple does not publish a comparable throughput figure.

<div style="display: flex; justify-content: center;">
  <img src="/assets/img/spark-vs-apple.svg" alt="SiliconBench agent split on Gemma 4 E4B and Qwen3.8-27B, vllm-metal on an M5 Pro against upstream vLLM on a DGX Spark: TTFT, end-to-end request latency, and output token throughput versus concurrency" style="max-width: 100%; width: 100%; height: auto;" />
</div>

On Gemma 4 E4B, the Mac leads at concurrency 1, with 21.1 output tokens per second to the Spark's 15.7. At concurrency 16, the Spark leads 225.1 to 63.6. On Qwen3.8-27B, where the Mac uses an 8-bit MLX conversion and the Spark uses Qwen's FP8 checkpoint, throughput at concurrency 4 is 5.3 on the Mac and 24.0 on the Spark, with TTFT at 14.7 s and 0.6 s, respectively.

At concurrency 16, Gemma 4 E4B averages 5.5 s per request on the Spark against 21.7 s on the Mac; on the 27B at concurrency 4, the gap is 11.2 s against 52.7 s.

Both expose the same OpenAI-compatible interface and V1 scheduling model, so the SiliconBench workload runs unchanged while the execution backend and hardware set the performance ceiling.

## Reproduction

The DGX Spark comparison reuses the agent split and SiliconBench client unchanged on a GB10 box with 121 GB of usable unified memory.

- **DGX Spark:** upstream vLLM `0.27.2rc1.dev568+gf25c580af` on GB10, using the same client and agent split with `--max-model-len 16384` and prefix caching enabled. The 27B arm serves `Qwen/Qwen3.8-27B-FP8`, a different checkpoint from the Mac's 8-bit MLX conversion.
- **DGX Spark memory:** the 27B uses `--gpu-memory-utilization 0.7`; the default `0.9` exhausted the shared memory pool. Nothing was preempted at concurrency 4 with `0.7`.
