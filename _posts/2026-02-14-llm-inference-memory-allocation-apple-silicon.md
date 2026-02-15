---
layout: post
title: "LLM Inference Engine's Memory Allocation on Apple Silicon"
date: 2026-02-14 10:00:00-0000
description: "Exploring memory allocation strategies for LLM inference engines on Apple Silicon's unified memory architecture."
tags: [LLM, inference, Apple Silicon, memory, systems]
categories: [research]
giscus_comments: false
related_posts: false
toc:
  beginning: true
---

## Introduction

LLM inference engines like [vLLM](https://github.com/vllm-project/vllm) were designed for discrete NVIDIA GPUs, where GPU memory is a dedicated, isolated resource. The memory allocator can safely assume it owns nearly all of VRAM — upstream vLLM defaults to claiming 90% of total GPU memory (`gpu_memory_utilization=0.9`), and if another process has already taken enough that this target can't be met, vLLM simply refuses to start.

This assumption breaks on Apple Silicon. The M-series chips use a Unified Memory Architecture (UMA): CPU, GPU, and Neural Engine all share a single physical memory pool. There is no dedicated "VRAM" to claim. A Mac running [vllm-metal](https://github.com/vllm-project/vllm-metal) is likely also running a browser, an IDE, and other applications — all competing for the same memory. Reserving 90% of total memory is not realistic in this scenario.

But the problem isn't just about lowering a threshold. We also want to support the case where a Mac *is* used as a dedicated inference server with no other significant memory consumers. 
<!-- And beneath the `gpu_memory_utilization` question lies a deeper architectural mismatch: the current memory allocation logic conflates two very different KV cache strategies, contiguous allocation (what MLX actually does) and paged blocks (what the scheduler thinks is happening), producing phantom block counts that don't correspond to any real memory layout. -->

The discussion tracks [vllm-metal#97](https://github.com/vllm-project/vllm-metal/issues/97).

---

## Why vllm-metal

An NVIDIA RTX 5090 ships with 32 GB of GDDR7 at a $1,999 MSRP — roughly **$62.47 per GB**. A Mac Studio with the M3 Ultra and 512 GB of unified memory starts at $9,499 — roughly **$18.55 per GB**, more than 3× cheaper. This is not an apple-to-Apple comparison (😉): CUDA VRAM is dedicated and faster, while unified memory is shared with the rest of the system and comes with lower bandwidth — which is precisely why the memory *allocator* matters so much on this platform.

---

## Related Work

To compare the three engines below, we use a common vocabulary:

- **memory hint**: the value reported by Metal's `recommendedMaxWorkingSetSize` — the OS's suggestion for how much GPU memory a process should use. This is a static, per-process hint; it does not reflect memory consumed by other apps.
- **RAM cap**: a software-imposed ceiling, typically a fraction of total system RAM (e.g., 2/3 or 3/4), intended to leave room for the OS and other apps.
- **utilization target**: the fraction of the memory budget an engine attempts to claim (e.g., 0.9 means "use up to 90%").

### vLLM
**vLLM** follows a profile-and-claim strategy. On CUDA, it queries the GPU for total and free VRAM at startup, then claims `total_vram × utilization_target` (default 0.9). If the requested amount exceeds what is actually free (because another process is already using the GPU), vLLM fails immediately rather than proceeding with a smaller budget. The claimed memory is managed as a paged block pool (PagedAttention), where KV cache blocks are allocated and freed at page granularity as requests arrive and leave.

GPU memory is filled in two stages. First, **model weights** are loaded onto the GPU. vLLM then runs a profiling forward pass with dummy inputs to measure peak non-KV memory usage (activations, temporaries). The remaining memory — `total_vram × utilization_target - weight_memory - profile_peak` — becomes the KV block pool:

$$
\text{kv_budget} = \text{total_vram} \times \text{utilization} - \text{weight_memory} - \text{profile_peak}
$$

### mistral.rs
**mistral.rs** adapts the same two-phase approach for Apple Silicon but introduces a RAM cap to leave room for the OS and other applications sharing the same memory pool:

$$
\text{ram_cap} = \begin{cases} \text{system_ram} \times 2/3 & \text{if } \text{system_ram} \leq 36\text{ GB} \\\\ \text{system_ram} \times 3/4 & \text{otherwise} \end{cases}
$$

In **Phase 1 (model weights)**, mistral.rs uses the memory hint directly, computing `memory_hint - current_process_allocation` and reserving `max(available × 0.02, 512 MB)` as headroom, then greedily places layers on GPU until the budget is exhausted.

In **Phase 2 (KV cache)**, the effective ceiling becomes `min(memory_hint, ram_cap)`. Since Apple typically reports a memory hint in the 66–75% range of system RAM — close to what the RAM cap already computes — the `min()` often selects the same value. The KV budget is then:

$$
\text{kv_budget} = \min(\text{memory_hint},\; \text{ram_cap}) \times \text{utilization} - \text{used}
$$

The two safety margins stack: 25–33% is reserved by the RAM cap, and a further 10% by the utilization target (default 0.9), leaving roughly 60–68% of system RAM as the effective ceiling before model weights are subtracted.

### llama.cpp
**llama.cpp** uses the memory hint as its ceiling, `memory_hint - current_process_allocation`, with no RAM cap and no utilization target. Memory consumed by other apps is invisible; the engine has no system-wide pressure signal. On macOS 15+, a background thread requests buffer residency every 500 ms to prevent OS eviction, an acknowledgment that the OS may reclaim memory under pressure even after allocation succeeds.

GPU memory is filled in three stages: **model weights** (last N layers offloaded to GPU, back-to-front), a **KV cache** (preallocated at a user-specified token capacity `n_ctx`), and a **compute scratch buffer** (worst-case activation buffer reused every forward pass). Unlike the paged designs above, llama.cpp's KV cache is contiguous — sequences share a ring buffer with 1-token granularity and an explicit defrag pass. Total KV memory is committed at startup:

$$
\text{KV}_{\text{total}} = n_{\text{kv_layers}} \times 2 \times n_{\text{kv_heads}} \times d_{\text{head}} \times n_{\text{ctx}} \times \text{dtype_size}
$$

vLLM assumes exclusive ownership of a discrete memory pool. mistral.rs introduces Apple Silicon-specific caps for desktop coexistence. llama.cpp pre-commits a user-specified amount with no system-wide awareness. vllm-metal inherits from all three but needs to handle both the mixed-use desktop case and the dedicated server case, ideally without requiring the user to manually tune a single magic number.

---

## vllm-metal status quo

https://github.com/vllm-project/vllm-metal/issues/97

path 1: mlx_lm contiguous allocation. This is the current implementation. should keep it simple and inherent the mlx_lm design.

path 2: vllm paged allocation. This is the proposed new implementation. The design should be aligned with vllm's paged allocation.

we need a clean refactor for the path 1, and a new design for the path 2. 

## My Proposal

concern 1. gpu_memory_utilization=0.9 the original vllm semantic is 0.9 of the total gpu memory. But for apple silicon, we need to save some room for os and other app. I don't want to set it as 0.9 of avaiable memory - mix the meaning of total memory vs available memory. This is confusing. 

concern 2. use portion of available memory is not good for mac. e.g.,  16GB, 32GB, 256GB, 512GB. saving 10% of 512 is overkill, while saving 10% of 16GB might be too little when the user have youtube playing. 

lesson learned from mistral.rs: 
https://github.com/EricLBuehler/mistral.rs/issues/1348

sudo sysctl iogpu.disable_wired_collector=1

On Apple Silicon Macs, the CPU and GPU share unified memory. macOS has a mechanism that actively reclaims wired (non-pageable) memory that was allocated to the GPU when the system is under memory pressure — this is the "wired collector."
Setting iogpu.disable_wired_collector=1 tells macOS: don't reclaim GPU wired memory, even when the system wants that memory back.

the setting does not persist across reboots unless you add them to a startup script.