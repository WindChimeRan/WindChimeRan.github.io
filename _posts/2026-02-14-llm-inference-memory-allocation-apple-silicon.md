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

But the problem isn't just about lowering a threshold. We also want to support the case where a Mac *is* used as a dedicated inference server with no other significant memory consumers. And beneath the `gpu_memory_utilization` question lies a deeper architectural mismatch: the current memory allocation logic conflates two very different KV cache strategies — contiguous allocation (what MLX actually does) and paged blocks (what the scheduler thinks is happening) — producing phantom block counts that don't correspond to any real memory layout.

This post describes the problem in detail, surveys related work, and proposes a redesigned memory allocation path for vllm-metal that is honest about its memory model and adapts to Apple Silicon's shared-memory reality. The discussion tracks [vllm-metal#97](https://github.com/vllm-project/vllm-metal/issues/97).

---

## Related Work

We first review vLLM's memory allocation on CUDA as the baseline design — it is the upstream that [vllm-metal](https://github.com/vllm-project/vllm-metal) adapts from. We then examine how mistral.rs and llama.cpp, both with native Metal support, approach the same problem on Apple Silicon. To compare them clearly, we use a common vocabulary:

- **memory hint**: the value reported by Metal's `recommendedMaxWorkingSetSize` — the OS's suggestion for how much GPU memory a process should use. This is a static, per-process hint; it does not reflect memory consumed by other apps.
- **RAM cap**: a software-imposed ceiling, typically a fraction of total system RAM (e.g., 2/3 or 3/4), intended to leave room for the OS and other apps.
- **utilization target**: the fraction of the memory budget an engine attempts to claim (e.g., 0.9 means "use up to 90%").

### vLLM
**vLLM** follows a profile-and-claim strategy. On CUDA, it queries the GPU for total and free VRAM at startup, then claims `total_vram × utilization_target` (default 0.9). If the requested amount exceeds what is actually free — because another process is already using the GPU — vLLM fails immediately rather than proceeding with a smaller budget. The claimed memory is managed as a paged block pool (PagedAttention), where KV cache blocks are allocated and freed at page granularity as requests arrive and leave.

GPU memory is filled in two stages. First, **model weights** are loaded onto the GPU. vLLM then runs a profiling forward pass with dummy inputs to measure peak non-KV memory usage (activations, temporaries). The remaining memory — `total_vram × utilization_target - weight_memory - profile_peak` — becomes the KV block pool:

$$
\text{kv_budget} = \text{total_vram} \times \text{utilization} - \text{weight_memory} - \text{profile_peak}
$$

This design assumes dedicated VRAM that the process can monopolize. On Apple Silicon, there is no dedicated VRAM — "GPU memory" is system RAM shared with everything else, so claiming 90% of the total is not realistic for desktop use.

### mistral.rs
**mistral.rs** adapts the same two-phase approach for Apple Silicon but introduces a RAM cap to leave room for the OS, browsers, and editors sharing the same memory pool:

$$
\text{ram_cap} = \begin{cases} \text{system_ram} \times 2/3 & \text{if } \text{system_ram} \leq 36\text{ GB} \\\\ \text{system_ram} \times 3/4 & \text{otherwise} \end{cases}
$$

The intent is reasonable, but the cap is applied inconsistently between the two phases.

In **Phase 1 (model weights)**, mistral.rs uses the memory hint directly (no cap). It queries `memory_hint - current_process_allocation` as available memory, reserves `max(available × 0.02, 512 MB)` as headroom, then greedily places layers on GPU until the budget is exhausted.

In **Phase 2 (KV cache)**, the RAM cap enters the picture. The engine derives `used` as `min(memory_hint, ram_cap) - (memory_hint - current_process_allocation)` — note that the first term is capped while the second is not. The KV budget is:

$$
\text{kv_budget} = \min(\text{memory_hint},\; \text{ram_cap}) \times \text{utilization} - \text{used}
$$

When `memory_hint ≈ ram_cap`, the capped and uncapped values are consistent and the formula works — though two safety margins stack (25–33% from the RAM cap, plus 10% from the utilization target), leaving ~32–40% of RAM reserved. When `memory_hint > ram_cap`, the uncapped free value is larger than it should be relative to the capped total, so `used` underestimates actual allocation and the KV budget inflates. On a 48 GB Mac with a 14 GB model:

```
memory_hint = 45 GB,  ram_cap = 36 GB

min(memory_hint, ram_cap)           = 36 GB    ← capped
memory_hint - current_allocation    = 31 GB    ← uncapped
used = 36 - 31                      =  5 GB    ← should be 14 GB

kv_budget = 36 × 0.9 - 5 = 27.4 GB
total GPU = 14 + 27.4     = 41.4 GB  (86% of RAM)
```

The 9 GB gap between the two ceilings (45 − 36) leaks directly into the KV budget as phantom free memory.

### llama.cpp
**llama.cpp** sidesteps dynamic budget computation entirely. It uses the memory hint as its ceiling — `memory_hint - current_process_allocation` — with no RAM cap and no utilization target. Memory consumed by other apps is invisible; the engine has no system-wide pressure signal. On macOS 15+, a background thread requests buffer residency every 500 ms to prevent OS eviction, an acknowledgment that the OS may reclaim memory under pressure even after allocation succeeds.

GPU memory is filled in three stages. First, **model weights**: the last N layers are offloaded to GPU (back-to-front); the rest stay on CPU. Second, the **KV cache**: fully preallocated at context creation — the user specifies a token capacity (`n_ctx`) and the entire buffer is allocated upfront. Third, a **compute scratch buffer**: a worst-case activation buffer sized for one micro-batch, reused on every forward pass.

Unlike the paged designs above, llama.cpp's KV cache is contiguous. It operates in one of two modes. In *shared* mode (default), all sequences share a single ring buffer of size `n_ctx` — simpler and better utilized when sequence lengths vary, but attention masks must explicitly exclude other sequences' tokens. In *partitioned* mode, each sequence gets its own ring buffer of size `n_ctx / max_seqs`, providing strict isolation at the cost of rigid capacity per sequence. Total memory is identical in both modes:

$$
\text{KV}_{\text{total}} = n_{\text{kv_layers}} \times 2 \times n_{\text{kv_heads}} \times d_{\text{head}} \times n_{\text{ctx}} \times \text{dtype_size}
$$

Each layer's K and V tensors are placed on the same device as that layer's weights. Slot management uses 1-token granularity with a ring buffer head pointer per partition, and an explicit defrag pass handles fragmentation. There is no paging and no dynamic resizing — the entire KV budget is committed at startup.

The common thread: none of these engines have a memory allocation strategy designed for a shared-memory desktop environment. vLLM assumes exclusive ownership of a discrete memory pool. mistral.rs adds Apple Silicon-specific caps but mixes two different memory ceilings in ways that are either overly conservative or, when the ceilings disagree, insufficiently conservative. llama.cpp pre-commits a user-specified amount and hopes it fits. A solution for Apple Silicon needs to handle both the mixed-use desktop case (other apps are running) and the dedicated server case (the machine is yours), ideally without requiring the user to manually tune a single magic number.

---

## My Proposal

### Key Idea

[TODO: One-sentence summary of your proposed memory allocation strategy]

### Design

[TODO: Architecture overview — how the allocator works at a high level]

[TODO: Diagram placeholder — memory layout / allocation flow]

### Implementation Details

[TODO: Specific techniques — e.g., memory pool design, KV cache placement, prefetching strategy]

[TODO: How it integrates with existing engines (llama.cpp / MLX)]

### Preliminary Results

[TODO: Any benchmarks, measurements, or expected improvements]

[TODO: Comparison with baseline approaches]

---

## Next Steps

[TODO: What comes next — further experiments, open questions, call for feedback]
