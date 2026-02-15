---
layout: post
title: "Rethinking vllm-metal's Memory Budget for Apple Silicon"
date: 2026-02-14 10:00:00-0000
description: "Exploring memory allocation strategies for LLM inference engines on Apple Silicon's unified memory architecture."
tags: [LLM, inference, Apple Silicon, systems, proposal]
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
**[vLLM](https://github.com/vllm-project/vllm)** follows a profile-and-claim strategy. On CUDA, it queries the GPU for total and free VRAM at startup, then claims `total_vram × utilization_target` (default 0.9). If the requested amount exceeds what is actually free (because another process is already using the GPU), vLLM fails immediately rather than proceeding with a smaller budget. The claimed memory is managed as a paged block pool (PagedAttention), where KV cache blocks are allocated and freed at page granularity as requests arrive and leave.

GPU memory is filled in two stages. First, **model weights** are loaded onto the GPU. vLLM then runs a profiling forward pass with dummy inputs to measure peak non-KV memory usage (activations, temporaries). The remaining memory — `total_vram × utilization_target - weight_memory - profile_peak` — becomes the KV block pool:

$$
\text{kv_budget} = \text{total_vram} \times \text{utilization} - \text{weight_memory} - \text{profile_peak}
$$

### mistral.rs
**[mistral.rs](https://github.com/EricLBuehler/mistral.rs)** adapts the same two-phase approach for Apple Silicon but introduces a RAM cap to leave room for the OS and other applications sharing the same memory pool:

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
**[llama.cpp](https://github.com/ggml-org/llama.cpp)** uses the memory hint as its ceiling, `memory_hint - current_process_allocation`, with no RAM cap and no utilization target. Memory consumed by other apps is invisible; the engine has no system-wide pressure signal. On macOS 15+, a background thread requests buffer residency every 500 ms to prevent OS eviction, an acknowledgment that the OS may reclaim memory under pressure even after allocation succeeds.

GPU memory is filled in three stages: **model weights** (last N layers offloaded to GPU, back-to-front), a **KV cache** (preallocated at a user-specified token capacity `n_ctx`), and a **compute scratch buffer** (worst-case activation buffer reused every forward pass). Unlike the paged designs above, llama.cpp's KV cache is contiguous — sequences share a ring buffer with 1-token granularity and an explicit defrag pass. Total KV memory is committed at startup:

$$
\text{KV}_{\text{total}} = n_{\text{kv_layers}} \times 2 \times n_{\text{kv_heads}} \times d_{\text{head}} \times n_{\text{ctx}} \times \text{dtype_size}
$$

vLLM assumes exclusive ownership of a discrete memory pool. mistral.rs introduces Apple Silicon-specific caps for desktop coexistence. llama.cpp pre-commits a user-specified amount with no system-wide awareness. vllm-metal inherits from all three but needs to handle both the mixed-use desktop case and the dedicated server case, ideally without requiring the user to manually tune a single magic number.

---

## vllm-metal Status Quo

vllm-metal has two KV cache paths. **Path 1 (contiguous allocation, MLX)** is the current default. **Path 2 (paged allocation, vLLM)** is under active development on the `paged-attention-v3` branch ([vllm-metal#70](https://github.com/vllm-project/vllm-metal/issues/70)). Both are tracked in [vllm-metal#97](https://github.com/vllm-project/vllm-metal/issues/97). This proposal introduces a `VLLM_METAL_USE_PAGED_ATTENTION` environment variable to select between them. Today, both paths share allocation and scheduling code written with legacy Path 2's paged assumptions, creating a mismatch for Path 1's contiguous runtime.

### Path 1: Contiguous Allocation (MLX)

The current implementation tangles Path 1 with legacy, half-baked paged attention memory calculations. The scheduler reasons in 16-token blocks and reports phantom block counts, but MLX allocates contiguous caches in 256-token steps. None of those blocks exist at runtime. The fix is a straightforward refactor: strip out the paged bookkeeping and fall back to mlx_lm's original contiguous design, where `make_prompt_cache()` manages per-request caches.

### Path 2: Paged Allocation (vLLM)

Path 2 would align with upstream vLLM's PagedAttention, maintaining a global block pool backed by Metal buffers. The same profile-and-claim strategy described in the Related Work section would apply: measure weight and activation memory at startup, then fill the remaining budget with KV blocks. The block pool's design depends on kernel work still in progress on the `paged-attention-v3` branch.

---

## Proposal: Tiered OS Reserve

Upstream vLLM's `gpu_memory_utilization=0.9` means "claim 90% of total GPU memory." On a discrete GPU with dedicated VRAM, this is safe because the only other consumer is the driver. Redefining it as a fraction of *available* system memory does not help: does 0.9 mean 90% of total RAM, or 90% of what happens to be free right now? The parameter is inherently confusing on UMA and vllm-metal should drop it entirely. A flat percentage also scales poorly. Reserving 10% of 16 GB leaves 1.6 GB for the OS and a browser; reserving 10% of 512 GB withholds 51.2 GB. OS overhead does not grow linearly with total RAM.

The alternative is to subtract a fixed, tier-based amount from system RAM before any utilization arithmetic begins:

$$
\text{os_reserve} = \begin{cases} 4\text{ GB} & \text{if } \text{system_ram} \leq 16\text{ GB} \\\\ 6\text{ GB} & \text{if } \text{system_ram} \leq 64\text{ GB} \\\\ 8\text{ GB} & \text{if } \text{system_ram} \leq 128\text{ GB} \\\\ 12\text{ GB} & \text{otherwise} \end{cases}
$$

At the tier boundaries, the reserved fraction decreases: 25% on a 16 GB machine, 9.4% on 64 GB, 6.3% on 128 GB, 2.3% on 512 GB.

The remaining memory after the OS reserve is the **inference budget**:

$$
\text{inference_budget} = \text{system_ram} - \text{os_reserve}
$$

The KV cache budget is the inference budget minus what the model itself consumes:

$$
\text{kv_budget} = \text{inference_budget} - \text{weight_memory} - \text{profile_peak}
$$

If the KV budget is zero or negative, the model does not fit within the inference budget. Following upstream vLLM's behavior, the engine should refuse to start and log an explicit message: either free memory, or set `VLLM_METAL_OS_RESERVE` to a smaller value. The tiered reserve is the default, designed for Macs running other applications alongside inference. For dedicated servers, `VLLM_METAL_OS_RESERVE=2` (in GB) replaces the tier lookup with an explicit 2 GB reserve. An environment variable, not a CLI flag, matches vllm-metal's existing configuration pattern (`VLLM_METAL_PREFIX_CACHE`, etc.).

### Lesson from mistral.rs: The Wired Collector

macOS distinguishes between pageable memory (which can be swapped to disk under pressure) and wired memory (which is pinned in physical RAM). Metal GPU buffers are wired. When the system comes under memory pressure, macOS invokes the **wired collector**, a kernel mechanism that reclaims GPU wired memory by evicting Metal buffers, causing silent performance degradation or crashes with no warning visible to the application.

The mistral.rs community discovered and documented this behavior in [mistral.rs#1348](https://github.com/EricLBuehler/mistral.rs/issues/1348). The workaround is a single sysctl: `sudo sysctl iogpu.disable_wired_collector=1`, which tells the kernel not to reclaim GPU wired memory even under pressure. The setting does not persist across reboots unless added to a startup script.

The tiered OS reserve reduces wired collector risk by leaving headroom below the pressure threshold. For dedicated servers configured with a minimal reserve via `VLLM_METAL_OS_RESERVE`, documenting `iogpu.disable_wired_collector=1` as a companion setting is prudent.