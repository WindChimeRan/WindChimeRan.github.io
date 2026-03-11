---
layout: post
title: "Developing vllm-metal's Paged Varlen FlashAttention Kernel"
description: ""
tags: [LLM, inference, Apple Silicon, systems, vllm-metal]
categories: [research]
giscus_comments: false
related_posts: false
toc:
  beginning: true
---

*Draft — work in progress.*

## Background

From reading the [nano-vllm](https://github.com/some-repo/nano-vllm) and vLLM source code, two imports stand out as foundational:

```python
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
```

These two functions encode the core requirements for a production attention kernel:

- **Paged KV cache** — the soul of vLLM. We are a vLLM plugin, so this is non-negotiable.
- **Varlen (variable-length) support** — the foundation for continuous batching, chunked prefilling, MQA Scorer DraftModel speculative decoding, and more.
- **FlashAttention IO-aware acceleration** — important but lower priority. The algorithm is supercomplicated and deeply bound to NVIDIA hardware. That said, the online softmax trick at its core should be straightforward to implement.

Of these, paged and varlen are the most important features. FlashAttention itself is in lower priority compared to these two.

The problem: both `flash_attn_varlen_func` and `flash_attn_with_kvcache` are either implemented fully in CUDA or bound to Triton — neither is available on Metal.

That is the starting point for this work.

## Vibe Research

To map the landscape, I launched four parallel Claude Code research agents:

1. **FlashAttention innovations** — surveying FlashAttention v1, v2, v3, v4, FlashInfer, etc.
2. **vLLM Triton unified attention kernel** — how vLLM's own Triton kernel works
3. **Metal vs NVIDIA platform capabilities** — what Metal gives us, what is NVIDIA-specific
4. **Other vLLM plugin repos** — what other platform ports (ROCm, XPU, etc.) have done

*TODO: Insert findings from each agent.*

### Round 2: Grounded Code Research Against Metal Kernel Repos

Using the final report from Round 1, I launched a second team of agents to compare findings against existing Metal FlashAttention implementations:

- [metal-flash-attention](https://github.com/philipturner/metal-flash-attention) (philipturner)
- [universal-metal-flash-attention](https://github.com/bghira/universal-metal-flash-attention) (bghira)
- [mlx-flashattention-steel](https://github.com/marcogva-hub/mlx-flashattention-steel/) (marcogva-hub)

**Caveat:** Some of these repos are vibe-coded and may not be reliable. The research focused narrowly on features identified in the Round 1 report — other features are bonus, not the goal.

**Methodology:** Same divide-and-conquer pattern — one agent per repo, then the main agent cross-validates findings and resolves conflicts.

**Core question:** Are there any code snippets we can directly vendor into vllm-metal? The research is grounded to our repo — we already have an existing `kernel_v1` and a triangle unit test for the future unified attention kernel.

*TODO: Insert Round 2 findings and vendorability assessment.*

### Overview: What We Can Do in Metal vs What Is NVIDIA-Specific

*TODO: Synthesis of both rounds — what's portable, what can be vendored, what needs reimplementation, what to skip.*

## FlashAttention Deep Dive

*TODO: Deep dive into FlashAttention v1, v2, v3, v4, FlashInfer, etc. — understanding what to port and what to adapt for Metal.*

## Metal vs CUDA: What Do We Have?

*TODO: Deep code research — existing Metal code that can be vendored directly, and what needs to be built from scratch.*
