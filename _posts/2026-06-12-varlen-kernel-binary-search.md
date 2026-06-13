---
layout: post
title: "One Launch for Any Batch: The Binary Search Inside vllm-metal's Varlen Attention"
date: 2026-06-12 00:00:00-0000
description: "How vllm-metal's paged attention kernels resolve which sequence owns each token with a per-threadgroup binary search, what that buys continuous batching on Apple Silicon, and what happened when we replaced it with an O(1) lookup map."
tags: [Metal, kernel, Apple Silicon, vllm-metal, systems]
categories: [research]
giscus_comments: false
related_posts: false
toc:
  beginning: true
---

The [previous post](/blog/2026/vllm-metal-vs-mlx-lm-kv-cache/) ended at the kernel boundary: vLLM's scheduler emits a varlen schedule, and the schedule pays off only if the attention kernel can consume it. When one GPU launch covers a batch that mixes prefill chunks, decode steps, and draft verification, how does each parallel worker discover which request it is working for?

## The ladder from chip to lane

This machine is an M1 Pro with 14 GPU cores. A Metal compute dispatch is a **grid** of **threadgroups**; a threadgroup runs on one GPU core, which keeps several resident and switches between them to hide memory latency. Within a threadgroup, threads execute in **simdgroups** that step through instructions in lockstep. Apple's own documentation maps the vocabulary to other dialects: a simdgroup is what CUDA calls a warp, a lane is a thread within it.

{% raw %}
<style>
.hier{
  --hr-ink:var(--global-text-color,#1a1a2e);
  --hr-muted:var(--global-text-color-light,#5a5a72);
  --hr-line:var(--global-divider-color,#e4e4ec);
  background:var(--global-card-bg-color,#fff);
  border:1.5px solid var(--hr-line);border-radius:12px;
  padding:14px 16px;margin:1.4em 0;color:var(--hr-ink);font-size:.85rem;
}
.hier .hbox{border:1.5px solid;border-radius:8px;padding:8px 12px;margin:8px 0 2px}
.hier .hrow{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap}
.hier .hl{font-weight:700;font-size:.82rem}
.hier .hm{color:var(--hr-muted);font-size:.78rem}
.hier .hbadge{margin-left:auto;font:600 .72rem ui-monospace,SFMono-Regular,Menlo,monospace;
  color:var(--hr-muted);border:1px solid var(--hr-line);border-radius:10px;padding:1px 8px;white-space:nowrap}
.hier .chip{border-color:#4477AA}
.hier .core{border-color:#009988}
.hier .tg{border-color:#EE7733;margin:20px 18px 2px 0;background:var(--global-card-bg-color,#fff);
  box-shadow:6px -6px 0 0 var(--global-card-bg-color,#fff),6px -6px 0 1.5px #EE7733,
    12px -12px 0 0 var(--global-card-bg-color,#fff),12px -12px 0 1.5px #EE7733}
.hier .sg{border-color:#CC3311}
.hier .lanes{display:flex;gap:2px;margin-top:6px;flex-wrap:wrap}
.hier .lane{width:9px;height:14px;border-radius:2px;background:#CC3311;opacity:.55}
</style>
<div class="hier">
  <div class="hbox chip">
    <div class="hrow"><span class="hl">M1 Pro</span><span class="hm">shared unified memory</span><span class="hbadge">14 GPU cores</span></div>
    <div class="hbox core">
      <div class="hrow"><span class="hl">GPU core</span><span class="hm">runs whole threadgroups; threadgroup memory lives in the core</span><span class="hbadge">many threadgroups, switched to hide latency</span></div>
      <div class="hbox tg">
        <div class="hrow"><span class="hl">threadgroup</span><span class="hm">up to 1,024 threads, 32 KB threadgroup memory</span><span class="hbadge">256 threads in the decode kernel</span></div>
        <div class="hbox sg">
          <div class="hrow"><span class="hl">simdgroup</span><span class="hm">32 lanes in lockstep</span><span class="hbadge">8 per threadgroup here</span></div>
          <div class="lanes"><div class="lane"></div><div class="lane"></div><div class="lane"></div><div class="lane"></div><div class="lane"></div><div class="lane"></div><div class="lane"></div><div class="lane"></div><div class="lane"></div><div class="lane"></div><div class="lane"></div><div class="lane"></div><div class="lane"></div><div class="lane"></div><div class="lane"></div><div class="lane"></div><div class="lane"></div><div class="lane"></div><div class="lane"></div><div class="lane"></div><div class="lane"></div><div class="lane"></div><div class="lane"></div><div class="lane"></div><div class="lane"></div><div class="lane"></div><div class="lane"></div><div class="lane"></div><div class="lane"></div><div class="lane"></div><div class="lane"></div><div class="lane"></div></div>
        </div>
      </div>
    </div>
  </div>
</div>
{% endraw %}

{% raw %}
<style>
.bench-table{overflow-x:auto;margin:0 0 1.4em}
.bench-table table{
  width:100%;border-collapse:collapse;
  font-size:.92rem;line-height:1.4;
  font-variant-numeric:tabular-nums;
  border-top:2px solid var(--global-text-color,#1a1a2e);
  border-bottom:2px solid var(--global-text-color,#1a1a2e);
}
.bench-table th{
  font-weight:600;padding:.45em .75em;white-space:nowrap;
  border-bottom:1.5px solid var(--global-text-color,#1a1a2e);
}
.bench-table td{padding:.4em .75em}
.bench-table th:first-child,.bench-table td:first-child{padding-left:.25em}
.bench-table th:last-child,.bench-table td:last-child{padding-right:.25em}
.bench-table td:first-child{font-weight:600}
</style>
{% endraw %}

<div class="bench-table" markdown="1">

| Metal | CUDA |
|---|---|
| grid | grid |
| threadgroup | thread block |
| simdgroup | warp |
| thread / lane | thread / lane |

</div>

The numbers in the figure are read from this machine (`threadExecutionWidth`, `maxTotalThreadsPerThreadgroup`, `maxThreadgroupMemoryLength`) and match the constants hard-coded in the kernels (`NUM_SIMD_LANES=32`, the 32 KB shared-memory budget documented in `paged_ops.cpp`).

## One threadgroup per unit of query work

vllm-metal ships two paged attention kernels in `kernels_v2/`, and the host picks per step:

**Pure-decode batches** run `paged_attention` in `pagedattention.metal`. The grid is `(num_heads, total_q_tokens)`: one threadgroup per query token per head, which for a real batch is thousands of threadgroups that the 14 cores work through a few at a time. Its 256 threads form 8 simdgroups that stride across the sequence's KV blocks, each maintaining online-softmax running state, merged across simdgroups at the end.

**Batches with more query tokens than sequences**, which means any batch containing prefill, run `paged_attention_tiled` in `pagedattention_tiled.metal` when the dtype and head size qualify; the per-token kernel is the fallback for everything else. The tiled kernel is FlashAttention-2 style: one threadgroup computes a 32-token block of query rows for one head (`BQ=32`), with four simdgroups each owning 8 rows, and Q·Kᵀ and P·V done as 8×8 `simdgroup_multiply_accumulate` tiles.

Both grids are flat over the packed token axis from the previous post. The padded alternative would be a grid over `(batch, max_seq_len)` with most threadgroups assigned to padding. Here every threadgroup corresponds to real work, with one exception: the tiled grid is sized `total_q_tokens/BQ + num_seqs`, a deliberate over-estimate of how many 32-token blocks the ragged sequences actually fill. Surplus threadgroups discover they are surplus and exit. That choice is inherited.

## Which sequence owns token 4,097?

A threadgroup wakes up knowing only its grid coordinates: head 3, query token 4,097. The KV cache is paged, so before it can do anything it needs `seq_idx`: which request's block table to walk, which context length bounds the causal mask, where its sequence starts in the packed query tensor. The batch is ragged, so token 4,097 could belong to any request.

The mapping from token to request lives in one array: `cu_seqlens_q`, the exclusive prefix sum of query lengths, `[0, len_0, len_0+len_1, ...]`, length `num_seqs + 1`. Finding the owner of token `t` means finding the largest `i` with `cu_seqlens_q[i] <= t`. The array is sorted. That is a binary search:

```c
// pagedattention.metal
inline int find_seq_idx(const device int32_t *cu_seqlens_q,
                        int q_token_idx, int num_seqs) {
  int lo = 0, hi = num_seqs;
  while (lo < hi) {
    int mid = (lo + hi + 1) / 2;
    if (cu_seqlens_q[mid] <= q_token_idx) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}
```

Step through it below. Click a token; the steps are the ones the kernel executes.

{% raw %}
<style>
.bsw{
  --bsw-ink:var(--global-text-color,#1a1a2e);
  --bsw-muted:var(--global-text-color-light,#5a5a72);
  --bsw-line:var(--global-divider-color,#e4e4ec);
  --bsw-out:#eef0f5; --bsw-hit:#009988; --bsw-miss:#CC3311;
  background:var(--global-card-bg-color,#fff);
  border:1.5px solid var(--bsw-line);border-radius:12px;
  padding:16px 18px;margin:1.5em 0;color:var(--bsw-ink);
}
html[data-theme="dark"] .bsw{ --bsw-out:#2b2b33; --bsw-miss:#e85c41; }
.bsw label{font-size:.88rem}
.bsw input[type=range]{vertical-align:middle;width:120px}
.bsw .bsw-h{font-weight:700;margin:1.05em 0 .3em}
.bsw .hint{font-size:.85rem;color:var(--bsw-muted);margin:.15em 0 .5em}
.bsw .strip{display:flex;gap:2px;flex-wrap:wrap;margin:.4em 0}
.bsw .tok{width:15px;height:26px;border-radius:3px;cursor:pointer;opacity:.85}
.bsw .tok.sel{outline:2.5px solid var(--bsw-ink);outline-offset:-2px;opacity:1}
.bsw .cuarr{font:.82rem ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--bsw-out);
  border-radius:6px;padding:6px 10px;margin:.4em 0;overflow-x:auto;white-space:nowrap}
.bsw .steps{font:.82rem ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--bsw-out);
  border-radius:6px;padding:6px 10px;margin:.4em 0;overflow-x:auto}
.bsw .steps .yes{color:var(--bsw-hit);font-weight:700}
.bsw .steps .no{color:var(--bsw-miss);font-weight:700}
.bsw .res{font-size:.92rem;margin:.5em 0 0}
</style>
<div class="bsw">
  <label>requests in the batch: <input type="range" id="bswn" min="2" max="16" value="6"> <b id="bswnv">6</b></label>
  <div class="hint">query lengths mix decode (1 token) and prefill chunks, like a real continuous-batching step</div>
  <div class="bsw-h">packed query tokens, colored by request</div>
  <div class="strip" id="bswstrip"></div>
  <div class="cuarr" id="bswcu"></div>
  <div class="bsw-h">binary search for the selected token</div>
  <div class="steps" id="bswsteps"></div>
  <p class="res" id="bswres"></p>
  <div class="hint">every thread of the threadgroup runs these same steps in lockstep on the same inputs</div>
</div>
<script>
(function(){
const LENS=[1,1,5,1,12,2,1,7,1,3,4,1,9,1,2,6];
const PAL=['#4477AA','#EE7733','#009988','#CC3311','#AA3377','#66CCEE','#228833','#CCBB44',
           '#4477AA','#EE7733','#009988','#CC3311','#AA3377','#66CCEE','#228833','#CCBB44'];
const $=id=>document.getElementById(id);
let nseq=6, sel=null;
function cu(){const a=[0];for(let i=0;i<nseq;i++)a.push(a[i]+LENS[i]);return a;}
function search(c,t){
  const steps=[];let lo=0,hi=nseq;
  while(lo<hi){
    const mid=Math.floor((lo+hi+1)/2), ok=c[mid]<=t;
    steps.push({lo,hi,mid,v:c[mid],ok});
    if(ok)lo=mid;else hi=mid-1;
  }
  return {steps,seq:lo};
}
function render(){
  $('bswnv').textContent=nseq;
  const c=cu(), total=c[nseq];
  if(sel===null||sel>=total) sel=Math.min(Math.floor(total*0.7),total-1);
  const strip=$('bswstrip');strip.innerHTML='';
  for(let s=0;s<nseq;s++)for(let k=0;k<LENS[s];k++){
    const t=c[s]+k,d=document.createElement('div');
    d.className='tok'+(t===sel?' sel':'');d.style.background=PAL[s];d.title='token '+t;
    d.addEventListener('click',()=>{sel=t;render();});
    strip.appendChild(d);
  }
  $('bswcu').textContent='cu_seqlens_q = ['+c.join(', ')+']   (num_seqs = '+nseq+')';
  const {steps,seq}=search(c,sel);
  $('bswsteps').innerHTML=steps.map((s,i)=>
    'step '+(i+1)+':  lo='+s.lo+' hi='+s.hi+' → mid='+s.mid+',  cu['+s.mid+']='+s.v+' ≤ '+sel+' ?  '+
    (s.ok?'<span class="yes">yes → lo=mid</span>':'<span class="no">no → hi=mid−1</span>')
  ).join('<br>');
  $('bswres').innerHTML='token <b>'+sel+'</b> belongs to <b style="color:'+PAL[seq]+'">request '+seq+
    '</b> after <b>'+steps.length+'</b> steps; the bound is ⌈log₂(num_seqs+1)⌉ = '+Math.ceil(Math.log2(nseq+1));
}
$('bswn').addEventListener('input',e=>{nseq=+e.target.value;sel=null;render();});
render();
})();
</script>
{% endraw %}

The search runs before any attention math, in every threadgroup, and nothing about it is parallel. All threads of the threadgroup, 256 in the decode kernel and 128 in the tiled one, execute it redundantly on identical inputs and take identical branches. Redundant execution is the right call: the alternative is one lane searching while the rest wait at a barrier, then a broadcast through threadgroup memory. Uniform redundant execution costs the same wall time as one lane and skips the synchronization.

The cost is at most $\lceil \log_2(N{+}1) \rceil$ iterations, each one dependent int32 load. At 16 concurrent requests that is 5 loads of a 68-byte array that every threadgroup touches, against a KV loop that streams megabytes. For a pure decode batch the search is computing the identity function: `cu_seqlens_q` is `[0, 1, 2, ...]`, so token `t` belongs to request `t`. The kernel pays those loads to learn that.

## Borrowed from Triton

The kernel's own comment names its ancestor: the approach is "the same approach used by the upstream vLLM unified Triton kernel," `triton_unified_attention.py:find_seq_idx`. vllm-metal adapted the upstream kernel's tests first, then translated the kernel to Metal against them. Upstream's `find_seq_idx` is also a binary search over the same prefix array, executed once per Triton program as scalar control flow (it has since moved to `triton_attention_helpers.py`).

The tiled kernel inherits a subtler trick. Its grid is in units of 32-token Q-blocks, but blocks are aligned per sequence, so block boundaries depend on where sequences start. Rather than materialize a second prefix array in block units, the tiled kernel and its Triton ancestor search the token-unit array through a transform: compare against `cu_seqlens_q[mid] / BQ + mid` instead of `cu_seqlens_q[mid]`. Each sequence contributes `len/BQ` full blocks plus one potential ragged tail, and the `+ mid` term counts the tails. One array serves both coordinate systems.

The over-provisioned grid has the same upstream origin, and upstream documents the reason: computing the exact number of Q-blocks would require reading the query lengths on the CPU, and the kernel's authors judged a few empty threadgroups cheaper than that synchronization. The pattern repeats: spend trivial GPU work to keep the CPU out of the launch path.

## What continuous batching asks of the kernel

Trace one step end to end. The vLLM scheduler hands vllm-metal a step that mixes, say, 32 decoding requests and one mid-prefill request. The model runner packs them into one flat token tensor, decodes first, and builds `cu_seqlens_q` as plain Python lists in `prepare_unified()`: something like `[0, 1, 2, ..., 32, 544]`. Every attention layer makes one paged-attention dispatch; the C++ side routes this batch to the tiled kernel, sizes the grid from `total_q_tokens`, and each threadgroup binary-searches its way to its request. No layer of this stack ever sees a `[batch, seq_len]` rectangle.

Each forward pass is one attention dispatch per layer regardless of batch composition (a small pure-decode batch adds a reduce pass via the split-KV path), which is what continuous batching asks for: requests join and leave the batch every step, query lengths change every step, and none of that changes the kernel, only the contents of a 136-byte prefix array. The speculative-decoding case from the previous post lands in the same array, though not as one piece: `prepare_unified()` expands a k+1-token verification segment into k+1 unit deltas, each with its own context length carrying the causal staircase.

## Measuring the obvious optimization

The search is $O(\log N)$ work per threadgroup to compute something the host already knows. The host builds `cu_seqlens_q`; it could just as easily build the inverse, a `token → seq_idx` array of length `total_q_tokens`, and the kernel would do one load instead of a loop. We implemented it behind an env flag and measured.

**Method.** A function constant adds a second path to both kernels: `seq_idx = seq_map[q_token_idx]` for the decode kernel, a Q-block map with sentinel entries for the tiled kernel's surplus threadgroups. The host builds the token map in `prepare_unified()` next to `cu_seqlens_q` and derives the Q-block map from it at the step's first tiled dispatch; each converts to an MLX array once per step. Correctness is checked by asserting bit-identical attention output against the binary-search path across decode, mixed, ragged-tail, and spec-verify shaped batches, and a dispatch-side counter asserts that each timing arm actually ran the path it claims. Timing is ABAB-interleaved across processes, 2 warmup batches then pooled timed repeats, medians and IQR reported. Shapes: 16 query heads, 4 KV heads, head size 128, fp16 (Qwen3-0.6B's query-head count and head size; the model itself has 8 KV heads).

**The search is not free everywhere.** Per-dispatch medians on the M1 Pro:

<div class="bench-table" markdown="1">

| batch | kernel | binary search (µs) | O(1) map (µs) | delta |
|---|---|---:|---:|---:|
| decode, 64 reqs, ctx 128 | per-token | 322.6 | 296.5 | −8.1% |
| decode, 256 reqs, ctx 128 | per-token | 1,277.1 | 1,171.0 | −8.3% |
| decode, 1,024 reqs, ctx 128 | per-token | 5,159.2 | 4,580.9 | −11.2% |
| decode, 64 reqs, ctx 2,048 | per-token | 4,074.3 | 4,043.2 | −0.8% |
| decode, 1,024 reqs, ctx 2,048 | per-token | 65,141.9 | 64,259.9 | −1.4% |
| prefill, 8 × 512 | tiled | 7,297.2 | 7,157.0 | −1.9% |
| mixed, 32 decode + 4 × 512 | tiled | 7,022.6 | 6,811.6 | −3.0% |

</div>

The pattern matches the arithmetic. At context 128 the KV loop visits 8 blocks, so the search's dependent loads are a visible fraction of each threadgroup's runtime, and the fraction grows with the batch: 64 requests take up to 7 search iterations, 1,024 up to 11. At context 2,048 the KV loop visits 128 blocks and the search disappears into it. No configuration regressed beyond noise. The host pays for the map: at 1,024 decoding requests, 86 µs to build the Python list and 10 µs to convert it, per step. That is under 0.1% of the attention time of the step it serves.

**End to end, at ordinary scale, nothing moves.** Serving Qwen3-0.6B at concurrency 16 with 1,024-token inputs, 3 ABAB rounds per arm: output throughput +0.9%, median TTFT −1.2%, median TPOT +0.5%, p99 TPOT +2.6%, all inside the round-to-round IQR. Three rounds per arm can only detect complete separation, and the TPOT point estimates lean slightly against the map, so the supported read is flat. At 1,024 tokens of context, decode sits in the regime where the table above says the search costs nothing, so this is the microbench's prediction, confirmed.

**The favorable regime, end to end.** Serving the same model at concurrency 64 with 128-token inputs and outputs keeps every decode step at context 128 to 256 with a 64-deep batch: the microbench's best case. The first three ABAB rounds per arm showed output throughput +3.1% and median TPOT −3.0%. Nine rounds per arm dissolved it: +0.9% mean throughput, −0.7% mean TPOT, 6 of 9 pairwise wins, exact permutation p = 0.26 and 0.45. The early +3.1% traced to three fast server processes that landed on both arms (Fisher p = 1.0) and never recurred. An 8-11% kernel-side win dilutes to under 1% of serving time, and separating a 1% effect from 1.3% per-round noise takes about thirty rounds per arm; we stopped at nine.

**Where this sits in the design space.** Kernels that need token-to-request resolution have settled into four patterns. Per-program search over the prefix array, as here and in upstream's unified Triton kernel. Grid axes that encode the request directly, as in the classic padded PagedAttention launch, paying threadgroups for padding instead. Host-precomputed work descriptors, as in FlashInfer's `plan()` phase and FlashMLA's tile scheduler metadata, which amortize scheduling onto the CPU once per step. And explicit precomputed index maps, which is the variant we tested; vLLM already ships them in FlexAttention's `doc_ids` and ROCm AITER's `token_to_batch` (non-default attention backends) and in Mamba2's per-chunk `seq_idx`, but not in its default attention path.

**Verdict.** The binary search is not literally free: replace it with a host-built map and the decode kernel returns up to 11% of its time at short context and large batch, bit-identically. It is free where it matters: those are the configurations where the kernel is cheapest, so serving throughput moves by less than the run-to-run noise. The upstream trade, a few empty threadgroups and a $\log_2 N$ loop in exchange for keeping the CPU out of the launch path, survives measurement, and the seq-map branch stays parked in its experiment worktree. The patch worth upstreaming from this exercise is the bug the bit-identical assertion flushed out of the unmodified baseline instead: an exception thrown from inside `mx.eval` leaves MLX's Metal eval state wedged, and a later, valid eval can return an unwritten output buffer.

## Links

- vllm-metal: [github.com/vllm-project/vllm-metal](https://github.com/vllm-project/vllm-metal), kernels in `vllm_metal/metal/kernels_v2/`
- Upstream ancestor: [`triton_unified_attention.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_unified_attention.py), introduced in [vllm#16828](https://github.com/vllm-project/vllm/pull/16828); `find_seq_idx` now lives in [`triton_attention_helpers.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_attention_helpers.py)
- Apple GPU execution model: [Scale compute workloads across Apple GPUs, WWDC22 session 10159](https://developer.apple.com/videos/play/wwdc2022/10159/), [Metal feature set tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- Previous post: [Contiguous vs Paged Varlen KV Cache](/blog/2026/vllm-metal-vs-mlx-lm-kv-cache/)
