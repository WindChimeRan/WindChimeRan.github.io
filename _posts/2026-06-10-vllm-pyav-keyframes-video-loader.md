---
layout: post
title: "Keyframe-Only Video Loading in vLLM: The Accuracy-Throughput Trade"
date: 2026-06-10 10:00:00-0000
description: "An opt-in lossy video loader for vLLM that decodes only I-frames: 1.77× end-to-end on offline video classification, −0.1 pt on NExTQA, −11.3 pt on MVBench."
tags: [LLM, inference, video, multimodal, vllm]
categories: [research]
giscus_comments: false
related_posts: false
toc:
  beginning: true
---

eBay has billions of videos and a steady stream of classification jobs over them: a video plus a classification prompt goes to a vision-language model, and the answer is one multiple-choice token. A typical job is 1,000–2,000 short low-resolution clips run offline through vLLM's `LLM.chat` with `max_tokens=1`, so there is no generation phase: the run is prefill, and every input-side cost sits on the critical path. Profiling these jobs (Qwen2.5-VL-7B-Instruct, one A100, TP=1) put video decode at 28–44% of wall time depending on the dataset. The GPU waits while the CPU turns compressed video into pixel arrays.

[PR #45203](https://github.com/vllm-project/vllm/pull/45203) is the workaround I proposed upstream: `pyav_keyframes`, an opt-in lossy video loader that decodes only keyframes, so decode work is at most `num_frames` single-frame decodes per clip regardless of clip length. What it buys and what it costs are both measurable. All numbers below are from public datasets, with full settings and tuning logs in [offline_video_vllm](https://github.com/WindChimeRan/offline_video_vllm).

## Where decode time goes

Video codecs store a complete image only at periodic keyframes (I-frames). The frames between them are motion-compensated deltas: P-frames reference earlier frames, B-frames reference earlier and later ones. A keyframe plus the frames that depend on it forms a GOP (group of pictures), typically 2–10 s in web encodes.

{% raw %}
<style>
.gop-strip{margin:1.2em 0}
.gop-strip .gop{display:flex;gap:3px;flex-wrap:wrap}
.gop-strip .fr{width:30px;height:38px;border-radius:5px;display:flex;align-items:center;justify-content:center;color:#fff;font:700 .72rem ui-monospace,SFMono-Regular,Menlo,monospace}
.gop-strip .I{background:#EE7733}
.gop-strip .P{background:#4477AA}
.gop-strip .B{background:#9dbbd8}
.gop-strip .legend{display:flex;gap:16px;flex-wrap:wrap;font-size:.85rem;color:var(--global-text-color-light,#5a5a72);margin:.4em 0 0}
.gop-strip .legend span::before{content:"";display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.gop-strip .lg-I::before{background:#EE7733}
.gop-strip .lg-P::before{background:#4477AA}
.gop-strip .lg-B::before{background:#9dbbd8}
</style>
<div class="gop-strip">
  <div class="gop">
    <div class="fr I">I</div><div class="fr B">B</div><div class="fr B">B</div><div class="fr P">P</div>
    <div class="fr B">B</div><div class="fr B">B</div><div class="fr P">P</div><div class="fr B">B</div>
    <div class="fr P">P</div>
    <div class="fr I">I</div><div class="fr B">B</div><div class="fr P">P</div><div class="fr B">B</div>
    <div class="fr B">B</div><div class="fr P">P</div><div class="fr B">B</div><div class="fr P">P</div>
    <div class="fr B">B</div>
  </div>
  <div class="legend"><span class="lg-I">I = keyframe (self-contained)</span> <span class="lg-P">P = delta vs past</span> <span class="lg-B">B = delta vs past + future</span></div>
</div>
{% endraw %}

Decoding an arbitrary frame means starting at its GOP's keyframe and decoding forward, because P and B frames are meaningless alone. A lossless sparse sampler pays that GOP-prefix decode for every target it touches, however few targets it keeps. Keyframes have neither problem: decoding one costs one frame decode, and finding them costs no decode at all, since the demuxer reads packet headers that already carry a keyframe flag and a timestamp.

## A loader that never decodes a delta frame

`pyav_keyframes` makes two passes over the container. The first demuxes the stream and records every keyframe timestamp; no pixels are decoded. The second spreads `num_frames` picks evenly over that keyframe list, then seeks to each pick and decodes exactly one frame. A 30-second clip and a 10-minute clip cost the same: one header sweep plus at most `num_frames` keyframe decodes.

When the budget exceeds the keyframe count, picks repeat instead of falling back to delta frames, and the repeats stay balanced: a 2-keyframe clip asked for 16 frames returns 8 copies of each. Repeated frames are decoded once and reported at their true source positions in the frame metadata, so a model like Qwen2.5-VL, which embeds each frame's time position, sees the same moment twice rather than motion that never happened.

The sliders below replay the pick logic: set how many keyframes the clip has and how many frames the caller asks for, then compare the decode work of lossless uniform sampling against keyframe-only sampling for the same request.

{% raw %}
<style>
.pick-widget{
  --pw-orange:#EE7733; --pw-red:#CC3311;
  --pw-ink:var(--global-text-color, #1a1a2e);
  --pw-muted:var(--global-text-color-light, #5a5a72);
  --pw-line:var(--global-divider-color, #e4e4ec);
  --pw-chip:#d9dbe4; --pw-kf-bg:#f0e4da; --pw-kf-ink:#7a3c12; --pw-out-bg:#eef0f5;
  background:var(--global-card-bg-color, #ffffff);
  border:1.5px solid var(--pw-line); border-radius:12px;
  padding:16px 18px; margin:1.5em 0; color:var(--pw-ink);
}
html[data-theme="dark"] .pick-widget{
  --pw-red:#e85c41; --pw-chip:#44444f; --pw-kf-bg:#3d2d1e; --pw-kf-ink:#f3c89e; --pw-out-bg:#2b2b33;
}
.pick-widget label{font-size:.9rem;margin-right:18px}
.pick-widget input[type=range]{vertical-align:middle;width:160px}
.pick-widget .pw-h{font-weight:700;margin:1.2em 0 .3em}
.pick-widget .hint{font-size:.85rem;color:var(--pw-muted)}
.pick-widget .chips{display:flex;gap:2px;flex-wrap:wrap;margin:.5em 0}
.pick-widget .chip{width:14px;height:26px;border-radius:3px;background:var(--pw-chip)}
.pick-widget .chip.kf{background:var(--pw-orange)}
.pick-widget .chip.dec{outline:2.5px solid var(--pw-red);outline-offset:-2px}
.pick-widget .kfrow{display:flex;gap:5px;flex-wrap:wrap;margin:.5em 0}
.pick-widget .kfchip{min-width:36px;padding:4px 5px;border-radius:6px;background:var(--pw-kf-bg);
  border:1.5px solid var(--pw-orange);text-align:center;
  font:600 .75rem ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--pw-kf-ink)}
.pick-widget .kfchip .cnt{display:block;font-size:.95rem;color:var(--pw-red)}
.pick-widget .kfchip.zero{opacity:.32;border-style:dashed}
.pick-widget .stat{font-size:.92rem;margin:.4em 0;color:var(--pw-muted)}
.pick-widget .stat b{color:var(--pw-ink)}
.pick-widget .out{font:.82rem ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--pw-out-bg);
  border-radius:6px;padding:6px 10px;margin:.4em 0;overflow-x:auto}
.pick-widget .legend{display:flex;gap:16px;flex-wrap:wrap;font-size:.85rem;color:var(--pw-muted);margin:.6em 0 0}
.pick-widget .legend span::before{content:"";display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.pick-widget .lg-I::before{background:var(--pw-orange)}
.pick-widget .lg-skip::before{background:var(--pw-chip)}
</style>
<div class="pick-widget">
  <label>keyframes in clip (<code>n_kf</code>): <input type="range" id="nkf" min="2" max="24" value="12"> <b id="nkfv">12</b></label>
  <label>budget (<code>num_frames</code>): <input type="range" id="nf" min="1" max="32" value="16"> <b id="nfv">16</b></label>
  <div class="pw-h">Which keyframes get picked (badge = times duplicated)</div>
  <div class="kfrow" id="kfrow"></div>
  <div class="out" id="picksout"></div>
  <div class="pw-h">Decode work on the timeline (GOP = 10 frames)</div>
  <div class="hint">lossless uniform sampling — must decode every outlined frame to reach its targets:</div>
  <div class="chips" id="lossless"></div>
  <div class="hint"><code>pyav_keyframes</code> — decodes only outlined keyframes:</div>
  <div class="chips" id="lossy"></div>
  <div class="stat" id="stats"></div>
  <div class="legend"><span class="lg-I">keyframe</span><span class="lg-skip">delta frame</span> <span style="color:var(--pw-red)">▢ outlined = actually decoded</span></div>
</div>
<script>
function npRound(x){ // numpy round-half-even
  const f=Math.floor(x), d=x-f;
  if(d>0.5) return f+1;
  if(d<0.5) return f;
  return (f%2===0)?f:f+1;
}
function linspace(a,b,k){
  if(k===1) return [a];
  const out=[]; for(let i=0;i<k;i++) out.push(a+(b-a)*i/(k-1)); return out;
}
const GOP=10;
function render(){
  const nkf=+document.getElementById('nkf').value;
  const nf=+document.getElementById('nf').value;
  document.getElementById('nkfv').textContent=nkf;
  document.getElementById('nfv').textContent=nf;
  const picks=linspace(0,nkf-1,nf).map(npRound);
  const counts=Array(nkf).fill(0); picks.forEach(p=>counts[p]++);
  // keyframe row
  const kfrow=document.getElementById('kfrow'); kfrow.innerHTML='';
  for(let i=0;i<nkf;i++){
    const d=document.createElement('div');
    d.className='kfchip'+(counts[i]?'':' zero');
    d.innerHTML='kf'+i+'<span class="cnt">'+(counts[i]?'×'+counts[i]:'—')+'</span>';
    kfrow.appendChild(d);
  }
  document.getElementById('picksout').textContent=
    'picks = ['+picks.join(', ')+']   →   frames_indices = ['+picks.map(p=>p*GOP).join(', ')+']';
  // timeline
  const total=nkf*GOP;
  const targets=linspace(0,total-1,Math.min(nf,total)).map(Math.floor); // lossless uniform targets
  const decodedLossless=new Set();
  targets.forEach(t=>{ const kf=Math.floor(t/GOP)*GOP; for(let f=kf;f<=t;f++) decodedLossless.add(f); });
  const decodedLossy=new Set(picks.map(p=>p*GOP));
  const mk=(host,decoded)=>{
    const el=document.getElementById(host); el.innerHTML='';
    for(let f=0;f<total;f++){
      const c=document.createElement('div');
      c.className='chip'+(f%GOP===0?' kf':'')+(decoded.has(f)?' dec':'');
      el.appendChild(c);
    }
  };
  mk('lossless',decodedLossless); mk('lossy',decodedLossy);
  document.getElementById('stats').innerHTML=
    'decoded frames — lossless: <b>'+decodedLossless.size+'</b> · pyav_keyframes: <b>'+
    decodedLossy.size+'</b> ('+(decodedLossless.size/decodedLossy.size).toFixed(1)+'× fewer)';
}
document.getElementById('nkf').addEventListener('input',render);
document.getElementById('nf').addEventListener('input',render);
render();
</script>
{% endraw %}

## The trade, measured

Setup: Qwen2.5-VL-7B-Instruct on one A100 (TP=1), offline `LLM.chat`, `max_tokens=1`, 16 frames per clip, 1,990 multiple-choice questions from NExTQA and MVBench. The two runs are identical except the video loader: lossless OpenCV sampling vs `pyav_keyframes`. End-to-end wall time drops from 674 s to 380 s and throughput rises from 2.95 to 5.23 req/s, a 1.77× speedup from the loader swap alone. At the decode stage, lossless sampling costs 193 ms on a 30 s clip and 3,124 ms on a 600 s clip; `pyav_keyframes` stays between 43 and 77 ms across all four test clips.

<div style="display: flex; justify-content: center;">
  <img src="/assets/img/pyav_keyframes_speed.png" alt="Decode time per clip and end-to-end wall time, lossless vs pyav_keyframes" style="max-width: 100%; height: auto;" />
</div>

<p align="center"><em>Left: decode time to extract 16 frames (log scale) across clip lengths and GOP settings. Right: end-to-end wall time on the N=1990 benchmark; the only change between bars is the video loader.</em></p>

Accuracy is the other side of the trade. NExTQA, scene-content QA, is unchanged at 79.6 vs 79.5. MVBench drops 11.3 points overall, and the drop concentrates in motion-sensitive subtasks: `action_antonym` loses 52.7 points, `moving_attribute` and `object_existence` lose 36.4 each, while 10 of 18 subtasks stay within ±2 points. Keyframes land on scene boundaries, so prompts about what a scene contains keep their signal; prompts about what changes between keyframes lose it.

<div style="display: flex; justify-content: center;">
  <img src="/assets/img/pyav_keyframes_accuracy.png" alt="Overall accuracy on NExTQA and MVBench, and MVBench per-subtask deltas" style="max-width: 100%; height: auto;" />
</div>

<p align="center"><em>Left: overall accuracy, num_frames=16. Right: MVBench per-subtask deltas, the worst and best of 18 subtasks; the remaining 10 move less than ±2 pt.</em></p>

The decode savings apply to every clip by construction; the accuracy cost lands only where the prompt needs inter-keyframe information. A classification job in the eBay mold, one-token answers about what the clip shows, matches the NExTQA column. Motion reasoning matches the `action_antonym` column and should stay on a lossless loader.

## Using it

The loader is opt-in; nothing changes unless a run selects the `pyav_keyframes` backend. Until the PR lands, the same loader ships as a single-file drop-in (`pyav_keyframes_v2`) in the experiment repo: importing the module registers it with vLLM's loader registry, and the README has the exact `LLM(...)` configuration.

## Links

- vLLM PR: [vllm-project/vllm#45203](https://github.com/vllm-project/vllm/pull/45203), "[Multimodal] Add lossy keyframe-only video loader (pyav_keyframes)"
- Experiments and benchmarks (public datasets): [WindChimeRan/offline_video_vllm](https://github.com/WindChimeRan/offline_video_vllm)
