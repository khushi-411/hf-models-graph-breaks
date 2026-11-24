# Graph-break sweep: results (2026-11-18/19, top-100 batch 2026-11-23)

Every model in `models.yaml` was run through both checks on **CPU and CUDA**, and every graph break
found was matched to a root cause with an issue draft: PyTorch drafts in [`issues/`](issues/), transformers and
HF Hub drafts in [`issues/hf/`](issues/hf/), and the one harness bug in [`issues/harness/`](issues/harness/).
Dynamic-shape recompiles were triaged too (see [Dynamic-shape recompiles](#dynamic-shape-recompiles)). Per-model detail is in
[`COVERAGE.md`](COVERAGE.md) (all checks and devices) and in
[`SUMMARY.md`](SUMMARY.md) / [`SUMMARY_CUDA.md`](SUMMARY_CUDA.md) /
[`DYNAMIC_SUMMARY.md`](DYNAMIC_SUMMARY.md) / [`DYNAMIC_CUDA_SUMMARY.md`](DYNAMIC_CUDA_SUMMARY.md).

**Status:** 209 models · 0 unmatched graph breaks · 0 unexplained errors · 77 issue drafts
(33 PyTorch, 43 transformers/HF Hub, 1 harness). Nothing is filed upstream yet. The Nov 23 batch (the 100
most-downloaded Hub models, [below](#nov-23-the-100-most-downloaded-hub-models)) added 52 models and 9 drafts.
A partial `dynamic.py --unbacked` sweep (Nov 24, [below](#unbacked-dynamic-shapes-partial-stopped)) added 2 drafts.
An Inkling deep dive (Nov 24, [below](#inkling-deep-dive-nov-24)) ran the five Inkling models through every
compile path and added 6 drafts (3 PyTorch, 3 transformers), plus a tested fix for the Inkling merge-loops crash.

## Setup

| | |
|---|---|
| torch | `2.16.0a0+git57f02f8` (source build). CUDA: rebuilt in `torch-dev` with sm_90 + sm_100, cuDNN 9.28, Triton 3.9.0. CPU: frozen copy of the same commit (`torch-snap-57f02f8`), conda g++ 13.4 |
| transformers | `5.24.0.dev0`, checkout `b7fe69cb19`. inkling2_mini(_vlm) and the Nov 22 batch (below): `046bdb8cc2` (Nov 21), needed for those models |
| Hardware | CUDA: NVIDIA B200 (`hopper-b200`). The Nov 22 runs (Nov 22 batch + inkling2_mini CUDA) are on B300 (`hopper-b300`, sm_103): the B200 partition is gone. Caveat: the `torch-dev` build ships CUDA 12.8, which doesn't support sm_103. Those runs completed, but other workloads on B300 hit NVRTC (`invalid --gpu-architecture`) and MoE kernel crashes, so treat B300 results with care; H100 (`hopper-dev`, sm_90) is fully supported. Nov 23 batch: B300 first, then every CUDA run that failed on B300 rerun on H100 (and quantized models + CUDA accuracy on H100 only). Quantized runs use `torch-dev-quant` (clone of `torch-dev` + accelerate, kernels 0.18.2, gguf). CPU: Xeon 8488C / 8124M (`hopper-cpu`) |
| Static check | `python -m graph_breaks.check --model X --device cpu\|cuda`: `torch._dynamo.explain`, config-only model with reduced layers, random weights, batch 2 / seq 32 |
| Dynamic check | `python -m graph_breaks.dynamic --model X --device cpu\|cuda`: `torch.compile(backend="inductor", dynamic=True)` over (batch, seq) = (2, 32), (3, 40), (4, 56) |

## Headline numbers

| check | device | model/scenario runs | errors | unique breaks* | total breaks |
|---|---|---|---|---|---|
| static | CPU | 535 | 20 | 624 | 1,619 |
| static | CUDA | 544 | 6 | 898 | 2,088 |
| dynamic | CPU | 405 | 33 | 440 | – |
| dynamic | CUDA | 412 | 28 | 549 | – |

(Nov 18–22 runs alone: static 414 / 414, dynamic 308 / 308 runs. New CPU errors from Nov 23 are the Qwen3.5 /
Qwen3.6 models, which need CPU BLAS; new CUDA errors are the four crash bugs in the Nov 23 section.)

\* distinct break locations per model/scenario, summed over runs. The same root cause counts once per
model and scenario it appears in, so these collapse to ~110 code locations and 60 root causes.
The new CPU errors are lara_mini and jev, which need CPU BLAS (same build limit as qwen3_8 / qwen3_next).

- Static CPU and CUDA have the **same root causes**. Only counts differ, e.g. deberta 12 vs 14 breaks,
  and ctrl breaks on CPU only (`nn-buffer-reassign`).
- All new **crashes** came from the dynamic-shape (Inductor) runs, which `check.py` alone never exercises.
- Errors are explained in `COVERAGE.md`: PyTorch crash bugs (15), harness problems (10), build limits (7,
  no CPU LAPACK/BLAS: qwen3_next and qwen3_8 on CPU), and 5 runs cancelled for compile time (below).

## Root causes by repo

### PyTorch (pytorch/pytorch): 23 · drafts in `issues/`

**Hard crashes (compile fails), new in this sweep:**

| issue | models | what |
|---|---|---|
| [inductor-cpp-half-reduction-redeclare](issues/inductor-cpp-half-reduction-redeclare.md) | pythia, phi (train, CPU) | fp16/bf16 reduction with two outputs → generated C++ declares a variable twice. Static and dynamic. Standalone repro. phi found by the accuracy check |
| [dynamo-guard-python-mod-value-range](issues/dynamo-guard-python-mod-value-range.md) | internvl (CPU + CUDA) | `size % 0.5` → `ValueRangeError: Invalid ranges [0:-0.5]` in guard building. Standalone repro |
| [inductor-merge-loops-minmax-not-comparable](issues/inductor-merge-loops-minmax-not-comparable.md) | inkling2, inkling2_vlm, inkling2_mini, inkling2_mini_vlm (eval, CUDA) | `merge_loops` → `sympy_subs` → `'-((0))' is not comparable`. `dynamic=True` and default `torch.compile`; CPU is fine. Model-level repro |
| [inductor-bounds-truncdiv-nan](issues/inductor-bounds-truncdiv-nan.md) (existing) | longformer, led | `Invalid NaN comparison` in value-range analysis |

**Recompiles / specialization / compile time (not graph breaks), new:**

| issue | models | what |
|---|---|---|
| [mamba-associative-scan-seq-guard](issues/mamba-associative-scan-seq-guard.md) | mamba, jamba, falcon_mamba | `associative_scan`: generic mode guards on `(seq//8)%2` (CPU); pointwise + autograd pins batch and seq (CUDA). Standalone repro |
| [inductor-cpu-tiling-stride-at-simplify](issues/inductor-cpu-tiling-stride-at-simplify.md) | funnel, mobilevit, gemma3_vlm, swin… | CPU compile takes hours (vs minutes on CUDA): `sympy.simplify` in `sizevars.stride_at` |
| [floordiv-simplify-compile-time](issues/floordiv-simplify-compile-time.md) | swin | 15–52 min compile on CUDA: `FloorDiv.eval` → `sympy.simplify` on nested size expressions |
| [significant-strides-equal-guard](issues/significant-strides-equal-guard.md) | qwen2_vl, qwen2_5_vl, qwen3_vl, qwen3_8 (train) | Inductor specializes `Eq(s87, 1176/1536)` |
| [dynamo-rnn-unsupported](issues/dynamo-rnn-unsupported.md) | encodec, wavtokenizer | LSTM breaks unless `allow_rnn=True` (feature request) |
| [sdpa-efficient-backward-bias-align-guard](issues/sdpa-efficient-backward-bias-align-guard.md) | t5, mt5, inkling (CUDA train; guard also in beit, swin without recompiling) | meta for efficient-attention backward branches on `attn_bias.size(-1) % 16` |
| [one-hot-num-classes-symint](issues/one-hot-num-classes-symint.md) | long_t5 | `one_hot` schema takes `int num_classes` → symbolic class count specialized |
| [native-eager-router-traced-by-dynamo](issues/native-eager-router-traced-by-dynamo.md) | deberta (CUDA) | Dynamo compiles `torch/_native/registry.py:eager_router` as a user frame |
| [dynamo-unknown-stride-guard-source](issues/dynamo-unknown-stride-guard-source.md) | encodec | recompile reason says "unknown source … please file a bug" (diagnostics only) |

**Existing drafts:** `graph-break-in-loop` (24 models), `check-tensor-all-item` (15), `ctc-loss-graph-break`,
`requires-grad-intermediate`, `genexpr-after-graph-break`, `sdpa-is-causal-symbool`, `nn-buffer-reassign`,
`conv-backward-stride-guard`, `cpp-acc-helper-reduction-guard`, `tolist-unbacked-length`.

### transformers (huggingface/transformers): 35 · drafts in `issues/hf/`

Model-code patterns Dynamo can't trace. Most drafts include a repro that checks the proposed fix gives one
graph (and the same output where shown).

| issue | models | pattern |
|---|---|---|
| [qwen-vl-grid-thw-data-dependent](issues/hf/qwen-vl-grid-thw-data-dependent.md) | qwen2_vl, qwen2_5_vl, qwen3_vl, qwen3_8, hy_v4_vl, lfm3_vl, inkling2_vlm, dots3_note_vlm | `grid_thw` `.tolist()` / `.item()` in vision tower + mRoPE (20–35 breaks/forward) |
| [vlm-split-image-features-tolist](issues/hf/vlm-split-image-features-tolist.md) | 7 VLMs | split image features by `.tolist()` only to re-concat |
| [vision-attn-split-lengths-tolist](issues/hf/vision-attn-split-lengths-tolist.md) | 5 VLMs | per-image `torch.split(lengths.tolist())` in vision attention |
| [placeholder-mask-bool-index-numel](issues/hf/placeholder-mask-bool-index-numel.md) | inkling2, dots3_note, omniasr | `inputs_embeds[mask].numel()` check |
| [encodec-conv-params-buffers](issues/hf/encodec-conv-params-buffers.md) | encodec, mimi, wavtokenizer | conv hyper-params as tensor buffers (20–35 breaks/forward) |
| [specaugment-numpy-mask](issues/hf/specaugment-numpy-mask.md), [ctc-labels-max-check](issues/hf/ctc-labels-max-check.md) | wav2vec2, hubert, wavlm | numpy SpecAugment; `labels.max()` check (train) |
| [sam3-spatial-shape-tensor-arange](issues/hf/sam3-spatial-shape-tensor-arange.md) | sam3 (new model) | `torch.arange(0, <tensor>)` in every DETR decoder layer |
| [rope-standardize-in-forward-typeddict-isinstance](issues/hf/rope-standardize-in-forward-typeddict-isinstance.md) | phimoe (dynamic) | config mutated in forward; hub validation `isinstance(…, TypedDict)` |
| [deberta-att-span-item](issues/hf/deberta-att-span-item.md) | deberta | tensor `att_span` → `.item()` per layer (12 → 1 graph with fix) |
| [problem-type-config-mutation](issues/hf/problem-type-config-mutation.md) | classification heads (vit) | 2nd training step always recompiles |
| [nemotron-h-pad-size-guard](issues/hf/nemotron-h-pad-size-guard.md) | nemotron_h (+ same code in bamba, granitemoehybrid, zamba2, falcon_h1) | `if pad_size > 0` → recompile |
| [looped-model-cache-deepcopy](issues/hf/looped-model-cache-deepcopy.md) | inkling2_mini, inkling2_mini_vlm, dory (new models) | looped models `copy.deepcopy` cache layers inside forward (train, eval with cache): 14–19 breaks |
| [jev-decision-bool-mask-select](issues/hf/jev-decision-bool-mask-select.md) | jev (new model) | decision head returns `logits[bool_mask]` → `nonzero` break at the end of forward |
| [placeholder-check-fstring-symint](issues/hf/placeholder-check-fstring-symint.md) | llava, gemma3_vlm, paligemma, omniasr (41 call sites) | error f-string built eagerly → SymInt feature count specialized |
| [detr-mask-interpolate-batch-as-channels](issues/hf/detr-mask-interpolate-batch-as-channels.md) | rt_detr (CUDA; same line in 15 DETR-family models) | `interpolate(pixel_mask[None])` → batch treated as channels → `batch <= 3` guard |
| [funnel-pos-list-range-seq-len](issues/hf/funnel-pos-list-range-seq-len.md) | funnel | `pos = list(range(seq_len))` → every seq length recompiles |
| [python-loop-over-seq-len](issues/hf/python-loop-over-seq-len.md) | minimax, rwkv | Python loop over a seq-len-derived `range` → recompile per length |
| [levit-attention-bias-cache](issues/hf/levit-attention-bias-cache.md) | levit | eval bias cache dict filled on first call → 2nd call always recompiles |
| others | one model each | [longformer/led `.item()`](issues/hf/longformer-global-attn-item.md), [xlm assert](issues/hf/xlm-lengths-assert-item.md), [fsmt mask](issues/hf/fsmt-padding-mask-any.md), [bigbird attn switch](issues/hf/bigbird-attn-type-switch.md), [switch_transformers expert loop](issues/hf/switch-router-module-to.md), [smolvlm](issues/hf/idefics3-image-nonzero.md), [granite_docling](issues/hf/granite-docling-tile-mask-index.md), [mistral3](issues/hf/mistral3-image-sizes-tolist.md), [llava_next](issues/hf/llava-next-anyres-image-size-tolist.md), [apertus1p5](issues/hf/apertus1p5-image-sizes-tolist.md), [phimoe LongRoPE](issues/hf/phimoe-longrope-seq-len-branch.md), [dynamic RoPE](issues/hf/dynamic-rope-data-dependent.md), [reformer](issues/hf/reformer-lsh.md), [output_capturing](issues/hf/output-capturing-reconstruction.md) (0 occurrences on current main), [layerdrop](issues/hf/layerdrop-rand-branch.md) (now only hubert-family), [loss warning](issues/hf/loss-fn-warning-once.md) |

`layerdrop-rand-branch` is **fixed on transformers main** for bart, mbart, marian, m2m_100, blenderbot,
pegasus, pegasus_x, opt, whisper, xglm, biogpt, fsmt, led (train: 9 breaks → 0); still present in
hubert-family models.

### Other repos: 2 · Zyphra draft in `issues/hf/` (HF Hub remote code), harness draft in `issues/harness/`

| issue | repo | what |
|---|---|---|
| [zonos3-moe-unique-expert-loop](issues/hf/zonos3-moe-unique-expert-loop.md) | Zyphra/ZONOS3-Mini-hf (remote code) | MoE loop over `torch.unique(...).tolist()` + `nonzero()`, 17 breaks/step. Repro with a fix |
| [harness-dynamic-loss-selection](issues/harness/harness-dynamic-loss-selection.md) | this repo | `dynamic.py` `_loss()` picks a non-scalar loss / int tensor → dac, encodec, mimi dynamic-train results are harness errors |

## Models added in this sweep

| requested | result |
|---|---|
| inkling2 small | already tracked (`inkling2`, `inkling2_vlm`) |
| Inkling 2 Mini → `inkling2_mini`, `inkling2_mini_vlm` | added (CPU + CUDA). 19 breaks with cache / train → `looped-model-cache-deepcopy`; the VLM also hits the existing VLM issues. CUDA dynamic eval crashes (`inductor-merge-loops-minmax-not-comparable`) |

### Nov 22: every model added to transformers main from Nov 1 to Nov 21

19 model directories were added upstream (checked against main `2f1bc5e321`). 12 were already tracked
(lfm3_vl, iquestcoder, internlm3, dots3_note, inkling2, granite_docling, omniasr, omniasr_ctc, apertus1p5,
wavtokenizer, hy_v4_vl, inkling2_loop → inkling2_mini). The other 7 were added and run on CPU and CUDA (B300):

| model | config from | result |
|---|---|---|
| lfm3 | `LiquidAI/LFM3-1.7B` | 1 graph everywhere. Dynamic CPU train recompiles: `conv-backward-stride-guard` (existing) |
| lara_mini | `LaraMiniConfig` (gated `translated/Lara-4-Mini-Base`) | 1 graph, no recompiles (CUDA). CPU: needs BLAS |
| dory | `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16` | 15 graphs / 14 breaks with cache and in train → `looped-model-cache-deepcopy` (extended to Dory) |
| exaone | `ExaoneConfig` (gated `LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct`) | 1 graph, no recompiles |
| sarvam_mla | `sarvamai/sarvam-105b` | 1 graph, no recompiles |
| sarvam_moe | `sarvamai/sarvam-30b` | 1 graph, no recompiles |
| jev | `JevConfig` (`autotrust/JEV-9B`) | 1 break at the end of forward → new issue `jev-decision-bool-mask-select`. CPU: needs BLAS |
| Qwen3.8-27B → `qwen3_8` | added. CUDA: 29 breaks/scenario, all existing VLM issues; 13 recompiles with dynamic shapes. CPU: not runnable (needs CPU BLAS) |
| SAM3 → `sam3` | added (Sam3Config defaults, gated hf_id). 9 (CPU) / 12 (CUDA) breaks → new issue `sam3-spatial-shape-tensor-arange` |
| ZONOS3-Mini → `zonos3_mini` | added with `trust_remote_code` (code reviewed, revision e39d30d). 17 breaks → new issue for Zyphra |
| Hy4-flash | **not tested**: needs unmerged transformers PR #50903 (declined) |
| SenseNova-Loop-9B | **not tested**: gated, no HF token |
| MiniMax-H3-Image, LongCat-Video-Avatar-1.6, FastFrames-L | **not tested**: diffusion image/video generators, not transformers models |

### Nov 23: the 100 most-downloaded Hub models

Source: `huggingface.co/api/models?sort=downloads&limit=100` on 2026-11-23 (downloads over the last 30 days).

| | count | |
|---|---|---|
| already tracked | 12 | gpt2, bert, t5, clip, qwen3, vit, xlm_roberta, qwen3_8, roberta, gpt_oss, distilbert, opt |
| added | 48 | one entry per checkpoint (same architecture included: different configs, Auto classes and heads) |
| quantized, added | 3 | `qwen3_8_fp8`, `deepseek_v32_fp8` (fine-grained FP8), `gpt_oss_mxfp4` (real MXFP4 checkpoint, 2 layers) |
| GGUF base model added | 1 | `qwen3_6_moe` (Qwen/Qwen3.6-35B-A3B, base of #79) |
| not transformers models | 28 | diffusion / ComfyUI checkpoints, CoreML, CTranslate2, chronos, pyannote, coqui, ultralytics, ... |
| not run | 9 | timm (needs torchvision); GGUF/ONNX/FP8 copies of covered models; Qwen3.8-Flash-Next (`qwen4_exp`) and DeepSeek-V4.1-Flash (`deepseek_v41`) need a newer transformers; Ornith-1.5 has no config |

GGUF repos are not run: transformers dequantizes GGUF on load, so the compiled graph is the base model's.

**Clean:** 30 of the 48 have 1 graph everywhere and no recompiles: every embedding / reranker model (BGE, E5,
MiniLM, MPNet, nomic, gte, snowflake, granite-embedding, Qwen3-Embedding, contriever), the dense LLMs (Qwen2.5,
Qwen3-8B/14B, Llama-3.1, Yi, DeepSeek-V3.2), CLIP ×2, mdeberta, roberta-large. 5 more have no breaks but recompile
under dynamic shapes: whisper ×3 (2 each), qwen3_4b_instruct and dinov2_base (1 each, train). The other 13 have
graph breaks or crashes.

**New root causes (9 drafts):**

| id | repo | kind | models |
|---|---|---|---|
| `inductor-cuda-int-sqrt-pow` | PyTorch | **wrong value** → device-side assert | gemma4, gemma4_moe, gemma4_e4b (dynamic, CUDA). `int(s ** 0.5)` is off by one for some perfect squares on CUDA (9, 7056, ...). Tested repro |
| `inductor-user-triton-constexpr-global` | PyTorch | crash | gpt_oss_mxfp4 (any compile). Runtime-defined Triton kernel loses its `constexpr_function` helpers. Tested repro |
| `aot-synthetic-base-symbolic-nbytes` | PyTorch | crash | gpt_oss_mxfp4 (dynamic). AOTAutograd builds a synthetic base for aliased, mutated inputs with `set_(storage)`, which reads a symbolic `nbytes`. Tested repro (CPU, `aot_eager`) |
| `inductor-cantsplit-square-floordiv` | PyTorch | crash | unlimited_ocr (dynamic CUDA train). `(a**2)//a` not simplified in the LayerNorm-backward mix-order reduction. Tested repro (conv → LayerNorm → Linear); workaround `triton.mix_order_reduction=False` |
| `dynamo-triton-kernels-wrapper-args` | PyTorch | 19 breaks | gpt_oss_mxfp4: Triton kernels taking triton_kernels `Tensor` wrappers |
| `fp8-deepgemm-fallback-compile` | transformers | crash | qwen3_8_fp8, deepseek_v32_fp8: DeepGEMM → Triton fallback works in eager only. Tested repro (CPU, no HF) |
| `glm5-dsa-pooled-keep-mask` | transformers | break | glm5_next: boolean mask in the LM's sparse-attention indexer |
| `gemma4-vision-pooler-mask-index` | transformers | break | gemma4 ×3 |
| `clap-is-longer-where` | transformers | break | clap |

Existing issues extended to new files: `vlm-split-image-features-tolist` and `placeholder-mask-bool-index-numel`
(gemma4), `vlm-split-image-features-tolist` and `vision-attn-split-lengths-tolist` (glm5_next), and the Qwen-VL
entries (qwen3_5_moe).

**Accuracy:** no mismatches except `gemma4_moe` train on CUDA (logits RMSE vs fp32 0.070 compiled vs 0.016 eager;
eval is accurate, dense gemma4 train is accurate). **Not a compiler bug:** a bf16 MoE routing flip. In fp32, compiled
matches eager to 5.5e-7 with identical router choices; in bf16, compiled picks a different expert for 2.3% of
tokens in layer 1 (1 of 44), which is enough for the logits gap. CPU eager bf16 is itself 0.059 RMSE from the fp32
reference for this model, so its tolerance check is sensitive to any rounding change.

**Harness fix found on the way:** `deepseek_v32` ran with both shrunk layers dense, because its config sets
`mlp_layer_types` and `first_k_dense_replace` alone didn't change it. Fixed and rerun; no other entry is affected
(checked every config with `mlp_layer_types`).

## Caveats

- **Counts are lower bounds.** Models run with 2–4 layers (per-layer breaks scale with depth), one input
  per scenario (paths it doesn't reach aren't counted), and after "Failed to handle graph break
  gracefully" Dynamo runs the rest of the frame eagerly, so breaks further inside it are not seen.
- New drafts from the dynamic-recompile triage were written from the recorded guards plus reading the
  torch / transformers source at the run commits. Their repros are sketches that haven't been run yet
  (marked in each draft).
- `qwen3_next` CPU static results are from the earlier build `9933f27` (0 breaks; it can't run on this CPU
  build). `rwkv` was run with the model built on GPU and moved to each device (its init needs LAPACK).

## Dynamic-shape recompiles

Every run that recompiles (74 CPU, 62 CUDA) was checked against its size-related guards. Each recompile
falls in one of these groups:

- **Its own root cause, with a draft:** conv-backward-stride-guard, cpp-acc-helper-reduction-guard
  (dinov2, exaone4, qwen3_moe, whisper, hubert), significant-strides-equal-guard,
  mamba-associative-scan-seq-guard (including the `vmap` batch-size guard in the same scan),
  nemotron-h-pad-size-guard, problem-type-config-mutation, and the 10 new drafts above.
- **Knock-on of an existing draft:**
  - `ir.py:clamp` and the conv-model `get_kernel_metadata` hits only specialize sizes 40/56, *after*
    the conv-backward recompile.
  - In the VLMs, `compare_sizes`, `broadcast_symbolic_shapes`, the `get_kernel_metadata` hits and the
    `len(kwargs)` / duck-sizing guards (`image_sizes.size(1) == batch`) are in resume frames after the
    `.tolist()` breaks, where sizes from `.tolist()` are Python ints.
  - big_bird (attention-type switch on the first call), switch_transformers, sam3, encodec/mimi
    `_pad1d`, omniasr, phimoe, inkling2_mini (cache deepcopy).
- **Not a dynamic-shape problem:** the `transformers/utils/generic.py` / `deprecation.py` wrappers
  recompile once per module class (`___check_type_id(args[0])`, `len(kwargs)`), during the *first* size.
  These are frames created by graph breaks, and they disappear with the breaks.

inkling2_mini train specializes in `codegen/common.py:doprint` and `ir.py:clamp` on CPU only. The CUDA run,
where the conv-backward bug doesn't apply, has neither and still recompiles once, from the duck-sizing guard
in `looped-model-cache-deepcopy`. So those two CPU sites are conv-backward knock-ons.

## Unbacked dynamic shapes (partial, stopped)

`python -m graph_breaks.dynamic --unbacked`: the same `dynamic=True` + Inductor run as above, with
`capture_scalar_outputs` and `capture_dynamic_output_shape_ops` on, so `.item()` / `.tolist()` / `nonzero()`
values are traced as unbacked SymInts instead of breaking the graph. Started Nov 24 over every model, CUDA on
B300 and CPU, and **stopped on request** partway (alphabetical order):

| | models with results | runs | ok | errors |
|---|---|---|---|---|
| CUDA (B300) | 78 of 209 (`albert` … `granitemoe`) | 150 | 114 | 36 |
| CPU | 21 of 206 (`albert` … `clip`) | 38 | 36 | 2 |

Five models only have their `eval_forward` run (CPU: beit, bloom, clap, clip; CUDA: deberta_v2, fsmt,
funnel, granite_docling, granitemoe). Results: `results_dynamic_unbacked[_cuda]/`, summaries
[`DYNAMIC_UNBACKED_SUMMARY.md`](DYNAMIC_UNBACKED_SUMMARY.md) / [`DYNAMIC_UNBACKED_CUDA_SUMMARY.md`](DYNAMIC_UNBACKED_CUDA_SUMMARY.md),
and `unbacked` rows in [`COVERAGE.md`](COVERAGE.md).

- **Graph breaks:** the same count as the backed dynamic run in every model that ran.
- **Recompiles:** fewer in 4 models, where the guard came from a value now traced as unbacked. clap 1 → 0
  (eval) and 2 → 0 (train), gemma3_vlm train 4 → 3, granite_docling eval 6 → 3. None went up.
- **Errors in models whose backed run passes** (the other 11 errors are the same as in the backed run):

| cause | models | status |
|---|---|---|
| [dynamo-pending-unbacked-symbol-not-found](issues/dynamo-pending-unbacked-symbol-not-found.md) (PyTorch) | deberta, encodec (CUDA) | new draft, internal Dynamo error. Both keep sizes in tensors and use them as ints. Not reduced |
| [inductor-conv-unbacked-spatial-hint](issues/inductor-conv-unbacked-spatial-hint.md) (PyTorch) | apertus1p5_vlm (CPU + CUDA) | new draft: convolution lowering guards on `256*u4*u5` (per-image crop size read from a tensor). Not reduced |
| `CUDA error: unspecified launch failure` | deepseek_v2, deepseek_v3, deepseek_v32, dots3_note, ernie4_5_moe, glm4_moe, granitemoe | all MoE, B300 only, unbacked only (backed passes on B300). **Not triaged:** not rerun on H100, so unknown whether B300-specific |
| NVRTC: no sm_103 in this CUDA 12.8 build | dots3_note_vlm, gemma3n, glm5_next | environment, same as the main sweep; not rerun on H100 |

## Accuracy check (stopped, incomplete)

`graph_breaks/accuracy.py` compares eager, compiled (Inductor) and an fp64 reference for every output and
gradient (`torch._dynamo.utils.same` with `fp64_ref`, as in PyTorch's benchmarks). It was run over all 157
models on CPU and the 55 recent + popular models on H100, then **stopped on request** before the corrected
rerun finished.

- Stored results (`results_accuracy/`, `results_accuracy_cuda/`) come from the first pass, before three fixes to
  the checker: a fp32-typed reference made `same` skip its fp64 test; MoE / fp32-cast models had no reference (now
  an fp32 fallback); SpecAugment masks (numpy) and drop-connect were still random in train mode. Most first-pass
  "mismatches" are those artifacts. **No precision issue has been confirmed or filed.**
- `results_accuracy_cuda_b300/`: an earlier CUDA attempt on B300, not usable (CUDA 12.8 doesn't support sm_103).
- Real findings it did produce, added to existing drafts: phi-2 train on CPU hits
  `inductor-cpp-half-reduction-redeclare`; the Inkling `merge_loops` crash also happens with default
  `torch.compile` on CUDA.
- To finish: rerun the mismatched runs with the fixed checker (`~/gb-jobs/run_accre_cpu.sbatch`,
  `~/gb-jobs/run_accre_cuda.sbatch`, lists in `~/gb-jobs/acc_rerun_*.txt`), then
  `python -m graph_breaks.accuracy --summarize` → `ACCURACY.md`.

## Inkling deep dive (Nov 24)

The five Inkling models (`inkling`, `inkling2`, `inkling2_vlm`, `inkling2_mini`, `inkling2_mini_vlm`) were run
through every check the harness has, on CPU (`torch-snap-57f02f8`) and CUDA (B300, `torch-dev` at `f29b6d5`):
static graph breaks, dynamic and unbacked shapes, accuracy (fixed checker), and the new compile paths in
[`graph_breaks/modes.py`](graph_breaks/modes.py) and [`graph_breaks/distributed.py`](graph_breaks/distributed.py).
Results: [`MODES.md`](MODES.md) (all modes + DDP/FSDP2), [`ACCURACY.md`](ACCURACY.md), and the `*_FULLDEPTH*`,
`*_IDENTITYFIX*` and `*_EXPERTS_EAGER*` summaries.

New checks (`python -m graph_breaks.modes`): `generate()` with a dynamic cache and with a static cache,
`torch.export` (strict / non-strict) + AOTInductor, `mode="reduce-overhead"` (CUDA graphs), `max-autotune`,
`freezing`, bf16 and fp16 autocast (+ GradScaler steps), 3-step AdamW training with a compiled optimizer, compiled
autograd, eager-vs-compiled perf. `python -m graph_breaks.distributed` (torchrun, 2 GPUs): DDP and FSDP2.
Switches: `GB_FULL_DEPTH=1` (real layer count / pattern, reduced width), `GB_PRETRAINED=1` (real checkpoint),
`GB_PATCH=identity` (apply the proposed merge-loops fix), `GB_EXPERTS_IMPL=eager`, `GB_RESULTS_SUFFIX`.

Root causes for Inkling: 8 PyTorch, 8 transformers (one transformers entry, `moe-eager-experts-loop`, is
`not-filing`: it only appears with the eager-experts workaround).

| PyTorch | effect on Inkling | state |
|---|---|---|
| [inductor-merge-loops-minmax-not-comparable](issues/inductor-merge-loops-minmax-not-comparable.md) | every CUDA inference path of the 4 Inkling-2 models crashes | root cause (`Identity` has no `eval`), 10-line repro, **fix tested**: with it all 4 compile and match eager in every mode (eval ~2.6–3.6x vs eager) |
| [symbolic-shapes-replacement-float-inverse-bound](issues/symbolic-shapes-replacement-float-inverse-bound.md) | unbacked: both VLMs crash | root cause (float division in `_set_replacement`), 15-line repro |
| [duck-sizing-specializes-unrelated-dims](issues/duck-sizing-specializes-unrelated-dims.md) | `generate()` recompiles decode graphs per batch size | root cause, 20-line repro |
| [grouped-mm-sm100-kernel-without-arch-specific-build](issues/grouped-mm-sm100-kernel-without-arch-specific-build.md) | bf16 MoE (inkling) crashes on a `TORCH_CUDA_ARCH_LIST=10.3` build | repro; device-only arch gate |
| [sdpa-is-causal-symbool](issues/sdpa-is-causal-symbool.md) | export / AOTI of both VLMs (the workaround guards on an unbacked length) | export section added |
| conv-backward-stride-guard, sdpa-efficient-backward-bias-align-guard, graph-break-in-loop | recompiles / break multiplication | unchanged |

transformers: the 5 VLM / looped-model break drafts, plus
[inkling-static-cache-sliding-position-bias](issues/hf/inkling-static-cache-sliding-position-bias.md) (static-cache
`generate()` fails in eager, all 5 models), [moe-grouped-mm-fallback-cudagraph-unsafe](issues/hf/moe-grouped-mm-fallback-cudagraph-unsafe.md)
(`reduce-overhead` crashes, 4 Inkling-2 models) and [position-bias-mask-fp16-overflow](issues/hf/position-bias-mask-fp16-overflow.md)
(fp16 autocast, inkling, eager).

Other results:
- **Accuracy:** no wrong-value bug. Every accuracy run passes, including full-depth inkling against fp64 (66 layers,
  1,386 gradients). The earlier inkling CPU "mismatch" (RMSE 3.4e-4) was first-pass checker noise.
- **Full depth** finds the same root causes as the reduced configs.
- **Perf** (B300, reduced width, with the merge-loops fix): Inkling-2 eval 2.6–3.6x, train 2.4–3.0x;
  Inkling-2-Mini train ~1.0x because `looped-model-cache-deepcopy` makes the text model run eagerly.
- **DDP:** correct for all 5. Its breaks (`Logger.set_runtime_stats_and_log`, `_inside_ddp_forward`) are by design.
  **FSDP2:** the per-unit `_dynamo.disable` break is by design in this build; inkling needs its fp32 submodules as
  separate units (FSDP2 uniform-dtype rule).
- **Attention fusion:** Inkling-2's token attention (sinks + position bias + mask + GQA) matches no Inductor SDPA
  pattern; the 2-per-layer `fuse_attention` rewrites are the cross-layer attention residual (`_sfdp_pattern_2`).
  An "SDPA + sinks" rewrite (`out * sigmoid(lse - sink)`) would cover it.

Open for Inkling: the real bf16 grouped-GEMM path and real Inkling-2-Mini weights (NVFP4) need a
`TORCH_CUDA_ARCH_LIST=10.3a` build; MoE ran with `experts_implementation="eager"` meanwhile. Real Inkling (952B) and
Inkling-2-Small (612B) weights don't fit one GPU.

The 10-popular-model extension of these checks was stopped on request. Only its full-depth runs executed (the
baseline tasks failed at launch because of a job-runner bug, since fixed); those results are in the
`*_fulldepth` folders but are not triaged.

## Still to do

1. **5 CPU dynamic runs not completed** (cancelled for compile time; they ran 1–3 h without finishing):
   swin eval/train, swinv2 eval/train, funnel train. List: `~/gb-jobs/cpu_dyn_todo_remaining.txt`.
   Rerun one-per-task with a time cap, no requeue:
   `sed -i 's/cpu_dyn_todo.txt/cpu_dyn_todo_remaining.txt/; s/--array=0-8/--array=0-4/' ~/gb-jobs/run_cpu_dyn_redo.sbatch && sbatch ~/gb-jobs/run_cpu_dyn_redo.sbatch`
2. Run the repro sketches in the 11 drafts that don't have a tested repro yet (the 9 from the recompile triage,
   jev, and the Dory part of `looped-model-cache-deepcopy`), and save them to `issues/repros/` / `issues/hf/repros/`.
3. Finish the accuracy check (see above), or drop it.
4. Review the drafts, file them, and set `issue:` / `status: open` in `known_issues.yaml`.
5. Nov 23 batch: `unlimited_ocr` CPU dynamic train hit the 2 h CPU compile cap (same CPU-only slowdown as
   above; its CPU accuracy runs all pass, without an fp64 reference because of an fp64 grouped-mm limit). The 5 new
   crash / wrong-value drafts all have tested repros now; the 4 graph-break drafts point at the model code. File
   them with item 4.
6. Unbacked sweep (stopped): resume with `sbatch ~/gb-jobs/unbacked/cuda.sbatch` / `cpu.sbatch` (skip models that
   already have both scenarios), rerun the MoE launch failures and NVRTC models on H100
   (`--partition=hopper-dev --gres=gpu:h100:1 --export=LIST=...`), and reduce the two new drafts.

## Repo changes (uncommitted)

- `graph_breaks/check.py`: `--results-dir`; CUDA results to `results_cuda/`; per-model `trust_remote_code`,
  `init_weights`; dict-aware config overrides; `sam3` / `zonos3_rows` inputs.
- `graph_breaks/dynamic.py`: `--results-dir`; CUDA results to `results_dynamic_cuda/`.
- `graph_breaks/summarize.py`: `--device cuda` → `SUMMARY_CUDA.md`; remote-code path normalization.
- `graph_breaks/coverage.py` (new) → `COVERAGE.md`: every break on every device/check with its issue.
- `graph_breaks/affected.py` (new): writes the "Affected models in this sweep" table into each draft (graph
  breaks, crashes and dynamic-shape recompiles per model, check and device).
- `models.yaml`: + qwen3_8, sam3, zonos3_mini, inkling2_mini(_vlm), lfm3, lara_mini, dory, exaone, sarvam_mla,
  sarvam_moe, jev. `known_issues.yaml`: 60 root causes.
- `graph_breaks/check.py`: `jev` inputs. `graph_breaks/dynamic.py`: `_loss()` handles a bare tensor output.
- `graph_breaks/accuracy.py` (new): eager vs compiled vs fp64 accuracy check (results in `results_accuracy*/`).
- Nov 23: `check.py` adds `clap` and `gemma4` inputs, no `labels` for headless (`AutoModel`) encoders,
  per-sequence labels for `*ForSequenceClassification`, `decoder_input_ids` for encoder-decoder VLMs, and per-model
  `quantize: true` / `pretrained: true` (also used by `accuracy.py`). `summarize.py` normalizes `kernels` hub
  paths. `coverage.py` maps the new crash errors. `models.yaml`: + 52 models.
- Nov 24: `coverage.py` / `affected.py` also read `results_dynamic_unbacked[_cuda]/` (`unbacked` rows in
  `COVERAGE.md`, two more columns in the drafts' affected-models tables). Job scripts in `~/gb-jobs/unbacked/`.
- Nov 24 (Inkling deep dive): `graph_breaks/modes.py` (new) → `MODES.md`, `graph_breaks/distributed.py` (new),
  `graph_breaks/__init__.py` (`GB_PATCH`), `GB_FULL_DEPTH` / `GB_PRETRAINED` / `GB_EXPERTS_IMPL` in `check.py` and
  `accuracy.py`, `--results-dir` for `accuracy.py`, `--results-dir` / `--out` for `summarize.py`, `--summary-file`
  for `dynamic.py`, `ACCURACY.md` covers every `results_accuracy*` folder. Job scripts in `~/gb-jobs/inkling/`.
- Drafts split by repo: `issues/` (PyTorch) + `issues/repros/`, `issues/hf/` (transformers, HF Hub) +
  `issues/hf/repros/`, `issues/harness/`. Results: `results*/`, summaries.
