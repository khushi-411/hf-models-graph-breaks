# Graph-break sweep: results (2026-11-18/19)

Every model in `models.yaml` was run through both checks on **CPU and CUDA**, and every graph break
found was matched to a root cause with an issue draft: PyTorch drafts in [`issues/`](issues/), transformers and
HF Hub drafts in [`issues/hf/`](issues/hf/), and the one harness bug in [`issues/harness/`](issues/harness/).
Dynamic-shape recompiles were triaged too (see [Dynamic-shape recompiles](#dynamic-shape-recompiles)). Per-model detail is in
[`COVERAGE.md`](COVERAGE.md) (all checks and devices) and in
[`SUMMARY.md`](SUMMARY.md) / [`SUMMARY_CUDA.md`](SUMMARY_CUDA.md) /
[`DYNAMIC_SUMMARY.md`](DYNAMIC_SUMMARY.md) / [`DYNAMIC_CUDA_SUMMARY.md`](DYNAMIC_CUDA_SUMMARY.md).

**Status:** 157 models · 0 unmatched graph breaks · 0 unexplained errors · 60 root causes, 60 issue drafts
(23 PyTorch, 36 transformers/HF Hub, 1 harness). Nothing is filed upstream yet.

## Setup

| | |
|---|---|
| torch | `2.16.0a0+git57f02f8` (source build). CUDA: rebuilt in `torch-dev` with sm_90 + sm_100, cuDNN 9.28, Triton 3.9.0. CPU: frozen copy of the same commit (`torch-snap-57f02f8`), conda g++ 13.4 |
| transformers | `5.24.0.dev0`, checkout `b7fe69cb19`. inkling2_mini(_vlm) and the Nov 22 batch (below): `046bdb8cc2` (Nov 21), needed for those models |
| Hardware | CUDA: NVIDIA B200 (`hopper-b200`). The Nov 22 runs (Nov 22 batch + inkling2_mini CUDA) are on B300 (`hopper-b300`, sm_103): the B200 partition is gone. Caveat: the `torch-dev` build ships CUDA 12.8, which doesn't support sm_103. Those runs completed, but other workloads on B300 hit NVRTC (`invalid --gpu-architecture`) and MoE kernel crashes, so treat B300 results with care; H100 (`hopper-dev`, sm_90) is fully supported. CPU: Xeon 8488C / 8124M (`hopper-cpu`) |
| Static check | `python -m graph_breaks.check --model X --device cpu\|cuda`: `torch._dynamo.explain`, config-only model with reduced layers, random weights, batch 2 / seq 32 |
| Dynamic check | `python -m graph_breaks.dynamic --model X --device cpu\|cuda`: `torch.compile(backend="inductor", dynamic=True)` over (batch, seq) = (2, 32), (3, 40), (4, 56) |

## Headline numbers

| check | device | model/scenario runs | errors | unique breaks* | total breaks |
|---|---|---|---|---|---|
| static | CPU | 414 | 8 | 460 | 1,239 |
| static | CUDA | 414 | 0 | 514 | 1,301 |
| dynamic | CPU | 308 | 25 | 326 | – |
| dynamic | CUDA | 308 | 15 | 333 | – |

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

## Still to do

1. **5 CPU dynamic runs not completed** (cancelled for compile time; they ran 1–3 h without finishing):
   swin eval/train, swinv2 eval/train, funnel train. List: `~/gb-jobs/cpu_dyn_todo_remaining.txt`.
   Rerun one-per-task with a time cap, no requeue:
   `sed -i 's/cpu_dyn_todo.txt/cpu_dyn_todo_remaining.txt/; s/--array=0-8/--array=0-4/' ~/gb-jobs/run_cpu_dyn_redo.sbatch && sbatch ~/gb-jobs/run_cpu_dyn_redo.sbatch`
2. Run the repro sketches in the 11 drafts that don't have a tested repro yet (the 9 from the recompile triage,
   jev, and the Dory part of `looped-model-cache-deepcopy`), and save them to `issues/repros/` / `issues/hf/repros/`.
3. Finish the accuracy check (see above), or drop it.
4. Review the drafts, file them, and set `issue:` / `status: open` in `known_issues.yaml`.

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
- Drafts split by repo: `issues/` (PyTorch) + `issues/repros/`, `issues/hf/` (transformers, HF Hub) +
  `issues/hf/repros/`, `issues/harness/`. Results: `results*/`, summaries.
