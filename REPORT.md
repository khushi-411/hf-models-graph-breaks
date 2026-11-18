# Graph-break sweep: results (2026-11-18/19)

Every model in `models.yaml` was run through both checks on **CPU and CUDA**, and every graph break
found was matched to a root cause with an issue draft in `issues/`. Per-model detail is in
[`COVERAGE.md`](COVERAGE.md) (all checks and devices) and in
[`SUMMARY.md`](SUMMARY.md) / [`SUMMARY_CUDA.md`](SUMMARY_CUDA.md) /
[`DYNAMIC_SUMMARY.md`](DYNAMIC_SUMMARY.md) / [`DYNAMIC_CUDA_SUMMARY.md`](DYNAMIC_CUDA_SUMMARY.md).

**Status:** 148 models · 0 unmatched graph breaks · 0 unexplained errors · 49 root causes, 49 issue drafts.
Nothing is committed or filed upstream yet.

## Setup

| | |
|---|---|
| torch | `2.16.0a0+git57f02f8` (source build). CUDA: rebuilt in `torch-dev` with sm_90 + sm_100, cuDNN 9.28, Triton 3.9.0. CPU: frozen copy of the same commit (`torch-snap-57f02f8`), conda g++ 13.4 |
| transformers | `5.24.0.dev0`, checkout `b7fe69cb19` |
| Hardware | CUDA: NVIDIA B200 (`hopper-b200`). CPU: Xeon 8488C / 8124M (`hopper-cpu`) |
| Static check | `python -m graph_breaks.check --model X --device cpu\|cuda`: `torch._dynamo.explain`, config-only model with reduced layers, random weights, batch 2 / seq 32 |
| Dynamic check | `python -m graph_breaks.dynamic --model X --device cpu\|cuda`: `torch.compile(backend="inductor", dynamic=True)` over (batch, seq) = (2, 32), (3, 40), (4, 56) |

## Headline numbers

| check | device | model/scenario runs | errors | unique breaks* | total breaks |
|---|---|---|---|---|---|
| static | CPU | 388 | 3 | 418 | 1,064 |
| static | CUDA | 388 | 0 | 470 | 1,126 |
| dynamic | CPU | 290 | 21 | 300 | – |
| dynamic | CUDA | 290 | 13 | 315 | – |

\* distinct break locations per model/scenario, summed over runs. The same root cause counts once per
model and scenario it appears in, so these collapse to ~110 code locations and 49 root causes.

- Static CPU and CUDA have the **same root causes**. Only counts differ, e.g. deberta 12 vs 14 breaks,
  and ctrl breaks on CPU only (`ctrl-buffer-reassign`).
- All new **crashes** came from the dynamic-shape (Inductor) runs, which `check.py` alone never exercises.
- Errors are explained in `COVERAGE.md`: PyTorch crash bugs (15), harness problems (10), build limits (7,
  no CPU LAPACK/BLAS: qwen3_next and qwen3_8 on CPU), and 5 runs cancelled for compile time (below).

## Root causes by repo

### PyTorch (pytorch/pytorch): 17

**Hard crashes (compile fails), new in this sweep:**

| issue | models | what |
|---|---|---|
| [inductor-cpp-half-reduction-redeclare](issues/inductor-cpp-half-reduction-redeclare.md) | pythia (train, CPU) | fp16/bf16 reduction with two outputs → generated C++ declares a variable twice. Static and dynamic. Standalone repro |
| [dynamo-guard-python-mod-value-range](issues/dynamo-guard-python-mod-value-range.md) | internvl (CPU + CUDA) | `size % 0.5` → `ValueRangeError: Invalid ranges [0:-0.5]` in guard building. Standalone repro |
| [inductor-merge-loops-minmax-not-comparable](issues/inductor-merge-loops-minmax-not-comparable.md) | inkling2, inkling2_vlm (eval, CUDA) | `merge_loops` → `sympy_subs` → `'-((0))' is not comparable`. Model-level repro |
| [inductor-bounds-truncdiv-nan](issues/inductor-bounds-truncdiv-nan.md) (existing) | longformer, led | `Invalid NaN comparison` in value-range analysis |

**Recompiles / specialization / compile time (not graph breaks), new:**

| issue | models | what |
|---|---|---|
| [mamba-associative-scan-seq-guard](issues/mamba-associative-scan-seq-guard.md) | mamba, jamba, falcon_mamba | `associative_scan`: generic mode guards on `(seq//8)%2` (CPU); pointwise + autograd pins batch and seq (CUDA). Standalone repro |
| [inductor-cpu-tiling-stride-at-simplify](issues/inductor-cpu-tiling-stride-at-simplify.md) | funnel, mobilevit, gemma3_vlm, swin… | CPU compile takes hours (vs minutes on CUDA): `sympy.simplify` in `sizevars.stride_at` |
| [floordiv-simplify-compile-time](issues/floordiv-simplify-compile-time.md) | swin | 15–52 min compile on CUDA: `FloorDiv.eval` → `sympy.simplify` on nested size expressions |
| [significant-strides-equal-guard](issues/significant-strides-equal-guard.md) | qwen2_vl, qwen2_5_vl, qwen3_vl, qwen3_8 (train) | Inductor specializes `Eq(s87, 1176/1536)` |
| [dynamo-rnn-unsupported](issues/dynamo-rnn-unsupported.md) | encodec, wavtokenizer | LSTM breaks unless `allow_rnn=True` (feature request) |

**Existing drafts:** `graph-break-in-loop` (24 models), `check-tensor-all-item` (15), `ctc-loss-graph-break`,
`requires-grad-intermediate`, `genexpr-after-graph-break`, `sdpa-is-causal-symbool`, `ctrl-buffer-reassign`,
`conv-backward-stride-guard`, `cpp-acc-helper-reduction-guard`, `tolist-unbacked-length`.

### transformers (huggingface/transformers): 27

Model-code patterns Dynamo can't trace. Most drafts include a repro that checks the proposed fix gives one
graph (and the same output where shown).

| issue | models | pattern |
|---|---|---|
| [qwen-vl-grid-thw-data-dependent](issues/qwen-vl-grid-thw-data-dependent.md) | qwen2_vl, qwen2_5_vl, qwen3_vl, qwen3_8, hy_v4_vl, lfm3_vl, inkling2_vlm, dots3_note_vlm | `grid_thw` `.tolist()` / `.item()` in vision tower + mRoPE (20–35 breaks/forward) |
| [vlm-split-image-features-tolist](issues/vlm-split-image-features-tolist.md) | 7 VLMs | split image features by `.tolist()` only to re-concat |
| [vision-attn-split-lengths-tolist](issues/vision-attn-split-lengths-tolist.md) | 5 VLMs | per-image `torch.split(lengths.tolist())` in vision attention |
| [placeholder-mask-bool-index-numel](issues/placeholder-mask-bool-index-numel.md) | inkling2, dots3_note, omniasr | `inputs_embeds[mask].numel()` check |
| [encodec-conv-params-buffers](issues/encodec-conv-params-buffers.md) | encodec, mimi, wavtokenizer | conv hyper-params as tensor buffers (20–35 breaks/forward) |
| [specaugment-numpy-mask](issues/specaugment-numpy-mask.md), [ctc-labels-max-check](issues/ctc-labels-max-check.md) | wav2vec2, hubert, wavlm | numpy SpecAugment; `labels.max()` check (train) |
| [sam3-spatial-shape-tensor-arange](issues/sam3-spatial-shape-tensor-arange.md) | sam3 (new model) | `torch.arange(0, <tensor>)` in every DETR decoder layer |
| [rope-standardize-in-forward-typeddict-isinstance](issues/rope-standardize-in-forward-typeddict-isinstance.md) | phimoe (dynamic) | config mutated in forward; hub validation `isinstance(…, TypedDict)` |
| [deberta-att-span-item](issues/deberta-att-span-item.md) | deberta | tensor `att_span` → `.item()` per layer (12 → 1 graph with fix) |
| [problem-type-config-mutation](issues/problem-type-config-mutation.md) | classification heads (vit) | 2nd training step always recompiles |
| [nemotron-h-pad-size-guard](issues/nemotron-h-pad-size-guard.md) | nemotron_h (+ same code in bamba, granitemoehybrid, zamba2, falcon_h1) | `if pad_size > 0` → recompile |
| others | one model each | longformer/led `.item()`, xlm assert, fsmt mask, bigbird attn switch, switch_transformers expert loop, smolvlm, granite_docling, mistral3, llava_next, apertus1p5, phimoe LongRoPE, dynamic RoPE, reformer, output_capturing, layerdrop (now only hubert-family), loss warning |

`layerdrop-rand-branch` is **fixed on transformers main** for bart, mbart, marian, m2m_100, blenderbot,
pegasus, pegasus_x, opt, whisper, xglm, biogpt, fsmt, led (train: 9 breaks → 0); still present in
hubert-family models.

### Other repos: 2

| issue | repo | what |
|---|---|---|
| [zonos3-moe-unique-expert-loop](issues/zonos3-moe-unique-expert-loop.md) | Zyphra/ZONOS3-Mini-hf (remote code) | MoE loop over `torch.unique(...).tolist()` + `nonzero()`, 17 breaks/step. Repro with a fix |
| [harness-dynamic-loss-selection](issues/harness-dynamic-loss-selection.md) | this repo | `dynamic.py` `_loss()` picks a non-scalar loss / int tensor → dac, encodec, mimi dynamic-train results are harness errors |

## Models added in this sweep

| requested | result |
|---|---|
| inkling2 small | already tracked (`inkling2`, `inkling2_vlm`) |
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
- **Dynamic-shape recompiles were not triaged one by one.** About 60 runs per device recompile; the ones
  investigated (mamba family, nemotron_h, swin, problem_type) have issues. The rest are probably
  downstream of the data-dependent breaks above, but that was not verified per model.
- `qwen3_next` CPU static results are from the earlier build `9933f27` (0 breaks; it can't run on this CPU
  build). `rwkv` was run with the model built on GPU and moved to each device (its init needs LAPACK).

## Still to do

1. **5 CPU dynamic runs not completed** (cancelled for compile time; they ran 1–3 h without finishing):
   swin eval/train, swinv2 eval/train, funnel train. List: `~/gb-jobs/cpu_dyn_todo_remaining.txt`.
   Rerun one-per-task with a time cap, no requeue:
   `sed -i 's/cpu_dyn_todo.txt/cpu_dyn_todo_remaining.txt/; s/--array=0-8/--array=0-4/' ~/gb-jobs/run_cpu_dyn_redo.sbatch && sbatch ~/gb-jobs/run_cpu_dyn_redo.sbatch`
2. Triage the remaining dynamic-shape recompiles.
3. Review the drafts, file them, and set `issue:` / `status: open` in `known_issues.yaml`.

## Repo changes (uncommitted)

- `graph_breaks/check.py`: `--results-dir`; CUDA results to `results_cuda/`; per-model `trust_remote_code`,
  `init_weights`; dict-aware config overrides; `sam3` / `zonos3_rows` inputs.
- `graph_breaks/dynamic.py`: `--results-dir`; CUDA results to `results_dynamic_cuda/`.
- `graph_breaks/summarize.py`: `--device cuda` → `SUMMARY_CUDA.md`; remote-code path normalization.
- `graph_breaks/coverage.py` (new) → `COVERAGE.md`: every break on every device/check with its issue.
- `models.yaml`: + qwen3_8, sam3, zonos3_mini. `known_issues.yaml`: 49 root causes. `issues/`: 36 new
  drafts + `repros/`. Results: `results*/`, summaries.
