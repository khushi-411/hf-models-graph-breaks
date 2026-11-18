<!-- Draft for: pytorch/pytorch (label: module: inductor, module: dynamic shapes). Once filed, put the URL in known_issues.yaml and set status: open -->

# [dynamic shapes] Inductor `merge_loops` crashes: `ValueError: The argument '-((0))' is not comparable` in `sympy_subs` → `Min/Max.__new__`

### System info

- torch: `2.16.0a0+git57f02f8` (source build, CUDA 12.8, sm_90 / sm_100, cuDNN 9.28, Triton 3.9.0)
- GPU: NVIDIA B200
- transformers: `5.24.0.dev0` (git `b7fe69cb19`)
- Python 3.15, Linux

### Description

Compiling the Inkling-2 text model (and the Inkling-2 VLM, same language model) with
`torch.compile(backend="inductor", dynamic=True)` on CUDA fails during scheduling, in the very first
compile (no graph is produced):

```
File "torch/_inductor/scheduler.py", line 6252, in _init
File "torch/_inductor/scheduler.py", line 7270, in merge_loops
File "torch/_inductor/scheduler.py", line 4172, in merge_loops
File "torch/_inductor/loop_body.py", line 226, in merge_loops
File "torch/_inductor/sizevars.py", line 365, in simplify_loops
File "torch/_inductor/sizevars.py", line 500, in _simplify_loops_impl
File "torch/_inductor/sizevars.py", line 1294, in stride_vars
File "torch/_inductor/sizevars.py", line 1280, in wrapper
File "torch/_inductor/sizevars.py", line 1314, in _stride_vars
File "torch/_inductor/utils.py", line 1738, in sympy_subs
File "torch/fx/experimental/_size_hinting.py", line 66, in _sympy_subs
File "torch/utils/_sympy/functions.py", line 649, in __new__
File "torch/utils/_sympy/functions.py", line 943, in _new_args_filter
    raise ValueError(f"The argument '{arg}' is not comparable.")
torch._inductor.exc.InductorError: ValueError: The argument '-((0))' is not comparable.
```

`_stride_vars` substitutes the iteration variables into an index expression that contains a `Min`/`Max`.
After substitution one argument is `-((0))`. It looks like a negated zero that was not evaluated, and
`MinMaxBase._new_args_filter` rejects it as "not comparable" instead of folding it to `0`.

- `eval_forward`: fails as above (inkling2 and inkling2_vlm).
- `train_forward`: compiles fine with the same model and sizes.
- Static shapes (`dynamic=False` / Dynamo-only `explain`) are fine.
- CPU dynamic run: not yet confirmed (the CPU run is in progress; will update).

The model has sliding-window attention (window and relative-position extent below the sequence length)
and a cross-layer attention-residual block. The `Min`/`Max` probably comes from a clamped window or
relative-position index (not yet confirmed).

### Reproduction

Model-level (minimal repro still to be extracted):

```bash
# from the hf-models-graph-breaks harness
python -m graph_breaks.dynamic --model inkling2 --scenario eval_forward --device cuda
```

which builds the model from its config (7 layers, random weights) and runs
`torch.compile(model, backend="inductor", dynamic=True)` on inputs of shape (2, 32), (3, 40), (4, 56).

### Expected behavior

`-((0))` should simplify to `0` (or be treated as comparable) during `sympy_subs`, and the graph should
compile, as it does in `train_forward`.
