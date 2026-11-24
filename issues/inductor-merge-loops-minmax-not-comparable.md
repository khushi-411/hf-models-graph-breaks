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

- `eval_forward`: fails as above (inkling2, inkling2_vlm, and the later inkling2_mini, inkling2_mini_vlm).
- `train_forward`: compiles fine with the same model and sizes.
- Dynamo-only `explain` is fine.
- Default `torch.compile` (`dynamic=None`, one input size) also fails on CUDA (H100): eval_forward for all four
  models, and eval_with_cache for inkling2 and inkling2_vlm. These models graph-break in the vision tower and
  the looped text model, and the resumed frames recompile, so automatic dynamic shapes are on by then. Seen in the
  accuracy check (`python -m graph_breaks.accuracy`).
- CPU: compiles and runs fine for all four models (dynamic and default), so this is CUDA (Triton) codegen only.
- Still present on main (`2.16.0a0+gitf29b6d5`, CUDA 13.2, B300). It is the main CUDA blocker for these models:
  every inference path that goes through Inductor fails on it. `graph_breaks/modes.py` on CUDA: eval with
  `freezing=True`, under `torch.autocast(bfloat16)`, `mode="reduce-overhead"`, AOTInductor
  (`aoti_compile_and_package` on the exported program, so not only `torch.compile`), `generate()` with a compiled
  forward (`dynamic=True`), and the perf benchmark. `mode="max-autotune-no-cudagraphs"` eval compiles (inkling2),
  as does training.

### Cause

Captured at the failure (inkling2 eval, B300, `f29b6d5`): `_stride_vars` substitutes `{q0, q1, q2, q3} -> 0` into

```
128*q0*s27 + 16*q1 + 128*q2 + Min(15, Max(0, q2 - Identity(q3)))
```

for `buf385 = Pointwise(size=[s52, 8, s27, s27 + 1])`, a fused `aten.cat` + the relative-position bias
(`distance.clamp(0, rel_extent - 1)` with `rel_extent = 16`, `where(distance < 0 | distance >= 16, 0, ...)`). The
`Identity` comes from the cat lowering (`lowering.py:2072`, `Identity(idx_load[dim] - inputs_ranges[i][0])`).

`torch.utils._sympy.functions.Identity` has no `eval`, so `Identity(0)` stays unevaluated. Its `is_comparable` is
True, but `-Identity(0)` is a `Mul`, whose comparability sympy decides by `evalf`, which `Identity` doesn't
implement, so `(-Identity(0)).is_comparable` is False and `Max(0, -Identity(0))` raises (printed `-((0))`).

### Reproduction

Pure sympy, CPU, 10 lines: [`issues/repros/inductor_sympy_subs_identity_minmax.py`](repros/inductor_sympy_subs_identity_minmax.py)

```python
import sympy
from torch._inductor.utils import sympy_subs
from torch.utils._sympy.functions import Identity

q2, q3 = sympy.symbols("q2 q3", integer=True, nonnegative=True)
expr = sympy.Min(15, sympy.Max(0, q2 - Identity(q3)))
sympy_subs(expr, {q2: sympy.Integer(0), q3: sympy.Integer(0)})
# ValueError: The argument '-((0))' is not comparable.
```

Fails on `57f02f8` and `f29b6d5`. Model-level: `python -m graph_breaks.dynamic --model inkling2 --scenario eval_forward --device cuda`.

### Possible fix

Let `Identity` fold numbers (it only exists to block expansion of symbolic expressions):

```python
class Identity(sympy.Function):
    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            return arg
```

With this patch the repro returns `0`, and symbolic `Identity(q3)` is unchanged. (Alternatives: implement
`_eval_evalf` on `Identity`, or strip `Identity` in `sympy_subs` / `_stride_vars` before substituting.)

<!-- affected-models:start (generated by `python -m graph_breaks.affected`, do not edit) -->

### Affected models in this sweep

4 model(s). Each cell lists the scenarios where this root cause shows up: a number is the graph breaks matched to it in that run; `recompiles (a / b / c)` is graphs compiled on the 1st / 2nd / 3rd input size with `dynamic=True`; `guard only` means the guard was installed but didn't fail on these sizes. Recompile counts are for the whole run, so a model with several root causes counts all of them. Scenarios: `eval` = eval_forward, `cache` = eval_with_cache, `train` = train_forward. Models are built from the checkpoint's config with reduced layers and random weights.

| model | config from | static CPU | static CUDA | dynamic CPU | dynamic CUDA | unbacked CPU | unbacked CUDA |
|---|---|---|---|---|---|---|---|
| inkling2 | `thinkingmachines/Inkling-2-Small-MXFP8` | – | – | – | eval: **crash** | – | eval: **crash** |
| inkling2_mini | `thinkingmachines/Inkling-2-Mini` | – | – | – | eval: **crash** | – | eval: **crash** |
| inkling2_mini_vlm | `thinkingmachines/Inkling-2-Mini` | – | – | – | eval: **crash** | – | – |
| inkling2_vlm | `thinkingmachines/Inkling-2-Small-MXFP8` | – | – | – | eval: **crash** | – | – |

<!-- affected-models:end -->
