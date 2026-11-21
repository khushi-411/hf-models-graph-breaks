<!-- Draft for: pytorch/pytorch (labels: module: inductor, module: dynamic shapes). Previously marked not-filing in known_issues.yaml (reason kept below). Once filed, put the URL in known_issues.yaml and set status: open -->

# [dynamic shapes] Inductor `ir.py:significant_strides_equal` specializes a dynamic size via `guard_or_false(Eq(s, c))` when strides happen to be equal

### System info

- torch: `2.16.0a0+git57f02f8` (source build, CUDA 12.8)
- transformers: `5.24.0.dev0` (git `b7fe69cb19`)
- Python 3.15, Linux; CPU and CUDA (B200)

### Description

With `torch.compile(model, backend="inductor", dynamic=True)`, Inductor's `significant_strides_equal`
(`torch/_inductor/ir.py`) compares strides with `guard_or_false(Eq(...))`. When the comparison involves a
backed dynamic symbol, that installs an equality guard and **specializes the symbol**:

| model (train_forward) | specialization site | guard |
|---|---|---|
| qwen2_vl | `torch/_inductor/ir.py:significant_strides_equal` | `Eq(s87, 1176)` |
| qwen2_5_vl | same | `Eq(s87, 1176)` |
| qwen3_vl | same | `Eq(s87, 1536)` |
| qwen3_8 (Qwen3.8, qwen3_5) | same | `Eq(s87, 1536)` |

Seen on CPU and CUDA. `s87` is `pixel_values.shape[-1]` in the vision tower's patch embedding
(`modeling_qwen2_vl.py:716`): `1176 = 3 channels * 2 temporal * 14 * 14` (`1536 = 3 * 2 * 16 * 16` for
patch size 16).

### Why this was marked "not filing" before

That dimension is fixed by the model config, so it never changes between real inputs and the
specialization causes **no recompile** in these models. It only shows up because `dynamic=True` marks every
dim dynamic, and `guard_or_false` guarding a backed symbol is by design.

### Why it may still be worth reporting

`significant_strides_equal` is a layout-equivalence check. Turning it into a value guard on a size means
any model where the compared size *does* vary (e.g. a variable channel or feature dim) would specialize and
recompile. A size-oblivious comparison (or `statically_known_true`) would answer the layout question
without installing a guard.

### Reproduction

```bash
python -m graph_breaks.dynamic --model qwen2_vl --scenario train_forward              # CPU
python -m graph_breaks.dynamic --model qwen2_vl --scenario train_forward --device cuda
```

`results_dynamic*/qwen2_vl/train_forward.json` -> `inductor_specialization_sites:
["torch/_inductor/ir.py:significant_strides_equal"]`, example guard `Eq(s87, 1176)`.

### Expected behavior

Equal-stride detection without specializing the size, or documentation that this guard is intended.
