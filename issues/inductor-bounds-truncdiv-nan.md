<!-- Draft for: pytorch/pytorch. Once filed, put the URL in known_issues.yaml and set status: open -->

# [dynamic shapes] Inductor crashes with `TypeError: Invalid NaN comparison` on trunc division of two dynamic sizes (value-range analysis of `truncdiv`)

### System info

- torch: `2.16.0a0+git6b3a750` (source build)
- transformers: `5.22.1`
- Python 3.14, Linux, CPU

### Description

`torch.div(a, b, rounding_mode="trunc")` on two SymInts is lowered to a pointwise
`ops.truncdiv(s0, s1)`. During codegen, Inductor's bound analysis (`BoundVars.get_bounds` →
`ValueRangeAnalysis.truncdiv` in `torch/_inductor/bounds.py`) computes the bounds through float
`truediv`:

```python
@classmethod
def truncdiv(cls, a, b):
    x = cls.truediv(a, b)   # TODO: slightly inaccurate ... through float truediv
```

`ValueRanges.truediv` → `coordinatewise_monotone_map` evaluates all four corner products. For two
unbounded positive sizes, `[2, int_oo] / [1, int_oo]`, one corner is `oo / oo = nan`, and
`min(products)` raises:

```
torch._inductor.exc.InductorError: TypeError: Invalid NaN comparison
  File "torch/_inductor/bounds.py", line 268, in truncdiv
  File "torch/utils/_sympy/value_ranges.py", line 710, in truediv
  File "torch/utils/_sympy/value_ranges.py", line 428, in coordinatewise_monotone_map
    return ValueRanges(min(products), max(products))
```

The whole compile fails; with static shapes the same code compiles.

Seen in Hugging Face longformer (`eval_forward` and `train_forward`,
`torch.compile(dynamic=True)`): `modeling_longformer.py:822`,
`torch.div(seq_len, window_overlap, rounding_mode="trunc")`. The failing loop body is just
`ops.truncdiv(ops.index_expr(s14), ops.index_expr(s55))`.

### Reproduction

```python
import torch

def fn(x, y):
    n = torch.div(x.shape[1], y.shape[1], rounding_mode="trunc")
    return x.new_zeros(()) + n

x, y = torch.randn(2, 512), torch.randn(2, 64)
torch.compile(fn, dynamic=False)(x, y)  # OK -> 8.0
torch.compile(fn, dynamic=True)(x, y)   # InductorError: TypeError: Invalid NaN comparison
```

Full script: [repros/inductor_bounds_truncdiv_nan.py](repros/inductor_bounds_truncdiv_nan.py).

### Expected behavior

It compiles. The bounds of a trunc division of two positive unbounded ints are `[0, int_oo]`.

### Possible fix

- Give `ValueRangeAnalysis.truncdiv` an integer implementation (as `floordiv` has) instead of
  going through float `truediv`. The existing TODO already notes the precision problem.
- And/or make `coordinatewise_monotone_map` / `truediv` handle `oo/oo` (skip NaN corners, or
  return `ValueRanges.unknown()`) instead of comparing NaN.
