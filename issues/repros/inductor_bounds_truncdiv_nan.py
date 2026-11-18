"""Inductor crashes with `TypeError: Invalid NaN comparison` on a trunc division of two dynamic sizes.

`torch.div(a, b, rounding_mode="trunc")` on two SymInts is lowered to `ops.truncdiv(s0, s1)`.
Inductor's bound analysis (torch/_inductor/bounds.py ValueRangeAnalysis.truncdiv) goes through
float truediv, and ValueRanges.coordinatewise_monotone_map evaluates the four corner products of
[2, int_oo] / [1, int_oo]. One of them is oo/oo = nan, and min() over it raises:

    InductorError: TypeError: Invalid NaN comparison

Seen in HF longformer (eval_forward and train_forward, dynamic=True):
modeling_longformer.py:822 `torch.div(seq_len, window_overlap, rounding_mode="trunc")`.
The whole compile fails; with static shapes it works.

Expected: compiles (bounds of a truncdiv of two unbounded positive ints are [0, int_oo]).
"""

import torch


def fn(x, y):
    n = torch.div(x.shape[1], y.shape[1], rounding_mode="trunc")
    return x.new_zeros(()) + n


x, y = torch.randn(2, 512), torch.randn(2, 64)
for dynamic in (False, True):
    torch._dynamo.reset()
    try:
        out = torch.compile(fn, dynamic=dynamic)(x, y)
        print(f"dynamic={dynamic}: OK ->", out.item())
    except Exception as e:
        print(f"dynamic={dynamic}: FAILS ->", f"{type(e).__name__}: {str(e).splitlines()[0][:120]}")
