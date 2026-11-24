"""Inductor `sympy_subs` crashes on Min/Max over an `Identity(...)` index: ValueError: The argument '-((0))' is not comparable.

Reduced from inkling2 eval on CUDA (dynamic=True): scheduler.merge_loops -> sizevars._stride_vars substitutes the loop
variables with 0 in a fused cat + relative-position index `Min(15, Max(0, q2 - Identity(q3)))` (from
`distance.clamp(0, rel_extent - 1)`; the Identity comes from the cat lowering). `Identity(0)` does not evaluate to
`Integer(0)`, so Max receives `-Identity(0)` (printed `-((0))`), which `MinMaxBase._new_args_filter` rejects.
"""
import sympy
import torch
from torch._inductor.utils import sympy_subs
from torch.utils._sympy.functions import Identity

q2, q3 = sympy.symbols("q2 q3", integer=True, nonnegative=True)
expr = sympy.Min(15, sympy.Max(0, q2 - Identity(q3)))
print("expr:", expr)
print("Identity(0) ->", sympy.srepr(Identity(sympy.Integer(0))))
print("subs:", sympy_subs(expr, {q2: sympy.Integer(0), q3: sympy.Integer(0)}))   # ValueError
