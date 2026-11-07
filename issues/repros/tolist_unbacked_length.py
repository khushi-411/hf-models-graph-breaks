"""`.tolist()` on a tensor with a data-dependent (unbacked) length crashes compilation
when capture_dynamic_output_shape_ops=True, instead of graph-breaking.

Seen in transformers Qwen2.5-VL / Qwen3-VL vision towers (cu_seqlens from
repeat_interleave(grid_thw) -> .tolist()). With the default flags those ops graph-break
(25 / 36 graphs); turning on capture_scalar_outputs + capture_dynamic_output_shape_ops
to remove the breaks turns them into a hard GuardOnDataDependentSymNode error instead.

Expected: compiled output (or a graph break with fullgraph=False).
Actual:   GuardOnDataDependentSymNode: Could not guard on data-dependent expression Eq(u0, 0)
          even with fullgraph=False.
"""

import torch
import torch._dynamo.config as dc


def f(x):
    n = x.nonzero().flatten()  # length is unbacked (u0)
    return torch.tensor(n.tolist()) + 1


def g(x):
    return torch.repeat_interleave(x, x).tolist()  # same, via repeat_interleave


x = torch.tensor([0, 2, 0, 3])
for fn in (f, g):
    for fullgraph in (False, True):
        torch._dynamo.reset()
        with dc.patch(capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True):
            try:
                out = torch.compile(fn, backend="eager", fullgraph=fullgraph)(x)
                print(f"{fn.__name__} fullgraph={fullgraph}: OK", out)
            except Exception as e:
                print(f"{fn.__name__} fullgraph={fullgraph}: {type(e).__name__}: {str(e).splitlines()[0][:120]}")
