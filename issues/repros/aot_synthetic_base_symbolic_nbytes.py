"""dynamic=True: two graph inputs that share a storage, have no ._base, may overlap, and one is mutated.

AOTAutograd's merge_view_inputs builds a synthetic base with `torch.empty(0).set_(alias.untyped_storage())`.
aten::set_(Storage) reads `storage.nbytes()`, which throws when the (fake) storage size is symbolic:

    RuntimeError: Expected !size_bytes_is_heap_allocated_ to be true, but got false.

Static shapes are fine. CPU or CUDA, any backend that goes through AOTAutograd (aot_eager is enough).
In gpt-oss (MXFP4, kernels-community/gpt-oss-triton-kernels) the inputs are int32 routing buffers carved out of
one allocation, passed into the graph after a graph break.

    python issues/repros/aot_synthetic_base_symbolic_nbytes.py
"""

import torch


def make(n, k):
    buf = torch.zeros(2 * n, dtype=torch.int32)
    return buf[:n].detach(), buf[k:k + n].detach()  # overlapping views, no ._base


def f(a, b):
    a.add_(1)
    return b.sum()


a, b = make(8, 3)
print("static: ", torch.compile(f, backend="aot_eager")(a, b).item(), "(expected 5)")
torch._dynamo.reset()
a, b = make(8, 3)
try:
    print("dynamic:", torch.compile(f, backend="aot_eager", dynamic=True)(a, b).item())
except Exception as e:
    print("dynamic:", type(e).__name__, [l for l in str(e).splitlines() if "RuntimeError" in l][:1])
