"""Inductor CPU codegen guards on a symbolic reduction size (need_use_acc_helper) -> recompile at 4096.

torch/_inductor/codegen/cpp.py, need_use_acc_helper:

    return V.graph.sizevars.guard_or_false(sympy.Gt(reduction_size, chunk_size))  # chunk_size = 4096

guard_or_false still installs a guard when the hint decides it, so a float `sum` over a dynamic dim
compiled at size <= 4096 recompiles once the size goes above 4096 (and vice versa). Whether to use
the cascade-sum helper is a codegen heuristic; it shouldn't add a guard.

Seen in HF hubert train_forward: "(-1)*s0 + s0*(s2 // 10) <= 4096" is its last remaining recompile
once conv-backward-stride-guard is fixed. CPU only.

Expected: 1 graph for all sizes.
Actual:   2 graphs; recompile reason "... <= 4096".
"""

import torch
from torch._dynamo.testing import CompileCounterWithBackend

torch._inductor.config.force_disable_caches = True
torch._logging.set_logs(recompiles=True)


def fn(x):
    return x.sum(-1)


cnt = CompileCounterWithBackend("inductor")
compiled = torch.compile(fn, backend=cnt, dynamic=True)
for n in (1000, 3000, 5000, 9000):
    compiled(torch.randn(4, n))
print("graphs compiled for reduction sizes 1000/3000/5000/9000 =", cnt.frame_count)
