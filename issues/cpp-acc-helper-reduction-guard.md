<!-- Draft for: pytorch/pytorch. Once filed, put the URL in known_issues.yaml and set status: open -->

# [dynamic shapes] Inductor CPU `need_use_acc_helper` installs a guard on the reduction size, so a float sum recompiles when the size crosses 4096

### System info

- torch: `2.16.0a0+git6b3a750` (source build)
- transformers: `5.22.1`
- Python 3.14, Linux, CPU

### Description

`CppKernel.need_use_acc_helper` in `torch/_inductor/codegen/cpp.py` decides whether a float `sum`
reduction uses the cascade-sum accumulator helper:

```python
reduction_size = functools.reduce(operator.mul, self.call_ranges[self.reduction_depth :])
chunk_size = 4096
return V.graph.sizevars.guard_or_false(sympy.Gt(reduction_size, chunk_size))
```

When the size hint decides the comparison, `guard_or_false` still installs a guard. With
`dynamic=True`, a graph compiled at a reduction size of 4096 or less carries the guard
`2 <= s1 <= 4096`, so the first input above 4096 recompiles (and the other way round).

The choice is a codegen heuristic for accuracy and performance. It doesn't need to be sound for
every size, so it shouldn't add a guard. It's the same pattern as #137093: a `guard_*` call on a
`sympy.Expr` in Inductor where only a hint is needed.

Seen in Hugging Face models (`torch.compile(dynamic=True)`, CPU):
- hubert `train_forward`: `(-1)*s0 + s0*(s2 // 10) <= 4096` is its **last remaining recompile**
  once [conv-backward-stride-guard](conv-backward-stride-guard.md) is fixed.
- nemotron_h `train_forward`: the guard specializes the batch size (`s52 = 2`).

### Reproduction

```python
import torch
from torch._dynamo.testing import CompileCounterWithBackend

torch._inductor.config.force_disable_caches = True

def fn(x):
    return x.sum(-1)

cnt = CompileCounterWithBackend("inductor")
compiled = torch.compile(fn, backend=cnt, dynamic=True)
for n in (1000, 3000, 5000, 9000):
    compiled(torch.randn(4, n))
print(cnt.frame_count)  # 2
# [__recompiles] 2 <= x.size()[1] <= 4096 ... (_inductor/codegen/cpp.py:2405 in need_use_acc_helper)
```

Full script: [repros/cpp_acc_helper_reduction_guard.py](repros/cpp_acc_helper_reduction_guard.py).

### Expected behavior

One graph for all reduction sizes.

### Possible fix

Decide from the hint without guarding, e.g.
`V.graph.sizevars.optimization_hint(reduction_size) > chunk_size`, or
`statically_known_true(...)` with a fallback. Alternatively, when the size is symbolic, always
emit the acc-helper path, which is correct for any size. That might cost a little speed on small
reductions, but there are no recompiles.
