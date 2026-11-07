<!-- Draft for: pytorch/pytorch. Once filed, put the URL in known_issues.yaml and set status: open -->

# Dynamo: reassigning an `nn.Buffer` attribute in forward graph-breaks with "Unsupported python_type() call" (`GetAttrVariable(TensorVariable(), persistent)`)

### System info

- torch: `2.16.0a0+git9933f27` (source build)
- transformers: `5.22.1`
- Python 3.15, Linux, CPU

### Description

If a buffer is created with `nn.Buffer(...)` and then reassigned in `forward`
(`self.pe = self.pe.to(device=..., dtype=...)`), Dynamo graph-breaks inside
`Module.__setattr__`. The same code compiles without a break when the buffer was created
with `register_buffer`.

In eager, `.to()` with an unchanged dtype and device returns the same object, which still
carries the `_is_buffer` / `persistent` Python attributes set by `nn.Buffer`. So
`isinstance(value, Buffer)` is True and `__setattr__` reads `value.persistent`
(`module.py:2050`). Dynamo models the result of `.to()` as a new `TensorVariable` that
doesn't track that attribute. It produces `GetAttrVariable(TensorVariable(), persistent)`,
then fails on `if persistent:` in `register_buffer`. The error message itself says
"This is likely to be a Dynamo bug".

Hit by transformers CTRL (`modeling_ctrl.py`: `self.pos_encoding = self.pos_encoding.to(...)`),
in every scenario: 2–3 extra graphs.

### Reproduction

```python
import torch, torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.pe = nn.Buffer(torch.randn(8, 4), persistent=False)
    def forward(self, x):
        self.pe = self.pe.to(device=x.device, dtype=x.dtype)
        return x + self.pe[: x.shape[0]]

torch.compile(M(), fullgraph=True)(torch.randn(4, 4))
# Unsupported python_type() call ... GetAttrVariable(TensorVariable(), persistent)
```

Full script, comparing against `register_buffer`: [repros/nn_buffer_reassign.py](repros/nn_buffer_reassign.py).

### Expected behavior

1 graph, as with `register_buffer`.
