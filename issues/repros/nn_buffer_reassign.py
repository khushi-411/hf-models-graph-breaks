"""Dynamo graph break when reassigning an `nn.Buffer` attribute inside forward.

Seen in transformers CTRL (modeling_ctrl.py: `self.pos_encoding = self.pos_encoding.to(...)`).
No transformers needed to reproduce.

Expected: 1 graph, 0 breaks (works when the buffer is created with register_buffer).
Actual:   2 graphs, 1 break: "Unsupported python_type() call ... GetAttrVariable(TensorVariable(), persistent)".
"""

import torch
import torch.nn as nn


class WithBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.pe = nn.Buffer(torch.randn(8, 4), persistent=False)

    def forward(self, x):
        self.pe = self.pe.to(device=x.device, dtype=x.dtype)
        return x + self.pe[: x.shape[0]]


class WithRegisterBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("pe", torch.randn(8, 4), persistent=False)

    def forward(self, x):
        self.pe = self.pe.to(device=x.device, dtype=x.dtype)
        return x + self.pe[: x.shape[0]]


for cls in (WithRegisterBuffer, WithBuffer):
    torch._dynamo.reset()
    e = torch._dynamo.explain(cls())(torch.randn(4, 4))
    print(f"{cls.__name__:20s} graphs={e.graph_count} breaks={e.graph_break_count}")
    for r in e.break_reasons:
        print("   ", str(r.reason).splitlines()[0], "|", str(r.reason).split("debug context: ")[-1].splitlines()[0])

torch._dynamo.reset()
torch.compile(WithBuffer(), fullgraph=True)(torch.randn(4, 4))  # raises Unsupported
