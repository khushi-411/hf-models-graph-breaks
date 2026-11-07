"""A graph break inside a for loop makes Dynamo skip the *whole frame* (eager fallback).

Dynamo can't resume after a break inside a loop (gb7000 "graph break in loop"), so it
compiles nothing in that frame, not even the code before and after the loop. In
transformers almost every model runs its layers in `for layer in self.layers:` in the
decoder/encoder forward, so any in-loop break (LayerDrop's `if torch.rand([]) < p`,
Longformer's per-layer `.item()`, ...) sends the entire decoder forward to eager.
nested_graph_breaks=True does not help here.

Expected: graphs for `x.sin()` and `x.cos()` (plus per-iteration graphs).
Actual:   0 graphs in this frame.
"""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])

    def forward(self, x):
        x = x.sin()
        for layer in self.layers:
            torch._dynamo.graph_break()  # stand-in for e.g. `if torch.rand([]) < self.layerdrop: continue`
            x = layer(x)
        return x.cos()


for nested in (False, True):
    graphs = []

    def backend(gm, example_inputs):
        graphs.append([str(n.target) for n in gm.graph.nodes if n.op in ("call_function", "call_method", "call_module")])
        return gm

    torch._dynamo.reset()
    with torch._dynamo.config.patch(nested_graph_breaks=nested):
        torch.compile(Decoder(), backend=backend)(torch.randn(2, 8))
    print(f"nested_graph_breaks={nested}: {len(graphs)} graphs {graphs}")
