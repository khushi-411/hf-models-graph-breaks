"""After a graph break, a generator expression that closes over `self` fails with
"Attempted to read undefined local variable ... Could not find a local variable with name `self`".

Seen in transformers Qwen2.5-VL vision tower (modeling_qwen2_5_vl.py, forward):
    position_embeddings = tuple(
        self.permute_input_for_window_attn(freq, window_index, seq_len=seq_len) for freq in position_embeddings
    )
which runs after earlier data-dependent breaks in the same frame (grid_thw .item()/.tolist()).
The resume function can't resume there, so the rest of the vision forward runs in eager.
"""

import sys

import torch
import torch.nn as nn


class M(nn.Module):
    def scale(self, t, k):
        return t * k

    def forward(self, x):
        k = 2
        x = x + 1
        torch._dynamo.graph_break()  # e.g. grid_thw.tolist()
        pair = (x.sin(), x.cos())
        return tuple(self.scale(t, k) for t in pair)


print(sys.version.split()[0], torch.__version__)
graphs = []


def backend(gm, example_inputs):
    graphs.append([str(n.target) for n in gm.graph.nodes if n.op in ("call_function", "call_method")])
    return gm


torch._logging.set_logs(graph_breaks=True)
out = torch.compile(M(), backend=backend)(torch.randn(3))
print(len(graphs), "graphs:", graphs)  # expected 2 graphs: [add] and [sin, cos, mul, mul]
