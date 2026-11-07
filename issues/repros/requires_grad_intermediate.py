"""Setting `.requires_grad = True` on an intermediate (a view of an input) graph-breaks.

transformers' wav2vec2-family feature encoder does this in training mode, for gradient
checkpointing:
    hidden_states = input_values[:, None]
    if self._requires_grad and self.training:
        hidden_states.requires_grad = True
Hit by wav2vec2, hubert, wavlm (and data2vec-audio, sew, unispeech, ... which copy it).

Expected: 1 graph.
Actual:   2 graphs, 1 break: "requires_grad_() intermediate leaked as output".
"""

import torch


def f(x):
    h = x[:, None]
    h.requires_grad = True
    return (h * 2).sum()


e = torch._dynamo.explain(f)(torch.randn(2, 8))
print(f"graphs={e.graph_count} breaks={e.graph_break_count}", [str(r.reason).splitlines()[0] for r in e.break_reasons])
