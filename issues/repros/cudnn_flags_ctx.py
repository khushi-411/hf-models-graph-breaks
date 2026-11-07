"""`with torch.backends.cudnn.flags(...)` is not traceable by Dynamo.

Dynamo treats `torch.backends.cudnn.flags` as a torch.* function and fails because it
returns a _GeneratorContextManager ("torch.* op returned non-Tensor"). The code before the
`with` is compiled; everything from the `with` on runs in eager.

transformers' ForCTCLoss (loss/loss_ctc.py) wraps F.ctc_loss in
`torch.backends.cudnn.flags(enabled=False)`: hit in training for wav2vec2, hubert, wavlm
and every other CTC model. (ctc_loss then breaks on its own; see ctc_loss_no_meta.py.)

Expected: 1 graph with log_softmax, mul and sum (the flag only matters for cuDNN kernel selection).
Actual:   1 graph with only log_softmax, plus a graph break.
"""

import torch


def f(x):
    y = x.log_softmax(-1)
    with torch.backends.cudnn.flags(enabled=False):
        y = y * 2
    return y.sum()


graphs = []


def backend(gm, example_inputs):
    graphs.append([str(n.target) for n in gm.graph.nodes if n.op in ("call_function", "call_method")])
    return gm


torch.compile(f, backend=backend)(torch.randn(4, 5))
print("graphs compiled:", graphs)
torch._dynamo.reset()
e = torch._dynamo.explain(f)(torch.randn(4, 5))
print([str(r.reason).splitlines()[0] for r in e.break_reasons])
