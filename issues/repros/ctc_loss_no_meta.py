"""F.ctc_loss always graph-breaks under torch.compile, even with static shapes.

aten._ctc_loss.Tensor is treated as a dynamic-output-shape op (log_alpha's last dim is
2*max(target_lengths)+1, i.e. data-dependent), and with
capture_dynamic_output_shape_ops=True it fails with "Dynamic shape operator (no meta kernel)".

Hits every CTC model's training step in transformers via loss/loss_ctc.py
(wav2vec2, hubert, wavlm, data2vec-audio, sew, unispeech, parakeet, ...).
"""

import torch
import torch.nn.functional as F

T, N, C, S = 50, 2, 20, 10
log_probs = torch.randn(T, N, C).log_softmax(-1)
targets = torch.randint(1, C, (N, S))
input_lengths, target_lengths = torch.full((N,), T), torch.full((N,), S)


def loss_fn(lp, t, il, tl):
    return F.ctc_loss(lp * 1.0, t, il, tl, zero_infinity=True) + 1


for cfg in ({}, {"capture_dynamic_output_shape_ops": True, "capture_scalar_outputs": True}):
    graphs = []

    def backend(gm, example_inputs):
        graphs.append([str(n.target) for n in gm.graph.nodes if n.op == "call_function"])
        return gm

    torch._dynamo.reset()
    with torch._dynamo.config.patch(**cfg):
        torch.compile(loss_fn, backend=backend)(log_probs, targets, input_lengths, target_lengths)
    print(cfg or "default flags", "->", len(graphs), "graphs:", graphs)  # expected: 1 graph containing ctc_loss
