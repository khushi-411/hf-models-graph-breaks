"""ZONOS3-Mini MoE (`Zonos3MiniLM._feed_forward`, modeling_zonos3.py:321): the expert loop iterates over
`torch.unique(choices).tolist()` and gathers tokens with `nonzero()`. Both are data-dependent, so every MoE
layer graph-breaks. A static loop over all experts with a routing mask gives the same output in one graph
(it computes every expert on every token, so it trades FLOPs for no host sync; grouped GEMM avoids that)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import torch
import torch.nn.functional as F
from _pattern import report

torch.manual_seed(0)
N_EXPERTS, TOPK, DIM, ROUTED, HIDDEN, N = 8, 2, 64, 32, 48, 24
router_w = torch.randn(N_EXPERTS, DIM) * 0.1
gate_up = torch.randn(N_EXPERTS, 2 * HIDDEN, ROUTED) * 0.1
down = torch.randn(N_EXPERTS, ROUTED, HIDDEN) * 0.1
latent_down = torch.randn(ROUTED, DIM) * 0.1
x = torch.randn(N, DIM)


def original(x):
    logits = F.linear(x, router_w)
    choices = torch.topk(logits, TOPK, -1).indices
    weights = logits.gather(-1, choices).softmax(-1)
    latent = F.linear(x, latent_down)
    ranked = latent.new_zeros(TOPK, len(x), latent.shape[-1])
    for expert in torch.unique(choices).tolist():                      # modeling_zonos3.py:321
        token, rank = (choices == expert).nonzero(as_tuple=True)         # :322
        h = F.linear(latent[token], gate_up[expert])
        out = F.linear(F.silu(h[:, :HIDDEN]) * h[:, HIDDEN:], down[expert])
        ranked[rank, token] = out * weights[token, rank, None]
    return ranked.sum(0)


def fixed(x):
    logits = F.linear(x, router_w)
    choices = torch.topk(logits, TOPK, -1).indices
    weights = logits.gather(-1, choices).softmax(-1)
    latent = F.linear(x, latent_down)
    output = torch.zeros_like(latent)
    for expert in range(N_EXPERTS):                                      # static loop, no host sync
        w = (weights * (choices == expert)).sum(-1, keepdim=True)        # 0 for tokens not routed here
        h = F.linear(latent, gate_up[expert])
        output = output + F.linear(F.silu(h[:, :HIDDEN]) * h[:, HIDDEN:], down[expert]) * w
    return output


for label, fn in [("unique().tolist() + nonzero", original), ("static expert loop + mask", fixed)]:
    report(label, fn, x)
    torch._dynamo.reset()
    e = torch._dynamo.explain(fn)(x)
    # A break Dynamo can't resume from makes it skip the frame (runs eagerly), so graph_break_count can stay 0.
    print(f"    break reasons recorded: {len(e.break_reasons)}")
torch.testing.assert_close(original(x), fixed(x), rtol=1e-4, atol=1e-5)
print("outputs match")
