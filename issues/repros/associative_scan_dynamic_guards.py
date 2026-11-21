"""torch._higher_order_ops.associative_scan under torch.compile(dynamic=True) recompiles for new input
sizes. Same call as transformers' Mamba `mamba_selective_scan` (modeling_mamba.py:237):
  - combine_mode="pointwise" (CUDA), with grad: the batch dim is specialized (guard Eq(batch, 2)),
    so every new batch size recompiles.
  - combine_mode="generic" (CPU): guard ((seq // 8) % 2) == 0, so new sequence lengths recompile.
Shapes: (batch, intermediate=64, seq, state=16), scan over dim=2."""
import torch
from torch._higher_order_ops.associative_scan import associative_scan

print(torch.__version__)


def combine_fn(left, right):
    a_left, b_left = left
    a_right, b_right = right
    return a_left * a_right, a_right * b_left + b_right


def scan(a, b, mode):
    _, states = associative_scan(combine_fn, (a, b), dim=2, combine_mode=mode)
    return states


import torch._dynamo.testing as T


def run(device, mode, train, sizes, label):
    torch._dynamo.reset()
    cnt = T.CompileCounterWithBackend("inductor")
    f = torch.compile(lambda a, b: scan(a, b, mode), dynamic=True, backend=cnt)
    for batch, seq in sizes:
        a = torch.rand(batch, 64, seq, 16, device=device, requires_grad=train)
        b = torch.randn(batch, 64, seq, 16, device=device, requires_grad=train)
        out = f(a, b)
        if train:
            out.sum().backward()
    print(f"[{device} {mode:9s} {'train' if train else 'eval '}] {label}: compiled graphs = {cnt.frame_count}")


vary_batch = [(2, 32), (3, 32), (4, 32)]
vary_seq = [(2, 32), (2, 40), (2, 56)]
run("cpu", "generic", False, vary_seq, "seq 32/40/56")
run("cpu", "generic", False, vary_batch, "batch 2/3/4")
if torch.cuda.is_available():
    for train in (False, True):
        run("cuda", "pointwise", train, vary_batch, "batch 2/3/4")
        run("cuda", "pointwise", train, vary_seq, "seq 32/40/56")
