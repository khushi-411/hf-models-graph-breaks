"""Inductor's convolution_backward lowering specializes dynamic sizes -> recompile on every new length.

torch/_inductor/kernel/conv.py, convolution_backward_lowering (added in #178945):

    guard = V.graph.sizevars.guard_int_seq
    stride_order = ir.get_stride_order(guard(layout_dx.stride))

guard_int on a symbolic stride (e.g. the sequence length L in a (B, C, L) input) installs
Eq(L, <hint>), so with dynamic=True every new sequence length recompiles in train mode.
Eval is fine: the forward conv lowering doesn't do this.

Hit by every HF model with a Conv1d over the sequence in train mode: the Mamba-style
causal depthwise conv (mamba, falcon_mamba, jamba, bamba, zamba2, falcon_h1, nemotron_h,
lfm2, granitemoehybrid) and the wav2vec2-family feature encoder (wav2vec2, wavlm, hubert).

Expected: 1 graph for all three lengths.
Actual:   3 graphs; recompile reason "size mismatch at index 2. expected 32, actual 40".
"""

import torch
from torch._dynamo.testing import CompileCounterWithBackend

torch._inductor.config.force_disable_caches = True  # a warm cache would hide the lowering
torch._logging.set_logs(recompiles=True)

conv = torch.nn.Conv1d(8, 8, kernel_size=4, groups=8, padding=3)  # Mamba-style causal depthwise conv


def fn(x):
    return conv(x)[..., : x.shape[-1]]


for train in (False, True):
    torch._dynamo.reset()
    cnt = CompileCounterWithBackend("inductor")
    compiled = torch.compile(fn, backend=cnt, dynamic=True)
    for n in (32, 40, 56):
        x = torch.randn(2, 8, n, requires_grad=train)
        out = compiled(x)
        if train:
            out.sum().backward()
    print(f"{'train' if train else 'eval '}: graphs compiled for lengths 32/40/56 = {cnt.frame_count}")
