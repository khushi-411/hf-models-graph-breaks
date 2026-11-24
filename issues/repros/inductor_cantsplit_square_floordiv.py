"""Inductor CantSplit with dynamic=True: `((s//16)**2)//(s//16) not divisible by (s//16)` (CUDA, train).

Patch-embedding conv -> channels-last permute -> LayerNorm -> Linear, then backward. The LayerNorm backward is
codegen'd as a mix-order reduction; one of its ranges is `(s//16)**2 // (s//16)` (unsimplified `a**2 // a`), and
`_split_iteration_ranges` can't split it by `s//16`. Static shapes and smaller sizes (C=64, or a 256/512 image)
compile fine. Workaround: `torch._inductor.config.triton.mix_order_reduction = False`.

Seen in Unlimited-OCR (baidu/Unlimited-OCR) SAM vision encoder, dynamic=True train on H100.

    python issues/repros/inductor_cantsplit_square_floordiv.py [--no-mix-order]
"""

import sys

import torch
import torch.nn as nn

if "--no-mix-order" in sys.argv:
    torch._inductor.config.triton.mix_order_reduction = False

C, S = 768, 1024


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, C, 16, 16)
        self.ln = nn.LayerNorm(C)
        self.fc = nn.Linear(C, C)

    def forward(self, x):
        x = self.patch(x).permute(0, 2, 3, 1)  # (B, S//16, S//16, C)
        return self.fc(self.ln(x))


m = M().cuda()
x = torch.randn(2, 3, S, S, device="cuda")
torch.compile(m)(x).sum().backward()
print("static: ok")
torch._dynamo.reset()
try:
    torch.compile(m, dynamic=True)(x).sum().backward()
    print("dynamic: ok")
except Exception as e:
    print("dynamic:", type(e).__name__, str(e).splitlines()[0][:200])
