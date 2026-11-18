"""Inductor CPU codegen: one fp16 reduction stored to two outputs declares
`auto at_Half_tmp_acc0_vec` twice -> CppCompileError ("conflicting declaration").
Pattern from GPT-NeoX's parallel residual (pythia): the attention and MLP output projections get the same
grad_output, so their bias grads are the same column sum, which Inductor fuses into one reduction with two
stores. fp16, train (backward), dynamic=True, CPU."""
import torch
from torch import nn

print(torch.__version__)


class ParallelResidual(nn.Module):
    def __init__(self, d=768):
        super().__init__()
        self.attn_out = nn.Linear(d, d)
        self.mlp_out = nn.Linear(d, d)

    def forward(self, a, b):
        return self.attn_out(a) + self.mlp_out(b)


def run(dtype, dynamic):
    torch._dynamo.reset()
    m = ParallelResidual().to(dtype)
    c = torch.compile(m, dynamic=dynamic)
    try:
        for seq in (32, 40):
            a = torch.randn(2, seq, 768, dtype=dtype, requires_grad=True)
            b = torch.randn(2, seq, 768, dtype=dtype, requires_grad=True)
            c(a, b).float().pow(2).sum().backward()
        print(f"[{dtype}, dynamic={dynamic}] OK")
    except Exception as e:
        print(f"[{dtype}, dynamic={dynamic}] {type(e).__name__}: {str(e).splitlines()[0][:100]}")


run(torch.float16, True)
run(torch.float16, False)
run(torch.float32, True)
run(torch.bfloat16, True)
