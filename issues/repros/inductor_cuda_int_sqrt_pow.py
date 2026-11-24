"""int(s ** 0.5) on a dynamic size is wrong under torch.compile on CUDA (Inductor: trunc(libdevice.pow(s, 0.5)))."""
import torch

def f(x):
    return torch.full((1,), int(x.shape[0] ** 0.5), device=x.device)

for device in ("cpu", "cuda"):
    torch._dynamo.reset()
    g = torch.compile(f, dynamic=True)
    bad = []
    for n in range(2, 200):
        x = torch.empty(n * n, device=device)
        if g(x).item() != n:
            bad.append(n * n)
    print(device, "wrong for perfect squares:", bad[:15], f"({len(bad)} of 198)")
