import torch
def make(batch=2, grid=6, device="cuda"):
    xs = torch.arange(grid, device=device).repeat(grid); ys = torch.arange(grid, device=device).repeat_interleave(grid)
    return torch.randn(batch, grid * grid, 8, device=device), torch.stack([xs, ys], -1)[None].repeat(batch, 1, 1)

def v_k(h, pos, length: int):                 # just k
    return torch.full((1,), int((h.shape[1] // length) ** 0.5), device=h.device)
def v_div(h, pos, length: int):               # floor div by sqrt-derived k
    k = int((h.shape[1] // length) ** 0.5)
    return torch.div(pos.clamp(min=0), k, rounding_mode="floor")
def v_div_const(h, pos, length: int):         # floor div by a literal 3
    return torch.div(pos.clamp(min=0), 3, rounding_mode="floor")
def v_idx(h, pos, length: int):               # full kernel_idxs
    k = int((h.shape[1] // length) ** 0.5)
    c = pos.clamp(min=0); max_x = c[..., 0].max(dim=-1, keepdim=True)[0] + 1
    ki = torch.div(c, k, rounding_mode="floor")
    return ki[..., 0] + (max_x // k) * ki[..., 1]

h, pos = make()
for fn in (v_k, v_div, v_div_const, v_idx):
    torch._dynamo.reset()
    ref = fn(h, pos, 4); out = torch.compile(fn, dynamic=True)(h, pos, 4); torch.cuda.synchronize()
    print(f"{fn.__name__:12s}", "ok" if torch.equal(ref, out) else f"MISMATCH eager={ref.flatten()[:12].tolist()} compiled={out.flatten()[:12].tolist()}")
