import torch
print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.get_arch_list())
G, M, K, N = 4, 64, 128, 256
for dt in (torch.float32, torch.bfloat16):
    a = torch.randn(M * G, K, device="cuda", dtype=dt)
    b = torch.randn(G, K, N, device="cuda", dtype=dt)
    offs = torch.arange(M, M * G + 1, M, device="cuda", dtype=torch.int32)
    try:
        out = torch._grouped_mm(a, b.transpose(-2, -1).contiguous().transpose(-2, -1), offs=offs)
        torch.cuda.synchronize()
        ref = torch.cat([a[i*M:(i+1)*M].float() @ b[i].float() for i in range(G)])
        print(dt, "ok, max err", (out.float() - ref).abs().max().item())
    except Exception as e:
        print(dt, "FAILED:", type(e).__name__, str(e).splitlines()[0]); break
