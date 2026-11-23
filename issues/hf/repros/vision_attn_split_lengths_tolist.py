"""Vision attention eager/SDPA path (qwen2_vl, hy_v4_vl, inkling2, dots3_note):
`torch.split(t, (cu_seqlens[1:] - cu_seqlens[:-1]).tolist(), dim=2)` per image, then attention per
chunk. A block-diagonal mask built from cu_seqlens gives the same result in one SDPA call."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import torch
import torch.nn.functional as F
from _pattern import report

cu_seqlens = torch.tensor([0, 16, 40, 64])
q = torch.randn(1, 4, 64, 8); k = torch.randn(1, 4, 64, 8); v = torch.randn(1, 4, 64, 8)


def original(q, k, v, cu_seqlens):
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    splits = [torch.split(t, lengths.tolist(), dim=2) for t in (q, k, v)]
    outs = [F.scaled_dot_product_attention(a, b, c) for a, b, c in zip(*splits)]
    return torch.cat(outs, dim=2)


def fixed(q, k, v, cu_seqlens):
    pos = torch.arange(q.shape[2])
    seg = torch.searchsorted(cu_seqlens[1:], pos, right=True)     # segment id per token
    mask = seg[:, None] == seg[None, :]                          # block-diagonal
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


report("split(lengths.tolist())", original, q, k, v, cu_seqlens)
report("block-diagonal mask", fixed, q, k, v, cu_seqlens)
torch.testing.assert_close(original(q, k, v, cu_seqlens), fixed(q, k, v, cu_seqlens))
print("outputs match")
