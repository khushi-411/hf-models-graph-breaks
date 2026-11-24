"""Duck sizing ties the batch size to num_kv_heads in a KV-cache decode step, so batch gets specialized.

The cache tensor is a graph input of shape (batch, kv_heads, seq, head_dim). On the first call batch == kv_heads == 2,
so duck sizing gives both dims the same symbol s0. The new key is built with kv_heads as a Python constant (from the
config), and torch.cat needs the head dims to match: s0 == 2, which also fixes batch to 2. The next batch size
recompiles, although dynamic=True was asked for. With use_duck_shape=False, or a first batch != kv_heads, it doesn't.
Seen in every Inkling model's generate() (decode graphs), see issues/duck-sizing-specializes-unrelated-dims.md.
"""
import sys
import torch
from torch._dynamo.testing import CompileCounter

KV_HEADS, HEAD_DIM, HIDDEN = 2, 8, 16


def decode_step(cache_k, x, w):
    b = x.shape[0]
    k = (x @ w).view(b, 1, KV_HEADS, HEAD_DIM).transpose(1, 2)   # (b, KV_HEADS, 1, HEAD_DIM)
    return torch.cat([cache_k, k], dim=-2)


def run(first_batch, duck=True):
    torch._dynamo.reset()
    torch.fx.experimental._config.use_duck_shape = duck
    cnt = CompileCounter()
    f = torch.compile(decode_step, backend=cnt, dynamic=True)
    w = torch.randn(HIDDEN, KV_HEADS * HEAD_DIM)
    for b in (first_batch, first_batch + 1, first_batch + 3):
        f(torch.randn(b, KV_HEADS, 5, HEAD_DIM), torch.randn(b, 1, HIDDEN), w)
    return cnt.frame_count


print("first batch == kv_heads, duck sizing  :", run(2), "graphs for 3 batch sizes")   # 2: batch specialized to 2
print("first batch != kv_heads, duck sizing  :", run(3), "graphs")                     # 1
print("first batch == kv_heads, no duck shape:", run(2, duck=False), "graphs")         # 1
