"""F.scaled_dot_product_attention(is_causal=<SymBool>) graph-breaks under dynamic shapes.

When `is_causal` is computed from tensor shapes (as HF's sdpa_attention_forward does, e.g.
`is_causal = query.shape[2] > 1 and ...`), it is a SymBool under dynamic=True. Dynamo passes it
straight to the op and fake-tensor propagation fails:

    TypeError: scaled_dot_product_attention(): argument 'is_causal' must be bool, not SymBool

so the frame graph-breaks (gb0112) and, inside HF's attention wrapper, the whole function falls back
to eager (moonshine train_forward: 1 graph static -> 16 graphs dynamic).

Expected: Dynamo guards the SymBool (or bool()s it) for bool schema args and keeps one graph.
"""

import torch
import torch.nn.functional as F


def attn(q, k, v):
    is_causal = q.shape[2] > 1  # SymBool when the sequence dim is dynamic
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


for dynamic in (False, True):
    torch._dynamo.reset()
    q = k = v = torch.randn(2, 4, 16, 8)
    try:
        torch.compile(attn, fullgraph=True, dynamic=dynamic)(q, k, v)
        print(f"dynamic={dynamic}: OK, one graph")
    except Exception as e:
        msg = [l for l in str(e).splitlines() if "SymBool" in l or "TypeError" in l]
        print(f"dynamic={dynamic}: FAILS with fullgraph=True ->", msg[:2])
