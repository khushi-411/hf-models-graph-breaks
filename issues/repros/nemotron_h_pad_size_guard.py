"""NemotronH `mamba2_chunk_scan` (modeling_nemotron_h.py) always pads the sequence to a multiple of the
chunk size, but only cuts the padding off again under `if pad_size > 0:`. With dynamic shapes that branch
installs a guard on `seq_len % chunk_size` and recompiles when it changes. Slicing unconditionally
(`output[:, :seq_len]` is a no-op without padding) removes the guard."""
import torch
import torch.nn.functional as F

CHUNK = 16


def chunk_scan(x, conditional_slice):
    seq_len = x.shape[1]
    pad_size = (CHUNK - seq_len % CHUNK) % CHUNK
    x = F.pad(x, (0, 0, 0, pad_size))                          # always padded, as in the model
    out = x.view(x.shape[0], -1, CHUNK, x.shape[-1]).cumsum(2).reshape(x.shape)
    if conditional_slice:
        if pad_size > 0:                                       # modeling_nemotron_h.py:372
            out = out[:, :seq_len]
    else:
        out = out[:, :seq_len]
    return out


for name, cond in [("if pad_size > 0: slice", True), ("always slice", False)]:
    torch._dynamo.reset()
    n = []
    c = torch.compile(lambda x: chunk_scan(x, cond), backend=lambda gm, ex: (n.append(1), gm.forward)[1], dynamic=True)
    for seq in (32, 40, 56):            # 32 needs no padding, 40 and 56 do
        x = torch.randn(2, seq, 8)
        torch.testing.assert_close(c(x), chunk_scan(x, cond))
    print(f"[{name}] compiled graphs over seq 32/40/56: {len(n)}")
