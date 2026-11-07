<!-- Draft for: pytorch/pytorch. Discussion/feature request rather than a bug. -->

# A graph break inside a `for` loop sends the whole frame to eager, so HF decoder stacks lose compilation for any in-loop break

### System info

- torch: `2.16.0a0+git9933f27` (source build)
- transformers: `5.22.1`
- Python 3.15, Linux

### Description

When a graph break happens inside a `for`/`while` loop, Dynamo raises `graph break in
loop` (gb7000). It then skips the whole frame, including the code before and after the
loop. `nested_graph_breaks=True` doesn't change this.

In transformers almost every encoder and decoder runs its layers as
`for layer in self.layers:` inside `forward`. So any break inside that loop, even a cheap
one, makes the whole decoder `forward` run in eager: embeddings, mask creation, final
norm, and the Python glue between layers. Each layer is then compiled as a separate frame,
with no fusion across layers and a guard check per layer per step. Breaks we hit inside
layer loops:

- LayerDrop: `if torch.rand([]) < self.layerdrop: continue`, in train mode, across 43
  model families. We hit it in bart, mbart, marian, pegasus, pegasus_x, blenderbot,
  m2m_100, opt, xglm, biogpt, led, whisper, wav2vec2, hubert, wavlm and fsmt.
- Longformer/LED: a per-layer `.item()` on the global-attention flag.

The HF breaks themselves should be fixed in HF. This issue is about how much a single
in-loop break costs: one data-dependent `if` in the loop body costs the whole frame.

### Reproduction

```python
import torch, torch.nn as nn

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])
    def forward(self, x):
        x = x.sin()
        for layer in self.layers:
            torch._dynamo.graph_break()
            x = layer(x)
        return x.cos()

graphs = []
def backend(gm, _): graphs.append(gm); return gm
torch.compile(Decoder(), backend=backend)(torch.randn(2, 8))
print(len(graphs))  # 0: sin/cos aren't compiled either
```

Full script: [repros/graph_break_in_loop.py](repros/graph_break_in_loop.py). On OPT in
train mode, `TORCH_LOGS=graph_breaks` shows the chain: the data-dependent branch at
`modeling_opt.py:378`, then `graph break in loop`, then `frame skipped: forward
(modeling_opt.py line 318)`.

### Expected behavior

Ideally, compile the code before the loop, run the loop with a resume point at the
break, and compile after the loop. At minimum, compile the prefix before the loop
rather than skipping the frame. Unrolled loops over an `nn.ModuleList` have a known trip
count, so one option is to emit a resume function per unrolled iteration.

### Notes

On an older build (`41b037b`), the same models showed a follow-on `Reconstruction failure`
for `DictItemsIterator`/`DictIterator` in transformers' `output_capturing.py` wrapper.
On `9933f27` that has been replaced by this loop fallback.
