<!-- Draft for: pytorch/pytorch. Once filed, put the URL in known_issues.yaml and set status: open -->

# [dynamic shapes] Meta for `_scaled_dot_product_efficient_attention_backward` branches on `attn_bias.size(-1) % 16`, so train mode recompiles when seq_len crosses a multiple of 16

### System info

- torch: `2.16.0a0+git57f02f8` (source build, CUDA 12.8). The same code is still on main (`41a828ca634`).
- transformers: `5.24.0.dev0` (git `b7fe69cb19`)
- Python 3.15, Linux, CUDA (B200)

### Description

The meta kernel for the memory-efficient SDPA backward pads the bias gradient's last dim to a
multiple of 16, using a Python conditional on a symbolic size (`torch/_meta_registrations.py:6981`):

```python
if attn_bias is not None and grad_input_mask[3]:
    lastDim = attn_bias.size(-1)
    lastDimAligned = lastDim if lastDim % 16 == 0 else lastDim + 16 - lastDim % 16
```

With `dynamic=True`, `lastDim % 16 == 0` installs `Eq(Mod(s, 16), 0)` (or `Ne(...)`). Every time the
key length switches between "multiple of 16" and "not a multiple of 16", the model recompiles. The
same line is in `meta__efficient_attention_backward` (`:7336`).

This only happens when SDPA gets an `attn_mask` that needs a gradient and picks the efficient kernel,
i.e. CUDA training. Hugging Face T5-family models pass the learned relative position bias as the mask.

`torch.compile(dynamic=True)`, Inductor, CUDA, `train_forward`, seq 32 / 40 / 56. The recompile reason
is exactly this guard:

```
- 0/0: (kwargs['input_ids'].size()[1] % 16) == 0  # (_meta_registrations.py:6981 in meta__scaled_dot_product_efficient_backward)
```

| model | graphs compiled per size (32 / 40 / 56) |
|---|---|
| t5 | 2 / 2 / 0 |
| mt5 | 2 / 2 / 0 |
| inkling | 2 / 1 / 0 |

Eval and CPU don't recompile.

### Reproduction (sketch, not yet run standalone)

```python
import torch
import torch.nn.functional as F

def f(q, k, v, bias):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias)

cf = torch.compile(f, dynamic=True)
for n in (32, 40, 56):
    q, k, v = (torch.randn(2, 4, n, 64, device="cuda", requires_grad=True) for _ in range(3))
    bias = torch.randn(1, 4, n, n, device="cuda", requires_grad=True)
    cf(q, k, v, bias).sum().backward()
# TORCH_LOGS=recompiles: (... % 16) == 0 from meta__scaled_dot_product_efficient_backward
```

### Possible fix

Round up without branching. The value is the same in both cases:

```python
lastDimAligned = ((lastDim + 15) // 16) * 16
```

Apply it in both meta functions.

<!-- affected-models:manual -->
### Affected models

All runs use `torch.compile(backend="inductor", dynamic=True)` over (batch, seq, audio samples) =
(2, 32, 16000), (3, 40, 17600), (4, 56, 19200), on a model built from the checkpoint's config with reduced
layers. "Graphs per size" is the number of graphs compiled on the 1st / 2nd / 3rd size (anything after the
1st size is a recompile).

| model | config from | scenario · device | graphs per size |
|---|---|---|---|
| t5 | `google-t5/t5-small` | train · CUDA | 2 / 2 / 0 |
| mt5 | `google/mt5-small` | train · CUDA | 2 / 2 / 0 |
| inkling | `thinkingmachines/Inkling` | train · CUDA | 2 / 1 / 0 |
| beit | `microsoft/beit-base-patch16-224` | train · CUDA | 2 / 0 / 0 (guard installed, no recompile) |
| swin | `microsoft/swin-tiny-patch4-window7-224` | train · CUDA | 2 / 0 / 0 (guard installed, no recompile) |

beit and swin also install this guard, but it didn't fail across the three sizes. Their bias's last dim
looks tied to the patch / window grid rather than the sequence length (not verified). Eval and CPU runs of all
five models don't reach this meta function.
