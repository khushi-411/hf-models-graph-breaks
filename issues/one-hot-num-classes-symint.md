<!-- Draft for: pytorch/pytorch. Once filed, put the URL in known_issues.yaml and set status: open -->

# [dynamic shapes] `F.one_hot(x, num_classes)` specializes a symbolic `num_classes` (schema takes `int`, not `SymInt`)

### System info

- torch: `2.16.0a0+git57f02f8` (source build). The schema is unchanged on main (`41a828ca634`).
- transformers: `5.24.0.dev0` (git `b7fe69cb19`)
- Python 3.15, Linux; CPU and CUDA (B200)

### Description

The schema is `one_hot(Tensor self, int num_classes=-1) -> Tensor` (`native_functions.yaml:6141`), and the
C++ kernel takes `int64_t num_classes` (`aten/src/ATen/native/Onehot.cpp`). When `num_classes` is derived
from a dynamic size, tracing the call converts the SymInt to a plain int and installs
`Eq(<expr>, <hint>)`. The output size then becomes static, and every new value recompiles.

The Python decomposition (`torch/_decomp/decompositions.py`, `aten.one_hot.default.py_impl`) already
works with a symbolic `num_classes` (`torch.arange(num_classes)` + comparison). Only the `int` in the
schema forces specialization.

LongT5 (`modeling_longt5.py:246`, `_create_global_aggregates`):

```python
one_hot_block_ids = nn.functional.one_hot(block_ids.type(torch.int64), global_seq_len + 1)[:, :, :-1]
```

where `global_seq_len = seq_len // global_block_size` (16). `torch.compile(dynamic=True)`, Inductor,
seq 32 / 40 / 56. The recompile reason:

```
- 0/0: (1 + (kwargs['input_ids'].size()[1] // 16)) == 3  # one_hot_block_ids = nn.functional.one_hot(...)  # modeling_longt5.py:246
```

Graphs compiled per size: 2 / 0 / 2 in eval and train, on CPU and on CUDA. 32 and 40 both give
`global_seq_len + 1 == 3`, and 56 gives 4, so it recompiles.

### Reproduction (sketch, not yet run standalone)

```python
import torch
import torch.nn.functional as F

def f(x):
    ids = torch.arange(x.shape[1]) // 16
    return F.one_hot(ids, x.shape[1] // 16 + 1).float().sum()

cf = torch.compile(f, dynamic=True)
for n in (32, 40, 56):
    cf(torch.randn(2, n))
# TORCH_LOGS=recompiles: Eq((s0//16) + 1, 3)
```

### Possible fix

Change the schema to `SymInt num_classes=-1` (and `c10::SymInt` in `Onehot.cpp`), so a symbolic class count
reaches the existing decomposition. `-1` (data-dependent) keeps its current handling.

Until then, LongT5 can avoid it on its side with the same math the decomposition does:
`(block_ids[..., None] == torch.arange(global_seq_len, device=block_ids.device)).to(hidden_states.dtype)`.
Negative `block_ids` (padding) never match, so the extra class and the `[:, :, :-1]` slice aren't needed.

<!-- affected-models:manual -->
### Affected models

All runs use `torch.compile(backend="inductor", dynamic=True)` over (batch, seq, audio samples) =
(2, 32, 16000), (3, 40, 17600), (4, 56, 19200), on a model built from the checkpoint's config with reduced
layers. "Graphs per size" is the number of graphs compiled on the 1st / 2nd / 3rd size (anything after the
1st size is a recompile).

| model | config from | scenario · device | graphs per size |
|---|---|---|---|
| long_t5 | `google/long-t5-tglobal-base` | eval · CPU | 2 / 0 / 2 |
| long_t5 | | train · CPU | 2 / 0 / 2 |
| long_t5 | | eval · CUDA | 2 / 0 / 2 |
| long_t5 | | train · CUDA | 2 / 0 / 2 |

No other model in the sweep calls `F.one_hot` with a size-derived `num_classes`.
