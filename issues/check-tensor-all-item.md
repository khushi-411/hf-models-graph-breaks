<!-- Draft for: pytorch/pytorch. Once filed, put the URL in known_issues.yaml and set status: open -->

# `torch._check_tensor_all` causes a graph break under `torch.compile` (calls `.item()` internally)

### System info

- torch: `2.16.0a0+git9933f27` (source build)
- transformers: `5.22.1`
- Python 3.15, Linux, CPU

### Description

`torch._check_tensor_all(cond)` is meant to work as a compiler-friendly assertion, like `torch._check`. Under Dynamo
it traces into the Python implementation, which does `cond._is_all_true().item()`. With
the default `capture_scalar_outputs=False`, that `.item()` is unsupported, so every call
causes a graph break. `torch._check` on a Python bool or SymBool compiles without one.

Hugging Face transformers runs into this through `torch_compilable_check` (`utils/import_utils.py`). Under tracing
that helper calls `torch._check_tensor_all(cond)` for tensor conditions. LLaVA-style
VLMs use it to validate the image placeholder count in `get_placeholder_mask`, so every
compiled forward with images breaks there.

| model | graphs today | graphs with the check not breaking |
|---|---|---|
| llava | 4 | **1** |
| gemma3 (VLM) | 4 | **1** |
| internvl | 9 | 7 |
| mistral3 | 14 | 12 |
| smolvlm (idefics3) | 13 | 10 |
| qwen2_5_vl | 25 | 22 |
| qwen3_vl | 36 | 33 |

The right column was measured with `TRANSFORMERS_DISABLE_TORCH_CHECK=1`, which skips the
check entirely. Setting `capture_scalar_outputs=True` also gives 1 graph for llava and gemma3.

### Reproduction

```python
import torch

def f(image_features, image_token_mask):
    torch._check_tensor_all(image_token_mask.sum() * image_features.shape[-1] == image_features.numel())
    return image_features * 2

e = torch._dynamo.explain(f)(torch.randn(4, 8), torch.ones(4, dtype=torch.bool))
print(e.graph_count, e.graph_break_count)  # 2 1
# Unsupported Tensor.item() call with capture_scalar_outputs=False
```

Full script: [repros/check_tensor_all_item.py](repros/check_tensor_all_item.py).

### Expected behavior

No graph break. The check should become a runtime assert in the graph (e.g.
`aten._assert_async.msg`), or be applied to the unbacked value when it's available.

### Possible fix

Give `torch._check_tensor_all` / `torch._check_tensor_all_with` a Dynamo handler that
emits `torch._assert_async(cond.all(), msg)` into the graph rather than tracing the
`.item()` path. Alternatively, lower it to `torch._check(cond.all().item())` only when
`capture_scalar_outputs` is on.
