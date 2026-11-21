<!-- Draft for: huggingface/transformers. Once filed, put the URL in known_issues.yaml and set status: open -->

# `torch.compile` graph break in XLM: `assert lengths.max().item() <= slen` input check in forward

### System info

- transformers: `5.24.0.dev0` (main, git `b7fe69cb19`)
- torch: `2.16.0a0+git57f02f8` (source build)
- Python 3.15, Linux; checked on CPU and on CUDA (B200)

### Description

`XLMModel.forward` validates its inputs with a `.item()` (`modeling_xlm.py:793`):

```python
# check inputs
assert lengths.size(0) == bs
assert lengths.max().item() <= slen
```

`lengths` is computed from the input when not passed, so this runs on every call. The `.item()` forces a
host sync and a graph break in **eval and train**.

Harness (config-only model, 2 layers): xlm `eval_forward` 3 graphs / 2 breaks, `train_forward` 3 / 2.

### Reproduction

`issues/repros/xlm_lengths_assert_item.py` (real `XLMModel`, 2 layers, eval):

```
graphs=4 breaks=3
   - Unsupported Tensor.item() call with capture_scalar_outputs=False
```

### Possible fix

Use a compile-friendly check instead of the Python `assert`:

```python
torch._check(lengths.size(0) == bs)
torch._assert_async(lengths.max() <= slen)   # device-side, no host sync
```

or skip it under `is_torchdynamo_compiling()`, like other input checks in the library.

### Affected

xlm (eval_forward, train_forward).
