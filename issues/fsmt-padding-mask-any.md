<!-- Draft for: huggingface/transformers. Once filed, put the URL in known_issues.yaml and set status: open -->

# `torch.compile` graph break in FSMT `make_padding_mask`: `if not padding_mask.any(): padding_mask = None`

### System info

- transformers: `5.24.0.dev0` (main, git `b7fe69cb19`)
- torch: `2.16.0a0+git57f02f8` (source build)
- Python 3.15, Linux; checked on CPU and on CUDA (B200)

### Description

`modeling_fsmt.py:263`:

```python
def make_padding_mask(input_ids, padding_idx=1):
    """True for pad tokens"""
    padding_mask = input_ids.eq(padding_idx)
    if not padding_mask.any():
        padding_mask = None
    return padding_mask
```

The `if` branches on a tensor (graph break), in **eval and train**. Returning `None` when there is no
padding is only an optimization: an all-False mask masks nothing.

Harness (config-only, reduced layers): fsmt `eval_forward` 3 graphs / 2 breaks, `train_forward` 5 / 4.

### Reproduction

`issues/repros/fsmt_padding_mask_any.py`:

```
[original]           graphs=2 breaks=1
    - generic_jump TensorVariable()
[always return mask] graphs=1 breaks=0
outputs match
```

### Possible fix

Always return the mask (`return input_ids.eq(padding_idx)`), or keep the shortcut only outside of
compilation.

### Affected

fsmt.
