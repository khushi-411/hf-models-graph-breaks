<!-- Draft for: huggingface/transformers. Once filed, put the URL in known_issues.yaml and set status: open -->

# `torch.compile`: vision attention (non-flash path) splits q/k/v per image with `lengths.tolist()`

### System info

- transformers: `5.24.0.dev0` (main, git `b7fe69cb19`)
- torch: `2.16.0a0+git57f02f8` (source build)
- Python 3.15, Linux; checked on CPU and on CUDA (B200)

### Description

The vision towers of qwen2_vl (`modeling_qwen2_vl.py:402`), hy_v4_vl (`:927`), inkling2 (`:1027`) and
dots3_note (`:1413`) handle variable-length images in the eager / SDPA path like this:

```python
# Other implementations: Process each chunk separately
lengths = cu_seqlens[1:] - cu_seqlens[:-1]
splits = [
    torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
]
```

`.tolist()` is a graph break (`Data dependent operator`) and the list comprehension around it can't be
resumed (`comprehension_graph_break`), in **every vision layer**.

### Reproduction

`issues/repros/vision_attn_split_lengths_tolist.py` (3 images of 16/24/24 tokens):

```
[split(lengths.tolist())] graphs=2 breaks=1
    - comprehension_graph_break
[block-diagonal mask]     graphs=1 breaks=0
outputs match
```

### Possible fix

Build a block-diagonal attention mask from `cu_seqlens` and run one SDPA call:

```python
pos = torch.arange(seq_len, device=...)
seg = torch.searchsorted(cu_seqlens[1:], pos, right=True)
mask = seg[:, None] == seg[None, :]
attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
```

This gives the same result (checked in the repro). The mask is O(seq²), so keep the flash / varlen path
for long sequences.

### Affected

qwen2_vl, hy_v4_vl, inkling2, dots3_note (and other copies of this attention).
