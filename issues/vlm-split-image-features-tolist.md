<!-- Draft for: huggingface/transformers. Once filed, put the URL in known_issues.yaml and set status: open -->

# `torch.compile`: VLM `get_image_features` splits features with `(grid_thw.prod(-1) // merge**2).tolist()`, only for the LM path to concatenate them again

### System info

- transformers: `5.24.0.dev0` (main, git `b7fe69cb19`)
- torch: `2.16.0a0+git57f02f8` (source build)
- Python 3.15, Linux; checked on CPU and on CUDA (B200)

### Description

Several VLMs return per-image feature tuples from `get_image_features`:

```python
# lfm3_vl (modeling_lfm3_vl.py:558), dots3_note (:1594) are the same
split_sizes = (image_grid_thw.prod(-1) // self.vision_tower.spatial_merge_size**2).tolist()
image_outputs.pooler_output = torch.split(image_features, split_sizes)
```

Same pattern: hy_v4_vl (`:1423`), granite_docling (`:1080`), inkling2 (`:1490`, which also moves
`image_grid_thw` to the CPU first), qwen2_vl.

`.tolist()` on a tensor is a graph break (`Data dependent operator`) on every forward with images. The
model's own forward then does `torch.cat(image_features, dim=0)` right away, so the split is not needed on
that path; it only exists for the public `get_image_features` return type.

### Reproduction

`issues/repros/vlm_split_image_features_tolist.py`:

```
[split(...tolist()) then cat] graphs=2 breaks=1
    - Data dependent operator
[no split]                    graphs=1 breaks=0
```

### Possible fix

Have the internal path use the concatenated features directly (e.g. a private `_get_image_features` that
returns the flat tensor, or a `split=False` flag), and only split for the public API.

### Affected

hy_v4_vl, granite_docling, lfm3_vl, inkling2, dots3_note, qwen2_vl. Related:
`qwen-vl-grid-thw-data-dependent`, `vision-attn-split-lengths-tolist`.
