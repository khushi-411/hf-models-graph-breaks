<!-- Draft for: huggingface/transformers. Once filed, put the URL in known_issues.yaml and set status: open -->

# DETR-family backbones interpolate `pixel_mask[None]`, treating the batch as channels (recompiles on CUDA when batch crosses 3→4)

### System info

- transformers: `5.24.0.dev0` (main, git `b7fe69cb19`)
- torch: `2.16.0a0+git57f02f8` (source build, CUDA 12.8)
- Python 3.15, Linux, CUDA (B200)

### Description

The conv backbone wrappers downsample the pixel mask like this (`modeling_rt_detr.py:425`):

```python
mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
```

`pixel_mask` is `(batch, H, W)`. `[None]` makes it `(1, batch, H, W)`, so `interpolate` sees the
**batch as the channel dim**. The result is correct, but on CUDA the `_upsample_nearest` decomposition picks
a memory format based on the channel count (`torch/_decomp/decompositions.py:3902`):

```python
if result.ndim == 4 and input.device.type == "cuda" and n_channels < 4:
    memory_format = torch.contiguous_format
```

With `dynamic=True`, that installs `batch <= 3` on the batch size. Batches 2 and 3 share one graph, and
batch 4 recompiles:

```
- 0/0: 2 <= kwargs['pixel_values'].size()[0] <= 3  # ... mask = nn.functional.interpolate(pixel_mask[None].float(), ...)  # modeling_rt_detr.py:425 in forward (_decomp/decompositions.py:3902 in _upsample_nearest)
```

rt_detr `eval_forward`, CUDA, batch 2 / 3 / 4: graphs per size 2 / 0 / 1. CPU: 2 / 0 / 0.

### Possible fix

Put the batch in the batch dim:

```python
mask = nn.functional.interpolate(pixel_mask[:, None].float(), size=feature_map.shape[-2:]).to(torch.bool)[:, 0]
```

The output is identical (nearest interpolation is per-channel), there's one channel, and nothing branches
on the batch size.

<!-- affected-models:manual -->
### Affected models

All runs use `torch.compile(backend="inductor", dynamic=True)` over batch 2 / 3 / 4, on a model built
from the checkpoint's config with reduced layers. "Graphs per size" is the number of graphs compiled for
the 1st / 2nd / 3rd batch size (anything after the 1st is a recompile).

| model | config from | scenario · device | graphs per size |
|---|---|---|---|
| rt_detr | `PekingU/rtdetr_r50vd` | eval · CUDA | 2 / 0 / 1 |
| rt_detr | | eval · CPU | 2 / 0 / 0 (no guard: the branch is CUDA-only) |

rt_detr is the only DETR-family model in the sweep. Same line, not run: conditional_detr, d_fine, dab_detr,
deformable_detr, detr, grounding_dino, lw_detr, mm_grounding_dino, pp_doclayout_v2/v3/v4, rf_detr,
rt_detr_v2, table_transformer.

The PyTorch-side heuristic (`n_channels < 4`) could also use `guard_or_false`/`statically_known_*` so a
symbolic channel count doesn't guard. That's a separate, smaller change.
