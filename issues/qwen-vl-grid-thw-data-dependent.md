<!-- Draft for: huggingface/transformers. Once filed, put the URL in known_issues.yaml and set status: open -->

# `torch.compile`: Qwen2/2.5/3-VL-style vision towers and mRoPE index are driven by `grid_thw` values (20-35 graph breaks per forward)

### System info

- transformers: `5.24.0.dev0` (main, git `b7fe69cb19`)
- torch: `2.16.0a0+git57f02f8` (source build)
- Python 3.15, Linux; checked on CPU and on CUDA (B200)

### Description

The vision tower and the multimodal RoPE index read image grid sizes out of the `image_grid_thw` tensor
in Python, in many places. The main ones, shared through `transformers/vision_utils.py`:

```python
# vision_utils.py:112  (rotary position ids for the vision tower)
for (t, h, w), merge_size in zip(grid_thw.tolist(), spatial_merge_size.tolist()):
    ...
# vision_utils.py:64   (cu_seqlens)
seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
```

plus model-specific ones, e.g. hy_v4_vl `scale_h = (int(height) + 0.1) / self.position_edge` (`:795`) and
the per-sample `current_input_ids[attention_mask[batch_idx].bool()]` in the mRoPE index (`:1360`).
Each `.tolist()` / `int(tensor)` / data-dependent output shape is a graph break.

Numbers are from the graph-break harness: model built from its config with random weights and reduced layers, one `torch._dynamo.explain` call per scenario.

| model | eval_forward | eval_with_cache | train_forward |
|---|---|---|---|
| qwen2_vl | 25 graphs / 24 breaks | 25 / 24 | 25 / 24 |
| qwen2_5_vl | 25 / 24 | 25 / 24 | 25 / 24 |
| qwen3_vl | 36 / 35 | 37 / 36 | 37 / 36 |
| hy_v4_vl | 23 / 22 | 23 / 22 | 23 / 22 |
| inkling2_vlm | 26 / 25 | 26 / 25 | 26 / 25 |
| dots3_note_vlm | 19 / 18 | 19 / 18 | 19 / 18 |
| lfm3_vl | 13 / 12 | 13 / 12 | 13 / 12 |

(Same on CUDA.) Turning on `capture_scalar_outputs` + `capture_dynamic_output_shape_ops` does not help: it
turns the breaks into a hard `GuardOnDataDependentSymNode` error (PyTorch side: `tolist-unbacked-length`).

### Possible fix

The grid sizes are known on the host before the forward (the processor computes them). Options:

- Accept precomputed `cu_seqlens`, rotary position ids and mRoPE position ids through kwargs (part of
  this already exists), and compute them in the processor / generation utilities instead of in forward.
- Accept `image_grid_thw` as a Python list of tuples as well as a tensor, and use it as-is.

### Affected

qwen2_vl, qwen2_5_vl, qwen3_vl, hy_v4_vl, lfm3_vl, inkling2, dots3_note (and other users of
`vision_utils`). Related: `vlm-split-image-features-tolist`, `vision-attn-split-lengths-tolist`.
