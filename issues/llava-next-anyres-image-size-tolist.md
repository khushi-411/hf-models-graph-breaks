<!-- Draft for: huggingface/transformers. Once filed, put the URL in known_issues.yaml and set status: open -->

# `torch.compile`: LLaVA-NeXT anyres tiling converts `image_sizes` to Python with `.tolist()` in three helpers

### System info

- transformers: `5.24.0.dev0` (main, git `b7fe69cb19`)
- torch: `2.16.0a0+git57f02f8` (source build)
- Python 3.15, Linux; checked on CPU and on CUDA (B200)

### Description

`modeling_llava_next.py` chooses the tile layout per image in Python:

```python
# get_anyres_image_grid_shape (:68), image_size_to_num_patches (:97), unpad_image (:129)
if not isinstance(image_size, (list, tuple)):
    ...
    image_size = image_size.tolist()
```

`image_sizes` arrives as a tensor, so each helper does a `.tolist()` (graph break), and the caller can't
resume inside its loop ("Failed to handle graph break gracefully").

Numbers are from the graph-break harness: model built from its config with random weights and reduced layers, one `torch._dynamo.explain` call per scenario. llava_next: 7 graphs / 6 breaks in eval_forward, eval_with_cache and train_forward (CPU and CUDA).

### Possible fix

The helpers already accept lists. Convert `image_sizes` to a list once, outside the compiled region
(processor / `generate`), or accept it as a list in `forward` and only fall back to `.tolist()` when a
tensor is passed.

### Affected

llava_next (and llava_next_video / llava_onevision, which share the anyres helpers).
Similar: `mistral3-image-sizes-tolist`, `apertus1p5-image-sizes-tolist`.
