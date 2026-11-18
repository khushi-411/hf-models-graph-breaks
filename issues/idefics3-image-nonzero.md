<!-- Draft for: huggingface/transformers. Once filed, put the URL in known_issues.yaml and set status: open -->

# `torch.compile`: Idefics3 / SmolVLM filter padding images with boolean indexing and split with `.tolist()`

### System info

- transformers: `5.24.0.dev0` (main, git `b7fe69cb19`)
- torch: `2.16.0a0+git57f02f8` (source build)
- Python 3.15, Linux; checked on CPU and on CUDA (B200)

### Description

`Idefics3Model.get_image_features` (`modeling_idefics3.py:582`) drops all-zero padding images and then
splits the features per sample:

```python
nb_values_per_image = pixel_values.shape[1:].numel()
real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image
num_real_images_per_batch = real_images_inds.view(batch_size, num_images).sum(dim=1)
pixel_values = pixel_values[real_images_inds].contiguous()          # :585 Dynamic shape operator
...
image_outputs.pooler_output = [
    f.reshape(-1, f.shape[-1]) for f in image_features.split(num_real_images_per_batch.tolist())   # :613
]
```

Both lines are graph breaks, and they run on every forward with images, even when no image is padding.

Numbers are from the graph-break harness: model built from its config with random weights and reduced layers, one `torch._dynamo.explain` call per scenario. smolvlm: 9 graphs / 8 breaks in eval_forward, eval_with_cache and train_forward (CPU and CUDA).
With `capture_scalar_outputs` + `capture_dynamic_output_shape_ops` it compiles to 1 graph.

### Possible fix

- Let the processor pass which images are real (it creates the padding), or pass a
  `pixel_attention_mask`-style per-image flag, and skip the boolean filter when all images are real.
- Keep the concatenated features for the LM path instead of splitting with `.tolist()` (same as
  `vlm-split-image-features-tolist`).

### Affected

smolvlm (Idefics3 architecture), idefics3. Related: `granite-docling-tile-mask-index`.
