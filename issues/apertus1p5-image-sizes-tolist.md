<!-- Draft for: huggingface/transformers. Once filed, put the URL in known_issues.yaml and set status: open -->

# `torch.compile`: Apertus 1.5 image tokenizer loops over `image_sizes.tolist()` and embeds tokens per image

### System info

- transformers: `5.24.0.dev0` (main, git `b7fe69cb19`)
- torch: `2.16.0a0+git57f02f8` (source build)
- Python 3.15, Linux; checked on CPU and on CUDA (B200)

### Description

`modeling_apertus1p5.py:814`:

```python
image_tokens = []
for image, (height, width) in zip(pixel_values, image_sizes.tolist()):
    codes = self.vision_tokenizer.encode(image[None, :, :height, :width]).image_tokens
    ...
```

and `:829`:

```python
image_tokens = self.get_image_tokens(pixel_values, image_sizes)
image_features = tuple(self.get_input_embeddings()(tokens) for tokens in image_tokens)
```

`.tolist()` is a graph break, and the generator expression after it can't be resumed ("Failed to handle
graph break gracefully"), so the whole image path runs eagerly.

Numbers are from the graph-break harness: model built from its config with random weights and reduced layers, one `torch._dynamo.explain` call per scenario. apertus1p5_vlm: 7 graphs / 6 breaks in eval_forward, eval_with_cache and train_forward.

### Possible fix

Accept `image_sizes` as a Python list (it comes from the processor), and embed the concatenated token ids
once instead of per image.

### Affected

apertus1p5 (VLM). Similar: `mistral3-image-sizes-tolist`, `llava-next-anyres-image-size-tolist`.
