<!-- Draft for: huggingface/transformers. Once filed, put the URL in known_issues.yaml and set status: open -->

# `torch.compile`: GraniteDocling selects real tiles with `pixel_values[pixel_attention_mask]` and splits with `.tolist()`

### System info

- transformers: `5.24.0.dev0` (main, git `b7fe69cb19`)
- torch: `2.16.0a0+git57f02f8` (source build)
- Python 3.15, Linux; checked on CPU and on CUDA (B200)

### Description

`modeling_granite_docling.py:1074`:

```python
tiles_per_sample = pixel_attention_mask.sum(-1)
image_outputs = self.vision_model(
    pixel_values=pixel_values[pixel_attention_mask].to(self.dtype), fine=fine, return_dict=True, **kwargs
)                                                          # Dynamic shape operator
tokens_per_tile = self.image_seq_len * 4 if fine else self.image_seq_len
split_sizes = (tiles_per_sample * tokens_per_tile).tolist()   # Data dependent operator
image_outputs.pooler_output = torch.split(image_outputs.pooler_output, split_sizes)
```

Boolean-mask indexing gives a data-dependent shape, and `.tolist()` is a graph break. With all tiles real
(the usual case for single-page inputs, and the harness input) the mask selects everything.

Numbers are from the graph-break harness: model built from its config with random weights and reduced layers, one `torch._dynamo.explain` call per scenario. granite_docling: 9 graphs / 8 breaks in eval_forward, eval_with_cache and train_forward.

### Possible fix

Skip the boolean indexing when `pixel_attention_mask.all()` is known on the host (the processor can pass
it), and keep the concatenated features for the LM path (see `vlm-split-image-features-tolist`).

### Affected

granite_docling. Similar: `idefics3-image-nonzero`.
