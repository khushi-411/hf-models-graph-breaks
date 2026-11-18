<!-- Draft for: huggingface/transformers. Once filed, put the URL in known_issues.yaml and set status: open -->

# `torch.compile`: DeBERTa (v1) graph-breaks once per layer because `compute_attention_span` returns a tensor

### System info

- transformers: `5.24.0.dev0` (main, git `b7fe69cb19`)
- torch: `2.16.0a0+git57f02f8` (source build)
- Python 3.15, Linux; checked on CPU and on CUDA (B200)

### Description

`modeling_deberta.py:123`:

```python
def compute_attention_span(query_layer, key_layer, max_relative_positions: int):
    return torch.tensor(min(max(query_layer.size(-2), key_layer.size(-2)), max_relative_positions))
```

The value only depends on shapes and a config int, but it is wrapped in a tensor. `disentangled_att_bias`
then slices with it (`modeling_deberta.py:300`):

```python
rel_embeddings = rel_embeddings[
    self.max_relative_positions - att_span : self.max_relative_positions + att_span, :
].unsqueeze(0)
```

Slicing with a 0-d tensor calls `.item()` -> graph break in every layer, in **eval and train**.

These helpers used to be `@torch.jit.script` functions; that was removed on main (torch removed
`torch.jit.script`), and since then Dynamo traces them and the break count went up:
harness (2 layers) deberta `eval_forward` 10 graphs on 5.22.1 -> 13 graphs on main (CPU),
15 graphs on CUDA. `deberta_v2` does not have this problem.

### Reproduction

`issues/repros/deberta_att_span_item.py` (real `DebertaModel`, 2 layers, relative attention on), with a
patched copy of the module where `compute_attention_span` returns a Python int:

```
[as is]           graphs=12 breaks=11 ['Unsupported Tensor.item() call with capture_scalar_outputs=False', ...]
[att_span as int] graphs=1 breaks=0
```

### Possible fix

```python
def compute_attention_span(query_layer, key_layer, max_relative_positions: int) -> int:
    return min(max(query_layer.size(-2), key_layer.size(-2)), max_relative_positions)
```

Check the other former `jit.script` helpers in the file for the same tensor-wrapping.

### Affected

deberta (eval_forward, train_forward), and models that copy `DebertaAttention` (e.g. sew-d).
