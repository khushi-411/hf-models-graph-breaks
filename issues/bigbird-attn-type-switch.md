<!-- Draft for: huggingface/transformers. Once filed, put the URL in known_issues.yaml and set status: open -->

# `torch.compile`: BigBird switches `block_sparse` -> `original_full` inside forward for short inputs (logger.warning + new nn.Linear modules)

### System info

- transformers: `5.24.0.dev0` (main, git `b7fe69cb19`)
- torch: `2.16.0a0+git57f02f8` (source build)
- Python 3.15, Linux; checked on CPU and on CUDA (B200)

### Description

When the input is too short for block-sparse attention, `BigBirdModel.forward` changes the attention
implementation on the fly (`modeling_big_bird.py:1580`):

```python
if self.attention_type == "block_sparse" and seq_length <= max_tokens_to_attend:
    # change attention_type from block_sparse to original_full
    sequence_length = input_ids.size(1) if input_ids is not None else inputs_embeds.size(1)
    logger.warning(
        "Attention type 'block_sparse' is not possible if sequence_length: ..."
    )
    self.set_attention_type("original_full")
```

`set_attention_type` builds new `BigBirdSelfAttention` modules (with new `nn.Linear`s) inside forward.
Dynamo graph-breaks on the `logger.warning`, on the `nn.Parameter()` constructor (by design) and on the
uninitialized `nn.Module`:

```
modeling_big_bird.py:1583  logging.Logger method not supported for non-export cases
torch/nn/modules/linear.py:108  Attempted to use `torch.nn.Parameter()` constructor with Dynamo
modeling_big_bird.py:133   Uninitialized nn.Module
```

Harness (config-only, 2 layers, 32 tokens): 3 graphs / 2 breaks in eval and train. The harness input is
short, so this is partly a test artifact, but any user with short inputs and the default
`attention_type="block_sparse"` hits it (the first compiled call also mutates the model). The block-sparse
path itself is not covered by this run.

### Possible fix

- Decide the attention type before compiling (e.g. in `__init__` / `from_pretrained` with an explicit
  `attention_type`), and in forward only fall back by *calling* the full-attention path, without building
  modules or reassigning `self.attention_type`.
- Use `logger.warning_once` (or skip the warning while compiling).

### Affected

big_bird, bigbird_pegasus.
