<!-- Draft for: huggingface/transformers. Once filed, put the URL in known_issues.yaml and set status: open -->

# `torch.compile`: LeViT fills `attention_bias_cache` (a dict) on the first eval forward, so the second call always recompiles

### System info

- transformers: `5.24.0.dev0` (main, git `b7fe69cb19`)
- torch: `2.16.0a0+git57f02f8` (source build)
- Python 3.15, Linux; CPU and CUDA (B200)

### Description

`LevitAttention` and `LevitAttentionSubsample` cache the gathered attention biases per device in eval mode
(`modeling_levit.py:192` and `:271`):

```python
def get_attention_biases(self, device):
    if self.training:
        return self.attention_biases[:, self.attention_bias_idxs]
    else:
        device_key = str(device)
        if device_key not in self.attention_bias_cache:
            self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
        return self.attention_bias_cache[device_key]
```

The first compiled call guards on the dict being empty and then mutates it, so the second call fails that
guard and recompiles:

```
- 0/0: not ___dict_contains('cpu', self._modules['levit']._modules['encoder']._modules['stages']._modules['0']._modules['layers']._modules['0']._modules['module'].attention_bias_cache)  # if device_key not in self.attention_bias_cache
```

`eval_forward`, `torch.compile(dynamic=True)`, CPU and CUDA: graphs per call 2 / 2 / 0. The second call
recompiles everything once, then it's stable. Train mode doesn't use the cache (2 / 0 / 0). It isn't
shape-related: a static compile hits it too, and so does any user who compiles and calls the model twice.
The cache also holds a tensor that's a view of a parameter, so it goes stale if the weights change after
the first eval call.

### Possible fix

Skip the cache under compile, where Inductor fuses the gather anyway:

```python
if self.training or torch.compiler.is_compiling():
    return self.attention_biases[:, self.attention_bias_idxs]
```

Or drop the cache entirely: the gather is cheap compared to attention.

<!-- affected-models:manual -->
### Affected models

All runs use `torch.compile(backend="inductor", dynamic=True)` over (batch, seq, audio samples) =
(2, 32, 16000), (3, 40, 17600), (4, 56, 19200), one call per size, on a model built from the checkpoint's
config with reduced layers. "Graphs per call" is the number of graphs compiled on the 1st / 2nd / 3rd call
(anything after the 1st call is a recompile).

| model | config from | scenario · device | graphs per call |
|---|---|---|---|
| levit | `facebook/levit-128S` | eval · CPU | 2 / 2 / 0 |
| levit | | eval · CUDA | 2 / 2 / 0 |
| levit | | train · CPU / CUDA | 2 / 0 / 0 (cache not used in training) |
