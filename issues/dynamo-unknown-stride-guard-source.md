<!-- Draft for: pytorch/pytorch. Once filed, put the URL in known_issues.yaml and set status: open -->

# Dynamo recompile reason: "unknown source hidden_states.stride()[0], please file a bug"

### System info

- torch: `2.16.0a0+git57f02f8` (source build, CUDA 12.8)
- transformers: `5.24.0.dev0` (git `b7fe69cb19`)
- Python 3.15, Linux; CPU and CUDA (B200)

### Description

Dynamo asks for this to be filed. With `torch.compile(dynamic=True)`, Inductor, the EnCodec conv wrapper
(`modeling_encodec.py:157`, `EncodecConv1d.forward` and `_pad1d`) is compiled as its own frame, because of the
graph breaks in `issues/hf/encodec-conv-params-buffers.md`. It then recompiles with:

```
Recompiling function forward in transformers/models/encodec/modeling_encodec.py:157
  - 4/2: hidden_states.stride()[0] == hidden_states.size()[1]  # (unknown source hidden_states.stride()[0], please file a bug)
  - 4/1: tensor 'hidden_states' stride mismatch at index 2. expected 1, actual 128
  - 4/0: tensor 'hidden_states' size mismatch at index 1. expected 1, actual 128
_pad1d:
  - hidden_states.stride()[0] == hidden_states.size()[2]*hidden_states.size()[1]  # (unknown source hidden_states.stride()[0], please file a bug)
```

The recompiles themselves are expected: the same frame is called from layers with different channel
counts and layouts. The bug is that a symbolic **stride** guard has no source attached. The guard is
still installed and correct, but the guard-failure message can't name where the symbol came from.

Seen only in `encodec` `eval_forward` (CPU and CUDA), graphs per size 33 / 4 / 0.

### Expected behavior

The stride symbol keeps its `TensorPropertySource(..., STRIDE, 0)` source, so the guard prints
`L['hidden_states'].stride()[0] == ...` like the size guards next to it.

Repro: not yet reduced. Likely needs a function that is first compiled with a static input, then recompiled
with automatic-dynamic strides (a non-contiguous input whose stride equals another size).

<!-- affected-models:manual -->
### Affected models

All runs use `torch.compile(backend="inductor", dynamic=True)` over (batch, seq, audio samples) =
(2, 32, 16000), (3, 40, 17600), (4, 56, 19200), on a model built from the checkpoint's config with reduced
layers. "Graphs per size" is the number of graphs compiled on the 1st / 2nd / 3rd size (anything after the
1st size is a recompile).

| model | config from | scenario · device | graphs per size |
|---|---|---|---|
| encodec | `facebook/encodec_24khz` | eval · CPU | 33 / 4 / 0 |
| encodec | | eval · CUDA | 33 / 4 / 0 |

encodec train is a harness error in both runs (`issues/harness/harness-dynamic-loss-selection.md`), so it
wasn't checked. mimi recompiles in the same `_pad1d` function but its guards all have a source.
