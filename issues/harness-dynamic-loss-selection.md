<!-- Draft for: this repo (hf-models-graph-breaks harness, graph_breaks/dynamic.py). Not an upstream issue. -->

# `graph_breaks.dynamic` train scenario: `_loss()` picks a non-scalar `loss` or an integer tensor for audio codecs, so `.backward()` fails

### System info

- torch: `2.16.0a0+git57f02f8`, transformers: `5.24.0.dev0` (git `b7fe69cb19`)
- Affects CPU and CUDA runs alike (it's harness logic, not device-specific)

### Description

For `train_forward`, `dynamic.py` runs the compiled model and then calls `_loss(out).backward()`:

```python
def _loss(out):
    loss = getattr(out, "loss", None)
    if loss is None:
        first = out[0] if isinstance(out, (tuple, list)) else next(
            v for v in out.values() if isinstance(v, torch.Tensor))
        loss = first.float().sum()
    return loss
```

This assumes a scalar `out.loss`, or a first tensor field that carries gradients (true for LM logits).
Audio codec outputs break both assumptions:

| model | output fields in train mode (name, shape, dtype, requires_grad) | what `_loss` picks | result |
|---|---|---|---|
| dac | `loss (2,) f32 True`, `audio_values`, `quantized_representation`, `audio_codes (2,9,86) i64 False`, ... | `out.loss`, shape `(2,)` | `RuntimeError: grad can be implicitly created only for scalar outputs` |
| encodec | `audio_codes (1,2,2,75) i64 False`, `audio_values (2,1,24000) f32 True` | `audio_codes` | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` |
| mimi | `audio_codes (2,32,13) i64 False`, `audio_values (2,1,24000) f32 True` | `audio_codes` | same as encodec |
| wavtokenizer | `audio_values (2,1,24000) f32 True`, `audio_codes ... i64 False` | `audio_values` | OK (only because of field order) |

So the dynamic-shape **train** results for dac, encodec and mimi are errors from the harness, not from
PyTorch or transformers, and say nothing about recompiles or Inductor specializations in those models.
(The committed results from the previous torch build show the same errors for dac and encodec.) `check.py` (Dynamo-only
`explain`) is not affected: it doesn't call `.backward()`.

### Reproduction

```bash
python -m graph_breaks.dynamic --model dac --scenario train_forward --device cuda
python -m graph_breaks.dynamic --model encodec --scenario train_forward --device cuda
```

### Proposed fix

Reduce a non-scalar `loss`, and otherwise pick the first **floating-point tensor that requires grad**:

```python
def _loss(out):
    loss = getattr(out, "loss", None)
    if isinstance(loss, torch.Tensor) and loss.requires_grad:
        return loss.float().sum() if loss.ndim else loss
    values = out if isinstance(out, (tuple, list)) else out.values()
    for v in values:
        if isinstance(v, torch.Tensor) and v.is_floating_point() and v.requires_grad:
            return v.float().sum()
    raise RuntimeError("no differentiable tensor in model output")
```

then rerun `dac`, `encodec`, `mimi` (and `wavtokenizer` for consistency) with `--scenario train_forward`
on CPU and CUDA.

### Affected

dac, encodec, mimi (`train_forward` in `results_dynamic/` and `results_dynamic_cuda/`).
