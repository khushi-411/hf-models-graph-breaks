<!-- Draft for: pytorch/pytorch. Once filed, put the URL in known_issues.yaml and set status: open -->

# CTC training can't compile in one graph: `F.ctc_loss` always graph-breaks and `torch.backends.cudnn.flags()` isn't traceable

### System info

- torch: `2.16.0a0+git9933f27` (source build)
- transformers: `5.22.1`
- Python 3.15, Linux, CPU

### Description

transformers' CTC loss (`loss/loss_ctc.py`, used by every `*ForCTC` model) is:

```python
labels_mask = (labels != ignore_index) & (labels != blank_token_id)
target_lengths = labels_mask.sum(-1)
flattened_targets = labels.masked_select(labels_mask)
with torch.backends.cudnn.flags(enabled=False):
    return nn.functional.ctc_loss(log_probs, flattened_targets, input_lengths, target_lengths, ...)
```

Three things in this snippet stop it compiling into one graph. Two of them are on the torch side.

1. **`torch.backends.cudnn.flags(...)` isn't traceable.** Dynamo treats it as a `torch.*`
   op and gives up with `torch.* op returned non-Tensor` (`_GeneratorContextManager`).
   The code before the `with` is compiled, and everything from the `with` on runs eagerly
   (see [repros/cudnn_flags_ctx.py](repros/cudnn_flags_ctx.py)). Closed issue #93586
   reported the same error message in a test.
2. **`F.ctc_loss` always graph-breaks, even with static shapes.** `aten._ctc_loss.Tensor`
   is treated as a dynamic-output-shape op, presumably because `log_alpha`'s last dim
   depends on `max(target_lengths)`. With `capture_dynamic_output_shape_ops=True` it
   fails with `Dynamic shape operator (no meta kernel)` instead. See
   [repros/ctc_loss_no_meta.py](repros/ctc_loss_no_meta.py).
3. `masked_select` is a dynamic-shape op. It goes away with
   `capture_dynamic_output_shape_ops=True`, so it's fine.

Affected in train mode (eval compiles to 1 graph): wav2vec2, hubert, wavlm, and by
extension data2vec-audio, sew, sew-d, unispeech(-sat), wav2vec2-conformer, wav2vec2-bert,
parakeet. On wav2vec2 the loss accounts for 2 of the 10 train-mode breaks. The rest are
on the HF side: numpy SpecAugment masking, LayerDrop, and a `labels.max()` check.

### Reproduction

```python
import torch, torch.nn.functional as F
T, N, C, S = 50, 2, 20, 10
lp = torch.randn(T, N, C).log_softmax(-1); t = torch.randint(1, C, (N, S))
il, tl = torch.full((N,), T), torch.full((N,), S)

graphs = []
def backend(gm, _): graphs.append(gm); return gm
torch.compile(lambda lp, t, il, tl: F.ctc_loss(lp * 1.0, t, il, tl) + 1, backend=backend)(lp, t, il, tl)
print(len(graphs))  # 2: ctc_loss itself runs eagerly between them
# graph break reason: "Dynamic shape operator", debug context: aten._ctc_loss.Tensor

def g(x):
    with torch.backends.cudnn.flags(enabled=False):
        return x.sum()
torch._dynamo.explain(g)(torch.randn(3))  # "torch.* op returned non-Tensor"
```

### Expected behavior

`ctc_loss` with static-shape inputs should trace into a single graph. With dynamic
targets it should trace once `capture_dynamic_output_shape_ops=True` is set.
`torch.backends.cudnn.flags` should be supported as a context manager, the way
`torch.backends.cuda.sdp_kernel` and similar flag managers are.

### Possible fix

- Register a fake/meta kernel for `aten._ctc_loss` (and `.Tensor`). Its
  `log_alpha` output size would be an unbacked symint (`2 * max(target_lengths) + 1`), or
  backed when `target_lengths` is a list. Alternatively, skip `log_alpha` when only the
  loss is needed.
- Handle `torch.backends.cudnn.flags` in Dynamo as a known context manager that
  sets and restores backend flags. Or treat it as a no-op inside the graph, since it only
  affects cuDNN kernel selection.
