<!-- Draft for: pytorch/pytorch. Once filed, put the URL in known_issues.yaml and set status: open -->

# Dynamo: `t.requires_grad = True` on a view of a graph input graph-breaks ("requires_grad_() intermediate leaked as output")

### System info

- torch: `2.16.0a0+git9933f27` (source build)
- transformers: `5.22.1`
- Python 3.15, Linux, CPU

### Description

The wav2vec2-family feature encoder in transformers does this in training mode, so that
gradient checkpointing has an input that requires grad:

```python
hidden_states = input_values[:, None]
if self._requires_grad and self.training:
    hidden_states.requires_grad = True
```

Dynamo graph-breaks on the assignment with `requires_grad_() intermediate leaked as
output`. Hit in train mode by wav2vec2, hubert and wavlm. data2vec-audio, sew,
unispeech and wav2vec2-conformer copy the same code.

### Reproduction

```python
import torch

def f(x):
    h = x[:, None]
    h.requires_grad = True
    return (h * 2).sum()

e = torch._dynamo.explain(f)(torch.randn(2, 8))
print(e.graph_count, e.graph_break_count)  # 2 1
```

Full script: [repros/requires_grad_intermediate.py](repros/requires_grad_intermediate.py).

### Expected behavior

Ideally no graph break. Setting `requires_grad` on a view of a graph input that doesn't
require grad is equivalent to treating that view as a new leaf. If that's out of scope,
a hint pointing to a workaround would help, since the HF code only needs the flag for
gradient checkpointing.

### Note

This may be better fixed on the transformers side, e.g. skip it under
`is_torchdynamo_compiling()` or only do it when gradient checkpointing is enabled. It's
filed here because the pattern is common and the break message doesn't suggest a workaround.
