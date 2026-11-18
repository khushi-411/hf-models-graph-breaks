<!-- Draft for: huggingface/transformers. Once filed, put the URL in known_issues.yaml and set status: open -->

# `torch.compile`: wav2vec2-family SpecAugment (`_compute_mask_indices`) uses numpy RNG and breaks the graph in train mode

### System info

- transformers: `5.24.0.dev0` (main, git `b7fe69cb19`)
- torch: `2.16.0a0+git57f02f8` (source build)
- Python 3.15, Linux; checked on CPU and on CUDA (B200)

### Description

In train mode with `mask_time_prob > 0` (the default is `0.05`), `_mask_hidden_states` calls
`_compute_mask_indices`, which is written in numpy (`modeling_wav2vec2.py:141`, copied into hubert
`:675`, wavlm `:864`, ...):

```python
# epsilon is used for probabilistic rounding
epsilon = np.random.rand(1).item()

def compute_num_masked_span(input_length): ...
spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
...
spec_aug_mask_idx = np.random.choice(
    np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
)
...  # np.concatenate / np.array to build the index arrays, then the mask is turned into a tensor
```

Dynamo can't trace numpy RNG (`Data dependent operator`) and can't resume inside the nested function
("Failed to handle graph break gracefully"), so the whole masking runs eagerly, splitting the graph.

### Reproduction

`issues/repros/specaugment_numpy_mask.py` (real `Wav2Vec2Model`, 2 layers):

```
mask_time_prob: 0.05
[train]                    graphs=7 breaks=6 ['Data dependent operator', 'requires_grad_() intermediate leaked as output']
[train, mask_time_prob=0]  graphs=5 breaks=4
[eval]                     graphs=1 breaks=0
```

SpecAugment accounts for 2 of the 6 breaks; the rest come from other issues in the same path
(`requires-grad-intermediate`, `ctc-labels-max-check`).

### Possible fix

A torch implementation of the span masking (`torch.rand` / `torch.randint` on device, an `arange`
comparison to build spans). The numpy version could stay for `Wav2Vec2ForPreTraining` data collators that
call `_compute_mask_indices` directly.

### Affected

wav2vec2, hubert, wavlm (train_forward), and every copy of `_compute_mask_indices`
(data2vec-audio, sew, sew-d, unispeech, unispeech-sat, wav2vec2-conformer, wav2vec2-bert, ...).
