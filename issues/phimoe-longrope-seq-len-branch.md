<!-- Draft for: huggingface/transformers. Once filed, put the URL in known_issues.yaml and set status: open -->

# `torch.compile`: PhiMoE rotary embedding branches on a tensor `seq_len` to pick LongRoPE short/long factors

### System info

- transformers: `5.24.0.dev0` (main, git `b7fe69cb19`)
- torch: `2.16.0a0+git57f02f8` (source build)
- Python 3.15, Linux; checked on CPU and on CUDA (B200)

### Description

`PhimoeRotaryEmbedding.forward` (`modeling_phimoe.py:81`):

```python
def forward(self, x, position_ids=None):
    mscale = None
    seq_len = torch.max(position_ids) + 1
    if self.config.rope_parameters["rope_type"] != "default" and seq_len:
        mscale = (
            self.config.rope_parameters["long_mscale"]
            if seq_len > self.config.rope_parameters["original_max_position_embeddings"]
            else self.config.rope_parameters["short_mscale"]
        )
```

`if ... and seq_len:` and `... if seq_len > original_max_position_embeddings else ...` are both
data-dependent branching on a 0-d tensor -> 2 graph breaks per forward, in **eval and train**.

Harness (config-only, reduced layers): phimoe 7 graphs / 6 breaks in eval_forward, eval_with_cache and
train_forward.

### Reproduction

`issues/repros/phimoe_longrope_seq_len_branch.py`:

```
[original]    graphs=3 breaks=2
    - generic_jump TensorVariable()
[torch.where] graphs=1 breaks=0
```

### Possible fix

Keep `mscale` as a tensor:

```python
seq_len = torch.max(position_ids) + 1
mscale = torch.where(seq_len > original_max, long_mscale, short_mscale)
```

(`if seq_len:` is always true for a non-empty input and can be dropped.) The factor selection for
`inv_freq` (short vs long factors) needs the same treatment; the shared
`longrope_frequency_update` in `modeling_rope_utils.py` has the same shape of check.

### Affected

phimoe; other LongRoPE users through `longrope_frequency_update`. Related: `dynamic-rope-data-dependent`.
